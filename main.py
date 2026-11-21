"""Browser entry point for the Improved UAV multi-trip geohazard planner.

Runs INSIDE the web page (PyScript). Not started with "python main.py".

Phase C/D final:
- Algorithm toggle buttons (TS4GM always on, as reference).
- Comparison mode uses api.run_comparison_async(..., browser_mode=True),
  which runs each algorithm's individual runs one at a time and awaits
  asyncio.sleep(0) between them. That gives the browser event loop a real
  turn, so Chrome/Edge no longer show "This page isn't working" even with
  three or four algorithms selected.
- browser_mode=True also caps each algorithm's max_iters at 15 for the
  browser only. check_phase4.py (the CLI) is unaffected.
- "Show details for: <algorithm>" dropdown, so the existing summary, trip
  table, route map and convergence panels are reused for the selected
  algorithm.
- "Comparison summary (CSV)" download button.
- Single-algorithm path runs run_planner() in a background thread via
  asyncio.to_thread, so the tab stays responsive for TS4GM-only runs.
"""

from __future__ import annotations

import asyncio
import html
import time

try:
    from pyscript import document  # type: ignore
    from pyodide.ffi import create_proxy, to_js  # type: ignore
    from js import Blob, Object, URL  # type: ignore

    IN_BROWSER = True
except ImportError:  # normal Python on a PC: no browser available
    IN_BROWSER = False

from uavplanner import generate_synthetic, instance_to_csv_text
from uavplanner.api import (
    ALGORITHM_LABELS,
    ALGORITHMS,
    comparison_convergence_svg,
    comparison_summary_html,
    history_csv,
    routes_csv,
    run_comparison_async,
    run_planner,
    summary_html,
    trip_color,
    trip_table_html,
    trips_csv,
)

_state = {
    "instance": None,
    "result": None,
    "results": {},
    "selected": "ts4gm",
    "toggles": {"ts4gm": True, "lns": True, "grasp": False, "ga": False},
    "is_comparison": False,
}
_proxies = []


# ----------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------
def route_map_svg(instance, result) -> str:
    """Map of the depot and all trips, one color per trip."""
    width, height, pad, legend_w = 760, 460, 30, 120
    plot_w = width - legend_w

    depot_x, depot_y = float(instance.depot[0]), float(instance.depot[1])
    xs, ys = [depot_x], [depot_y]
    for t in result["trips"]:
        for sid in t["route"]:
            s = instance.site(sid)
            xs.append(float(s.x))
            ys.append(float(s.y))

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    scale = min((plot_w - 2 * pad) / span_x, (height - 2 * pad) / span_y)

    def px(x: float) -> float:
        return pad + (x - min_x) * scale

    def py(y: float) -> float:
        return height - pad - (y - min_y) * scale

    parts = []
    for k, t in enumerate(result["trips"]):
        pts = [(depot_x, depot_y)]
        for sid in t["route"]:
            s = instance.site(sid)
            pts.append((float(s.x), float(s.y)))
        pts.append((depot_x, depot_y))
        coords = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pts)
        parts.append(
            f'<polyline points="{coords}" fill="none" stroke="{trip_color(k)}" '
            f'stroke-width="1.6" stroke-opacity="0.85"/>'
        )
    for k, t in enumerate(result["trips"]):
        for sid in t["route"]:
            s = instance.site(sid)
            r = 2 + float(s.profit)
            parts.append(
                f'<circle cx="{px(float(s.x)):.1f}" cy="{py(float(s.y)):.1f}" '
                f'r="{r:.1f}" fill="{trip_color(k)}" stroke="#333" '
                f'stroke-width="0.5"/>'
            )
    parts.append(
        f'<rect x="{px(depot_x) - 6:.1f}" y="{py(depot_y) - 6:.1f}" '
        f'width="12" height="12" fill="#000"/>'
    )
    lx = plot_w + 10
    parts.append(
        f'<rect x="{lx}" y="14" width="10" height="10" fill="#000"/>'
        f'<text x="{lx + 16}" y="23" font-size="12">Depot</text>'
    )
    for k, t in enumerate(result["trips"]):
        y = 42 + 20 * k
        parts.append(
            f'<rect x="{lx}" y="{y - 9}" width="10" height="10" '
            f'fill="{trip_color(k)}"/>'
            f'<text x="{lx + 16}" y="{y}" font-size="12">Trip {t["rank"]}</text>'
        )
    parts.append(
        f'<text x="{lx}" y="{height - 12}" font-size="11" fill="#555">'
        f"Bigger dot = higher profit</text>"
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;background:#fff;'
        f'border:1px solid #ccc;font-family:sans-serif">' + "".join(parts) + "</svg>"
    )


def history_svg(result) -> str:
    """Line chart: best objective found at each iteration."""
    values = [float(v) for v in result["history"]]
    if len(values) < 2:
        return "<p>No convergence data was recorded for this run.</p>"

    width, height = 640, 300
    left, right, top, bottom = 60, 20, 20, 40
    initial = float(result["initial_objective"])

    y_min = min(values + [initial])
    y_max = max(values + [initial])
    if y_max - y_min < 1e-9:
        y_max = y_min + 1.0
    n = len(values)

    def px(i: int) -> float:
        return left + i / (n - 1) * (width - left - right)

    def py(v: float) -> float:
        return height - bottom - (v - y_min) / (y_max - y_min) * (height - top - bottom)

    coords = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(values))
    parts = [
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" '
        f'y2="{height - bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{py(initial):.1f}" x2="{width - right}" '
        f'y2="{py(initial):.1f}" stroke="#999" stroke-dasharray="5,4"/>',
        f'<text x="{width - right}" y="{py(initial) - 4:.1f}" font-size="11" '
        f'fill="#777" text-anchor="end">initial solution</text>',
        f'<polyline points="{coords}" fill="none" stroke="#1f77b4" stroke-width="2"/>',
        f'<text x="{left - 6}" y="{top + 4}" font-size="11" text-anchor="end">'
        f"{y_max:.2f}</text>",
        f'<text x="{left - 6}" y="{height - bottom}" font-size="11" '
        f'text-anchor="end">{y_min:.2f}</text>',
        f'<text x="{left}" y="{height - bottom + 16}" font-size="11">0</text>',
        f'<text x="{width - right}" y="{height - bottom + 16}" font-size="11" '
        f'text-anchor="end">{n - 1}</text>',
        f'<text x="{(left + width - right) / 2:.0f}" y="{height - 8}" '
        f'font-size="12" text-anchor="middle">Iteration</text>',
        f'<text x="14" y="{(top + height - bottom) / 2:.0f}" font-size="12" '
        f'text-anchor="middle" transform="rotate(-90 14 '
        f'{(top + height - bottom) / 2:.0f})">Best objective</text>',
    ]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;background:#fff;'
        f'border:1px solid #ccc;font-family:sans-serif">' + "".join(parts) + "</svg>"
    )


# ----------------------------------------------------------------------
# Page helpers
# ----------------------------------------------------------------------
def _el(element_id: str):
    return document.getElementById(element_id)


def _set_html(element_id: str, content: str) -> None:
    _el(element_id).innerHTML = content


def _status(message: str) -> None:
    _el("status").textContent = message


def _download(filename: str, text: str, mimetype: str = "text/csv") -> None:
    options = to_js({"type": mimetype}, dict_converter=Object.fromEntries)
    blob = Blob.new(to_js([text]), options)
    url = URL.createObjectURL(blob)
    link = document.createElement("a")
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)


def _set_downloads_enabled(enabled: bool) -> None:
    for element_id in ("dl-trips", "dl-routes", "dl-history"):
        _el(element_id).disabled = not enabled


def _show_instance_info() -> None:
    inst = _state["instance"]
    total_profit = sum(s.profit for s in inst.sites)
    _set_html(
        "data-info",
        f"<b>Data:</b> synthetic example with {inst.n} sites, "
        f"total profit {total_profit}, "
        f"T<sub>max</sub> = {inst.t_max:.0f} s per trip.",
    )


# ----------------------------------------------------------------------
# Toggles and selector
# ----------------------------------------------------------------------
def _refresh_toggle_styles() -> None:
    for name in ("lns", "grasp", "ga"):
        button = _el(f"toggle-{name}")
        if button is None:
            continue
        on = bool(_state["toggles"][name])
        button.className = "toggle-btn on" if on else "toggle-btn"
        button.setAttribute("aria-pressed", "true" if on else "false")


def _on_toggle(name: str) -> None:
    if name == "ts4gm":
        return
    if name not in _state["toggles"]:
        return
    _state["toggles"][name] = not _state["toggles"][name]
    _refresh_toggle_styles()
    _refresh_run_button_label()


def _selected_algorithms() -> list:
    order = ["ts4gm", "lns", "grasp", "ga"]
    return [n for n in order if _state["toggles"].get(n)]


def _refresh_run_button_label() -> None:
    names = _selected_algorithms()
    button = _el("run-btn")
    if button is None:
        return
    if len(names) >= 2:
        button.textContent = f"Run comparison ({len(names)} algorithms)"
    else:
        button.textContent = "Run planner"


def _show_algorithm_selector(names: list) -> None:
    container = _el("algorithm-selector-row")
    if container is None:
        return
    if len(names) <= 1:
        container.style.display = "none"
        return
    select = _el("algorithm-select")
    if select is None:
        return
    select.innerHTML = "".join(
        f'<option value="{name}">{ALGORITHM_LABELS.get(name, name)}</option>'
        for name in names
    )
    select.value = _state["selected"]
    container.style.display = "flex"


def _render_selected_algorithm() -> None:
    name = _state["selected"]
    result = _state["results"].get(name)
    if result is None:
        return
    inst = _state["instance"]

    _state["result"] = result

    _set_html("summary", summary_html(inst, result))
    _set_html("trip-table", trip_table_html(inst, result))
    _set_html("route-chart", route_map_svg(inst, result))
    _set_html("history-chart", history_svg(result))

    label = ALGORITHM_LABELS.get(name, name)
    _status(
        f"Showing details for {label}. "
        f"Objective {result['objective']:.4f}, "
        f"{result['n_trips']} trips, "
        f"{result['elapsed_s']:.1f} s."
    )


def _on_select_algorithm(event=None) -> None:
    select = _el("algorithm-select")
    if select is None:
        return
    name = str(select.value)
    if name in _state["results"]:
        _state["selected"] = name
        _render_selected_algorithm()


# ----------------------------------------------------------------------
# Background-thread execution for the single-algorithm path
# ----------------------------------------------------------------------
async def _await_in_thread(fn, *args):
    """Run `fn(*args)` in a background thread, awaiting it via a 50 ms poll
    loop that yields to the browser. If asyncio.to_thread is unavailable,
    falls back to a synchronous call.
    """
    to_thread = getattr(asyncio, "to_thread", None)
    started = time.perf_counter()

    if to_thread is None:
        _status("Running (sync fallback — the tab may freeze briefly)…")
        await asyncio.sleep(0.05)
        return fn(*args)

    task = asyncio.ensure_future(to_thread(fn, *args))
    last_update = 0.0

    while not task.done():
        await asyncio.sleep(0.05)
        now = time.perf_counter()
        if now - last_update >= 0.5:
            last_update = now
            elapsed = now - started
            _status(
                f"Running… {elapsed:.1f} s elapsed. "
                f"The page stays responsive; results will appear here."
            )

    return await task


async def _run_job() -> None:
    run_button = _el("run-btn")
    run_button.disabled = True
    _set_downloads_enabled(False)
    _status("Running… 0.0 s elapsed.")
    await asyncio.sleep(0.1)

    try:
        preset = str(_el("preset-select").value)
        seed0 = 1
        names = _selected_algorithms()
        inst = _state["instance"]

        if len(names) >= 2:
            results = await run_comparison_async(
                names, inst, preset=preset, seed0=seed0,
                yield_callback=asyncio.sleep,
                browser_mode=True,
            )

            _state["results"] = {r["algorithm"]: r for r in results}
            _state["selected"] = "ts4gm"
            _state["is_comparison"] = True

            _set_html("comparison-summary", comparison_summary_html(results))
            _set_html("comparison-convergence", comparison_convergence_svg(results))

            _show_algorithm_selector([r["algorithm"] for r in results])
            _render_selected_algorithm()
            _el("dl-comparison").disabled = False

            total = sum(r["elapsed_s"] for r in results)
            _status(
                f"Comparison done in {total:.1f} seconds for "
                f"{len(results)} algorithms. "
                f"Use the dropdown to switch the detail view."
            )
        else:
            result = await _await_in_thread(run_planner, inst, preset, seed0)

            _state["result"] = result
            _state["results"] = {result["algorithm"]: result}
            _state["selected"] = result["algorithm"]
            _state["is_comparison"] = False

            _set_html("comparison-summary", "")
            _set_html("comparison-convergence", "")
            _el("algorithm-selector-row").style.display = "none"

            _set_html("summary", summary_html(inst, result))
            _set_html("trip-table", trip_table_html(inst, result))
            _set_html("route-chart", route_map_svg(inst, result))
            _set_html("history-chart", history_svg(result))
            _el("dl-comparison").disabled = True

            _status(f"Done in {result['elapsed_s']:.1f} seconds.")

        _set_downloads_enabled(True)
    except Exception as exc:
        _status(f"Something went wrong: {html.escape(str(exc))}")
    finally:
        run_button.disabled = False


def _on_run_click(event=None) -> None:
    asyncio.ensure_future(_run_job())


def _on_download_trips(event=None) -> None:
    if _state["result"] is not None:
        _download("trips_summary.csv", trips_csv(_state["result"]))


def _on_download_routes(event=None) -> None:
    if _state["result"] is not None:
        _download("routes.csv", routes_csv(_state["instance"], _state["result"]))


def _on_download_history(event=None) -> None:
    if _state["result"] is not None:
        _download("convergence.csv", history_csv(_state["result"]))


def _on_download_comparison(event=None) -> None:
    if not _state["is_comparison"] or not _state["results"]:
        return
    names = list(_state["results"].keys())
    header = "algorithm,trips,objective,improvement,runtime_s,feasible,iterations_recorded"
    lines = [header]
    for name in names:
        r = _state["results"][name]
        lines.append(
            f'{name},{r["n_trips"]},{r["objective"]:.6f},{r["improvement"]:+.6f},'
            f'{r["elapsed_s"]:.3f},{str(r["feasible"]).lower()},{len(r["history"])}'
        )
    _download("comparison_summary.csv", "\n".join(lines) + "\n")


def _on_download_instance(event=None) -> None:
    _download("example_synthetic_88.csv", instance_to_csv_text(_state["instance"]))


def _connect(element_id: str, handler) -> None:
    element = _el(element_id)
    if element is None:
        return
    proxy = create_proxy(handler)
    _proxies.append(proxy)
    element.addEventListener("click", proxy)


def start() -> None:
    _state["instance"] = generate_synthetic(seed=1)
    _show_instance_info()

    for name in ("lns", "grasp", "ga"):
        _connect(f"toggle-{name}", lambda e=None, n=name: _on_toggle(n))
    _refresh_toggle_styles()
    _refresh_run_button_label()

    select = _el("algorithm-select")
    if select is not None:
        proxy = create_proxy(_on_select_algorithm)
        _proxies.append(proxy)
        select.addEventListener("change", proxy)

    _connect("run-btn", _on_run_click)
    _connect("dl-trips", _on_download_trips)
    _connect("dl-routes", _on_download_routes)
    _connect("dl-history", _on_download_history)
    _connect("dl-comparison", _on_download_comparison)
    _connect("dl-instance", _on_download_instance)

    _set_downloads_enabled(False)
    _el("dl-comparison").disabled = True
    _el("run-btn").disabled = False
    _status("Ready. Choose algorithms, a preset, and press Run.")


if IN_BROWSER:
    start()