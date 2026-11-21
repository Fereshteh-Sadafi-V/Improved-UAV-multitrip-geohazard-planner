"""One-call interface used by the web page (and by check_phase*.py).

Public functions used by the browser and the check scripts:

Existing (unchanged):
    run_planner(instance, preset, seed0)
    summary_html, trip_table_html
    trips_csv, routes_csv, history_csv
    trip_color
    PRESETS
    run_algorithm(name, instance, preset, seed0)
    run_comparison(names, instance, preset, seed0)            -- synchronous
    comparison_summary_html(results)
    comparison_convergence_svg(results)
    ALGORITHMS, ALGORITHM_LABELS

Phase C / D additions:
    run_comparison_async(names, instance, preset, seed0,
                         yield_callback=None, browser_mode=False)
        -- async, cooperative; used by the browser only.
        Calls each algorithm's individual runs one at a time, awaiting
        yield_callback(0) between each, so the browser event loop gets a
        real turn between algorithms and between each algorithm's n_runs.
        When browser_mode=True, per-algorithm max_iters is capped at 15
        (BROWSER_MAX_ITERS) for responsiveness on slow laptops. The CLI
        never sets browser_mode, so the paper-style behaviour (and the
        verified results from check_phase4.py) are unchanged.

Nothing else changes. ts4gm.py, lns.py, grasp.py, ga.py and algo_common.py
are not modified; differences in their return shapes are normalised here.
"""

from __future__ import annotations

import csv
import html
import io
import time
from typing import Any, Callable, Dict, List, Sequence

from .algo_common import EPS
from .ga import run_ga
from .grasp import run_grasp
from .initial_paper import build_paper_initial_solution
from .lns import run_lns
from .model import check_solution, objective, trip_time
from .ordering import sort_trips_by_profit
from .ts4gm import run_ts4gm

# Colors used for the small colored squares in the trip table.
_TRIP_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# Browser-only cap. See run_comparison_async.
BROWSER_MAX_ITERS = 15


def trip_color(index: int) -> str:
    """Return a color for trip number `index` (0 = first trip)."""
    return _TRIP_COLORS[index % len(_TRIP_COLORS)]


PRESETS: Dict[str, Dict[str, int]] = {
    "quick": {
        "n_runs": 1, "max_iters": 30, "z_in": 15,
        "grasp_ls_iters": 2, "grasp_max_iters": 10,
        "ga_pop_size": 30, "ga_z_in": 10,
    },
    "standard": {
        "n_runs": 3, "max_iters": 100, "z_in": 30,
        "grasp_ls_iters": 3, "grasp_max_iters": 40,
        "ga_pop_size": 80, "ga_z_in": 30,
    },
    "paper": {
        "n_runs": 5, "max_iters": 200, "z_in": 50,
        "grasp_ls_iters": 5, "grasp_max_iters": 200,
        "ga_pop_size": 300, "ga_z_in": 50,
    },
}


# =========================================================================
# Existing TS4GM-only entry point (unchanged behaviour)
# =========================================================================
def run_planner(instance, preset: str = "quick", seed0: int = 1) -> Dict[str, Any]:
    """Run TS4GM with a preset and return a plain dictionary of results."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Choose from {list(PRESETS)}.")
    p = PRESETS[preset]

    initial = build_paper_initial_solution(instance)
    initial_obj = objective(instance, initial)

    started = time.perf_counter()
    from .ts4gm import run_ts4gm_trials
    res = run_ts4gm_trials(
        instance,
        n_runs=p["n_runs"],
        seed0=seed0,
        max_iters=p["max_iters"],
        z_in=p["z_in"],
        start_method="paper",
    )
    elapsed = time.perf_counter() - started

    solution = sort_trips_by_profit(instance, res.best_solution)
    problems = check_solution(instance, solution)
    score = objective(instance, solution)

    trips = []
    for rank, trip in enumerate(solution, start=1):
        raw = sum(instance.site(i).profit for i in trip)
        trips.append(
            {
                "rank": rank,
                "n_sites": len(trip),
                "time_s": trip_time(instance, trip),
                "raw_profit": raw,
                "discounted_profit": raw * instance.rho ** (rank - 1),
                "route": list(trip),
            }
        )

    return {
        "algorithm": "ts4gm",
        "seed": seed0,
        "preset": preset,
        "solution": solution,
        "trips": trips,
        "objective": score,
        "initial_objective": initial_obj,
        "improvement": score - initial_obj,
        "n_trips": len(solution),
        "elapsed_s": elapsed,
        "history": list(getattr(res, "history_best", [])),
        "problems": problems,
        "feasible": problems == [],
    }


def _fmt_hms(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def summary_html(instance, result: Dict[str, Any]) -> str:
    """Short HTML box with the headline numbers."""
    if result["feasible"]:
        status = '<span style="color:#2ca02c;font-weight:bold">Feasible</span>'
    else:
        status = '<span style="color:#d62728;font-weight:bold">NOT feasible</span>'

    rows = [
        ("Preset", html.escape(str(result["preset"]))),
        ("Solution status", status),
        ("Number of trips", str(result["n_trips"])),
        ("Initial objective", f'{result["initial_objective"]:.4f}'),
        ("Final objective", f'<b>{result["objective"]:.4f}</b>'),
        ("Improvement", f'{result["improvement"]:+.4f}'),
        ("Run time (seconds)", f'{result["elapsed_s"]:.1f}'),
    ]
    body = "".join(
        f'<tr><td style="padding:3px 12px 3px 0">{label}</td>'
        f'<td style="padding:3px 0">{value}</td></tr>'
        for label, value in rows
    )

    problems_html = ""
    if result["problems"]:
        items = "".join(f"<li>{html.escape(str(x))}</li>" for x in result["problems"])
        problems_html = (
            '<p style="color:#d62728;margin:8px 0 0 0">Problems found:</p>'
            f"<ul>{items}</ul>"
        )

    return (
        '<div style="border:1px solid #ccc;padding:10px;background:#fafafa;'
        'font-family:sans-serif;font-size:14px">'
        f"<table>{body}</table>{problems_html}</div>"
    )


def trip_table_html(instance, result: Dict[str, Any]) -> str:
    """HTML table with one row per trip and a totals row."""
    head_cells = [
        "Trip", "Sites", "Time (s)", "Time (h:mm:ss)",
        "Raw profit", "Discounted profit", "Route (site IDs)",
    ]
    head = "".join(
        f'<th style="border:1px solid #ccc;padding:4px 8px;background:#eee;'
        f'text-align:left">{c}</th>'
        for c in head_cells
    )

    cell = 'style="border:1px solid #ccc;padding:4px 8px;vertical-align:top"'
    rows = []
    total_sites = total_time = total_raw = total_disc = 0.0
    for k, t in enumerate(result["trips"]):
        swatch = (
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background:{trip_color(k)};margin-right:6px"></span>'
        )
        route = "0 &rarr; " + " &rarr; ".join(str(s) for s in t["route"]) + " &rarr; 0"
        rows.append(
            "<tr>"
            f"<td {cell}>{swatch}{t['rank']}</td>"
            f"<td {cell}>{t['n_sites']}</td>"
            f"<td {cell}>{t['time_s']:.1f}</td>"
            f"<td {cell}>{_fmt_hms(t['time_s'])}</td>"
            f"<td {cell}>{t['raw_profit']}</td>"
            f"<td {cell}>{t['discounted_profit']:.4f}</td>"
            f"<td {cell}>{route}</td>"
            "</tr>"
        )
        total_sites += t["n_sites"]
        total_time += t["time_s"]
        total_raw += t["raw_profit"]
        total_disc += t["discounted_profit"]

    rows.append(
        "<tr style='font-weight:bold'>"
        f"<td {cell}>Total</td>"
        f"<td {cell}>{int(total_sites)}</td>"
        f"<td {cell}>{total_time:.1f}</td>"
        f"<td {cell}>{_fmt_hms(total_time)}</td>"
        f"<td {cell}>{total_raw}</td>"
        f"<td {cell}>{total_disc:.4f}</td>"
        f"<td {cell}></td>"
        "</tr>"
    )

    return (
        '<table style="border-collapse:collapse;font-family:sans-serif;font-size:13px">'
        f"<tr>{head}</tr>{''.join(rows)}</table>"
    )


def trips_csv(result: Dict[str, Any]) -> str:
    """CSV text: one row per trip."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(
        ["trip", "n_sites", "time_s", "raw_profit", "discounted_profit", "route"]
    )
    for t in result["trips"]:
        route = "0 " + " ".join(str(s) for s in t["route"]) + " 0"
        w.writerow(
            [
                t["rank"],
                t["n_sites"],
                f'{t["time_s"]:.3f}',
                t["raw_profit"],
                f'{t["discounted_profit"]:.6f}',
                route,
            ]
        )
    w.writerow([])
    w.writerow(["total_objective", f'{result["objective"]:.6f}'])
    return buf.getvalue()


def routes_csv(instance, result: Dict[str, Any]) -> str:
    """CSV text: one row per stop (depot included), in flight order."""
    depot_x = float(instance.depot[0])
    depot_y = float(instance.depot[1])

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["trip", "position", "site_id", "x", "y", "profit"])
    for t in result["trips"]:
        trip_no = t["rank"]
        w.writerow([trip_no, 0, 0, depot_x, depot_y, 0])
        for pos, sid in enumerate(t["route"], start=1):
            s = instance.site(sid)
            w.writerow([trip_no, pos, sid, float(s.x), float(s.y), s.profit])
        w.writerow([trip_no, len(t["route"]) + 1, 0, depot_x, depot_y, 0])
    return buf.getvalue()


def history_csv(result: Dict[str, Any]) -> str:
    """CSV text: best objective found at each iteration."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["iteration", "best_objective"])
    for i, v in enumerate(result["history"]):
        w.writerow([i, f"{float(v):.6f}"])
    return buf.getvalue()


# =========================================================================
# Unified dispatcher, comparison view, comparison HTML
# =========================================================================
def _result_to_dict(
    result: Any,
    algorithm: str,
    instance,
    initial_obj: float,
    elapsed: float,
    preset: str,
    seed0: int,
) -> Dict[str, Any]:
    """Turn any of TS4GMResult / LNSResult / AlgoResult into the same dict
    shape that run_planner returns. All four algorithms go through here.
    """
    solution = sort_trips_by_profit(instance, result.best_solution)
    problems = check_solution(instance, solution)
    score = objective(instance, solution)

    trips = []
    for rank, trip in enumerate(solution, start=1):
        raw = sum(instance.site(i).profit for i in trip)
        trips.append(
            {
                "rank": rank,
                "n_sites": len(trip),
                "time_s": trip_time(instance, trip),
                "raw_profit": raw,
                "discounted_profit": raw * instance.rho ** (rank - 1),
                "route": list(trip),
            }
        )

    return {
        "algorithm": algorithm,
        "seed": seed0,
        "preset": preset,
        "solution": solution,
        "trips": trips,
        "objective": score,
        "initial_objective": initial_obj,
        "improvement": score - initial_obj,
        "n_trips": len(solution),
        "elapsed_s": elapsed,
        "history": list(getattr(result, "history_best", [])),
        "problems": problems,
        "feasible": problems == [],
    }


# -------------------------------------------------------------------------
# Synchronous adapters (used by the CLI and the existing sync run_comparison)
# -------------------------------------------------------------------------
def _run_ts4gm_adapter(instance, preset: str, seed0: int) -> Dict[str, Any]:
    p = PRESETS[preset]
    initial_obj = objective(instance, build_paper_initial_solution(instance))
    started = time.perf_counter()
    from .ts4gm import run_ts4gm_trials
    res = run_ts4gm_trials(
        instance,
        n_runs=p["n_runs"], seed0=seed0,
        max_iters=p["max_iters"], z_in=p["z_in"],
        start_method="paper",
    )
    elapsed = time.perf_counter() - started
    return _result_to_dict(res, "ts4gm", instance, initial_obj, elapsed, preset, seed0)


def _run_lns_adapter(instance, preset: str, seed0: int) -> Dict[str, Any]:
    p = PRESETS[preset]
    initial_obj = objective(instance, build_paper_initial_solution(instance))
    started = time.perf_counter()
    from .lns import run_lns_trials
    res = run_lns_trials(
        instance,
        n_runs=p["n_runs"], seed0=seed0,
        max_iters=p["max_iters"], z_in=p["z_in"],
        start_method="paper",
    )
    elapsed = time.perf_counter() - started
    return _result_to_dict(res, "lns", instance, initial_obj, elapsed, preset, seed0)


def _run_grasp_adapter(instance, preset: str, seed0: int) -> Dict[str, Any]:
    p = PRESETS[preset]
    initial_obj = objective(instance, build_paper_initial_solution(instance))
    started = time.perf_counter()
    from .grasp import run_grasp_trials
    res = run_grasp_trials(
        instance,
        n_runs=p["n_runs"], seed0=seed0,
        max_iters=p.get("grasp_max_iters", p["max_iters"]),
        ls_iters=p["grasp_ls_iters"],
        start_method="paper",
    )
    elapsed = time.perf_counter() - started
    return _result_to_dict(res, "grasp", instance, initial_obj, elapsed, preset, seed0)


def _run_ga_adapter(instance, preset: str, seed0: int) -> Dict[str, Any]:
    p = PRESETS[preset]
    initial_obj = objective(instance, build_paper_initial_solution(instance))
    started = time.perf_counter()
    from .ga import run_ga_trials
    res = run_ga_trials(
        instance,
        n_runs=p["n_runs"], seed0=seed0,
        max_iters=p["max_iters"], z_in=p["ga_z_in"],
        pop_size=p["ga_pop_size"],
        start_method="paper",
    )
    elapsed = time.perf_counter() - started
    return _result_to_dict(res, "ga", instance, initial_obj, elapsed, preset, seed0)


ALGORITHMS: Dict[str, Callable[[Any, str, int], Dict[str, Any]]] = {
    "ts4gm": _run_ts4gm_adapter,
    "lns":   _run_lns_adapter,
    "grasp": _run_grasp_adapter,
    "ga":    _run_ga_adapter,
}

ALGORITHM_LABELS: Dict[str, str] = {
    "ts4gm": "TS4GM",
    "lns":   "LNS",
    "grasp": "GRASP",
    "ga":    "GA",
}


def run_algorithm(
    name: str,
    instance,
    preset: str = "quick",
    seed0: int = 1,
) -> Dict[str, Any]:
    """Run one algorithm by name and return the same dict shape as run_planner."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Choose from {list(PRESETS)}.")
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{name}'. Choose from {list(ALGORITHMS)}.")
    return ALGORITHMS[name](instance, preset, seed0)


def run_comparison(
    names: Sequence[str],
    instance,
    preset: str = "quick",
    seed0: int = 1,
) -> List[Dict[str, Any]]:
    """Run several algorithms synchronously (used by check_phase4.py).

    The browser uses run_comparison_async below; this synchronous version
    is kept byte-identical for the CLI checks.
    """
    if len(names) == 0:
        raise ValueError("run_comparison needs at least one algorithm name.")
    unknown = [n for n in names if n not in ALGORITHMS]
    if unknown:
        raise ValueError(
            f"Unknown algorithm(s) {unknown}. Choose from {list(ALGORITHMS)}."
        )
    return [run_algorithm(n, instance, preset=preset, seed0=seed0) for n in names]


# -------------------------------------------------------------------------
# Async / cooperative path (used by the browser only)
# -------------------------------------------------------------------------
def _browser_max_iters(preset: str, browser_mode: bool) -> int:
    """Per-algorithm max_iters to use inside the async path.

    With browser_mode=True, cap at BROWSER_MAX_ITERS so each algorithm's
    synchronous burst is short enough that the browser watchdog does not
    fire, even with three or four algorithms selected. The CLI never sets
    browser_mode, so the paper-style behaviour is unchanged.
    """
    base = PRESETS[preset]["max_iters"]
    if browser_mode:
        return min(base, BROWSER_MAX_ITERS)
    return base


async def _run_ts4gm_async(instance, preset, seed0, yield_callback, browser_mode):
    p = PRESETS[preset]
    max_iters = _browser_max_iters(preset, browser_mode)
    initial_obj = objective(instance, build_paper_initial_solution(instance))
    started = time.perf_counter()
    best_result = None
    for k in range(p["n_runs"]):
        r = run_ts4gm(
            instance, seed=seed0 + k,
            max_iters=max_iters, z_in=p["z_in"],
            start_method="paper",
        )
        if best_result is None or r.best_objective > best_result.best_objective + EPS:
            best_result = r
        await yield_callback(0)
    elapsed = time.perf_counter() - started
    return _result_to_dict(best_result, "ts4gm", instance, initial_obj, elapsed, preset, seed0)


async def _run_lns_async(instance, preset, seed0, yield_callback, browser_mode):
    p = PRESETS[preset]
    max_iters = _browser_max_iters(preset, browser_mode)
    initial_obj = objective(instance, build_paper_initial_solution(instance))
    started = time.perf_counter()
    best_result = None
    for k in range(p["n_runs"]):
        r = run_lns(
            instance, seed=seed0 + k,
            max_iters=max_iters, z_in=p["z_in"],
            start_method="paper",
        )
        if best_result is None or r.best_objective > best_result.best_objective + EPS:
            best_result = r
        await yield_callback(0)
    elapsed = time.perf_counter() - started
    return _result_to_dict(best_result, "lns", instance, initial_obj, elapsed, preset, seed0)


async def _run_grasp_async(instance, preset, seed0, yield_callback, browser_mode):
    p = PRESETS[preset]
    # GRASP already has its own quick cap; apply browser_mode on top of it.
    base = p.get("grasp_max_iters", p["max_iters"])
    max_iters = min(base, BROWSER_MAX_ITERS) if browser_mode else base
    initial_obj = objective(instance, build_paper_initial_solution(instance))
    started = time.perf_counter()
    best_result = None
    for k in range(p["n_runs"]):
        r = run_grasp(
            instance, seed=seed0 + k,
            max_iters=max_iters,
            ls_iters=p["grasp_ls_iters"],
            start_method="paper",
        )
        if best_result is None or r.best_objective > best_result.best_objective + EPS:
            best_result = r
        await yield_callback(0)
    elapsed = time.perf_counter() - started
    return _result_to_dict(best_result, "grasp", instance, initial_obj, elapsed, preset, seed0)


async def _run_ga_async(instance, preset, seed0, yield_callback, browser_mode):
    p = PRESETS[preset]
    max_iters = _browser_max_iters(preset, browser_mode)
    initial_obj = objective(instance, build_paper_initial_solution(instance))
    started = time.perf_counter()
    best_result = None
    for k in range(p["n_runs"]):
        r = run_ga(
            instance, seed=seed0 + k,
            max_iters=max_iters, z_in=p["ga_z_in"],
            pop_size=p["ga_pop_size"],
            start_method="paper",
        )
        if best_result is None or r.best_objective > best_result.best_objective + EPS:
            best_result = r
        await yield_callback(0)
    elapsed = time.perf_counter() - started
    return _result_to_dict(best_result, "ga", instance, initial_obj, elapsed, preset, seed0)


_ASYNC_RUNNERS = {
    "ts4gm": _run_ts4gm_async,
    "lns":   _run_lns_async,
    "grasp": _run_grasp_async,
    "ga":    _run_ga_async,
}


async def run_comparison_async(
    names: Sequence[str],
    instance,
    preset: str = "quick",
    seed0: int = 1,
    yield_callback=None,
    browser_mode: bool = False,
) -> List[Dict[str, Any]]:
    """Async cooperative comparison. Used by the browser.

    Between algorithms, and between each algorithm's independent runs, it
    awaits `yield_callback(0)`. In Pyodide this is typically asyncio.sleep,
    which keeps the browser responsive and prevents Chrome/Edge from killing
    the tab while a long algorithm runs.

    If `browser_mode` is True, per-algorithm max_iters is capped at
    BROWSER_MAX_ITERS (15). This is the browser-only speed cap; the CLI
    never sets it, so the paper-style behaviour from check_phase4.py is
    unchanged.
    """
    if len(names) == 0:
        raise ValueError("run_comparison_async needs at least one algorithm name.")
    unknown = [n for n in names if n not in ALGORITHMS]
    if unknown:
        raise ValueError(
            f"Unknown algorithm(s) {unknown}. Choose from {list(ALGORITHMS)}."
        )
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Choose from {list(PRESETS)}.")

    if yield_callback is None:
        return run_comparison(names, instance, preset=preset, seed0=seed0)

    out: List[Dict[str, Any]] = []
    for name in names:
        runner = _ASYNC_RUNNERS[name]
        out.append(
            await runner(instance, preset, seed0, yield_callback, browser_mode)
        )
        await yield_callback(0)
    return out


# -------------------------------------------------------------------------
# Comparison HTML: summary table + convergence chart (Fig. 5 equivalent)
# -------------------------------------------------------------------------
def comparison_summary_html(results: Sequence[Dict[str, Any]]) -> str:
    """One row per algorithm, plus the initial-solution baseline row."""
    if not results:
        return "<p>No algorithms were run.</p>"

    head_cells = [
        "Algorithm", "Status", "Trips", "Objective",
        "Improvement", "Runtime (s)",
    ]
    head = "".join(
        f'<th style="border:1px solid #ccc;padding:4px 8px;background:#eee;'
        f'text-align:left">{c}</th>'
        for c in head_cells
    )

    cell = 'style="border:1px solid #ccc;padding:4px 8px"'
    rows = []
    initial = results[0]["initial_objective"]
    rows.append(
        "<tr style='background:#fafafa;color:#555'>"
        f"<td {cell}><i>Paper-style start (baseline)</i></td>"
        f"<td {cell}><span style='color:#2ca02c'>feasible</span></td>"
        f"<td {cell}>&mdash;</td>"
        f"<td {cell}>{initial:.4f}</td>"
        f"<td {cell}>0.0000</td>"
        f"<td {cell}>&mdash;</td>"
        "</tr>"
    )

    best_obj = max(r["objective"] for r in results)
    for r in results:
        if r["feasible"]:
            status = '<span style="color:#2ca02c">feasible</span>'
        else:
            status = '<span style="color:#d62728;font-weight:bold">NOT feasible</span>'
        obj_cell = f'{r["objective"]:.4f}'
        if abs(r["objective"] - best_obj) < 1e-9:
            obj_cell = f"<b>{obj_cell}</b>"
        rows.append(
            "<tr>"
            f"<td {cell}>{html.escape(ALGORITHM_LABELS.get(r['algorithm'], r['algorithm']))}</td>"
            f"<td {cell}>{status}</td>"
            f"<td {cell}>{r['n_trips']}</td>"
            f"<td {cell}>{obj_cell}</td>"
            f"<td {cell}>{r['improvement']:+.4f}</td>"
            f"<td {cell}>{r['elapsed_s']:.1f}</td>"
            "</tr>"
        )

    return (
        '<table style="border-collapse:collapse;font-family:sans-serif;font-size:13px">'
        f"<tr>{head}</tr>{''.join(rows)}</table>"
    )


def comparison_convergence_svg(
    results: Sequence[Dict[str, Any]],
    width: int = 720,
    height: int = 360,
) -> str:
    """One polyline per algorithm (best-so-far objective per iteration).

    Equivalent to Fig. 5 of Wang & Wen (2026).
    """
    if not results:
        return "<p>No algorithms were run.</p>"

    max_len = max(len(r["history"]) for r in results)
    all_values = [v for r in results for v in r["history"]]
    initial = results[0]["initial_objective"]
    all_values.append(initial)

    left, right, top, bottom = 70, 170, 34, 46
    vmin = min(all_values)
    vmax = max(all_values)
    if vmax - vmin < 1e-9:
        vmin -= 1.0
        vmax += 1.0

    def X(i: int) -> float:
        return left + (i / max(max_len - 1, 1)) * (width - left - right)

    def Y(v: float) -> float:
        return top + (vmax - v) / (vmax - vmin) * (height - top - bottom)

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" style="max-width:{width}px;background:#fff;'
        f'border:1px solid #ccc;font-family:sans-serif;font-size:12">'
    ]

    parts.append(
        f'<text x="{width / 2}" y="20" text-anchor="middle" font-size="14" '
        f'font-weight="bold">Convergence: best objective per iteration</text>'
    )

    for t in range(5):
        v = vmin + (vmax - vmin) * t / 4
        y = Y(v)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            f'stroke="#eeeeee"/>'
        )
        parts.append(
            f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11">{v:.2f}</text>'
        )

    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" '
        f'stroke="#444"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" '
        f'y2="{height - bottom}" stroke="#444"/>'
    )

    for i in sorted({0, max_len // 2, max_len - 1}):
        parts.append(
            f'<text x="{X(i):.1f}" y="{height - bottom + 16}" '
            f'text-anchor="middle" font-size="11">{i}</text>'
        )
    parts.append(
        f'<text x="{(left + width - right) / 2:.0f}" y="{height - 6}" '
        f'text-anchor="middle" font-size="12">Iteration</text>'
    )

    parts.append(
        f'<line x1="{left}" y1="{Y(initial):.1f}" x2="{width - right}" '
        f'y2="{Y(initial):.1f}" stroke="#999" stroke-dasharray="5,4"/>'
    )
    parts.append(
        f'<text x="{width - right - 4}" y="{Y(initial) - 4:.1f}" '
        f'text-anchor="end" font-size="11" fill="#777">'
        f'initial solution {initial:.2f}</text>'
    )

    lx = width - right + 10
    for k, r in enumerate(results):
        color = trip_color(k)
        hist = r["history"]
        if len(hist) >= 2:
            pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(hist))
            parts.append(
                f'<polyline points="{pts}" fill="none" stroke="{color}" '
                f'stroke-width="2"/>'
            )
        else:
            parts.append(
                f'<circle cx="{X(0):.1f}" cy="{Y(hist[0]):.1f}" r="3" '
                f'fill="{color}"/>'
            )
        label = ALGORITHM_LABELS.get(r["algorithm"], r["algorithm"])
        yy = top + 16 + 22 * k
        parts.append(
            f'<rect x="{lx}" y="{yy - 10}" width="12" height="12" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{lx + 18}" y="{yy}" font-size="12">'
            f'{html.escape(label)}: {r["objective"]:.3f}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)