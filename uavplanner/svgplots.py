"""Tiny SVG chart helpers (pure Python, no matplotlib, so the browser loads fast)."""

from __future__ import annotations

from typing import Dict, Sequence

PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#7f7f7f", "#bcbd22",
]


def _open(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" style="max-width:{width}px;background:#ffffff;'
        f'border:1px solid #cccccc" font-family="Arial, sans-serif" font-size="12">'
    )


def _text(x, y, s, anchor="start", size=12, weight="normal", fill="#222222"):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{s}</text>'
    )


# ---------------------------------------------------------------------------
# Map of sites and routes
# ---------------------------------------------------------------------------
def routes_svg(inst, solution, width: int = 640, height: int = 640,
               title: str = "Sites and routes") -> str:
    """Draw the depot, all sites (size = profit) and one coloured line per trip.

    If `solution` is empty, only the sites are drawn (used for the data preview).
    """
    pad = 40
    depot = (float(inst.depot[0]), float(inst.depot[1]))
    sites: Dict[int, tuple] = {
        i: (float(inst.site(i).x), float(inst.site(i).y)) for i in range(1, inst.n + 1)
    }

    xs = [depot[0]] + [p[0] for p in sites.values()]
    ys = [depot[1]] + [p[1] for p in sites.values()]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    span = max(maxx - minx, maxy - miny) or 1.0
    scale = (min(width, height) - 2 * pad) / span
    offx = (width - scale * (maxx - minx)) / 2
    offy = (height - scale * (maxy - miny)) / 2

    def px(p):
        return (offx + (p[0] - minx) * scale, height - offy - (p[1] - miny) * scale)

    trip_of = {}
    for k, trip in enumerate(solution):
        for sid in trip:
            trip_of[sid] = k

    parts = [_open(width, height), _text(width / 2, 20, title, "middle", 14, "bold")]

    # Route lines
    for k, trip in enumerate(solution):
        color = PALETTE[k % len(PALETTE)]
        pts = [px(depot)] + [px(sites[s]) for s in trip] + [px(depot)]
        pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        parts.append(
            f'<polyline points="{pts_str}" fill="none" stroke="{color}" '
            f'stroke-width="1.6" stroke-opacity="0.85"/>'
        )

    # Sites (radius grows with profit)
    for sid, p in sites.items():
        x, y = px(p)
        profit = float(inst.site(sid).profit)
        color = PALETTE[trip_of[sid] % len(PALETTE)] if sid in trip_of else "#555555"
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{2.0 + 1.2 * profit:.1f}" '
            f'fill="{color}" stroke="#ffffff" stroke-width="0.8">'
            f'<title>Site {sid} | profit {profit:g}</title></circle>'
        )

    # Depot
    dx, dy = px(depot)
    parts.append(
        f'<rect x="{dx - 6:.1f}" y="{dy - 6:.1f}" width="12" height="12" '
        f'fill="#000000" stroke="#ffffff" stroke-width="1"><title>Depot</title></rect>'
    )

    # Legend
    legend_h = 18 * (len(solution) + 1) + 8
    parts.append(
        f'<rect x="8" y="30" width="170" height="{legend_h}" fill="#ffffff" '
        f'fill-opacity="0.85" stroke="#dddddd"/>'
    )
    parts.append('<rect x="14" y="38" width="10" height="10" fill="#000000"/>')
    parts.append(_text(32, 47, "Depot"))
    for k, trip in enumerate(solution):
        color = PALETTE[k % len(PALETTE)]
        raw = sum(float(inst.site(s).profit) for s in trip)
        yy = 38 + 18 * (k + 1)
        parts.append(f'<rect x="14" y="{yy}" width="10" height="10" fill="{color}"/>')
        parts.append(_text(32, yy + 9, f"Trip {k + 1} ({len(trip)} sites, profit {raw:g})"))

    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Convergence line chart
# ---------------------------------------------------------------------------
def convergence_svg(values: Sequence[float], width: int = 640, height: int = 320,
                    title: str = "Best objective per iteration") -> str:
    left, right, top, bottom = 70, 20, 34, 42
    n = len(values)
    vmin, vmax = min(values), max(values)
    if vmax - vmin < 1e-9:
        vmin -= 1.0
        vmax += 1.0

    def X(i):
        return left + (i / max(n - 1, 1)) * (width - left - right)

    def Y(v):
        return top + (vmax - v) / (vmax - vmin) * (height - top - bottom)

    parts = [_open(width, height), _text(width / 2, 20, title, "middle", 14, "bold")]

    # Y grid and labels
    for t in range(5):
        v = vmin + (vmax - vmin) * t / 4
        y = Y(v)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            f'stroke="#eeeeee"/>'
        )
        parts.append(_text(left - 6, y + 4, f"{v:.2f}", "end", 11))

    # Axes
    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#444444"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" '
        f'y2="{height - bottom}" stroke="#444444"/>'
    )

    # X labels
    for i in sorted({0, (n - 1) // 2, n - 1}):
        parts.append(_text(X(i), height - bottom + 16, str(i), "middle", 11))
    parts.append(_text((left + width - right) / 2, height - 6, "Iteration", "middle", 12))

    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(values))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#1f77b4" stroke-width="2"/>')
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Grouped bars: raw vs discounted profit per trip
# ---------------------------------------------------------------------------
def profit_bars_svg(raw: Sequence[float], discounted: Sequence[float],
                    width: int = 640, height: int = 320,
                    title: str = "Profit per trip (raw vs discounted)") -> str:
    left, right, top, bottom = 50, 20, 44, 40
    k = len(raw)
    vmax = max(max(raw), max(discounted), 1.0)
    plot_h = height - top - bottom
    group_w = (width - left - right) / max(k, 1)
    bar_w = group_w * 0.32

    parts = [_open(width, height), _text(width / 2, 20, title, "middle", 14, "bold")]
    parts.append(f'<rect x="{left}" y="28" width="10" height="10" fill="#9ecae1"/>')
    parts.append(_text(left + 16, 37, "Raw profit"))
    parts.append(f'<rect x="{left + 100}" y="28" width="10" height="10" fill="#1f77b4"/>')
    parts.append(_text(left + 116, 37, "Discounted profit"))

    base = height - bottom
    parts.append(
        f'<line x1="{left}" y1="{base}" x2="{width - right}" y2="{base}" stroke="#444444"/>'
    )

    for i in range(k):
        x0 = left + i * group_w + group_w * 0.15
        for j, (val, color) in enumerate(((raw[i], "#9ecae1"), (discounted[i], "#1f77b4"))):
            h = val / vmax * plot_h
            x = x0 + j * (bar_w + 2)
            parts.append(
                f'<rect x="{x:.1f}" y="{base - h:.1f}" width="{bar_w:.1f}" '
                f'height="{h:.1f}" fill="{color}"/>'
            )
            parts.append(_text(x + bar_w / 2, base - h - 4, f"{val:.1f}", "middle", 10))
        parts.append(_text(x0 + bar_w + 1, base + 16, f"Trip {i + 1}", "middle", 11))

    parts.append("</svg>")
    return "".join(parts)