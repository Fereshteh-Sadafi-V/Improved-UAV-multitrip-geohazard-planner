"""Phase 2 checks.

Run from the project folder:
    python check_phase2.py
"""

import csv
import math
import time
from pathlib import Path

from uavplanner import (
    Instance,
    Site,
    check_solution,
    generate_synthetic,
    objective,
    profit_by_trip,
    trip_time,
)
from uavplanner.initial import build_initial_solution
from uavplanner.local_search import improve_trip, or_opt, two_opt

results = []


def report(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


# ------------------------------------------------------------
# 1. Small known route: a square with a crossing.
# ------------------------------------------------------------
small = Instance(
    depot=(0.0, 0.0),
    sites=[
        Site(1, 10.0, 0.0, 1.0, 0.0),
        Site(2, 10.0, 10.0, 1.0, 0.0),
        Site(3, 0.0, 10.0, 1.0, 0.0),
    ],
    t_max=100.0,
    speed=1.0,
)

crossing_route = [1, 3, 2]
shorter_route = two_opt(small, crossing_route)

report(
    "2-opt removes the square crossing",
    math.isclose(trip_time(small, shorter_route), 40.0, abs_tol=1e-9),
    f"(time = {trip_time(small, shorter_route):.3f} s)",
)
report(
    "2-opt preserves all sites",
    sorted(shorter_route) == sorted(crossing_route),
)
report(
    "2-opt does not modify the input list",
    crossing_route == [1, 3, 2],
)

relocated_route = or_opt(small, crossing_route)
report(
    "Or-opt preserves all sites",
    sorted(relocated_route) == sorted(crossing_route),
)
report(
    "Or-opt does not increase route time",
    trip_time(small, relocated_route)
    <= trip_time(small, crossing_route) + 1e-9,
)

# ------------------------------------------------------------
# 2. Larger route: combined local improvement.
# ------------------------------------------------------------
inst = generate_synthetic(seed=1)
sample_route = list(range(1, 13))
improved_sample = improve_trip(inst, sample_route)

report(
    "Combined local search preserves sites",
    sorted(improved_sample) == sorted(sample_route),
)
report(
    "Combined local search does not increase time",
    trip_time(inst, improved_sample)
    <= trip_time(inst, sample_route) + 1e-9,
)

# ------------------------------------------------------------
# 3. Build the initial solution.
# ------------------------------------------------------------
print()
print("Building the initial solution. Please wait...")

started = time.perf_counter()
solution = build_initial_solution(inst)
elapsed = time.perf_counter() - started

problems = check_solution(inst, solution)
visited = [site_id for trip in solution for site_id in trip]
raw_profits = profit_by_trip(inst, solution)
score = objective(inst, solution)

report(
    "Initial solution is feasible",
    problems == [],
    f"(problems = {problems})",
)
report(
    "All 88 sites visited exactly once",
    len(visited) == inst.n
    and sorted(visited) == list(range(1, inst.n + 1)),
)
report(
    "No empty trips",
    all(len(trip) > 0 for trip in solution),
)
report(
    "Raw collected profit is 200",
    math.isclose(sum(raw_profits), 200.0, abs_tol=1e-9),
)
report(
    "Discounted objective is positive and at most 200",
    0.0 < score <= 200.0 + 1e-9,
)

# ------------------------------------------------------------
# 4. Print results.
# ------------------------------------------------------------
print()
print("--- Initial solution ---")
print(f"Number of trips       : {len(solution)}")
print(f"Discounted objective  : {score:.6f}")
print(f"Construction time     : {elapsed:.3f} s")
print()

for r, trip in enumerate(solution, start=1):
    duration = trip_time(inst, trip)
    raw_profit = raw_profits[r - 1]
    discounted_profit = raw_profit * inst.rho ** (r - 1)

    print(
        f"Trip {r}: "
        f"{len(trip)} sites | "
        f"time {duration:.2f} s | "
        f"raw profit {raw_profit:.0f} | "
        f"discounted profit {discounted_profit:.3f}"
    )
    print("  Route: 0 -> " + " -> ".join(map(str, trip)) + " -> 0")

# ------------------------------------------------------------
# 5. Save CSV results.
# ------------------------------------------------------------
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

summary_path = output_dir / "phase2_trip_summary.csv"
routes_path = output_dir / "phase2_routes.csv"

with summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "trip",
        "number_of_sites",
        "trip_time_s",
        "raw_profit",
        "discounted_profit",
    ])

    for r, trip in enumerate(solution, start=1):
        raw_profit = raw_profits[r - 1]
        writer.writerow([
            r,
            len(trip),
            round(trip_time(inst, trip), 6),
            raw_profit,
            round(raw_profit * inst.rho ** (r - 1), 6),
        ])

with routes_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["trip", "visit_order", "site_id", "x", "y"])

    for r, trip in enumerate(solution, start=1):
        writer.writerow([r, 0, 0, *inst.depot])

        for order, site_id in enumerate(trip, start=1):
            site = inst.site(site_id)
            writer.writerow([r, order, site_id, site.x, site.y])

        writer.writerow([r, len(trip) + 1, 0, *inst.depot])

print()
print(f"Saved {summary_path}")
print(f"Saved {routes_path}")
print()
print(
    "ALL PHASE 2 CHECKS PASSED"
    if all(results)
    else "SOME CHECKS FAILED - please send me this output"
)

raise SystemExit(0 if all(results) else 1)