"""Phase 3 checks: TS4GM (paper-style protocol).

Run from the project folder:
    python check_phase3.py            (full: 5 trials x 200 iterations)
    python check_phase3.py --quick    (fast test: 1 trial x 30 iterations)
"""

import csv
import sys
import time
from pathlib import Path

from uavplanner import (
    check_solution,
    generate_synthetic,
    objective,
    profit_by_trip,
    trip_time,
)
from uavplanner.initial_paper import build_paper_initial_solution
from uavplanner.ordering import sort_trips_by_profit
from uavplanner.ts4gm import run_ts4gm_trials

QUICK = "--quick" in sys.argv

N_RUNS = 1 if QUICK else 5
MAX_ITERS = 30 if QUICK else 200
Z_IN = 15 if QUICK else 50
SEED0 = 1
EPS = 1e-6

results = []


def report(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


# ------------------------------------------------------------
# 1. Instance and paper-style initial solution.
# ------------------------------------------------------------
inst = generate_synthetic(seed=1)

print(f"Mode: {'QUICK' if QUICK else 'FULL'} "
      f"({N_RUNS} trial(s), max_iters={MAX_ITERS}, z_in={Z_IN})")
print(f"Instance: {inst.n} sites, Tmax = {inst.t_max} s")
print()

report("Instance has 88 sites", inst.n == 88, f"(n = {inst.n})")

initial = build_paper_initial_solution(inst)
initial_obj = objective(inst, initial)
report(
    "Paper-style initial solution is feasible",
    check_solution(inst, initial) == [],
    f"(objective = {initial_obj:.4f}, trips = {len(initial)})",
)

# ------------------------------------------------------------
# 2. Run TS4GM.
# ------------------------------------------------------------
print()
print("Running TS4GM. Please wait...")

started = time.perf_counter()
res = run_ts4gm_trials(
    inst,
    n_runs=N_RUNS,
    seed0=SEED0,
    max_iters=MAX_ITERS,
    z_in=Z_IN,
    start_method="paper",
)
elapsed = time.perf_counter() - started

# The search already returns sorted trips; sorting again is a harmless safety net.
solution = sort_trips_by_profit(inst, res.best_solution)
problems = check_solution(inst, solution)
visited = [site_id for trip in solution for site_id in trip]
raw_profits = profit_by_trip(inst, solution)
times = [trip_time(inst, trip) for trip in solution]
score = objective(inst, solution)
discounted_sum = sum(
    raw_profits[i] * inst.rho ** i for i in range(len(raw_profits))
)

print()
report(
    "TS4GM solution is feasible",
    problems == [],
    f"(problems = {problems})",
)
report(
    "All 88 sites visited exactly once",
    len(visited) == inst.n and sorted(visited) == list(range(1, inst.n + 1)),
)
report("No empty trips", all(len(trip) > 0 for trip in solution))
report(
    "Every trip time <= Tmax",
    all(t <= inst.t_max + EPS for t in times),
    f"(longest trip = {max(times):.1f} s)",
)
report(
    "Raw collected profit is 200",
    abs(sum(raw_profits) - 200.0) < EPS,
)
report(
    "Trips are in non-increasing raw-profit order",
    all(raw_profits[i] >= raw_profits[i + 1] for i in range(len(raw_profits) - 1)),
)
report(
    "Reported objective equals recomputed objective",
    abs(score - res.best_objective) < EPS,
    f"(reported = {res.best_objective:.6f}, recomputed = {score:.6f})",
)
report(
    "Displayed discounted profits sum to the objective",
    abs(discounted_sum - score) < EPS,
    f"(sum = {discounted_sum:.6f}, objective = {score:.6f})",
)
report(
    "TS4GM objective >= paper-style initial objective",
    score >= initial_obj - EPS,
    f"({initial_obj:.4f} -> {score:.4f})",
)
report(
    "Best-so-far history never decreases",
    all(
        res.history_best[i + 1] >= res.history_best[i] - EPS
        for i in range(len(res.history_best) - 1)
    ),
)
report(
    "iterations_done <= max_iters",
    res.iterations_done <= MAX_ITERS,
    f"(iterations_done = {res.iterations_done})",
)
report("n_trips matches the solution", res.n_trips == len(solution))

# ------------------------------------------------------------
# 3. Print results.
# ------------------------------------------------------------
print()
print("--- TS4GM solution ---")
print(f"Number of trips       : {len(solution)}")
print(f"Discounted objective  : {score:.6f}")
print(f"Initial objective     : {initial_obj:.6f}")
print(f"Improvement           : {score - initial_obj:+.6f}")
print(f"Run time              : {elapsed:.1f} s")
print()

# Trips are already sorted from highest to lowest raw profit (the paper's ordering).
for rank, trip in enumerate(solution, start=1):
    k = rank - 1
    discounted = raw_profits[k] * inst.rho ** (rank - 1)
    print(
        f"Trip {rank}: "
        f"{len(trip)} sites | "
        f"time {times[k]:.2f} s | "
        f"raw profit {raw_profits[k]:.0f} | "
        f"discounted profit {discounted:.3f}"
    )
    print("  Route: 0 -> " + " -> ".join(map(str, trip)) + " -> 0")

# ------------------------------------------------------------
# 4. Save CSV results.
# ------------------------------------------------------------
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

summary_path = output_dir / "phase3_trip_summary.csv"
routes_path = output_dir / "phase3_routes.csv"
history_path = output_dir / "phase3_history.csv"

with summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "trip", "number_of_sites", "trip_time_s", "raw_profit", "discounted_profit",
    ])
    for rank, trip in enumerate(solution, start=1):
        k = rank - 1
        writer.writerow([
            rank,
            len(trip),
            round(times[k], 6),
            raw_profits[k],
            round(raw_profits[k] * inst.rho ** (rank - 1), 6),
        ])

with routes_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["trip", "visit_order", "site_id", "x", "y"])
    for rank, trip in enumerate(solution, start=1):
        writer.writerow([rank, 0, 0, *inst.depot])
        for pos, site_id in enumerate(trip, start=1):
            site = inst.site(site_id)
            writer.writerow([rank, pos, site_id, site.x, site.y])
        writer.writerow([rank, len(trip) + 1, 0, *inst.depot])

with history_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["iteration", "best_objective"])
    for it, value in enumerate(res.history_best):
        writer.writerow([it, round(value, 6)])

print()
print(f"Saved {summary_path}")
print(f"Saved {routes_path}")
print(f"Saved {history_path}")
print()
print(
    "ALL PHASE 3 CHECKS PASSED"
    if all(results)
    else "SOME CHECKS FAILED - please send me this output"
)

raise SystemExit(0 if all(results) else 1)