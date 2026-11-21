"""Try LNS: run the LNS comparison algorithm on the 88-site instance.

Run from the project folder:
    python try_lns.py                 (full: 5 runs x 200 iterations)
    python try_lns.py --quick         (fast test: 1 run x 30 iterations)
    python try_lns.py --seed 7        (use a different random seed)
    python try_lns.py --runs 3 --iters 100

The instance is the same one used in Phase 3: generate_synthetic(seed=1).
The --seed option only changes the random choices inside LNS.
"""

import argparse
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

try:
    from uavplanner.lns import run_lns_trials
except ImportError as error:
    print("Could not import uavplanner/lns.py.")
    print(f"Python says: {error}")
    print()
    print("This usually means lns.py asks ts4gm.py for a helper function that")
    print("ts4gm.py does not have. Please send me this message together with")
    print("your uavplanner/ts4gm.py file.")
    raise SystemExit(2)

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
parser = argparse.ArgumentParser(description="Run LNS on the 88-site instance.")
parser.add_argument("--quick", action="store_true",
                    help="fast test: 1 run, 30 iterations")
parser.add_argument("--seed", type=int, default=1,
                    help="first random seed for LNS (default 1)")
parser.add_argument("--runs", type=int, default=None,
                    help="number of independent runs (default 5, quick 1)")
parser.add_argument("--iters", type=int, default=None,
                    help="max iterations per run (default 200, quick 30)")
args = parser.parse_args()

N_RUNS = args.runs if args.runs is not None else (1 if args.quick else 5)
MAX_ITERS = args.iters if args.iters is not None else (30 if args.quick else 200)
Z_IN = 15 if args.quick else 50
SEED0 = args.seed
EPS = 1e-6

results = []


def report(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def show_progress(it, max_iters, best_obj, best_trips):
    if it == 1 or it % 20 == 0 or it == max_iters:
        print(f"  iteration {it:>3}/{max_iters}  "
              f"best objective = {best_obj:.4f}  trips = {best_trips}")


# ------------------------------------------------------------
# 1. Instance and shared starting solution
# ------------------------------------------------------------
inst = generate_synthetic(seed=1)

print(f"Mode: {'QUICK' if args.quick else 'FULL'} "
      f"({N_RUNS} run(s), max_iters={MAX_ITERS}, z_in={Z_IN}, first seed={SEED0})")
print(f"Instance: {inst.n} sites, Tmax = {inst.t_max} s")
print()

initial = build_paper_initial_solution(inst)
initial_obj = objective(inst, initial)
report(
    "Paper-style initial solution is feasible",
    check_solution(inst, initial) == [],
    f"(objective = {initial_obj:.4f}, trips = {len(initial)})",
)

# ------------------------------------------------------------
# 2. Run LNS
# ------------------------------------------------------------
print()
print("Running LNS. Please wait...")

started = time.perf_counter()
res = run_lns_trials(
    inst,
    n_runs=N_RUNS,
    seed0=SEED0,
    max_iters=MAX_ITERS,
    z_in=Z_IN,
    start_method="paper",
    on_iteration=show_progress,
)
elapsed = time.perf_counter() - started

solution = sort_trips_by_profit(inst, res.best_solution)
problems = check_solution(inst, solution)
visited = [site_id for trip in solution for site_id in trip]
raw_profits = profit_by_trip(inst, solution)
times = [trip_time(inst, trip) for trip in solution]
score = objective(inst, solution)
discounted_sum = sum(raw_profits[i] * inst.rho ** i for i in range(len(raw_profits)))
total_profit = sum(inst.site(i).profit for i in range(1, inst.n + 1))

# ------------------------------------------------------------
# 3. Checks
# ------------------------------------------------------------
print()
report("LNS solution is feasible", problems == [], f"(problems = {problems})")
report(
    f"All {inst.n} sites visited exactly once",
    len(visited) == inst.n and sorted(visited) == list(range(1, inst.n + 1)),
)
report("No empty trips", all(len(trip) > 0 for trip in solution))
report(
    "Every trip time <= Tmax",
    all(t <= inst.t_max + EPS for t in times),
    f"(longest trip = {max(times):.1f} s)",
)
report(
    f"Raw collected profit is {total_profit:.0f}",
    abs(sum(raw_profits) - total_profit) < EPS,
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
    "LNS objective >= paper-style initial objective",
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
# 4. Print results
# ------------------------------------------------------------
print()
print("--- LNS solution ---")
print(f"Number of trips       : {len(solution)}")
print(f"Discounted objective  : {score:.6f}")
print(f"Initial objective     : {initial_obj:.6f}")
print(f"Improvement           : {score - initial_obj:+.6f}")
print(f"Run time (all runs)   : {elapsed:.1f} s")
print()

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
# 5. Save CSV results
# ------------------------------------------------------------
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

summary_path = output_dir / "lns_trip_summary.csv"
routes_path = output_dir / "lns_routes.csv"
history_path = output_dir / "lns_history.csv"

with summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["trip", "number_of_sites", "trip_time_s", "raw_profit", "discounted_profit"]
    )
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
    writer.writerow(["iteration", "best_objective", "best_n_trips"])
    for it, (value, n_trips) in enumerate(zip(res.history_best, res.history_trips)):
        writer.writerow([it, round(value, 6), n_trips])

print()
print(f"Saved {summary_path}")
print(f"Saved {routes_path}")
print(f"Saved {history_path}")
print()
print(
    "ALL LNS CHECKS PASSED"
    if all(results)
    else "SOME CHECKS FAILED - please send me this output"
)

raise SystemExit(0 if all(results) else 1)