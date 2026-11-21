"""Try GA: run the GA comparison algorithm on the 88-site instance.

Run from the project folder:
    python try_ga.py                       (default: FAST, ~1-2 minutes)
    python try_ga.py --paper               (paper mode: 5 runs x 200 iters, slow)
    python try_ga.py --quick               (same as default)
    python try_ga.py --seed 7              (different random seed)
    python try_ga.py --runs 3 --iters 50 --pop-size 60
    python try_ga.py --max-seconds 60      (hard wall-time budget, seconds)
    python try_ga.py --z-in 20             (early-stop after 20 idle generations)

The instance is the same one used in Phase 3: generate_synthetic(seed=1).
The --seed option only changes the random choices inside GA.

Speed note (see README.md, "Simplifications we made"):
The PAPER settings (5 runs x 200 generations x 300 individuals) are extremely
slow on an 88-site instance in plain Python. For day-to-day use this driver
therefore defaults to 1 run x 15 generations x 30 individuals, with a hard
wall-time budget of 120 s. Use --paper to restore the exact paper protocol.
ga.py itself still defaults to the paper values, so anyone who calls
run_ga_trials() directly still gets paper behaviour.
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
    from uavplanner.ga import run_ga_trials
except ImportError as error:
    print("Could not import uavplanner/ga.py.")
    print(f"Python says: {error}")
    print()
    print("This usually means ga.py, lns.py or algo_common.py is missing,")
    print("or ga.py asks ts4gm.py/lns.py for a helper that does not exist.")
    print("Please send me this message together with your")
    print("uavplanner/ga.py, uavplanner/lns.py and uavplanner/algo_common.py.")
    raise SystemExit(2)

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Run GA on the 88-site instance."
)
parser.add_argument("--quick", action="store_true",
                    help="fast test (also the default): 1 run, 15 iters, pop 30")
parser.add_argument("--paper", action="store_true",
                    help="paper mode: 5 runs, 200 iters, pop 300 (slow)")
parser.add_argument("--seed", type=int, default=1,
                    help="first random seed for GA (default 1)")
parser.add_argument("--runs", type=int, default=None,
                    help="number of independent runs (overrides mode defaults)")
parser.add_argument("--iters", type=int, default=None,
                    help="max generations per run (overrides mode defaults)")
parser.add_argument("--pop-size", type=int, default=None,
                    help="population size (paper-style: 300)")
parser.add_argument("--crossover-prob", type=float, default=0.8,
                    help="crossover probability (paper-style: 0.8)")
parser.add_argument("--mutation-prob", type=float, default=0.25,
                    help="mutation probability (paper-style: 0.25)")
parser.add_argument("--tournament-k", type=int, default=3,
                    help="tournament size k (paper-style: 3)")
parser.add_argument("--z-in", type=int, default=None,
                    help="stop after z_in idle generations (paper-style: 50; "
                         "0 or -1 disables)")
parser.add_argument("--max-11-candidates", type=int, default=None,
                    help="cap 1-1 move candidates per iteration")
parser.add_argument("--max-seconds", type=float, default=None,
                    help="hard wall-time budget per trial (0 = no limit)")
args = parser.parse_args()

if args.quick and args.paper:
    parser.error("Choose at most one of --quick and --paper.")

if args.paper:
    QUICK_MODE = False
elif args.quick:
    QUICK_MODE = True
else:
    QUICK_MODE = True

if QUICK_MODE:
    default_runs       = 1
    default_iters      = 15
    default_pop        = 30
    default_z_in       = 10
    default_m11        = 30
    default_max_seconds = 120.0
else:
    default_runs       = 5
    default_iters      = 200
    default_pop        = 300
    default_z_in       = 50
    default_m11        = 150
    default_max_seconds = 0.0

N_RUNS        = args.runs               if args.runs               is not None else default_runs
MAX_ITERS     = args.iters              if args.iters              is not None else default_iters
POP_SIZE      = args.pop_size           if args.pop_size           is not None else default_pop
Z_IN          = args.z_in               if args.z_in               is not None else default_z_in
MAX_11        = args.max_11_candidates  if args.max_11_candidates  is not None else default_m11
MAX_SECONDS   = args.max_seconds        if args.max_seconds        is not None else default_max_seconds
CROSSOVER     = args.crossover_prob
MUTATION      = args.mutation_prob
TOURNAMENT_K  = args.tournament_k
SEED0 = args.seed
EPS = 1e-6

if Z_IN is not None and Z_IN <= 0:
    Z_IN = None   # disable early stop

results = []


def report(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def show_progress(it, max_iters, best_obj, best_trips):
    if it == 1 or it % 20 == 0 or it == max_iters:
        print(
            f"  generation {it:>3}/{max_iters}  "
            f"best objective = {best_obj:.4f}  trips = {best_trips}"
        )


# ------------------------------------------------------------
# 1. Instance and shared starting solution
# ------------------------------------------------------------
inst = generate_synthetic(seed=1)

mode_label = "PAPER (explicit --paper)" if not QUICK_MODE else "QUICK (default)"
print(
    f"Mode: {mode_label} "
    f"({N_RUNS} run(s), max_iters={MAX_ITERS}, pop_size={POP_SIZE}, "
    f"crossover={CROSSOVER}, mutation={MUTATION}, k={TOURNAMENT_K}, "
    f"z_in={Z_IN}, max_11={MAX_11}, max_seconds={MAX_SECONDS}, "
    f"first seed={SEED0})"
)
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
# 2. Run GA
# ------------------------------------------------------------
print()
print("Running GA. Please wait...")

started = time.perf_counter()
res = run_ga_trials(
    inst,
    n_runs=N_RUNS,
    seed0=SEED0,
    max_iters=MAX_ITERS,
    z_in=Z_IN,
    start_method="paper",
    pop_size=POP_SIZE,
    crossover_prob=CROSSOVER,
    mutation_prob=MUTATION,
    tournament_k=TOURNAMENT_K,
    max_11_candidates=MAX_11,
    on_iteration=show_progress,
    max_trial_seconds=MAX_SECONDS,
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
report("GA solution is feasible", problems == [], f"(problems = {problems})")
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
    "GA objective >= paper-style initial objective",
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
print("--- GA solution ---")
print(f"Mode                  : {mode_label}")
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

summary_path = output_dir / "ga_trip_summary.csv"
routes_path  = output_dir / "ga_routes.csv"
history_path = output_dir / "ga_history.csv"

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
    "ALL GA CHECKS PASSED"
    if all(results)
    else "SOME CHECKS FAILED - please send me this output"
)

raise SystemExit(0 if all(results) else 1)