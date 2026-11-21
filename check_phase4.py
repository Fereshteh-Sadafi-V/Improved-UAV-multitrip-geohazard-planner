"""Phase 4 checks: run all four algorithms via the unified API and compare.

Run from the project folder:
    python check_phase4.py            (quick preset, 1 seed)
    python check_phase4.py --paper    (paper preset: slow, ~30+ min)

This is the headless equivalent of the browser comparison view and of
Fig. 5 in Wang & Wen (2026), Section 5.2.
"""

import csv
import sys
import time
from pathlib import Path

from uavplanner import check_solution, generate_synthetic, objective
from uavplanner.api import (
    ALGORITHMS,
    ALGORITHM_LABELS,
    run_comparison,
)

QUICK = "--paper" not in sys.argv
PRESET = "quick" if QUICK else "paper"
SEED0 = 1
EPS = 1e-6

names = ["ts4gm", "lns", "grasp", "ga"]

results = []


def report(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


# ------------------------------------------------------------
# 1. Instance
# ------------------------------------------------------------
inst = generate_synthetic(seed=1)
print(f"Mode: {'QUICK' if QUICK else 'PAPER'} preset, seed = {SEED0}")
print(f"Instance: {inst.n} sites, Tmax = {inst.t_max} s")
print()

# ------------------------------------------------------------
# 2. Run all four
# ------------------------------------------------------------
print("Running all four algorithms through api.run_comparison. Please wait...")
started = time.perf_counter()
res_list = run_comparison(names, inst, preset=PRESET, seed0=SEED0)
elapsed = time.perf_counter() - started

report("run_comparison returned one result per algorithm", len(res_list) == len(names),
       f"(got {len(res_list)})")

# ------------------------------------------------------------
# 3. Per-result checks
# ------------------------------------------------------------
expected_keys = {
    "algorithm", "seed", "preset", "solution", "trips", "objective",
    "initial_objective", "improvement", "n_trips", "elapsed_s",
    "history", "problems", "feasible",
}
for r in res_list:
    alg = r["algorithm"]
    label = ALGORITHM_LABELS.get(alg, alg)
    report(f"{label}: result has the expected keys",
           expected_keys.issubset(r.keys()),
           f"(missing = {sorted(expected_keys - r.keys())})")
    report(f"{label}: algorithm field matches its slot", alg in ALGORITHMS)
    report(f"{label}: solution is feasible",
           r["feasible"] and check_solution(inst, r["solution"]) == [])
    report(f"{label}: objective matches recomputed objective",
           abs(r["objective"] - objective(inst, r["solution"])) < EPS,
           f'({r["objective"]:.6f})')
    report(f"{label}: improvement = objective - initial",
           abs(r["improvement"] - (r["objective"] - r["initial_objective"])) < EPS)
    report(f"{label}: history never decreases",
           all(r["history"][i + 1] >= r["history"][i] - EPS
               for i in range(len(r["history"]) - 1)))
    report(f"{label}: n_trips == len(solution)",
           r["n_trips"] == len(r["solution"]))
    report(f"{label}: all 88 sites visited exactly once",
           sorted(s for trip in r["solution"] for s in trip)
           == list(range(1, inst.n + 1)))

# ------------------------------------------------------------
# 4. Print comparison table (paper's Table 4 / Fig. 5 in text form)
# ------------------------------------------------------------
initial = res_list[0]["initial_objective"]
print()
print("--- Comparison (same instance, same seed, same preset) ---")
print(f"Paper-style start objective : {initial:.4f}")
print()
header = f"{'Algorithm':<10} {'Trips':>5} {'Objective':>12} {'Improvement':>13} {'Runtime(s)':>12}"
print(header)
print("-" * len(header))
for r in res_list:
    label = ALGORITHM_LABELS.get(r["algorithm"], r["algorithm"])
    print(
        f"{label:<10} {r['n_trips']:>5} "
        f"{r['objective']:>12.4f} {r['improvement']:>+13.4f} "
        f"{r['elapsed_s']:>12.1f}"
    )
print("-" * len(header))
print(f"Total run time (wall-clock): {elapsed:.1f} s")
print()

# ------------------------------------------------------------
# 5. Save CSVs
# ------------------------------------------------------------
out = Path("results")
out.mkdir(exist_ok=True)

summary_path = out / "phase4_comparison.csv"
with summary_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["algorithm", "trips", "objective", "improvement",
                "runtime_s", "feasible", "iterations_recorded"])
    for r in res_list:
        w.writerow([
            r["algorithm"], r["n_trips"], f'{r["objective"]:.6f}',
            f'{r["improvement"]:+.6f}', f'{r["elapsed_s"]:.1f}',
            r["feasible"], len(r["history"]),
        ])

for r in res_list:
    hist_path = out / f'phase4_history_{r["algorithm"]}.csv'
    with hist_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "best_objective"])
        for i, v in enumerate(r["history"]):
            w.writerow([i, f"{float(v):.6f}"])

print(f"Saved {summary_path}")
for r in res_list:
    print(f'Saved results/phase4_history_{r["algorithm"]}.csv')
print()
print(
    "ALL PHASE 4 CHECKS PASSED"
    if all(results)
    else "SOME CHECKS FAILED - please send me this output"
)

raise SystemExit(0 if all(results) else 1)