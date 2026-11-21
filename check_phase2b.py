"""Phase 2b checks. Run from the project folder:
    python check_phase2b.py
"""

import csv
import math
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
from uavplanner.initial_paper import build_paper_initial_solution

results = []


def report(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


# 1. Tiny case where the rule decides the route.
#    From the depot the two nearest are sites 1 and 2 -> pick 2 (profit 5),
#    even though site 3 has profit 9 (it is too far to be shortlisted).
#    From site 2 the two nearest are 1 and 3 -> pick 3 (profit 9). Then 1.
tiny = Instance(
    depot=(0.0, 0.0),
    sites=[
        Site(1, 10.0, 0.0, 1.0, 0.0),
        Site(2, 0.0, 11.0, 5.0, 0.0),
        Site(3, 50.0, 0.0, 9.0, 0.0),
    ],
    t_max=1000.0,
    speed=1.0,
)
tiny_solution = build_paper_initial_solution(tiny)
report(
    "Rule check: two nearest, highest profit wins",
    tiny_solution == [[2, 3, 1]],
    f"(got {tiny_solution})",
)

# 2. The 88-site synthetic example.
inst = generate_synthetic(seed=1)
solution = build_paper_initial_solution(inst)
visited = [s for trip in solution for s in trip]
raw = profit_by_trip(inst, solution)

report("Solution is feasible",
       check_solution(inst, solution) == [])
report("All 88 sites visited exactly once",
       sorted(visited) == list(range(1, inst.n + 1)))
report("No empty trips", all(len(t) > 0 for t in solution))
report("Raw collected profit is 200",
       math.isclose(sum(raw), 200.0, abs_tol=1e-9))

# 3. Information only (not pass/fail).
improved = build_initial_solution(inst)
print()
print("--- Comparison (information only) ---")
print(f"Paper-style start : {len(solution)} trips, "
      f"objective {objective(inst, solution):.3f}")
print(f"Phase 2 start     : {len(improved)} trips, "
      f"objective {objective(inst, improved):.3f}")
print()

for r, trip in enumerate(solution, start=1):
    print(f"Trip {r}: {len(trip)} sites | "
          f"time {trip_time(inst, trip):.2f} s | "
          f"raw profit {raw[r - 1]:.0f}")

out = Path("results")
out.mkdir(exist_ok=True)
path = out / "phase2b_paper_start.csv"

with path.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["trip", "number_of_sites", "trip_time_s", "raw_profit",
                "discounted_profit"])
    for r, trip in enumerate(solution, start=1):
        w.writerow([r, len(trip), round(trip_time(inst, trip), 6),
                    raw[r - 1], round(raw[r - 1] * inst.rho ** (r - 1), 6)])

print(f"\nSaved {path}")
print("ALL PHASE 2b CHECKS PASSED" if all(results)
      else "SOME CHECKS FAILED - please send me this output")
raise SystemExit(0 if all(results) else 1)