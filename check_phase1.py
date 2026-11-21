"""Phase 1 self-check. Run from the project folder:  python check_phase1.py"""

import math
import os

from uavplanner import (
    check_solution,
    discounted_total,
    generate_synthetic,
    instance_to_csv_text,
    objective,
    parse_csv_text,
    trip_time,
)

results = []


def report(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


# 1. The paper's Table 5 total (trip profits 81, 81, 32, 6 with rho = 0.9)
value = discounted_total([81, 81, 32, 6], 0.9)
report("Paper Table 5 objective = 184.194", math.isclose(value, 184.194, abs_tol=1e-9),
       f"(got {value:.3f})")

# 2. Synthetic example matches the paper's profit mix
inst = generate_synthetic(seed=1)
counts = {v: sum(1 for s in inst.sites if s.profit == v) for v in (4, 3, 2, 1)}
report("88 synthetic sites", inst.n == 88, f"(got {inst.n})")
report("Total profit = 200", sum(s.profit for s in inst.sites) == 200)
report("Profit mix 15/21/25/27", counts == {4: 15, 3: 21, 2: 25, 1: 27}, f"(got {counts})")

# 3. CSV round trip
text = instance_to_csv_text(inst)
inst2 = parse_csv_text(text)
report("CSV round trip keeps 88 sites", inst2.n == 88)
report("CSV round trip keeps total profit", sum(s.profit for s in inst2.sites) == 200)

# 4. The checker catches a bad solution (everything in one trip)
bad = [list(range(1, inst.n + 1))]
problems = check_solution(inst, bad)
report("Checker rejects a single over-long trip", len(problems) > 0)

# 5. The checker accepts a valid (but poor) solution: one site per trip
trivial = [[i] for i in range(1, inst.n + 1)]
report("Checker accepts one-site-per-trip solution", check_solution(inst, trivial) == [])

# 6. Objective sanity: discounting must lower the value versus no discounting
report("Objective is below the undiscounted total of 200", objective(inst, trivial) < 200)

# 7. No site is impossible to reach
report("All sites reachable within T_max", inst.unreachable_sites() == [])

# Information for you
total_service = sum(s.service_time for s in inst.sites)
print()
print("--- Information ---")
print(f"Total service time          : {total_service:.0f} s")
print(f"T_max per trip              : {inst.t_max:.0f} s")
print(f"Time of one trip visiting all sites in numeric order (a bad route): "
      f"{trip_time(inst, bad[0]):.0f} s")
print(f"Trips needed at the very least (service time only): "
      f"{math.ceil(total_service / inst.t_max)}")

# Save the example data
os.makedirs("data", exist_ok=True)
with open(os.path.join("data", "example_synthetic_88.csv"), "w", encoding="utf-8") as f:
    f.write(text)
print("Saved data/example_synthetic_88.csv")

print()
print("ALL CHECKS PASSED" if all(results) else "SOME CHECKS FAILED - please send me this output")