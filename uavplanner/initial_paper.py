"""Paper-style 2-Nearest Greedy Insertion initializer (TS4GM, Section 4.1).

Interpretation (the paper does not give every detail):
1. Each trip starts empty at the depot.
2. Take all unassigned sites that still fit in the trip
   (including the return to the depot).
3. Keep the two nearest to the LAST visited vertex (depot at first).
4. Append the one with the highest profit.
   Ties: nearer to the last vertex, then smaller site ID.
5. When nothing fits, close the trip and start a new one.

No local search is applied here.
"""

from __future__ import annotations

from .model import Instance, Solution, check_solution, trip_time

EPS = 1e-9


def build_paper_initial_solution(instance: Instance) -> Solution:
    """Construct a feasible complete solution with no empty trips."""
    unreachable = instance.unreachable_sites()

    if unreachable:
        ids = ", ".join(str(i) for i in unreachable)
        raise ValueError(
            "Cannot build a complete solution. These sites cannot fit "
            f"even in individual trips: {ids}"
        )

    remaining = set(range(1, instance.n + 1))
    solution: Solution = []

    while remaining:
        route = []

        while remaining:
            last = route[-1] if route else 0

            feasible = [
                s for s in sorted(remaining)
                if trip_time(instance, route + [s])
                <= instance.t_max + EPS
            ]

            if not feasible:
                break

            feasible.sort(key=lambda s: (instance.travel_time(last, s), s))
            two_nearest = feasible[:2]

            chosen = max(
                two_nearest,
                key=lambda s: (
                    instance.site(s).profit,
                    -instance.travel_time(last, s),
                    -s,
                ),
            )

            route.append(chosen)
            remaining.remove(chosen)

        if not route:
            raise RuntimeError("Could not start a new trip.")

        solution.append(route)

    problems = check_solution(instance, solution)

    if problems:
        raise RuntimeError(
            "The initializer produced an invalid solution:\n"
            + "\n".join(problems)
        )

    return solution