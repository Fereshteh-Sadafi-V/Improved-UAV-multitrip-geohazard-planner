"""Documented two-nearest greedy insertion initializer.

This is a project implementation choice. Exact equivalence to the
paper's initializer has not yet been verified.

Rules
-----
1. Start each trip with the highest-profit unassigned site.
   Ties: shortest depot travel time, then smallest site ID.
2. Find the best feasible insertion for each remaining site.
3. Shortlist the two feasible sites nearest to any vertex of the
   current route, including the depot.
4. Choose the shortlisted site with the smallest insertion-time
   increase. Ties: higher profit, then smaller site ID.
5. If no insertion fits, improve the route and try again.
6. Close the trip if no insertion fits after route improvement.

All sites must be visited exactly once.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from .model import Instance, Solution, Trip, check_solution, trip_time
from .local_search import improve_trip

EPS = 1e-9


def best_feasible_insertion(
    instance: Instance,
    trip: Sequence[int],
    site_id: int,
) -> Optional[Tuple[float, int]]:
    """Return (extra seconds, insertion position), or None if infeasible."""
    route = list(trip)
    current_time = trip_time(instance, route)
    service = instance.site(site_id).service_time
    best = None

    for position in range(len(route) + 1):
        previous = 0 if position == 0 else route[position - 1]
        following = 0 if position == len(route) else route[position]

        extra_time = (
            instance.travel_time(previous, site_id)
            + instance.travel_time(site_id, following)
            - instance.travel_time(previous, following)
            + service
        )

        if current_time + extra_time <= instance.t_max + EPS:
            option = (extra_time, position)

            if best is None or option < best:
                best = option

    return best


def _choose_insertion(
    instance: Instance,
    route: Trip,
    remaining: set[int],
) -> Optional[Tuple[int, int]]:
    """Return (site ID, insertion position) using the two-nearest rule."""
    candidates = []
    anchors = [0] + route

    for site_id in sorted(remaining):
        insertion = best_feasible_insertion(instance, route, site_id)

        if insertion is None:
            continue

        extra_time, position = insertion

        nearest_time = min(
            instance.travel_time(anchor, site_id)
            for anchor in anchors
        )

        candidates.append(
            (nearest_time, site_id, extra_time, position)
        )

    if not candidates:
        return None

    # First shortlist the two nearest feasible sites.
    candidates.sort(key=lambda item: (item[0], item[1]))
    shortlist = candidates[:2]

    # Then select the smaller insertion cost.
    chosen = min(
        shortlist,
        key=lambda item: (
            item[2],
            -instance.site(item[1]).profit,
            item[1],
            item[3],
        ),
    )

    return chosen[1], chosen[3]


def build_initial_solution(instance: Instance) -> Solution:
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
        seed = min(
            remaining,
            key=lambda site_id: (
                -instance.site(site_id).profit,
                instance.travel_time(0, site_id),
                site_id,
            ),
        )

        route = [seed]
        remaining.remove(seed)

        while remaining:
            choice = _choose_insertion(instance, route, remaining)

            if choice is None:
                # A shorter route may make another insertion feasible.
                route = improve_trip(instance, route)
                choice = _choose_insertion(instance, route, remaining)

                if choice is None:
                    break

            site_id, position = choice
            route.insert(position, site_id)
            remaining.remove(site_id)

        route = improve_trip(instance, route)
        solution.append(route)

    problems = check_solution(instance, solution)

    if problems:
        raise RuntimeError(
            "The initializer produced an invalid solution:\n"
            + "\n".join(problems)
        )

    return solution