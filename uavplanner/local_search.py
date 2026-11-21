"""Within-trip route improvement.

2-opt:
    Reverse a consecutive section of a trip.

Or-opt:
    Relocate a consecutive block of 1, 2, or 3 sites without
    reversing the block.

These moves preserve the sites assigned to the trip. Therefore,
they preserve its profit and service time. Only travel time changes.

Both searches use best improvement and accept only strictly
time-reducing moves.
"""

from __future__ import annotations

from typing import Sequence

from .model import Instance, Trip, trip_time

EPS = 1e-9


def two_opt(
    instance: Instance,
    trip: Sequence[int],
    max_passes: int = 100,
) -> Trip:
    """Return an improved copy of trip using repeated best 2-opt moves."""
    route = list(trip)

    for _ in range(max_passes):
        current_time = trip_time(instance, route)
        best_time = current_time
        best_route = None

        for start in range(len(route) - 1):
            for stop in range(start + 2, len(route) + 1):
                candidate = (
                    route[:start]
                    + list(reversed(route[start:stop]))
                    + route[stop:]
                )

                candidate_time = trip_time(instance, candidate)

                if candidate_time < best_time - EPS:
                    best_time = candidate_time
                    best_route = candidate

        if best_route is None:
            break

        route = best_route

    return route


def or_opt(
    instance: Instance,
    trip: Sequence[int],
    max_block: int = 3,
    max_passes: int = 100,
) -> Trip:
    """Return an improved copy using block relocations.

    Insertion positions refer to the route after removing the block.
    """
    if max_block < 1:
        raise ValueError("max_block must be at least 1.")

    route = list(trip)

    for _ in range(max_passes):
        current_time = trip_time(instance, route)
        best_time = current_time
        best_route = None

        # Moving the entire route cannot improve its order.
        largest_block = min(max_block, len(route) - 1)

        for block_size in range(1, largest_block + 1):
            for start in range(len(route) - block_size + 1):
                block = route[start:start + block_size]
                remainder = (
                    route[:start] + route[start + block_size:]
                )

                for position in range(len(remainder) + 1):
                    candidate = (
                        remainder[:position]
                        + block
                        + remainder[position:]
                    )

                    if candidate == route:
                        continue

                    candidate_time = trip_time(instance, candidate)

                    if candidate_time < best_time - EPS:
                        best_time = candidate_time
                        best_route = candidate

        if best_route is None:
            break

        route = best_route

    return route


def improve_trip(
    instance: Instance,
    trip: Sequence[int],
    max_rounds: int = 10,
) -> Trip:
    """Alternate 2-opt and Or-opt until no improvement or the round limit."""
    route = list(trip)

    for _ in range(max_rounds):
        before = trip_time(instance, route)

        route = two_opt(instance, route)
        route = or_opt(instance, route)

        after = trip_time(instance, route)

        if after >= before - EPS:
            break

    return route