"""Core data model and objective function.

Conventions used everywhere in this project
-------------------------------------------
* Vertex 0 is the docking station (ADS). Sites are numbered 1..n.
* A trip is a list of site numbers, e.g. [5, 2, 9]. The flight
  0 -> 5 -> 2 -> 9 -> 0 is implied, so 0 is not written in the list.
* A solution is a list of trips. The first trip is trip r = 1.
* Distances are in metres, times in seconds, speed in metres per second.

Paper formulas implemented here (Wang & Wen, 2026):
* travel time  t_ij = distance(i, j) / speed                      (eq. 1)
* trip time    = flight time + sum of service times <= T_max      (eq. 5)
* profit       p_i(r) = p_i * rho^(r-1)                           (eq. 2)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

Trip = List[int]
Solution = List[Trip]


@dataclass(frozen=True)
class Site:
    """One geohazard site (GS)."""

    id: int
    x: float  # metres
    y: float  # metres
    profit: float  # urgency, e.g. 1 (low) to 4 (very high)
    service_time: float = 120.0  # seconds spent monitoring the site


class Instance:
    """A complete problem: docking station, sites and parameters."""

    def __init__(
        self,
        depot: Tuple[float, float],
        sites: Sequence[Site],
        t_max: float = 7200.0,
        rho: float = 0.9,
        speed: float = 16.0,
    ) -> None:
        if t_max <= 0:
            raise ValueError("T_max (maximum flight time per trip) must be positive.")
        if not (0 < rho <= 1):
            raise ValueError("rho (profit decrease rate) must be above 0 and at most 1.")
        if speed <= 0:
            raise ValueError("UAV speed must be positive.")
        if len(sites) == 0:
            raise ValueError("There must be at least one site.")

        for expected_id, s in enumerate(sites, start=1):
            if s.id != expected_id:
                raise ValueError("Sites must be numbered 1, 2, 3, ... in order.")
            if s.profit < 0:
                raise ValueError(f"Site {s.id} has a negative profit.")
            if s.service_time < 0:
                raise ValueError(f"Site {s.id} has a negative service time.")

        self.depot = (float(depot[0]), float(depot[1]))
        self.sites: List[Site] = list(sites)
        self.n = len(self.sites)
        self.t_max = float(t_max)
        self.rho = float(rho)
        self.speed = float(speed)

        # Travel-time matrix. Index 0 = docking station.
        points = [self.depot] + [(s.x, s.y) for s in self.sites]
        self._t = [
            [math.hypot(a[0] - b[0], a[1] - b[1]) / self.speed for b in points]
            for a in points
        ]

    def site(self, i: int) -> Site:
        return self.sites[i - 1]

    def travel_time(self, i: int, j: int) -> float:
        return self._t[i][j]

    def unreachable_sites(self) -> List[int]:
        """Sites that can never be monitored, because even a trip containing
        only that site is longer than T_max."""
        return [
            s.id
            for s in self.sites
            if 2 * self.travel_time(0, s.id) + s.service_time > self.t_max
        ]


def trip_time(instance: Instance, trip: Sequence[int]) -> float:
    """Flight time (0 -> trip -> 0) plus the service time of every site."""
    if not trip:
        return 0.0
    total = instance.travel_time(0, trip[0])
    for a, b in zip(trip, trip[1:]):
        total += instance.travel_time(a, b)
    total += instance.travel_time(trip[-1], 0)
    total += sum(instance.site(i).service_time for i in trip)
    return total


def discounted_total(trip_profits: Sequence[float], rho: float) -> float:
    """Total collected profit when trip r collected trip_profits[r-1]
    (before discounting): sum of trip_profit * rho^(r-1)."""
    return sum(p * rho**r for r, p in enumerate(trip_profits))


def profit_by_trip(instance: Instance, solution: Sequence[Sequence[int]]) -> List[float]:
    """Raw (undiscounted) profit collected in each trip."""
    return [sum(instance.site(i).profit for i in trip) for trip in solution]


def objective(instance: Instance, solution: Sequence[Sequence[int]]) -> float:
    """Objective function (2): total collected profit with the decrease rho."""
    return discounted_total(profit_by_trip(instance, solution), instance.rho)


def check_solution(instance: Instance, solution: Sequence[Sequence[int]]) -> List[str]:
    """Return a list of problems in plain English. Empty list = solution is valid."""
    problems: List[str] = []
    seen = {}
    for r, trip in enumerate(solution, start=1):
        if len(trip) == 0:
            problems.append(f"Trip {r} is empty.")
            continue
        all_known = True
        for i in trip:
            if not (1 <= i <= instance.n):
                problems.append(f"Trip {r} contains an unknown site number {i}.")
                all_known = False
            elif i in seen:
                problems.append(f"Site {i} appears in trip {seen[i]} and again in trip {r}.")
            else:
                seen[i] = r
        if all_known:
            t = trip_time(instance, trip)
            if t > instance.t_max + 1e-9:
                problems.append(
                    f"Trip {r} takes {t:.0f} s, which is more than T_max = {instance.t_max:.0f} s."
                )
    missing = [i for i in range(1, instance.n + 1) if i not in seen]
    if missing:
        preview = ", ".join(str(i) for i in missing[:10])
        more = " ..." if len(missing) > 10 else ""
        problems.append(f"{len(missing)} site(s) are not visited: {preview}{more}")
    return problems