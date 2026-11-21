"""Shared building blocks for the comparison algorithms (Version 1).

Every algorithm (TS4GM, LNS, GRASP, GA) uses the same:
- starting solution (so comparisons are fair),
- objective function and feasibility check (from model.py),
- trip ordering rule: trips are always sorted by raw profit, highest first,
  because the objective discounts trip r by rho^(r-1),
- result type (AlgoResult), so the browser can show them side by side.

Seeds: every algorithm receives an integer seed. The same seed on the same
instance gives the same result (fixed-seed repeatability).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from .initial import build_initial_solution
from .initial_paper import build_paper_initial_solution
from .local_search import improve_trip
from .model import Instance, Solution, check_solution, objective, trip_time

EPS = 1e-9

# progress callback: (algorithm_name, iteration, max_iterations, best_objective)
ProgressCallback = Callable[[str, int, int, float], None]


@dataclass
class AlgoResult:
    """Result of one algorithm run (same shape for every algorithm)."""

    algorithm: str
    best_solution: Solution
    best_objective: float
    iterations_done: int
    history_best: List[float]      # best objective after each iteration
    history_trips: List[int]       # number of trips of the best solution after each iteration
    n_trips: int
    raw_profit_by_trip: List[float]
    seed: int
    runtime_seconds: float


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def clone_solution(sol: Sequence[Sequence[int]]) -> Solution:
    return [list(t) for t in sol]


def drop_empty_trips(sol: Sequence[Sequence[int]]) -> Solution:
    return [list(t) for t in sol if len(t) > 0]


def raw_profit_by_trip(instance: Instance, sol: Sequence[Sequence[int]]) -> List[float]:
    return [sum(instance.site(i).profit for i in trip) for trip in sol]


def sort_trips_by_profit(instance: Instance, sol: Sequence[Sequence[int]]) -> Solution:
    """Return a copy with trips sorted by raw profit, highest first."""
    raw = raw_profit_by_trip(instance, sol)
    order = sorted(range(len(sol)), key=lambda k: (-raw[k], k))
    return [list(sol[k]) for k in order]


def sorted_objective(instance: Instance, sol: Sequence[Sequence[int]]) -> float:
    """Objective of the solution with its trips in the best (sorted) order."""
    return objective(instance, sort_trips_by_profit(instance, drop_empty_trips(sol)))


def trip_is_feasible(instance: Instance, trip: Sequence[int]) -> bool:
    return trip_time(instance, trip) <= instance.t_max + EPS


def solution_is_valid(instance: Instance, sol: Sequence[Sequence[int]]) -> bool:
    """True when check_solution finds no problems."""
    return not check_solution(instance, sol)


def improve_all_trips(instance: Instance, sol: Sequence[Sequence[int]]) -> Solution:
    """Apply the intra-trip search (2-opt + Or-opt) to every trip."""
    out: Solution = []
    for t in sol:
        out.append(list(improve_trip(instance, list(t))) if t else [])
    return out


def report_progress(
    callback: Optional[ProgressCallback],
    name: str,
    iteration: int,
    max_iters: int,
    best_obj: float,
) -> None:
    if callback is not None:
        callback(name, iteration, max_iters, best_obj)


# --------------------------------------------------------------------------
# Shared starting solution
# --------------------------------------------------------------------------
def build_start_solution(instance: Instance, start_method: str = "paper") -> Solution:
    """Starting solution shared by all algorithms.

    "paper"  -> 2-nearest greedy insertion (Wang & Wen, Section 4.1)
    "phase2" -> the Phase-2 initial solution

    The result has intra-trip improvement applied and trips sorted by profit.
    """
    if start_method == "paper":
        start = build_paper_initial_solution(instance)
    elif start_method == "phase2":
        start = build_initial_solution(instance)
    else:
        raise ValueError("start_method must be 'paper' or 'phase2'.")

    start = drop_empty_trips(improve_all_trips(instance, clone_solution(start)))
    return sort_trips_by_profit(instance, start)


# --------------------------------------------------------------------------
# Best-so-far tracker
# --------------------------------------------------------------------------
class BestTracker:
    """Keeps the best valid solution found and the convergence histories."""

    def __init__(self, instance: Instance, start_solution: Sequence[Sequence[int]]):
        self.instance = instance
        self.best: Solution = sort_trips_by_profit(instance, drop_empty_trips(start_solution))
        self.best_obj: float = objective(instance, self.best)
        self.history_best: List[float] = [self.best_obj]
        self.history_trips: List[int] = [len(self.best)]

    def offer(self, solution: Sequence[Sequence[int]]) -> bool:
        """Offer a solution. It is accepted only if it is valid and strictly better."""
        ordered = sort_trips_by_profit(self.instance, drop_empty_trips(solution))
        obj = objective(self.instance, ordered)
        if obj > self.best_obj + EPS and solution_is_valid(self.instance, ordered):
            self.best = ordered
            self.best_obj = obj
            return True
        return False

    def end_iteration(self) -> None:
        """Record the current best (call once per iteration)."""
        self.history_best.append(self.best_obj)
        self.history_trips.append(len(self.best))


# --------------------------------------------------------------------------
# Final result
# --------------------------------------------------------------------------
def start_timer() -> float:
    return time.perf_counter()


def finalize_result(
    name: str,
    instance: Instance,
    tracker: BestTracker,
    iterations_done: int,
    seed: int,
    started_at: float,
) -> AlgoResult:
    """Validate the best solution and package it as an AlgoResult."""
    best = sort_trips_by_profit(instance, drop_empty_trips(tracker.best))
    problems = check_solution(instance, best)
    if problems:
        raise RuntimeError(f"{name} produced an invalid solution:\n" + "\n".join(problems))

    return AlgoResult(
        algorithm=name,
        best_solution=best,
        best_objective=objective(instance, best),
        iterations_done=iterations_done,
        history_best=list(tracker.history_best),
        history_trips=list(tracker.history_trips),
        n_trips=len(best),
        raw_profit_by_trip=raw_profit_by_trip(instance, best),
        seed=seed,
        runtime_seconds=time.perf_counter() - started_at,
    )