"""LNS (Large Neighborhood Search) for the multi-trip UAV planner.

Comparison algorithm for Wang & Wen (2026), Section 5.2.

Paper setup followed here:
- Same initial solution as TS4GM (paper-style 2-nearest greedy insertion).
- Same intra-trip moves (2-opt, Or-opt) and inter-trip moves
  (1-0, 0-n, 1-1) as TS4GM; they are reused directly from ts4gm.py.
- Max 200 iterations.

Design choices (the paper does not specify them):
- Destroy: remove q sites, q random in [5%, 20%] of n, using one of
    * random removal
    * worst removal (largest time saved per unit of profit, randomised)
    * last-trip removal (empties the lowest-priority trip)
- Repair: reinsert removed sites, highest profit first (with a little
  noise), into the earliest feasible trip at the cheapest position.
  If no trip fits, a new trip is opened.
- Local search: after repair, intra-trip improvement on every trip, then
  a short best-improvement descent with the inter-trip moves.
- Acceptance: simulated-annealing style (better always accepted, worse
  accepted with probability exp(delta / T), T cooling geometrically).
- Stop: max_iters, or z_in iterations without a new global best.

Trip ordering: as in ts4gm.py, trips are always kept sorted by raw
profit (highest first), because the objective discounts later trips.

Optional hooks (used by the browser interface):
- on_iteration(it, max_iters, best_obj, best_n_trips)
- should_stop() -> bool   (return True to cancel the run)
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Set, Tuple

from .initial import build_initial_solution
from .initial_paper import build_paper_initial_solution
from .model import Instance, Solution, check_solution, objective, trip_time
from .ts4gm import (
    EPS,
    _apply_intra_trip_all,
    _apply_move,
    _clone_solution,
    _feasible_trip,
    _generate_0_n_moves,
    _generate_1_0_moves,
    _generate_1_1_moves,
    _normalize_no_empty,
    _raw_profit_by_trip,
    _sort_trips,
    _sorted_objective,
)


@dataclass
class LNSResult:
    best_solution: Solution
    best_objective: float
    iterations_done: int
    history_best: List[float]          # best objective after each iteration
    history_trips: List[int]           # number of trips of the best solution
    n_trips: int
    raw_profit_by_trip: List[float]
    seed: int = 1
    runtime_seconds: float = 0.0
    cancelled: bool = False
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _all_vertices(sol: Sequence[Sequence[int]]) -> List[int]:
    return [v for trip in sol for v in trip]


def _remove_vertices(sol: Sequence[Sequence[int]], removed: Set[int]) -> Solution:
    return _normalize_no_empty([[v for v in trip if v not in removed] for trip in sol])


def _trip_time_or_zero(instance: Instance, trip: Sequence[int]) -> float:
    return trip_time(instance, trip) if trip else 0.0


# --------------------------------------------------------------------------
# Destroy operators: each returns (partial_solution, removed_vertices)
# --------------------------------------------------------------------------
def _destroy_random(
    instance: Instance, sol: Solution, q: int, rng: random.Random
) -> Tuple[Solution, List[int]]:
    verts = _all_vertices(sol)
    removed = rng.sample(verts, min(q, len(verts)))
    return _remove_vertices(sol, set(removed)), removed


def _destroy_worst(
    instance: Instance, sol: Solution, q: int, rng: random.Random, p: float = 3.0
) -> Tuple[Solution, List[int]]:
    """Remove sites that cost a lot of flight time for little profit.

    Randomised selection (exponent p): larger p means greedier choices.
    """
    scored: List[Tuple[float, int]] = []
    for trip in sol:
        base = trip_time(instance, trip)
        for idx, v in enumerate(trip):
            rest = trip[:idx] + trip[idx + 1:]
            saving = base - _trip_time_or_zero(instance, rest)
            profit = max(instance.site(v).profit, 0.1)
            scored.append((saving / profit, v))
    scored.sort(key=lambda x: -x[0])

    removed: List[int] = []
    while scored and len(removed) < q:
        k = int(len(scored) * (rng.random() ** p))
        removed.append(scored.pop(k)[1])
    return _remove_vertices(sol, set(removed)), removed


def _destroy_last_trip(
    instance: Instance, sol: Solution, q: int, rng: random.Random
) -> Tuple[Solution, List[int]]:
    """Empty (part of) the last, lowest-priority trip to help cut trips."""
    pool = list(sol[-1])
    if len(pool) > q:
        removed = rng.sample(pool, q)
    else:
        removed = pool
        others = [v for trip in sol[:-1] for v in trip]
        extra = min(len(others), max(0, q - len(pool)))
        if extra > 0:
            removed = removed + rng.sample(others, extra)
    return _remove_vertices(sol, set(removed)), removed


_DESTROY_OPERATORS = (_destroy_random, _destroy_worst, _destroy_last_trip)


# --------------------------------------------------------------------------
# Repair operator
# --------------------------------------------------------------------------
def _repair(
    instance: Instance,
    partial: Solution,
    removed: Sequence[int],
    rng: random.Random,
) -> Optional[Solution]:
    """Greedy reinsertion. Returns None if some site cannot be placed."""
    sol = _clone_solution(partial)

    # Highest profit first; a little noise keeps the search diverse.
    order = sorted(removed, key=lambda v: -instance.site(v).profit + rng.random())

    for v in order:
        placed = False
        for k in range(len(sol)):  # earliest trip first = highest discounted profit
            best_pos = None
            best_added = None
            base_time = trip_time(instance, sol[k]) if sol[k] else 0.0
            for pos in range(len(sol[k]) + 1):
                cand = sol[k][:pos] + [v] + sol[k][pos:]
                if _feasible_trip(instance, cand):
                    added = trip_time(instance, cand) - base_time
                    if best_added is None or added < best_added - EPS:
                        best_added = added
                        best_pos = pos
            if best_pos is not None:
                sol[k].insert(best_pos, v)
                placed = True
                break
        if not placed:
            new_trip = [v]
            if not _feasible_trip(instance, new_trip):
                return None
            sol.append(new_trip)
    return _normalize_no_empty(sol)


# --------------------------------------------------------------------------
# Local search with the TS4GM inter-trip moves (best-improvement descent)
# --------------------------------------------------------------------------
def _inter_trip_descent(
    instance: Instance,
    solution: Solution,
    rng: random.Random,
    passes: int,
    max_11_candidates: Optional[int],
) -> Tuple[Solution, float]:
    cur, _ = _sort_trips(instance, solution)
    cur_obj = objective(instance, cur)

    for _ in range(passes):
        moves = []
        moves.extend(_generate_1_0_moves(instance, cur))
        moves.extend(_generate_0_n_moves(instance, cur))
        m11 = _generate_1_1_moves(instance, cur)
        if max_11_candidates is not None and len(m11) > max_11_candidates:
            m11 = rng.sample(m11, max_11_candidates)
        moves.extend(m11)
        if not moves:
            break
        rng.shuffle(moves)

        best_cand = None
        best_obj = cur_obj
        for mv in moves:
            cand, _, _ = _apply_move(instance, cur, mv)
            cand_obj = _sorted_objective(instance, cand)
            if cand_obj > best_obj + EPS and not check_solution(instance, cand):
                best_cand = cand
                best_obj = cand_obj

        if best_cand is None:
            break  # local optimum for these moves
        cur, _ = _sort_trips(instance, best_cand)
        cur_obj = best_obj

    return cur, cur_obj


# --------------------------------------------------------------------------
# Main LNS run
# --------------------------------------------------------------------------
def run_lns(
    instance: Instance,
    seed: int = 1,
    max_iters: int = 200,
    z_in: Optional[int] = 50,
    start_method: str = "paper",  # "paper" or "phase2"
    max_11_candidates: Optional[int] = 150,
    ls_passes: int = 2,
    destroy_min_frac: float = 0.05,
    destroy_max_frac: float = 0.20,
    on_iteration: Optional[Callable[[int, int, float, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> LNSResult:
    t_start = time.perf_counter()
    rng = random.Random(seed)

    if start_method == "paper":
        current = build_paper_initial_solution(instance)
    elif start_method == "phase2":
        current = build_initial_solution(instance)
    else:
        raise ValueError("start_method must be 'paper' or 'phase2'.")

    current = _normalize_no_empty(_apply_intra_trip_all(instance, _clone_solution(current)))
    current, _ = _sort_trips(instance, current)
    cur_obj = objective(instance, current)

    best = _clone_solution(current)
    best_obj = cur_obj

    n = instance.n
    q_min = max(2, int(destroy_min_frac * n))
    q_max = max(q_min, int(destroy_max_frac * n))

    temp_start = 0.005 * abs(best_obj) if best_obj else 1.0
    temp_end = 0.0002 * abs(best_obj) if best_obj else 0.01

    history_best = [best_obj]
    history_trips = [len(best)]
    no_improve = 0
    iterations_done = 0
    cancelled = False

    for it in range(1, max_iters + 1):
        if z_in is not None and no_improve >= z_in:
            break
        if should_stop is not None and should_stop():
            cancelled = True
            break
        iterations_done = it

        # --- destroy ---
        q = rng.randint(q_min, q_max)
        destroy = rng.choice(_DESTROY_OPERATORS)
        partial, removed = destroy(instance, current, q, rng)

        # --- repair + local search ---
        cand: Optional[Solution] = None
        cand_obj = None
        rebuilt = _repair(instance, partial, removed, rng)
        if rebuilt is not None:
            rebuilt = _normalize_no_empty(_apply_intra_trip_all(instance, rebuilt))
            cand, cand_obj = _inter_trip_descent(
                instance, rebuilt, rng, ls_passes, max_11_candidates
            )
            if check_solution(instance, cand):  # non-empty list = problems
                cand = None

        # --- acceptance ---
        if cand is not None and cand_obj is not None:
            frac = (it - 1) / max(1, max_iters - 1)
            temp = temp_start * (temp_end / temp_start) ** frac
            delta = cand_obj - cur_obj
            if delta >= -EPS or rng.random() < math.exp(delta / max(temp, 1e-12)):
                current = cand
                cur_obj = cand_obj

            if cand_obj > best_obj + EPS:
                best = _clone_solution(cand)
                best_obj = cand_obj
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        history_best.append(best_obj)
        history_trips.append(len(best))

        if on_iteration is not None:
            on_iteration(it, max_iters, best_obj, len(best))

    best = _normalize_no_empty(best)
    best, _ = _sort_trips(instance, best)
    best_obj = objective(instance, best)
    problems = check_solution(instance, best)
    if problems:
        raise RuntimeError("LNS produced invalid solution:\n" + "\n".join(problems))

    return LNSResult(
        best_solution=best,
        best_objective=best_obj,
        iterations_done=iterations_done,
        history_best=history_best,
        history_trips=history_trips,
        n_trips=len(best),
        raw_profit_by_trip=_raw_profit_by_trip(instance, best),
        seed=seed,
        runtime_seconds=time.perf_counter() - t_start,
        cancelled=cancelled,
    )


def run_lns_trials(
    instance: Instance,
    n_runs: int = 5,
    seed0: int = 1,
    max_iters: int = 200,
    z_in: Optional[int] = 50,
    start_method: str = "paper",
    max_11_candidates: Optional[int] = 150,
    ls_passes: int = 2,
    on_iteration: Optional[Callable[[int, int, float, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> LNSResult:
    """Paper-style protocol: several independent runs, keep the best."""
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1.")

    t_start = time.perf_counter()
    best_res: Optional[LNSResult] = None
    for k in range(n_runs):
        res = run_lns(
            instance=instance,
            seed=seed0 + k,
            max_iters=max_iters,
            z_in=z_in,
            start_method=start_method,
            max_11_candidates=max_11_candidates,
            ls_passes=ls_passes,
            on_iteration=on_iteration,
            should_stop=should_stop,
        )
        if best_res is None or res.best_objective > best_res.best_objective + EPS:
            best_res = res
        if res.cancelled:
            break

    assert best_res is not None
    best_res.runtime_seconds = time.perf_counter() - t_start  # total over all runs
    return best_res