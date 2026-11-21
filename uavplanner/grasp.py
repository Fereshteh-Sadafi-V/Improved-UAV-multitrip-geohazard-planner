"""GRASP metaheuristic (comparison baseline, Version 2 — aligned with the paper).

Paper parameters (Wang & Wen, 2026, Section 5.2):
- Same starting solution as TS4GM and LNS ("the same initial solutions").
- Greediness rate 0.25.
- Local search of 5 iterations per GRASP iteration.
- Same inter-trip operations (1-0 swap, 0-n swap, 1-1 swap) as TS4GM.
- Same intra-trip operations (2-opt, Or-opt) as TS4GM.
- Maximum 200 iterations per trial; 5 trials per case, best reported.

Design choices we made ourselves (the paper does not specify them), all
documented in README.md:
- Candidate list size = ceil(greediness * |improving candidates|), minimum 1.
  The paper only gives the rate, not the exact rounding.
- GRASP has no early-stop counter (no z_in). The paper's finding is that
  GRASP fails to reach 4 trips within 200 iterations; an early stop would
  hide that.
- Optional safety caps max_trial_seconds, max_moves_per_ls and
  max_candidates_eval are disabled by default (paper mode). They are meant
  only for the browser, where runtime is expensive.
- All trial results are kept in AlgoResult form (from algo_common), so the
  web page can show every algorithm side by side.
"""

from __future__ import annotations

import math
import random
import time
from typing import Callable, List, Optional, Sequence, Tuple

from .algo_common import (
    AlgoResult,
    BestTracker,
    EPS,
    ProgressCallback,
    build_start_solution,
    clone_solution,
    drop_empty_trips,
    finalize_result,
    report_progress,
    sort_trips_by_profit,
    sorted_objective,
    start_timer,
)
from .model import Instance, Solution, check_solution
from .ts4gm import (
    _apply_move,
    _generate_0_n_moves,
    _generate_1_0_moves,
    _generate_1_1_moves,
)

# --------------------------------------------------------------------------
# Paper parameters (Section 5.2)
# --------------------------------------------------------------------------
GREEDINESS_RATE = 0.25
LOCAL_SEARCH_ITERS = 5
DEFAULT_MAX_ITERS = 200
DEFAULT_N_RUNS = 5
DEFAULT_MAX_11_CANDIDATES = 150  # same cap as TS4GM, for a fair comparison

# Safety / performance defaults. All zero (or None) = paper mode, no caps.
DEFAULT_MAX_TRIAL_SECONDS = 0.0
DEFAULT_MAX_MOVES_PER_LS: Optional[int] = None
DEFAULT_MAX_CANDIDATES_EVAL: Optional[int] = None

# on_iteration callback: (iteration, max_iters, best_obj, best_n_trips)
IterationCallback = Callable[[int, int, float, int], None]


# --------------------------------------------------------------------------
# Neighbourhood and candidate selection
# --------------------------------------------------------------------------
def _neighbourhood(
    instance: Instance,
    current: Solution,
    rng: random.Random,
    max_11_candidates: Optional[int],
    max_moves_per_ls: Optional[int],
) -> list:
    """All inter-trip moves of the paper's neighbourhood for `current`.

    Sampling caps are optional; when None the full neighbourhood is built.
    """
    moves = []
    moves.extend(_generate_1_0_moves(instance, current))
    moves.extend(_generate_0_n_moves(instance, current))
    m11 = _generate_1_1_moves(instance, current)
    if max_11_candidates is not None and len(m11) > max_11_candidates:
        m11 = rng.sample(m11, max_11_candidates)
    moves.extend(m11)

    if max_moves_per_ls is not None and max_moves_per_ls > 0 and len(moves) > max_moves_per_ls:
        moves = rng.sample(moves, max_moves_per_ls)
    return moves


def _improving_candidates(
    instance: Instance,
    current: Solution,
    current_obj: float,
    moves: list,
    rng: random.Random,
    max_candidates_eval: Optional[int],
) -> List[Tuple[float, Solution]]:
    """Feasible candidates that strictly improve `current_obj`.

    The candidate is scored with its trips sorted by raw profit, which is the
    best order for the objective (same convention as TS4GM and LNS).
    """
    if max_candidates_eval is not None and max_candidates_eval > 0 and len(moves) > max_candidates_eval:
        moves = rng.sample(moves, max_candidates_eval)

    scored: List[Tuple[float, Solution]] = []
    for mv in moves:
        cand, _, _ = _apply_move(instance, current, mv)
        if check_solution(instance, cand):  # non-empty list = problems found
            continue
        obj = sorted_objective(instance, cand)
        if obj > current_obj + EPS:
            scored.append((obj, cand))
    return scored


def _restricted_candidate_list(
    scored: List[Tuple[float, Solution]],
    greediness: float,
) -> List[Tuple[float, Solution]]:
    """Keep the best ceil(greediness * |scored|) candidates (minimum 1).

    Our own rounding: the paper gives the rate only.
    """
    scored.sort(key=lambda item: -item[0])
    size = max(1, math.ceil(greediness * len(scored)))
    return scored[:size]


# --------------------------------------------------------------------------
# One randomized local search (one GRASP iteration)
# --------------------------------------------------------------------------
def _randomized_local_search(
    instance: Instance,
    start: Solution,
    rng: random.Random,
    greediness: float,
    ls_iters: int,
    max_11_candidates: Optional[int],
    max_moves_per_ls: Optional[int],
    max_candidates_eval: Optional[int],
    deadline: Optional[float],
) -> Solution:
    current = clone_solution(start)
    current_obj = sorted_objective(instance, current)

    for _ in range(ls_iters):
        if deadline is not None and time.perf_counter() >= deadline:
            break

        moves = _neighbourhood(
            instance,
            current,
            rng,
            max_11_candidates=max_11_candidates,
            max_moves_per_ls=max_moves_per_ls,
        )
        if not moves:
            break

        improving = _improving_candidates(
            instance,
            current,
            current_obj,
            moves,
            rng=rng,
            max_candidates_eval=max_candidates_eval,
        )
        if not improving:
            break  # local optimum reached early

        rcl = _restricted_candidate_list(improving, greediness)
        current_obj, cand = rng.choice(rcl)
        current = sort_trips_by_profit(instance, drop_empty_trips(cand))

    return current


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------
def run_grasp(
    instance: Instance,
    seed: int = 1,
    max_iters: int = DEFAULT_MAX_ITERS,
    start_method: str = "paper",
    greediness: float = GREEDINESS_RATE,
    ls_iters: int = LOCAL_SEARCH_ITERS,
    max_11_candidates: Optional[int] = DEFAULT_MAX_11_CANDIDATES,
    progress: Optional[ProgressCallback] = None,
    on_iteration: Optional[IterationCallback] = None,
    max_trial_seconds: float = DEFAULT_MAX_TRIAL_SECONDS,
    max_moves_per_ls: Optional[int] = DEFAULT_MAX_MOVES_PER_LS,
    max_candidates_eval: Optional[int] = DEFAULT_MAX_CANDIDATES_EVAL,
) -> AlgoResult:
    """One GRASP trial with a fixed seed.

    - `progress`     : ProgressCallback, called as progress(name, it, max_iters, best_obj).
    - `on_iteration` : (it, max_iters, best_obj, best_n_trips), same as lns.run_lns.
    - `max_trial_seconds`, `max_moves_per_ls`, `max_candidates_eval` are our
      own safety caps; leave them at defaults for paper behaviour.
    """
    if not 0.0 < greediness <= 1.0:
        raise ValueError("greediness must be in (0, 1].")
    if max_iters < 1:
        raise ValueError("max_iters must be >= 1.")
    if ls_iters < 1:
        raise ValueError("ls_iters must be >= 1.")
    if max_trial_seconds < 0:
        raise ValueError("max_trial_seconds must be >= 0.")
    if max_moves_per_ls is not None and max_moves_per_ls < 0:
        raise ValueError("max_moves_per_ls must be None or >= 0.")
    if max_candidates_eval is not None and max_candidates_eval < 0:
        raise ValueError("max_candidates_eval must be None or >= 0.")

    started = start_timer()
    rng = random.Random(seed)
    start = build_start_solution(instance, start_method)
    tracker = BestTracker(instance, start)

    deadline = (time.perf_counter() + max_trial_seconds) if max_trial_seconds > 0 else None

    iterations_done = 0
    for it in range(1, max_iters + 1):
        if deadline is not None and time.perf_counter() >= deadline:
            break

        iterations_done = it
        sol = _randomized_local_search(
            instance=instance,
            start=start,
            rng=rng,
            greediness=greediness,
            ls_iters=ls_iters,
            max_11_candidates=max_11_candidates,
            max_moves_per_ls=max_moves_per_ls,
            max_candidates_eval=max_candidates_eval,
            deadline=deadline,
        )
        tracker.offer(sol)
        tracker.end_iteration()

        report_progress(progress, "GRASP", it, max_iters, tracker.best_obj)
        if on_iteration is not None:
            on_iteration(it, max_iters, tracker.best_obj, len(tracker.best))

    return finalize_result(
        "GRASP", instance, tracker, iterations_done, seed, started
    )


def run_grasp_trials(
    instance: Instance,
    n_runs: int = DEFAULT_N_RUNS,
    seed0: int = 1,
    max_iters: int = DEFAULT_MAX_ITERS,
    start_method: str = "paper",
    greediness: float = GREEDINESS_RATE,
    ls_iters: int = LOCAL_SEARCH_ITERS,
    max_11_candidates: Optional[int] = DEFAULT_MAX_11_CANDIDATES,
    progress: Optional[ProgressCallback] = None,
    on_iteration: Optional[IterationCallback] = None,
    max_trial_seconds: float = DEFAULT_MAX_TRIAL_SECONDS,
    max_moves_per_ls: Optional[int] = DEFAULT_MAX_MOVES_PER_LS,
    max_candidates_eval: Optional[int] = DEFAULT_MAX_CANDIDATES_EVAL,
) -> AlgoResult:
    """Paper protocol: n_runs independent trials, best result reported."""
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1.")

    best_res: Optional[AlgoResult] = None
    total_started = start_timer()
    for k in range(n_runs):
        res = run_grasp(
            instance=instance,
            seed=seed0 + k,
            max_iters=max_iters,
            start_method=start_method,
            greediness=greediness,
            ls_iters=ls_iters,
            max_11_candidates=max_11_candidates,
            progress=progress,
            on_iteration=on_iteration,
            max_trial_seconds=max_trial_seconds,
            max_moves_per_ls=max_moves_per_ls,
            max_candidates_eval=max_candidates_eval,
        )
        if best_res is None or res.best_objective > best_res.best_objective + EPS:
            best_res = res

    assert best_res is not None
    # Total wall-clock over all trials (same convention as run_lns_trials).
    best_res.runtime_seconds = time.perf_counter() - total_started
    return best_res