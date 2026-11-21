"""TS4GM metaheuristic (paper-aligned baseline for Version 1).

Based on Wang & Wen (2026), Section 4:
- Initial solution: 2-nearest greedy insertion ("paper" start) or Phase-2 start.
- Intra-trip moves: 2-opt and Or-opt (via improve_trip).
- Inter-trip moves: 1-0 swap, 0-n swap, 1-1 swap.
- Tabu list per trip: a vertex removed from trip r cannot return to trip r
  for tau[r] iterations.
    tau_max  = floor(n / 4)
    tau_init = int(0.5 * tau_max)
    tau[r] -= 1 when a new best solution is found,
    tau[r] += 3 after 5 consecutive non-improving iterations,
    reset to tau_init if outside [1, tau_max].
- Aspiration: a tabu move is allowed if it beats the global best objective.
- Stop: max_iters reached OR z_in iterations without improvement.
- Multiple trials: run_ts4gm_trials keeps the best of n_runs.

Trip ordering (important):
- The objective discounts trip r by rho^(r-1), so the best order is always
  "highest raw profit first" (as in the paper's Table 5).
- The search therefore (a) scores every candidate with its trips sorted by
  raw profit, and (b) keeps the current solution sorted after every accepted
  move. Tabu memory and tau values are remapped when trips change position,
  so "trip index" always means "rank by raw profit".

Speed note:
- max_11_candidates limits how many 1-1 swap candidates are sampled per
  iteration (the 1-1 neighbourhood is by far the largest).
  Set it to None to evaluate the full 1-1 neighbourhood (slower).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .initial import build_initial_solution
from .initial_paper import build_paper_initial_solution
from .local_search import improve_trip
from .model import Instance, Solution, check_solution, objective, trip_time

EPS = 1e-9

Move = Tuple[str, Dict]


@dataclass
class TS4GMResult:
    best_solution: Solution
    best_objective: float
    iterations_done: int
    history_best: List[float]
    n_trips: int
    raw_profit_by_trip: List[float]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _clone_solution(sol: Sequence[Sequence[int]]) -> Solution:
    return [list(t) for t in sol]


def _normalize_no_empty(solution: Solution) -> Solution:
    return [t for t in solution if len(t) > 0]


def _raw_profit_by_trip(instance: Instance, sol: Sequence[Sequence[int]]) -> List[float]:
    return [sum(instance.site(i).profit for i in trip) for trip in sol]


def _trip_order(instance: Instance, sol: Sequence[Sequence[int]]) -> List[int]:
    """Trip indices sorted by raw profit, highest first (ties keep their order)."""
    raw = _raw_profit_by_trip(instance, sol)
    return sorted(range(len(sol)), key=lambda k: (-raw[k], k))


def _sort_trips(
    instance: Instance, sol: Sequence[Sequence[int]]
) -> Tuple[Solution, List[int]]:
    """Return (sorted copy of the solution, order) where order[new] = old index."""
    order = _trip_order(instance, sol)
    return [list(sol[k]) for k in order], order


def _sorted_objective(instance: Instance, sol: Sequence[Sequence[int]]) -> float:
    """Objective of the solution with its trips in the best (sorted) order."""
    ordered, _ = _sort_trips(instance, sol)
    return objective(instance, ordered)


def _feasible_trip(instance: Instance, trip: Sequence[int]) -> bool:
    return trip_time(instance, trip) <= instance.t_max + EPS


def _improve(instance: Instance, trip: List[int]) -> List[int]:
    if not trip:
        return trip
    return list(improve_trip(instance, trip))


def _apply_intra_trip_all(instance: Instance, solution: Solution) -> Solution:
    return [_improve(instance, t) for t in solution]


# --------------------------------------------------------------------------
# Move generators
# --------------------------------------------------------------------------
def _generate_1_0_moves(instance: Instance, current: Solution) -> List[Move]:
    """Move one vertex from trip r to the nearest feasible later trip q > r."""
    moves: List[Move] = []
    h = len(current)
    for r in range(h):
        for idx, v in enumerate(current[r]):
            for q in range(r + 1, h):
                best_pos = None
                best_time = None
                for pos in range(len(current[q]) + 1):
                    cand_q = current[q][:pos] + [v] + current[q][pos:]
                    if _feasible_trip(instance, cand_q):
                        t = trip_time(instance, cand_q)
                        if best_time is None or t < best_time - EPS:
                            best_time = t
                            best_pos = pos
                if best_pos is not None:
                    moves.append(
                        (
                            "1-0",
                            {
                                "from_trip": r,
                                "to_trip": q,
                                "from_idx": idx,
                                "vertex": v,
                                "to_pos": best_pos,
                            },
                        )
                    )
                    break  # nearest feasible subsequent trip only
    return moves


def _generate_0_n_moves(instance: Instance, current: Solution) -> List[Move]:
    """Move as many vertices as possible from a later trip q into an earlier trip r."""
    moves: List[Move] = []
    h = len(current)
    for r in range(h - 1):
        for q in range(r + 1, h):
            if not current[q]:
                continue
            src = list(current[q])
            dst = list(current[r])
            moved: List[int] = []
            changed = True
            while changed and src:
                changed = False
                best_pick = None  # (key, src_idx, dst_pos, vertex)
                for s_idx, v in enumerate(src):
                    best_pos = None
                    best_t = None
                    for pos in range(len(dst) + 1):
                        cand = dst[:pos] + [v] + dst[pos:]
                        if _feasible_trip(instance, cand):
                            t = trip_time(instance, cand)
                            if best_t is None or t < best_t - EPS:
                                best_t = t
                                best_pos = pos
                    if best_pos is not None:
                        key = (best_t, -instance.site(v).profit, v, best_pos)
                        if best_pick is None or key < best_pick[0]:
                            best_pick = (key, s_idx, best_pos, v)
                if best_pick is not None:
                    _, s_idx, best_pos, v = best_pick
                    dst.insert(best_pos, v)
                    src.pop(s_idx)
                    moved.append(v)
                    changed = True
            if moved:
                moves.append(
                    (
                        "0-n",
                        {
                            "to_trip": r,
                            "from_trip": q,
                            "moved_vertices": moved,
                            "new_dst": dst,  # exact feasible route for trip r
                        },
                    )
                )
    return moves


def _generate_1_1_moves(instance: Instance, current: Solution) -> List[Move]:
    """Swap one vertex between two different trips (feasible swaps only)."""
    moves: List[Move] = []
    h = len(current)
    for r in range(h):
        for q in range(r + 1, h):
            for i, vr in enumerate(current[r]):
                for j, vq in enumerate(current[q]):
                    cand_r = current[r][:i] + [vq] + current[r][i + 1:]
                    cand_q = current[q][:j] + [vr] + current[q][j + 1:]
                    if _feasible_trip(instance, cand_r) and _feasible_trip(instance, cand_q):
                        moves.append(
                            (
                                "1-1",
                                {
                                    "trip_a": r,
                                    "trip_b": q,
                                    "idx_a": i,
                                    "idx_b": j,
                                    "va": vr,
                                    "vb": vq,
                                },
                            )
                        )
    return moves


# --------------------------------------------------------------------------
# Apply a move
# --------------------------------------------------------------------------
def _apply_move(
    instance: Instance,
    current: Solution,
    move: Move,
) -> Tuple[Solution, List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Apply a move, run intra-trip improvement on changed trips only.

    Returns (new_solution, removals, additions), where
      removals  = [(vertex, trip_it_was_removed_from)]
      additions = [(vertex, trip_it_was_added_to)]
    """
    kind, d = move
    cand = _clone_solution(current)
    removals: List[Tuple[int, int]] = []
    additions: List[Tuple[int, int]] = []
    changed: List[int] = []

    if kind == "1-0":
        r, q = d["from_trip"], d["to_trip"]
        v = cand[r].pop(d["from_idx"])
        cand[q].insert(d["to_pos"], v)
        removals.append((v, r))
        additions.append((v, q))
        changed = [r, q]

    elif kind == "1-1":
        a, b = d["trip_a"], d["trip_b"]
        i, j = d["idx_a"], d["idx_b"]
        va, vb = cand[a][i], cand[b][j]
        cand[a][i] = vb
        cand[b][j] = va
        removals.extend([(va, a), (vb, b)])
        additions.extend([(vb, a), (va, b)])
        changed = [a, b]

    elif kind == "0-n":
        r, q = d["to_trip"], d["from_trip"]
        moved = set(d["moved_vertices"])
        cand[q] = [v for v in cand[q] if v not in moved]
        cand[r] = list(d["new_dst"])
        for v in d["moved_vertices"]:
            removals.append((v, q))
            additions.append((v, r))
        changed = [r, q]

    else:
        raise ValueError(f"Unknown move type: {kind}")

    for k in changed:
        cand[k] = _improve(instance, cand[k])

    return _normalize_no_empty(cand), removals, additions


def _is_tabu(
    tabu_until: Dict[Tuple[int, int], int],
    iteration: int,
    additions: Sequence[Tuple[int, int]],
) -> bool:
    for v, to_trip in additions:
        if tabu_until.get((v, to_trip), -1) >= iteration:
            return True
    return False


# --------------------------------------------------------------------------
# Main TS4GM run
# --------------------------------------------------------------------------
def run_ts4gm(
    instance: Instance,
    seed: int = 1,
    max_iters: int = 200,
    z_in: int = 50,
    start_method: str = "paper",  # "paper" or "phase2"
    max_11_candidates: Optional[int] = 150,
) -> TS4GMResult:
    rng = random.Random(seed)

    if start_method == "paper":
        current = build_paper_initial_solution(instance)
    elif start_method == "phase2":
        current = build_initial_solution(instance)
    else:
        raise ValueError("start_method must be 'paper' or 'phase2'.")

    current = _normalize_no_empty(_apply_intra_trip_all(instance, _clone_solution(current)))
    current, _ = _sort_trips(instance, current)  # trip 1 = highest raw profit
    best = _clone_solution(current)
    best_obj = objective(instance, best)

    n = instance.n
    tau_max = max(1, n // 4)
    tau_init = max(1, int(0.5 * tau_max))
    tau = [tau_init for _ in current]

    tabu_until: Dict[Tuple[int, int], int] = {}
    no_improve = 0          # iterations since last new global best (stop rule)
    streak = 0              # consecutive non-improving iterations (tau rule)
    history_best = [best_obj]
    iterations_done = 0

    for it in range(1, max_iters + 1):
        if no_improve >= z_in:
            break
        iterations_done = it

        if len(tau) < len(current):
            tau.extend([tau_init] * (len(current) - len(tau)))
        elif len(tau) > len(current):
            tau = tau[: len(current)]

        moves: List[Move] = []
        moves.extend(_generate_1_0_moves(instance, current))
        moves.extend(_generate_0_n_moves(instance, current))
        m11 = _generate_1_1_moves(instance, current)
        if max_11_candidates is not None and len(m11) > max_11_candidates:
            m11 = rng.sample(m11, max_11_candidates)
        moves.extend(m11)

        if not moves:
            break

        rng.shuffle(moves)

        best_cand = None
        best_cand_obj = None
        best_removals: List[Tuple[int, int]] = []

        for mv in moves:
            cand, removals, additions = _apply_move(instance, current, mv)

            if check_solution(instance, cand):  # non-empty list = problems found
                continue

            # Score the candidate with its trips in the best (sorted) order.
            cand_obj = _sorted_objective(instance, cand)

            tabu_violation = _is_tabu(tabu_until, it, additions)
            aspiration = cand_obj > best_obj + EPS
            if tabu_violation and not aspiration:
                continue

            if best_cand is None or cand_obj > best_cand_obj + EPS:
                best_cand = cand
                best_cand_obj = cand_obj
                best_removals = removals

        if best_cand is None:
            # every candidate was infeasible or tabu
            no_improve += 1
            streak += 1
            if streak >= 5:
                tau = [t + 3 for t in tau]
                streak = 0
            tau = [t if 1 <= t <= tau_max else tau_init for t in tau]
            history_best.append(best_obj)
            continue

        trips_before = len(current)
        current = best_cand

        if len(current) != trips_before:
            # a trip disappeared, so trip indices shifted: reset tabu memory
            tabu_until = {}
            tau = [tau_init for _ in current]
            current, _ = _sort_trips(instance, current)
        else:
            for v, removed_from in best_removals:
                if removed_from < len(tau):
                    tabu_until[(v, removed_from)] = it + tau[removed_from]
            # Re-sort the trips and move tabu memory / tau along with them.
            current, order = _sort_trips(instance, current)
            new_pos = {old: new for new, old in enumerate(order)}
            tabu_until = {
                (v, new_pos[k]): until
                for (v, k), until in tabu_until.items()
                if k in new_pos
            }
            if len(tau) == len(order):
                tau = [tau[k] for k in order]

        if best_cand_obj > best_obj + EPS:
            best = _clone_solution(current)
            best_obj = best_cand_obj
            no_improve = 0
            streak = 0
            tau = [t - 1 for t in tau]
        else:
            no_improve += 1
            streak += 1
            if streak >= 5:
                tau = [t + 3 for t in tau]
                streak = 0

        tau = [t if 1 <= t <= tau_max else tau_init for t in tau]
        history_best.append(best_obj)

    best = _normalize_no_empty(best)
    best, _ = _sort_trips(instance, best)
    best_obj = objective(instance, best)  # exact objective of the sorted solution
    problems = check_solution(instance, best)
    if problems:
        raise RuntimeError("TS4GM produced invalid solution:\n" + "\n".join(problems))

    return TS4GMResult(
        best_solution=best,
        best_objective=best_obj,
        iterations_done=iterations_done,
        history_best=history_best,
        n_trips=len(best),
        raw_profit_by_trip=_raw_profit_by_trip(instance, best),
    )


def run_ts4gm_trials(
    instance: Instance,
    n_runs: int = 5,
    seed0: int = 1,
    max_iters: int = 200,
    z_in: int = 50,
    start_method: str = "paper",
    max_11_candidates: Optional[int] = 150,
) -> TS4GMResult:
    """Paper-style protocol: run several independent trials and keep the best."""
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1.")

    best_res: Optional[TS4GMResult] = None
    for k in range(n_runs):
        res = run_ts4gm(
            instance=instance,
            seed=seed0 + k,
            max_iters=max_iters,
            z_in=z_in,
            start_method=start_method,
            max_11_candidates=max_11_candidates,
        )
        if best_res is None or res.best_objective > best_res.best_objective + EPS:
            best_res = res

    assert best_res is not None
    return best_res