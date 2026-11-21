"""GA (Genetic Algorithm) for the multi-trip UAV planner.

Comparison algorithm for Wang & Wen (2026), Section 5.2.

Paper setup followed here:
- Same starting solution as TS4GM / LNS / GRASP ("the same initial solutions").
- Same inter-trip moves (1-0, 0-n, 1-1) and intra-trip moves (2-opt, Or-opt)
  as TS4GM. They are reused from ts4gm.py.
- Paper GA parameters:
    tournament size k = 3
    population size    = 300
    crossover prob.    = 0.8
    mutation prob.     = 0.25
    max_iters          = 200
- Best of 5 independent runs is reported (paper protocol for all four
  algorithms; see Section 5.2).

Design choices we made ourselves (the paper does not specify them),
all documented in README.md ("Simplifications we made"):
- Chromosome = Solution = list of trips of site IDs (solution-as-genome).
  Crossover keeps a contiguous block of whole trips from one parent and
  fills the rest from the other; this preserves early high-profit trips
  across generations, which a flat-permutation OX would destroy because
  the objective discounts later trips by rho^(r-1).
- Crossover probability 0.8 means: with prob. 0.8 two parents produce a
  child; otherwise a clone of the fitter parent is used.
- Mutation probability 0.25 is applied per child, not per gene. A mutated
  child applies exactly one mutation operator (chosen uniformly):
    (i) a random feasible inter-trip move,
    (ii) improve_trip on one random trip,
    (iii) a feasible 1-1 swap between two trips.
- Repair: after crossover or mutation, lns._repair re-inserts any site
  that could not be placed; if it fails, the child is rejected and the
  fitter parent is cloned.
- Early stop: z_in consecutive generations without a new global best
  (same semantics as lns.run_lns). This is our addition for a fair
  comparison with LNS.
- Elitism: the global best individual always survives to the next
  generation.
- Initial population: one paper start + (POP_SIZE - 1) randomised
  variants, each made by applying random inter-trip moves to the paper
  start and then improve_trip on every trip.
"""

from __future__ import annotations

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
    improve_all_trips,
    report_progress,
    sort_trips_by_profit,
    sorted_objective,
    start_timer,
)
from .lns import _repair
from .model import Instance, Solution, check_solution
from .ts4gm import (
    _apply_move,
    _generate_0_n_moves,
    _generate_1_0_moves,
    _generate_1_1_moves,
)

# on_iteration callback: (generation, max_generations, best_obj, best_n_trips)
IterationCallback = Callable[[int, int, float, int], None]

# --------------------------------------------------------------------------
# Paper parameters (Section 5.2)
# --------------------------------------------------------------------------
DEFAULT_POP_SIZE = 300
DEFAULT_CROSSOVER_PROB = 0.8
DEFAULT_MUTATION_PROB = 0.25
DEFAULT_TOURNAMENT_K = 3
DEFAULT_MAX_ITERS = 200
DEFAULT_N_RUNS = 5
DEFAULT_Z_IN = 50
DEFAULT_MAX_11_CANDIDATES = 150  # same cap as TS4GM / GRASP


# --------------------------------------------------------------------------
# Initial population
# --------------------------------------------------------------------------
def _random_variant(
    instance: Instance,
    start: Solution,
    rng: random.Random,
    n_moves: int,
    max_11_candidates: Optional[int],
) -> Solution:
    """Apply `n_moves` random feasible inter-trip moves to a clone of start,
    then run improve_trip on every trip. Falls back to the (already
    feasible) current solution whenever a move yields an invalid child.
    """
    current = clone_solution(start)
    for _ in range(n_moves):
        moves = []
        moves.extend(_generate_1_0_moves(instance, current))
        moves.extend(_generate_0_n_moves(instance, current))
        m11 = _generate_1_1_moves(instance, current)
        if max_11_candidates is not None and len(m11) > max_11_candidates:
            m11 = rng.sample(m11, max_11_candidates)
        moves.extend(m11)
        if not moves:
            break
        mv = rng.choice(moves)
        cand, _, _ = _apply_move(instance, current, mv)
        if check_solution(instance, cand):
            continue
        current = sort_trips_by_profit(instance, drop_empty_trips(cand))
    current = improve_all_trips(instance, current)
    current = sort_trips_by_profit(instance, drop_empty_trips(current))
    return current


def _initial_population(
    instance: Instance,
    start: Solution,
    pop_size: int,
    rng: random.Random,
    max_11_candidates: Optional[int],
) -> List[Solution]:
    """One paper start + (pop_size - 1) randomised variants."""
    pop: List[Solution] = [clone_solution(start)]
    for _ in range(pop_size - 1):
        n_moves = rng.randint(2, 6)
        pop.append(
            _random_variant(
                instance, start, rng, n_moves, max_11_candidates
            )
        )
    return pop


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------
def _tournament(
    population: Sequence[Solution],
    fitness: Sequence[float],
    k: int,
    rng: random.Random,
) -> int:
    """Return the index of the winner of a k-way tournament."""
    idxs = rng.sample(range(len(population)), min(k, len(population)))
    return max(idxs, key=lambda i: fitness[i])


# --------------------------------------------------------------------------
# Crossover
# --------------------------------------------------------------------------
def _crossover(
    instance: Instance,
    parent_a: Solution,
    parent_b: Solution,
    rng: random.Random,
) -> Optional[Solution]:
    """Whole-trip block crossover + repair.

    Keeps a contiguous block of trips from parent A, fills the missing sites
    from parent B (in B's trip order), then repairs the child with
    lns._repair. Returns None if repair fails (then the caller clones the
    fitter parent instead).
    """
    if len(parent_a) == 0 or len(parent_b) == 0:
        return None

    # Block of whole trips from A.
    start = rng.randrange(len(parent_a))
    length = rng.randint(1, len(parent_a))
    block = [list(t) for t in parent_a[start:start + length]]
    taken = {v for trip in block for v in trip}

    # Collect the remaining sites from B, in B's trip order.
    remaining: List[int] = []
    for trip in parent_b:
        for v in trip:
            if v not in taken:
                remaining.append(v)

    # Child starts as the A-block; missing sites go into a new trip which is
    # then handed to _repair for re-insertion.
    child = clone_solution(block)
    if remaining:
        child.append(list(remaining))

    # _repair expects a partial solution and a list of sites to (re)insert.
    # If a site is already inside a trip, _repair skips it correctly (it only
    # inserts the sites in `removed`). So we pass `remaining` as `removed`.
    repaired = _repair(instance, child, remaining, rng)
    if repaired is None:
        return None

    repaired = sort_trips_by_profit(instance, drop_empty_trips(repaired))
    if check_solution(instance, repaired):
        return None
    return repaired


# --------------------------------------------------------------------------
# Mutation
# --------------------------------------------------------------------------
def _mutation(
    instance: Instance,
    solution: Solution,
    rng: random.Random,
    max_11_candidates: Optional[int],
) -> Solution:
    """Apply exactly one random mutation, uniformly from three kinds.

    (i)  a random feasible inter-trip move,
    (ii) improve_trip on one random trip,
    (iii) a feasible 1-1 swap between two trips.
    Falls back to the (already feasible) solution if the chosen mutation
    yields nothing usable.
    """
    kind = rng.randrange(3)

    if kind == 0:
        moves = []
        moves.extend(_generate_1_0_moves(instance, solution))
        moves.extend(_generate_0_n_moves(instance, solution))
        m11 = _generate_1_1_moves(instance, solution)
        if max_11_candidates is not None and len(m11) > max_11_candidates:
            m11 = rng.sample(m11, max_11_candidates)
        moves.extend(m11)
        if not moves:
            return clone_solution(solution)
        mv = rng.choice(moves)
        cand, _, _ = _apply_move(instance, solution, mv)
        if check_solution(instance, cand):
            return clone_solution(solution)
        return sort_trips_by_profit(instance, drop_empty_trips(cand))

    if kind == 1:
        if not solution:
            return clone_solution(solution)
        k = rng.randrange(len(solution))
        cand = clone_solution(solution)
        from .local_search import improve_trip  # local import keeps the file's
        # dependency on local_search explicit and avoids a module-level cycle
        cand[k] = list(improve_trip(instance, cand[k]))
        return sort_trips_by_profit(instance, drop_empty_trips(cand))

    # kind == 2: 1-1 swap between two different trips
    if len(solution) < 2:
        return clone_solution(solution)
    m11 = _generate_1_1_moves(instance, solution)
    if not m11:
        return clone_solution(solution)
    mv = rng.choice(m11)
    cand, _, _ = _apply_move(instance, solution, mv)
    if check_solution(instance, cand):
        return clone_solution(solution)
    return sort_trips_by_profit(instance, drop_empty_trips(cand))


# --------------------------------------------------------------------------
# One GA run
# --------------------------------------------------------------------------
def run_ga(
    instance: Instance,
    seed: int = 1,
    max_iters: int = DEFAULT_MAX_ITERS,
    z_in: Optional[int] = DEFAULT_Z_IN,
    start_method: str = "paper",
    pop_size: int = DEFAULT_POP_SIZE,
    crossover_prob: float = DEFAULT_CROSSOVER_PROB,
    mutation_prob: float = DEFAULT_MUTATION_PROB,
    tournament_k: int = DEFAULT_TOURNAMENT_K,
    max_11_candidates: Optional[int] = DEFAULT_MAX_11_CANDIDATES,
    progress: Optional[ProgressCallback] = None,
    on_iteration: Optional[IterationCallback] = None,
    max_trial_seconds: float = 0.0,          # <-- NEW
) -> AlgoResult:
    """One GA run with a fixed seed.

    - `progress`     : ProgressCallback, called as progress(name, it, max_iters, best_obj).
    - `on_iteration` : (it, max_iters, best_obj, best_n_trips), same as lns.run_lns.
    - `z_in`         : stop after this many generations without a new best.
    """
    if max_iters < 1:
        raise ValueError("max_iters must be >= 1.")
    if pop_size < 2:
        raise ValueError("pop_size must be >= 2.")
    if not 0.0 <= crossover_prob <= 1.0:
        raise ValueError("crossover_prob must be in [0, 1].")
    if not 0.0 <= mutation_prob <= 1.0:
        raise ValueError("mutation_prob must be in [0, 1].")
    if tournament_k < 1:
        raise ValueError("tournament_k must be >= 1.")
    if z_in is not None and z_in < 0:
        raise ValueError("z_in must be None or >= 0.")

    started = start_timer()
    if max_trial_seconds < 0:
        raise ValueError("max_trial_seconds must be >= 0.")
    deadline = (
        time.perf_counter() + max_trial_seconds
        if max_trial_seconds > 0
        else None
    )
    rng = random.Random(seed)
    start = build_start_solution(instance, start_method)

    pop = _initial_population(instance, start, pop_size, rng, max_11_candidates)
    fitness = [sorted_objective(instance, ind) for ind in pop]

    best_idx = max(range(len(pop)), key=lambda i: fitness[i])
    best = clone_solution(pop[best_idx])
    best_obj = fitness[best_idx]

    tracker = BestTracker(instance, start)
    # BestTracker starts at `start`; offer the GA best so history is exact.
    tracker.offer(best)
    # tracker.end_iteration() is called once per generation, so the initial
    # entry stays aligned with generation 0.

    history_best = [tracker.best_obj]
    history_trips = [len(tracker.best)]
    no_improve = 0
    iterations_done = 0

    for it in range(1, max_iters + 1):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        if z_in is not None and no_improve >= z_in:
            break
        iterations_done = it

        # Elitism: carry the best individual forward.
        new_pop: List[Solution] = [clone_solution(best)]
        new_fitness: List[float] = [best_obj]

        while len(new_pop) < pop_size:
            i_a = _tournament(pop, fitness, tournament_k, rng)
            i_b = _tournament(pop, fitness, tournament_k, rng)
            parent_a, parent_b = pop[i_a], pop[i_b]

            if rng.random() < crossover_prob:
                child = _crossover(instance, parent_a, parent_b, rng)
                if child is None:
                    child = clone_solution(parent_a if fitness[i_a] >= fitness[i_b] else parent_b)
            else:
                child = clone_solution(parent_a if fitness[i_a] >= fitness[i_b] else parent_b)

            if rng.random() < mutation_prob:
                child = _mutation(instance, child, rng, max_11_candidates)

            if check_solution(instance, child):
                # Repair failed at the child level; keep the fitter parent.
                child = clone_solution(parent_a if fitness[i_a] >= fitness[i_b] else parent_b)
            else:
                child = sort_trips_by_profit(instance, drop_empty_trips(child))

            obj = sorted_objective(instance, child)
            new_pop.append(child)
            new_fitness.append(obj)

        pop, fitness = new_pop, new_fitness

        gen_best_idx = max(range(len(pop)), key=lambda i: fitness[i])
        if fitness[gen_best_idx] > best_obj + EPS:
            best = clone_solution(pop[gen_best_idx])
            best_obj = fitness[gen_best_idx]
            no_improve = 0
        else:
            no_improve += 1

        tracker.offer(best)
        tracker.end_iteration()

        report_progress(progress, "GA", it, max_iters, tracker.best_obj)
        if on_iteration is not None:
            on_iteration(it, max_iters, tracker.best_obj, len(tracker.best))

    return finalize_result("GA", instance, tracker, iterations_done, seed, started)


# --------------------------------------------------------------------------
# Public: trials
# --------------------------------------------------------------------------
def run_ga_trials(
    instance: Instance,
    n_runs: int = DEFAULT_N_RUNS,
    seed0: int = 1,
    max_iters: int = DEFAULT_MAX_ITERS,
    z_in: Optional[int] = DEFAULT_Z_IN,
    start_method: str = "paper",
    pop_size: int = DEFAULT_POP_SIZE,
    crossover_prob: float = DEFAULT_CROSSOVER_PROB,
    mutation_prob: float = DEFAULT_MUTATION_PROB,
    tournament_k: int = DEFAULT_TOURNAMENT_K,
    max_11_candidates: Optional[int] = DEFAULT_MAX_11_CANDIDATES,
    progress: Optional[ProgressCallback] = None,
    on_iteration: Optional[IterationCallback] = None,
    max_trial_seconds: float = 0.0,
) -> AlgoResult:
    """Paper protocol: n_runs independent runs, keep the best."""
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1.")

    best_res: Optional[AlgoResult] = None
    total_started = start_timer()
    for k in range(n_runs):
        res = run_ga(
            instance=instance,
            seed=seed0 + k,
            max_iters=max_iters,
            z_in=z_in,
            start_method=start_method,
            pop_size=pop_size,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            tournament_k=tournament_k,
            max_11_candidates=max_11_candidates,
            progress=progress,
            on_iteration=on_iteration,
            max_trial_seconds=max_trial_seconds,
        )
        if best_res is None or res.best_objective > best_res.best_objective + EPS:
            best_res = res

    assert best_res is not None
    best_res.runtime_seconds = time.perf_counter() - total_started
    return best_res