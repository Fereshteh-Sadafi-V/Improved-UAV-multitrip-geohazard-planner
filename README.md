# Improved UAV Multi-trip Geohazard Planner

A free tool that plans **multi-trip routes for a drone (UAV)** that must inspect
many geohazard sites, such as landslides, collapses and debris flows. It runs
**in your web browser**. There is nothing to install and no account to create.

The planning method is the **TS4GM tabu search** from:

> Wang, W., & Wen, C. (2026). Multi-trip path planning for UAV with autonomous
> docking station in geohazards monitoring. *Computers & Industrial Engineering,
> 211*, 111578. https://doi.org/10.1016/j.cie.2025.111578

The tool also implements three comparison algorithms — **LNS**, **GRASP** and
**GA** — so the user can reproduce, on their own data, the comparative study in
Section 5.2 of that paper, with the same starting solution and the same
neighbourhood operators.

This project is an **independent re-implementation** based on the paper. It is
not the original authors' code, and it is not affiliated with or endorsed by them.

> **Try it online:** https://YOUR-GITHUB-USERNAME.github.io/Improved-UAV-multitrip-geohazard-planner/

---

## What problem does this tool solve?

Geohazards such as landslides and debris flows are often triggered by heavy rain
or earthquakes. Drones are a low-cost way to monitor them, but a drone can only
fly for a limited time on one battery (the paper's drone flies about 100 to 140
minutes). If there are many sites, one flight is not enough.

So the drone is based at an **autonomous docking station (ADS)**. After each
flight it returns to the station to recharge and upload its data, then flies
the next **trip**.

The tool decides:

1. **Which sites** are inspected on each trip.
2. **In what order** they are visited.

The rules are:

- Every site is inspected **exactly once**.
- Each trip must start and end at the docking station and must **not exceed the
  maximum flight time** (called `Tmax`).
- The goal is to **collect as much profit as possible** while using **as few
  trips as possible**.

## What is "profit"?

Each site gets a **profit from 1 to 4** that shows how urgent it is to inspect.
The paper derives it from two expert ratings: how **stable** the site is
(Good, Weak, Poor, Very Poor) and how **dangerous** it is (Minor, Moderate,
High, Very High). Profit 4 means the most urgent.

Earlier monitoring is more useful for warnings and emergency response, so profit
**decreases with each trip**. A site's profit on trip *r* is:
profit on trip r = original profit × ρ^(r-1) (ρ is between 0 and 1)


The paper uses **ρ = 0.9**. This pushes urgent sites into early trips.

**Worked example (from the paper's real case, ρ = 0.9).** The four trips collect
original profits of 81, 81, 32 and 6:

| Trip | Original profit | Multiplier | Counted profit |
|---|---|---|---|
| 1 | 81 | 1.000 | 81.000 |
| 2 | 81 | 0.900 | 72.900 |
| 3 | 32 | 0.810 | 25.920 |
| 4 | 6 | 0.729 | 4.374 |
| **Total** | | | **184.194** |

---

## How to use the tool (3 steps)

1. **Open the web page** (link above). Wait until the status line says
   *"Ready. Choose algorithms, a preset, and press Run."* The first load can
   take a minute because Python is starting inside your browser.
2. **Choose which algorithms to compare**:
   - **TS4GM** is the reference algorithm from the paper and is always included
     (you cannot turn it off).
   - **LNS**, **GRASP**, **GA** are toggle buttons. Click one to turn it on or
     off. The default is **LNS on** (so the page starts with a 2-algorithm
     comparison), and **GRASP** and **GA off**.
   - The Run button label updates to show how many algorithms are selected.
3. **Press Run** and read the results. The page stays responsive while the
   algorithms run; the status line shows elapsed seconds and which algorithm is
   executing.

> Only the **Quick** preset is available from the web page. Standard and
> Paper-style presets are far too slow for a browser tab, so they are only
> available from the command line (see "Running it on your own computer"
> below). The page will tell you this.

## What you will see

- **Comparison summary table**: one row per algorithm, plus the paper-style
  initial solution as a baseline. The best objective is in bold.
- **Convergence chart** (equivalent to Fig. 5 of the paper): one line per
  algorithm, with the initial-solution value as a dashed grey baseline.
- **Show details for** dropdown: switch between the algorithms' individual
  results without re-running.
- **Results summary and trip table** for the selected algorithm.
- **Route map** for the selected algorithm.
- **Convergence chart** for the selected algorithm.
- **Download buttons (CSV files)**:
  - *Trip summary*, *Routes*, *Convergence*: for the selected algorithm.
  - *Comparison summary*: one row per algorithm (only appears after a
    comparison run).

---

## The four algorithms in this project

All four use the **same starting solution**, the same **instance**, the same
**seed** and the same **preset** — this is the fairness requirement from the
paper's Section 5.2.

- **TS4GM** (the paper's method). Tabu search with the paper's inter-trip moves
  (1→0, 0←n, 1↔1) and intra-trip moves (2-opt, Or-opt), tabu list per trip with
  the paper's dynamic τ rule and aspiration criterion. Maximum 200 iterations,
  early stop after `z_in = 50` non-improving iterations. Best of 5 trials.
- **LNS**. Same neighbourhood operators as TS4GM, random / worst / last-trip
  destroy, greedy repair, simulated-annealing acceptance. Maximum 200
  iterations, early stop after `z_in = 50` non-improving iterations. Best of 5
  trials.
- **GRASP**. Paper parameters: greediness rate **0.25**, local search of **5**
  iterations per GRASP iteration, maximum **200** iterations per trial, **5**
  trials per case. No early stop (GRASP does not converge within 200 iterations
  on the paper's case — that is one of the paper's findings).
- **GA**. Paper parameters: tournament selection with **k = 3**, population
  size **300**, crossover probability **0.8**, mutation probability **0.25**,
  maximum **200** generations. Early stop after `z_in = 50` non-improving
  generations. Best of 5 runs.

The paper's result is that **TS4GM** finds the best solutions and converges
fastest; the LNS algorithm converges to the paper's trip count within 200
iterations; the GA converges in some cases; the GRASP algorithm does not
converge within 200 iterations on the paper's case. **These conclusions only
appear when the algorithms run to the paper's iteration limits** (see the
Paper-style preset on the command line). The browser's Quick preset is
heavily truncated and may produce different rankings.

---

## About the example data

The tool currently uses **synthetic (artificial) example data**: 88 sites, the
same number as the paper's case, but with invented locations and profits.

The paper's real data (southern Shaanxi Province, China, 2,000 km²) is available
from its authors on request. Because our data is different, **the numbers
produced by this tool cannot be compared directly with the paper's results**
(for example, the paper's 184.194). A default service time of **120 seconds per
site** is used here, while the paper sets service time per site from its type
and area.

Because the search involves randomness, two runs may give slightly different
results. This is normal for this kind of method. The paper runs each case five
times and keeps the best result.

---

## How does TS4GM work? (plain language)

TS4GM is a **tabu search**. It starts with a reasonable plan and keeps improving
it step by step.

1. **Starting plan.** Sites are added to trips one by one, always choosing the
   most profitable of the two nearest unvisited sites ("2-Nearest Greedy
   Insertion").
2. **Improvement moves.** The search tries small changes:
   - *Inside one trip:* reorder the route to make it shorter (**2-opt**,
     **Or-opt**), which can free time for more sites.
   - *Between trips:* move or swap sites so urgent sites fly earlier
     (**1-0 swap**, **0-n swap**, **1-1 swap**).
3. **Tabu list (memory).** A site that was just removed from a trip is temporarily
   not allowed back into it. This stops the search from going in circles. The
   ban is lifted if the move gives a new best result.
4. **Stopping.** The search stops after a set number of iterations, or when it
   has not improved for a set number of iterations.

Finding the perfect plan is extremely hard for many sites (the problem is
NP-hard), so TS4GM looks for a very good plan in reasonable time.

---

## Simplifications we made (not in the paper)

The paper describes the model and the four algorithms, but it does not spell
out every implementation detail. Wherever we had to make a choice, it is listed
below. **None of these change the mathematical model or the published
parameter values.** They are implementation convenience choices, chosen so the
tool fits in a browser and finishes in reasonable time.

**In the browser page**

- Only the **Quick** preset is available in the browser. Standard and
  Paper-style are far too slow for a browser tab (a Paper-style TS4GM run is
  several minutes even on a fast desktop, and a four-algorithm comparison is
  much longer). Both presets remain available from the command line.
- **Default toggles** are TS4GM (reference) + LNS on, GRASP and GA off. This is
  a lighter default that works on smaller laptops. The user can turn any
  combination on.
- The web page's **Quick** preset uses reduced iteration counts for GRASP
  compared to the other algorithms (GRASP's per-iteration cost is much higher).
  The paper's parameters are fully restored in **Standard** and **Paper-style**.

**Presets and iteration counts**

- The paper fixes only two things for the comparison:
  `max_iters = 200` and `n_runs = 5`. Everything else (per-algorithm iteration
  counts in Quick and Standard, GA `z_in`, etc.) is our choice. The exact
  numbers are visible in `uavplanner/api.py` under `PRESETS`.
- **GA early stop.** The paper specifies `z_in` only for TS4GM. We apply the
  same early-stop rule to the GA (`ga_z_in`) so all four algorithms can be
  compared fairly. GRASP has **no** early stop — that is why the paper reports
  GRASP failing to reach the paper's trip count within 200 iterations.

**Starting solutions**

- The paper says the LNS and GRASP use "the same initial solutions" as TS4GM;
  we apply that to the GA too.
- The GA's **initial population** is one paper-style start plus `pop_size − 1`
  randomised variants (random inter-trip moves applied to the paper start, then
  `improve_trip` on every trip). The paper does not specify a GA initialisation
  strategy.
- The GA **crossover** keeps a contiguous block of whole trips from one parent
  and fills the remaining sites from the other parent, then repairs with
  `lns._repair`. A flat-permutation order-crossover was rejected because the
  paper's objective is not trip-symmetric (later trips are discounted by
  `ρ^(r-1)`); whole-trip blocks preserve early high-profit trips across
  generations, which matches the paper's aim. The paper does not specify
  crossover structure.
- The GA **mutation** is applied per child (probability `0.25`) and chooses
  one of: a random feasible inter-trip move, `improve_trip` on one random trip,
  or a feasible 1-1 swap between two trips. The paper gives only the
  probability value.
- The GA **repair** on a failed child: the child is discarded and the fitter
  parent is cloned. The paper does not specify a repair strategy.

**GRASP**

- Candidate list size = `ceil(greediness * |improving candidates|)`, minimum 1.
  The paper gives the rate (`0.25`) only, not the rounding rule.
- GRASP has **no** early-stop counter, matching the paper's reported behaviour
  that GRASP does not converge within 200 iterations on the paper's case.

**LNS**

- The paper says the LNS is "adapted to use the same inter-trip and intra-trip
  operations ... and the same initial solutions with 8 trips". We implement
  that with three destroy operators (random, worst, last-trip), greedy repair,
  simulated-annealing acceptance, and a short best-improvement descent on the
  inter-trip moves after each repair. The paper does not specify the destroy
  and repair operators.

**Drivers and CLI**

- The `try_grasp.py` and `try_ga.py` drivers default to a **fast** mode
  (`try_grasp`: 1 run × 8 iterations × 2 LS iterations, with a 120 s wall-time
  budget; `try_ga`: 1 run × 15 generations × pop 30). The paper protocol is
  restored with `--paper`. The library modules (`grasp.py`, `ga.py`) keep the
  paper defaults, so anyone who calls `run_grasp_trials()` / `run_ga_trials()`
  directly gets the paper values.
- `check_phase4.py` (headless equivalent of the browser comparison view)
  defaults to the **Quick** preset; use `--paper` for the paper's parameters.

**Other**

- `index.html` loads all Python modules into PyScript's filesystem. The page's
  heavy work runs in a background thread (`asyncio.to_thread`) so the browser
  tab stays responsive. That is an implementation detail and does not affect
  results.

---

## What is planned next

- **Upload your own CSV file** with your own sites. *(Not available yet.)*

---

## Running it on your own computer (optional)

You only need this if you want to work on the code. You need Python 3 installed.

1. Download or clone this repository.
2. Open a terminal in the project folder.
3. Start a small local web server:
python -m http.server 8000


4. Open `http://localhost:8000/index.html` in your browser.

Do not open `index.html` by double-clicking it. Browsers block the Python files
from loading that way.

### Command-line verification scripts

These are the checks used during development. Each prints PASS/FAIL lines and
saves CSVs to `results/`:
python check_phase1.py
python check_phase2.py
python check_phase2b.py
python check_phase3.py (TS4GM paper protocol; add --quick to speed up)
python check_phase4.py (all four algorithms; add --paper for paper mode)
python try_grasp.py (GRASP; --quick default, --paper for paper mode)
python try_lns.py (LNS; add --quick to speed up)
python try_ga.py (GA; --quick default, --paper for paper mode)


Every script writes CSVs to the `results/` folder. The paper-style runs on the
88-site instance take minutes; the quick modes take under two minutes.

---

## Project structure

| File or folder | Purpose |
|---|---|
| `index.html` | The web page you interact with |
| `main.py` | Connects the buttons and charts to the planner |
| `uavplanner/` | The planning code (four algorithms and helpers) |
| `check_phase1.py` ... `check_phase4.py` | Development verification scripts |
| `try_grasp.py`, `try_lns.py`, `try_ga.py` | Standalone drivers for the comparison algorithms |
| `results/` | Output folder for CSV files written by the scripts |
| `LICENSE` | The MIT license |

Inside `uavplanner/`:

| Module | Purpose |
|---|---|
| `model.py` | Data model, travel time, objective, feasibility check |
| `data.py` | Synthetic data and CSV reading/writing |
| `local_search.py` | 2-opt and Or-opt (intra-trip improvement) |
| `initial.py` | Documented two-nearest greedy insertion initializer |
| `initial_paper.py` | Paper-style 2-Nearest Greedy Insertion initializer |
| `ordering.py` | Sort trips by raw profit (highest first) |
| `ts4gm.py` | TS4GM algorithm |
| `lns.py` | LNS algorithm |
| `grasp.py` | GRASP algorithm |
| `ga.py` | GA algorithm |
| `algo_common.py` | Shared building blocks used by all four algorithms |
| `svgplots.py` | Small SVG chart helpers (map and convergence charts) |
| `api.py` | One-call interface used by the browser and the scripts |
| `__init__.py` | Package exports |

## Reference

Wang, W., & Wen, C. (2026). Multi-trip path planning for UAV with autonomous
docking station in geohazards monitoring. *Computers & Industrial Engineering,
211*, 111578. https://doi.org/10.1016/j.cie.2025.111578

## Authors

Mostafa Hajiaghaei-Keshteli, Arya Bavar, Avesta Bavar, Fereshteh Sadafi Vishkaei,

## License

Released under the MIT License. See the `LICENSE` file. In short, you may use,
copy, modify and share this software, provided the copyright notice is kept.
The software comes without any warranty.