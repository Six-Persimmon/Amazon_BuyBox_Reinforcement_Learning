# MARL Pricing Simulation with Rule-Based Strategies

## 1. Overview (Aligned with `exp02`)
This package implements a multi-agent reinforcement learning (MARL) framework for price competition among $N$ sellers in an oligopoly. Agents do **not** choose continuous prices. Instead, they select from a discrete set of **rule-based pricing strategies** (e.g., undercut, match, above, reset). The goal is to study dynamics such as Edgeworth cycles, collusion, and equilibrium outcomes over a discretized price grid.

The `exp02` experiment is a **parameter sweep** over $(N, \mu, K)$ and strategy sets. It reuses the same environment and learning pipeline described below, but runs **many seeds in parallel**, saves results to Parquet, and organizes outputs by experiment name. The overview in this README is consistent with `exp02`; the main difference is that `exp02` is orchestration code for large-scale batch runs, not a single toy script.

## 2. Model & Environment

### A. Demand and Profit
The market uses a Logit demand system. For seller $i$ with price $p_i$:

$$
D_i(p) = \frac{\exp(\frac{a - p_i}{\mu})}{\sum_{j=1}^{N} \exp(\frac{a - p_j}{\mu}) + \exp(\frac{a_0}{\mu})}
$$

Profit is $\pi_i = (p_i - c) \cdot D_i$.

### B. Price Grid Construction
Prices are discretized into a grid of size $M$ (`num_grids`). We compute the stage-game Nash price $p^N$ and monopoly price $p^M$ via numerical optimization, then build a linear grid. In the current environment implementation, the grid is constructed so that one grid point lies below $p^N$ and one grid point lies above $p^M$:

$$
	ext{step} = \frac{p^M - p^N}{M-3}, \quad p_{grid} = \text{linspace}(p^N-\text{step},\; p^M+\text{step},\; M)
$$

### C. State, Observation, Action
* **State ($S_t$):** the full vector of price indices for all sellers.
* **Observation ($O_{i,t}$):** the **lowest price index** in the market.
* **Action:** a rule that maps the current price index and the competitors’ minimum price to the next index.

### D. Learning
Agents use independent Q-learning with $\epsilon$-greedy exploration and exponential decay. Q-tables are initialized via a heuristic computed from the environment lookup table to accelerate convergence.

## 3. How `exp02` Uses the Framework
`pricing_marl/experiments/exp02_heatmap_scan.py` runs large-scale sweeps:
1. Builds a `Config` for each $(N, \mu, K)$ and selected strategy set.
2. Initializes a `PricingEnvironment`, which computes the price grid and loads/creates a lookup table.
3. Runs multiple seeds in parallel via `joblib`, each seed executing the full train–evaluate pipeline.
4. Saves evaluation data to Parquet in `pricing_marl/data/results/<experiment_name>/N_<n>`.

## 4. Repository Structure (pricing_marl)

### Top-level
* `README.md` — This document.
* `experiments/` — Entry points for batch experiments.
* `src/` — Core environment, agents, and simulation logic.
* `analysis/` — Visualization, figure generation, and analysis utilities.
* `data/` — Lookup tables and experiment results (Parquet).

### `experiments/`
* `exp01_initial_study.py` — Early experiment runner (single study sweep).
* `exp02_heatmap_scan.py` — Main batch sweep used for large-scale experiments (the script you plan to run on HPC).

### `src/`
* `config.py` — Central configuration dataclass (model parameters, RL parameters, paths).
* `environment.py` — Pricing environment: computes $p^N$, $p^M$, builds the grid, creates/loads lookup tables, and provides fast step evaluation.
* `strategies.py` — Rule definitions and integer strategy IDs.
* `agent.py` — Independent Q-learning agents and heuristic initialization from the lookup table.
* `simulation.py` — End-to-end run: training (until convergence) + evaluation (detailed trajectories).
* `runner.py` — Batch orchestrator for experiments; parallel execution + Parquet export.
* `simulation_Dec24_2025.py` — Archived variant of the simulation loop.
* `(old)run_experiment.py` — Legacy runner script.

### `analysis/`
* `PARQUET_DATA_EXPLANATION.md` — Schema and usage notes for saved Parquet outputs.
* `plot_utils.py` — Common plotting utilities.
* `visualize_physics.ipynb` — Micro-dynamics visualization for rule combinations.
* `viz_heatmap_*.py` — Scripts to aggregate and visualize sweep results.
* `viz_history.ipynb`, `demo_exp_data.ipynb`, `viz_all_runs_grid.ipynb` — Exploratory notebooks.
* `figures/` — Output figures organized by experiment.

### `data/`
* `lookup_tables/` — Cached lookup tables keyed by config.
* `results/` — Experiment outputs (Parquet files + JSON configs).

## 5. Notes for HPC Deployment (exp02)
* `exp02_heatmap_scan.py` is the entry point for large sweeps.
* The script is embarrassingly parallel across $(N, \mu, K)$ and across random seeds.
* Parallelism can be controlled via environment variables:
	* `PRICING_MARL_N_JOBS` (highest priority)
	* `SLURM_CPUS_PER_TASK` (used if the above is not set)
	* If neither is set, joblib uses all available cores.
* Parquet outputs can be large; ensure adequate scratch storage and I/O bandwidth.
* Lookup tables are cached per config; consider a shared filesystem to reuse them across jobs.
* Lookup tables are protected by a file lock with timeout and retry to avoid duplicate computation.
* Parquet compression uses `zstd` for portability and better HPC compatibility.

## 6. 2026-02-25 Grid Floor Fix (Important)

Data stored as 2026-02-28 backup. Figure stored as 2026-03-11 backpack

### Why this change was needed
In the old grid construction, we used:

$$
\text{step} = \frac{p^M - p^N}{M-3}, \quad p_{grid} = \text{linspace}(p^N-\text{step},\; p^M+\text{step},\; M)
$$

This guarantees:
* Nash index = 1
* Monopoly index = 8 (when `num_grids=10`)

But for low `mu`, `p_grid[0]` can fall below marginal cost `c` (default `c=1`), creating irrational negative-profit punishment states.

### Environment change in `src/environment.py`
Grid logic is still the old design (Nash at 1, Monopoly at 8), with one added floor:

* Build the grid exactly as before.
* If `price_grid[0] < c`, set `price_grid[0] = c`.
* Otherwise do nothing.

This means most parameter cells are unchanged; only low-`mu` cells where old minimum was below cost are affected.

Implementation details:
* Added fields:
  * `grid_min_before_floor`
  * `grid_floor_applied`
  * `grid_min_after_floor`
* Lookup cache filename now conditionally adds suffix `_floorC` **only** when floor is applied.  
  This avoids accidentally reusing an old lookup table built with the pre-fix grid.

### `exp02` change in `experiments/exp02_heatmap_scan.py`
Added optional env-var filters for targeted reruns (without editing code):

* `PRICING_MARL_FILTER_N` (comma-separated ints)
* `PRICING_MARL_FILTER_MU` (comma-separated floats)
* `PRICING_MARL_FILTER_K` (comma-separated ints)
* `PRICING_MARL_ROUNDS_PER_CONFIG` (optional int override)
* `PRICING_MARL_EXPERIMENT_SET` (`3strats` or `4strats`)

These are used to rerun only affected cells instead of full heatmap.

### New analysis helper
`analysis/scan_grid_floor_cases.py` was added to:
* parse `N_VALUES/MU_VALUES/K_VALUES/ROUNDS_PER_CONFIG` from `exp02_heatmap_scan.py`
* compute which `(N, mu)` cells trigger `price_grid[0] < c`
* print summary tables (including `K x mu` style)
* generate CSV files:
  * `analysis/tables/grid_floor_scan_by_n_mu.csv`
  * `analysis/tables/grid_floor_rerun_manifest.csv`

## 7. Affected Cells Under Current `exp02` Grid

With current `exp02` axes (`N=[2,3,5,7,10]`, `mu=[0.01..0.31]`, `K=[10..100]`, `c=1`, `num_grids=10`), floor is applied for:

* `N=2`: `mu in {0.01, 0.04}`
* `N=3`: `mu in {0.01, 0.04, 0.07}`
* `N=5`: `mu in {0.01, 0.04, 0.07}`
* `N=7`: `mu in {0.01, 0.04, 0.07, 0.10}`
* `N=10`: `mu in {0.01, 0.04, 0.07, 0.10}`

To regenerate this list:

```bash
python analysis/scan_grid_floor_cases.py
```

## 8. 2026-03-01 exp02 Update (Q-table Snapshot + New Defaults)

This section documents the latest `exp02` changes made on **2026-03-01**.

Data: results 2026_3_12 Figure: 2026_3_12

### A. Q-table snapshots now include both `init` and `final`
Goal: store each seller's Q-table at two key times:
* `init`: right after heuristic initialization (before any training update)
* `final`: right when training ends and evaluation (greedy policy) starts

Implementation:
* `src/simulation.py`: builds per-run Q snapshots for both `init` and `final`.
* `src/runner.py`: saves that snapshot together with evaluation parquet.

Output files (under the same `results` folder as before):
* Evaluation trajectory: `run_<run_id>.parquet`
* Q snapshot: `run_<run_id>_qtable.parquet`

Typical path:
* `pricing_marl/data/results/<experiment_name>/N_<N>/run_<run_id>_qtable.parquet`

Q snapshot schema:
* `run_id`: run/seed id
* `seller_id`: seller index (`0 ... N-1`)
* `state_idx`: observation/state grid index
* `action_idx`: local action index in `active_strategies`
* `action_id`: global strategy id (e.g., undercut/match/above/reset id)
* `q_value`: Q(s, a)
* `qtable_type`: snapshot type (`init` or `final`)
* `training_stop_episode`: episode id when training ended (converged or hit max episodes)
  * `init` rows use `-1`
  * `final` rows use actual stop episode

Row count per run:
* `2 * num_sellers * num_grids * num_actions`

Example (real smoke-test sample):

| run_id | seller_id | state_idx | action_idx | action_id | q_value  | qtable_type | training_stop_episode |
|---|---:|---:|---:|---:|---:|---|---:|
| 0 | 0 | 0 | 0 | 0 | 0.000000 | init | -1 |
| 0 | 0 | 0 | 1 | 1 | 1.176599 | final | 5 |
| 0 | 0 | 0 | 2 | 2 | 2.307650 | final | 5 |

### B. `eval_H` reduced
In `experiments/exp02_heatmap_scan.py`, evaluation horizon is now:
* `eval_H = 2_000`

### C. Convergence threshold changed
In `experiments/exp02_heatmap_scan.py`, convergence threshold is now:
* `converge_period = 100_000`  (previously its 10_000)

Important definition (current code behavior in `src/simulation.py`):
* Convergence is checked by comparing the **full greedy policy matrix** of all sellers across all states between consecutive training episodes.
* If unchanged for `converge_period` consecutive episodes, run is marked converged.
* This is **not** "atomic actions unchanged for X steps"; it is "greedy policy unchanged for X episodes."

### D. `mu=0.01` removed from exp02 sweep
`MU_VALUES` in `experiments/exp02_heatmap_scan.py` now excludes `0.01`.
Current list:
* `[0.04, 0.07, 0.1, 0.13, 0.16, 0.19, 0.22, 0.25, 0.28, 0.31]`

### E. Server rerun strategy (reuse lookup tables)
For this update, lookup-table-defining parameters are unchanged relative to the post-2026-02-25 setup.  
So on HPC, a clean rerun can be done by:
1. Back up local `data/results` (optional but recommended).
2. Keep server `data/lookup_tables` as-is.
3. Clear server `data/results` only.
4. Re-submit `heatmap.sbatch`.
5. Monitor with `progress_report.py` using `paired progress` (eval + qtable).
6. Download server `data/results` back to local.

Reference: `HPC_PROGRESS_GUIDE.md` section **"2026-03-01 新实验重跑清单（含 Q-table 快照）"**.

### F. Analysis scripts and parquet selection rule
Because each run directory now contains both eval and qtable parquet files:
* eval: `run_<id>.parquet`
* qtable: `run_<id>_qtable.parquet`

Analysis scripts must read **eval files only** (`run_<id>.parquet` with numeric `<id>`).  
The main `analysis/*.py` loaders were updated accordingly to avoid mixing qtable files into metric calculations.

## 9. 2026-03-05 Init Q-table Bug Fix + Bulk Backfill Guide

This section documents a bug discovered during large HPC reruns and the official repair workflow.

### A. Problem observed
In downloaded qtable files, `qtable_type="init"` and `qtable_type="final"` were sometimes identical.

Root cause:
* In older `src/simulation.py`, both snapshots were generated **after** training.
* Result: `init` label existed, but values were actually post-training values.

### B. Code fix
`src/simulation.py` is fixed so that:
* `init` snapshot is captured immediately after agent initialization (before training updates).
* `final` snapshot is captured at training end (same as before).

After this fix, new runs should produce meaningfully different `init` vs `final` in most cases.

### C. Impact on existing results
For already-generated qtable files (from the buggy version):
* `final` rows are still valid.
* `init` rows are incorrect and should be repaired.
* Evaluation trajectory files (`run_<id>.parquet`) are unaffected.

### D. Bulk repair script (post-download, local)
Use `fix_initial_qtables.py` after all server results are downloaded to local `pricing_marl/data/results`.

Script behavior:
* Recomputes true init Q-values from config + lookup table.
* Replaces only `qtable_type == "init"` rows in each qtable parquet.
* Leaves non-init rows (e.g., `final`) unchanged.
* Does not touch evaluation parquet.
* Config priority:
  * First try `Config_N_<N>.json`.
  * If missing or parse fails, fallback to parsing `scan_<label>_mu*_k*` directory name.

Recommended commands:

```bash
cd pricing_marl

# 1) quick test (no write)
python fix_initial_qtables.py --results-root data/results --dry-run --limit-files 20

# 2) full scan (no write)
python fix_initial_qtables.py --results-root data/results --dry-run

# 3) apply repair
python fix_initial_qtables.py --results-root data/results
```

### E. Expected repaired output
For each `run_<id>_qtable.parquet`:
* still contains both `init` and `final`.
* row count remains `2 * N * num_grids * num_actions`.
* `init` rows now have `training_stop_episode = -1` and recomputed pre-training Q-values.
* `final` rows remain original post-training values.

Quick validation example:

```bash
python - <<'PY'
import glob, random, pandas as pd
p = random.choice(sorted(glob.glob('data/results/scan_*/N_*/run_*_qtable.parquet')))
q = pd.read_parquet(p)
key = ['run_id','seller_id','state_idx','action_idx','action_id']
m = q[q.qtable_type=='init'].merge(q[q.qtable_type=='final'], on=key, suffixes=('_i','_f'))
print('sample:', p)
print('qtable_type counts:', q['qtable_type'].value_counts().to_dict())
print('max_abs_diff(init,final):', float((m['q_value_i']-m['q_value_f']).abs().max()))
PY
```

### F. Debug notebook update
`analysis/debug_pkl_file_check.ipynb` now includes qtable diagnostics:
* verifies the `2 * N` table structure (`N` sellers × `init/final`).
* applies the same init-fix logic in-memory for inspection.
* prints per-seller init/final Q-tables with `action_id -> action_name` mapping for readability.

## 10. 2026-03-12 K=30 Rerun with Old Convergence Rule

This section documents the next rerun plan after the `100_000`-episode convergence definition produced unsatisfactory results.

### A. Convergence rule reverted
Current `exp02` code in `experiments/exp02_heatmap_scan.py` uses:
* `converge_period = 10_000`

This restores the earlier rule:
* if the full greedy-policy matrix of all sellers is unchanged for `10_000` consecutive training episodes, the run is marked converged.

This supersedes the `100_000` setting described in Section 8C for current reruns.

### B. Q-table recording status
Current `src/simulation.py` correctly records both Q snapshots:
* `init`: captured immediately after heuristic Q initialization, before any training update
* `final`: captured when training ends, right before greedy evaluation begins

Expected qtable structure per run:
* `2 * num_sellers * num_grids * num_actions` rows
* `qtable_type in {"init", "final"}`

### C. New rerun scope
The new rerun is intentionally restricted to:
* `K = 30` only
* all `N` values in current `exp02`
* all `mu` values in current `exp02`
* both strategy sets (`3strats`, `4strats`)
* `100` runs per parameter combination
* `eval_H = 2_000`
* qtable saving enabled

### D. New result location
The K=30 rerun writes to a separate folder:
* `pricing_marl/data/result_K30_qtable`

This avoids mixing the new rerun with prior `data/results` outputs.

### E. Lookup table reuse
This rerun should **not** require new lookup tables.
Reason:
* lookup tables depend on environment-side parameters such as `N`, `mu`, strategy set, grid construction, and `K`
* this rerun keeps the same relevant environment settings as the already-generated `K=30` cells
* changing `converge_period`, `eval_H`, or result directory does not affect lookup-table contents

### F. New sbatch entry point
Use:
* `heatmap_k30_qtable_turingvm.sbatch`

This sbatch:
* runs on `turingvm`
* forces `PRICING_MARL_FILTER_K="30"`
* writes outputs to `data/result_K30_qtable`
* keeps `100` rounds per configuration

### G. Progress monitoring
For this K=30-only rerun, `progress_report.py` should be pointed to the custom result root:

```bash
cd ~/bigdata/pricing_marl
python progress_report.py \
  --root ~/bigdata/pricing_marl/data/result_K30_qtable \
  --rounds 100 \
  --grid-file experiments/exp02_heatmap_scan.py \
  --n-count 5 \
  --mu-count 10 \
  --k-count 1 \
  --recent 20
```

Interpretation:
* expected runs per strategy = `5 * 10 * 1 * 100 = 5000`
* full completion across both strategy sets = `10000` paired runs

## 10. Decision: Go back to 2026_02_28 Data (eval_H = 10_000, Converge Criteria = 10_000)
