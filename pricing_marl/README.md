# MARL Simulation of Rule-Based Algorithmic Pricing

Code for the multi-agent reinforcement learning (MARL) simulation section of
**"Sophisticated Learners, Simple Rules: The Strategic Selection of Simple Rules in Algorithmic Pricing"**
(Liu, Wang, Ghose). This README covers only the simulation part of the paper
(Section 3 and Appendix B); the analytical model and the empirical Amazon-data
analysis live elsewhere.

[TOC]



---

## 1. What the simulation does

`N` sellers compete in a logit-demand oligopoly. Sellers do **not** choose
prices directly. Instead, each seller's Q-learning agent picks a **rule-based
pricing algorithm** (a "repricer rule") at the start of each outer episode, and
that rule then mechanically sets prices for `K` inner periods:

- **Outer loop (episode `t`):** each seller observes the lowest market price
  and ε-greedily selects one rule from its action set.
- **Inner loop (`k = 1..K`):** the chosen rules run autonomously; each period,
  every seller reprices as a function of the lowest competitor price, demand
  realizes, profits accrue.
- At the end of the block, each seller receives the average inner-period
  profit as its reward and updates its Q-table (Algorithm 1 in the paper).

**Pricing rules** (`src/strategies.py`, Table 1 in the paper):

| ID | Rule | Behavior |
|----|------|----------|
| 0 | `Undercut` | one grid step below the lowest competitor price |
| 1 | `Match` | match the lowest competitor price |
| 2 | `Above` | one grid step above the lowest competitor price |
| 3 | `Undercut+Reset` | undercut, but jump to the grid max when the market hits the grid floor |
| 4 | `Match+Reset` | match, with the same reset behavior (not used in the paper's main experiments) |

The two action sets used in the paper are `3strats` = {Undercut, Match, Above}
and `4strats` = {Undercut, Match, Above, Undercut+Reset}.

**Demand / profit** (`src/environment.py`): standard logit with an outside
option, `D_i = exp((a - p_i)/mu) / (exp(a0/mu) + Σ_j exp((a - p_j)/mu))`,
profit `r_i = (p_i - c) D_i`. Baseline parameters follow Calvano et al. (2020):
`a = 2`, `c = 1`, `a0 = 0`, `γ = 0.95`, `β = 1e-5`, `m = 10` grid points.

**Price grid**: for each `(N, mu)` we solve the static Bertrand–Nash price
`p^N` and monopoly price `p^M` numerically, then build a linear grid with step
`g = (p^M − p^N)/(m−3)` extending one step below `p^N` and one step above
`p^M`. A cost floor is applied to the lowest grid point only: if
`p_grid[0] < c` it is raised to `c` (Appendix B.1, Eq. 27). Lookup-table cache
files get a `_floorC` suffix when the floor binds, so pre-fix tables are never
reused by accident.

**Lookup table** (Appendix B.4): because all sellers are synchronized to the
scalar observation (lowest price index) at the start of each episode, the
`K`-period inner dynamics are deterministic given the joint rule profile. The
environment therefore pre-computes `L(state, sorted rule profile) → (rewards,
next state, stats)` once per configuration and caches it under
`data/lookup_tables/` (file-locked so parallel HPC jobs don't recompute it).

**Learning & evaluation** (`src/agent.py`, `src/simulation.py`): independent
Q-learning, ε = exp(−βt), Q-tables initialized with the heuristic
`Q_0(o,a) = E_{a_-i}[R] / (1−γ)` (Eq. 6). Training stops early when every
agent's greedy policy is unchanged for `converge_period = 10,000` consecutive
episodes (cap: 2–5M episodes). After training, the frozen greedy policies are
evaluated for `eval_H` atomic steps and the full trajectory (prices, profits,
rule actions, lowest/mean price, normalized profit gain Δ) is written to
Parquet, together with `init`/`final` Q-table snapshots.

---

## 2. Repository layout

```
pricing_marl/
├── README.md               # This document
├── src/                    # Core framework (fixed, exogenous K)
│   ├── config.py           #   Config dataclass: market, RL, and path parameters
│   ├── strategies.py       #   Rule definitions + integer IDs
│   ├── environment.py      #   Demand, Nash/monopoly solver, price grid, lookup tables
│   ├── agent.py            #   Independent Q-learning agents + heuristic init
│   ├── simulation.py       #   Train-until-convergence + greedy evaluation for one run
│   └── runner.py           #   Batch orchestration: parallel seeds → Parquet + config JSON
├── src_ext_K_action/       # Extended framework where actions are (rule, K) pairs
│                           #   (used by exp03/exp04; same module layout as src/)
├── experiments/            # Entry points (one per experiment, see §3)
├── analysis/               # Figure/table generation + notebooks (see §4)
│   ├── figures/            #   Generated figures (by script; dated backups in figures/archive/)
│   ├── tables/             #   Generated CSV/LaTeX tables
│   ├── robust_exp03_K_action/     # exp03 analysis notebooks + outputs
│   ├── robust_exp04_fix_K_action/ # exp04 analysis notebook + paper figures
│   ├── robust_exp05_k_1_scan/     # exp05 analysis notebook + figures
│   └── dev/                #   Debug / exploratory / one-off material (not in the paper)
├── hpc/                    # Slurm sbatch entry points, progress monitors, HPC_GUIDE.md
├── data/
│   ├── lookup_tables/      # Cached inner-loop lookup tables (auto-generated)
│   ├── results/            # exp02 main sweep results  ← paper heatmaps
│   ├── result_K30_qtable/  # exp02 K=30 rerun with Q-table snapshots ← paper Table 2
│   ├── results_exp03/      # exp03 endogenous-K results
│   ├── results_exp04/      # exp04 fixed-heterogeneous-K results ← paper Figs 11/18/19
│   ├── results_exp05/      # exp05 K=1 scan results
│   └── archive/            # Dated backups of earlier runs (see §5)
├── docs/archive/           # Superseded docs (old changelog README, old HPC log)
└── requirements.txt
```

Result files, per run (seed): `run_<id>.parquet` (evaluation trajectory) and
`run_<id>_qtable.parquet` (init/final Q snapshots), under
`<results root>/scan_<label>_mu<mu>_k<K>/N_<N>/`, plus a `Config_N_<N>.json`
with the full configuration. Schema details:
[analysis/PARQUET_DATA_EXPLANATION.md](analysis/PARQUET_DATA_EXPLANATION.md).

---

## 3. Experiments

All experiments are run as `python experiments/<script>.py` (locally) or via
the corresponding `.sbatch` file in `hpc/` (Slurm; see
[hpc/HPC_GUIDE.md](hpc/HPC_GUIDE.md)). Runs are fully independent across
seeds, so they parallelize perfectly with no inter-process communication
("embarrassingly parallel" in the technical sense), via `joblib`; worker
count comes from `PRICING_MARL_N_JOBS`, else
`SLURM_CPUS_PER_TASK`, else all cores. Each script also supports env-var
filters (`PRICING_MARL_<EXP>_FILTER_N`, `..._FILTER_MU`, etc.) for targeted
re-runs without editing code.

### exp02 — Main heatmap sweep (paper Sections 3.3–3.4)
`experiments/exp02_heatmap_scan.py` — the core experiment of the paper.

- Grid: `N ∈ {2,3,5,7,10}`, `mu ∈ {0.04, 0.07, …, 0.31}` (10 values),
  `K ∈ {10, 20, …, 100}` (10 values), both `3strats` and `4strats`,
  100 seeds per cell.
- Settings for the paper data: `eval_H = 10,000`, `converge_period = 10,000`.
- Output: `data/results/scan_<label>_mu<mu>_k<K>/N_<N>/` — this is the dataset
  behind the paper's (µ, K) heatmaps and price-history figures.

A restricted **K=30 rerun with Q-table snapshots** (same code, filtered via
`PRICING_MARL_FILTER_K=30`, launched by `hpc/heatmap_k30_qtable_turingvm.sbatch`)
produced `data/result_K30_qtable/`. It backs the baseline table (Table 2) and
the one-shot-deviation (Nash equilibrium / state robustness) analysis.

### exp03 — Endogenous commitment length K
`experiments/exp03_k_choice_scan.py`, built on `src_ext_K_action/`.

Each seller's action is a composite `(pricing rule, K)` with
`K ∈ {10, 30, 60}` on a base block of `base_K = 10`; sellers revise
asynchronously when their own commitment window expires. `N = 3`, all 10 `mu`
values, both strategy sets, 30 seeds. Output: `data/results_exp03/`.
(Robustness material; the current paper draft reports the fixed-K version,
exp04, in Appendix B.6.)

### exp04 — Fixed heterogeneous K (paper Appendix B.6, Figs 11/18/19)
`experiments/exp04_fix_k_choice.py`, also on `src_ext_K_action/`.

Same composite-action machinery as exp03, but each seller's K is **fixed ex
ante** via `fixed_k_by_agent`: sellers 0 and 1 always have `K = 10`, the focal
seller 2 gets `K ∈ {10, 30, 60}` across the three profiles `(10,10,10)`,
`(10,10,30)`, `(10,10,60)`. `N = 3`, all `mu` values, both strategy sets, 30
seeds. Output: `data/results_exp04/`.

### exp05 — K = 1 boundary case
`experiments/exp05_k_1_scan.py` — the exp02 pipeline with `K = 1` (rules are
re-chosen every period, i.e. no commitment). `N ∈ {2,3,5,7,10}`, all `mu`
values, both strategy sets, 30 seeds. Output: `data/results_exp05/`.
(Robustness / reviewer-response material, not in the current draft.)

### exp01 — Early exploratory study
`experiments/exp01_initial_study.py` — first-pass batch runs over a few
strategy sets at fixed `mu`. Superseded by exp02; kept for reference only.

---

## 4. Replicating the paper's figures and tables

The pipeline is always: **run experiment → Parquet in `data/` → analysis
script/notebook → figure/table file**. Everything below assumes the working
directory is `pricing_marl/` (scripts) or `pricing_marl/analysis/` (notebooks;
they locate the project root automatically).

| Paper item | Content | Generated by | Data | Output file(s) |
|---|---|---|---|---|
| Fig. 4 | Two-level timeline diagram | TikZ in the LaTeX source (not in this repo) | — | — |
| Fig. 5 | Inner-loop price/profit trajectories, Scenarios A/B/C | [analysis/visualize_physics.ipynb](analysis/visualize_physics.ipynb) (direct micro-simulation, no lookup table) | none (simulated in-notebook) | `analysis/figures/price_rule_demo_fig/scenario_{a,b,c}_3p_*.png` |
| Table 2 | Baseline Δ, lowest price, Nash Eq. %, State Robust % (`K=30`, `mu=0.25`) | [analysis/count_nash_equilibrium_runs.py](analysis/count_nash_equilibrium_runs.py) → CSV, then [analysis/tab_exp_res_pub.ipynb](analysis/tab_exp_res_pub.ipynb) | `data/result_K30_qtable/` | `analysis/tables/nash_equilibrium_summary_result_K30_qtable.csv`, `analysis/tables/baseline_mu0.25_k30.tex` |
| Figs. 6, 7 | Δ heatmaps over (µ, K), 3 Rules / 4 Rules, N ∈ {3,5,7,10} | [analysis/viz_heatmap_k_mu_by_n_pub.py](analysis/viz_heatmap_k_mu_by_n_pub.py) | `data/results/` | `analysis/figures/heatmaps_k_mu_by_n_pub/heatmap_delta_{3,4}strats_N<N>.png` |
| Fig. 8 | Δ(4 Rules) − Δ(3 Rules) difference heatmaps | [analysis/viz_heatmap_strat_delta_diff.py](analysis/viz_heatmap_strat_delta_diff.py) | `data/results/` | `analysis/figures/heatmaps_strat_diff/heatmap_delta_diff_N<N>.png` |
| Fig. 9 (N=10) and Figs. 16, 17 (other N) | Equilibrium rule-usage-share heatmaps | [analysis/viz_heatmap_k_mu_by_n_pub.py](analysis/viz_heatmap_k_mu_by_n_pub.py) (same run as Figs. 6–7) | `data/results/` | `analysis/figures/heatmaps_k_mu_by_n_pub/heatmap_action_share_{3,4}strats_N<N>.png` |
| Fig. 10 | Price histories of representative runs, N=10, K=30, µ ∈ {0.04, 0.10, 0.25} | [analysis/viz_history.ipynb](analysis/viz_history.ipynb) (last 150 blocks of one run) | `data/results/` | `analysis/figures/viz_price_history_selected/price_history_strats{3,4}_N10_k30_mu{0.04,0.1,0.25}_run1_last150.png` |
| Fig. 11 | Market-level Δ under K heterogeneity (N=3) | [analysis/robust_exp04_fix_K_action/desc_exp04_fix_k_action_overview.ipynb](analysis/robust_exp04_fix_K_action/desc_exp04_fix_k_action_overview.ipynb) | `data/results_exp04/` | `analysis/robust_exp04_fix_K_action/hetero_fix_K_avg_delta.png` |
| Fig. 18 | Focal-seller Δ under K heterogeneity | same notebook as Fig. 11 | `data/results_exp04/` | `analysis/robust_exp04_fix_K_action/hetero_fix_K_focal_seller_delta.png` |
| Fig. 19 | Rule-usage shares under K heterogeneity | same notebook as Fig. 11 | `data/results_exp04/` | `analysis/robust_exp04_fix_K_action/hetero_fix_K_action_share.png` |

Supporting definitions for Table 2's last two columns: for each run,
`count_nash_equilibrium_runs.py` takes seller 0's converged policy, injects a
one-shot deviation to every non-best rule at the deviation state, replays the
deterministic dynamics, and labels the run **Nash Eq.** if no deviation raises
the deviator's profit over the following 5 K-blocks, and **State Robust** if
the market returns to the pre-deviation lowest-price state within that window.
[analysis/viz_demo_deviate_nash.ipynb](analysis/viz_demo_deviate_nash.ipynb)
visualizes a single such deviation experiment
(`analysis/figures/viz_price_deviatoin_eq/`).

### End-to-end replication recipe

```bash
# 0. Environment
pip install -r requirements.txt          # numpy, pandas, scipy, joblib, filelock, pyarrow, ...

# 1. Main sweep (very heavy: 2 sets × 10 mu × 10 K × 5 N × 100 seeds; use HPC)
python experiments/exp02_heatmap_scan.py                     # → data/results
PRICING_MARL_FILTER_K=30 python experiments/exp02_heatmap_scan.py  # → K=30 qtable rerun
                                                             #   (set results root as in
                                                             #    hpc/heatmap_k30_qtable_turingvm.sbatch)

# 2. K-heterogeneity extension
python experiments/exp04_fix_k_choice.py                     # → data/results_exp04

# 3. Figures / tables
python analysis/viz_heatmap_k_mu_by_n_pub.py                 # Figs 6, 7, 9, 16, 17
python analysis/viz_heatmap_strat_delta_diff.py              # Fig 8
python analysis/count_nash_equilibrium_runs.py               # Table 2 inputs
# then execute: analysis/tab_exp_res_pub.ipynb               # Table 2 (LaTeX)
#               analysis/viz_history.ipynb                   # Fig 10
#               analysis/visualize_physics.ipynb             # Fig 5
#               analysis/robust_exp04_fix_K_action/desc_exp04_fix_k_action_overview.ipynb  # Figs 11/18/19
```

Lookup tables are created on first use and cached in `data/lookup_tables/`;
they depend only on `(N, m, mu, a, c, a0, xi, K, strategy set)` — not on RL
hyper-parameters — so they can be shared across reruns and machines.

---

## 5. Data folders: canonical vs. backup

| Folder | Status |
|---|---|
| `data/results/` | **Canonical exp02 sweep** (eval_H = 10,000, converge = 10,000). Byte-identical to the `2026_02_28` backup — the project reverted to this dataset after a 2026-03 rerun with different convergence settings was rejected. |
| `data/result_K30_qtable/` | **Canonical K=30 rerun** with init/final Q-table snapshots (eval_H = 2,000). Used for Table 2 and deviation analysis. |
| `data/results_exp03/`, `results_exp04/`, `results_exp05/` | Canonical outputs of exp03/exp04/exp05. |
| `data/lookup_tables/` | Active lookup-table cache. |
| `data/archive/results_2026_02_28/` | Backup of the canonical exp02 data (same content as `data/results/`). |
| `data/archive/results_2026_3_12/` | Rejected 2026-03-12 rerun (converge = 100,000 experiment, eval_H = 2,000). Kept for reference only. |
| `data/archive/lookup_tables_2026_02_28/`, `data/archive/lookup_tables_3_12/` | Backups of the lookup cache. |
| `analysis/figures/archive/` | Dated figure backups matching the data backups above. |

---

## 6. HPC notes

Everything HPC-related lives in `hpc/` — see **[hpc/HPC_GUIDE.md](hpc/HPC_GUIDE.md)**
for setup, submission, monitoring, and a checklist for adding new experiments.
In short:

- Slurm entry points: `hpc/heatmap.sbatch` / `hpc/heatmap_turingvm.sbatch`
  (exp02), `hpc/heatmap_k30_qtable_turingvm.sbatch` (K=30 qtable rerun),
  `hpc/exp03_k_choice_{scrc,turingvm}.sbatch`,
  `hpc/exp04_fix_k_choice_scrc.sbatch`, `hpc/exp05_k_1_scan_scrc.sbatch`.
  Submit from the project root: `sbatch hpc/<file>.sbatch`.
- Progress monitors: `hpc/progress_report.py` (exp02) and
  `hpc/exp0{3,4,5}_progress_report.py`; they count paired
  `run_<id>.parquet` + `run_<id>_qtable.parquet` files against the expected
  grid ("paired progress").
- Parquet is compressed with `zstd`; lookup tables are file-locked so
  concurrent jobs on a shared filesystem don't recompute them.
- The historical operations log is archived at
  `docs/archive/HPC_PROGRESS_GUIDE.md`.

---

## 7. Auxiliary files (not part of the paper pipeline)

Everything that exists for debugging, one-off investigations, or historical
reasons is grouped away from the paper pipeline:

- **`analysis/dev/`** — all debug/exploratory analysis material:
  one-off diagnostics (`scan_grid_floor_cases.py`, which identified the cells
  to rerun after the grid cost-floor fix; `debug_negative_delta_case.py`;
  `debug_pkl_file_check.ipynb`), investigation notes (`INVESTIGATION_*.md`),
  and exploratory visualizations (`viz_heatmap_k_mu_by_n.py` — working
  version of the pub script, `viz_heatmap_n_mu_by_k.py`,
  `viz_heatmap_eq_diversity_k_mu_by_n.py`, `viz_line_HighLowMu_N.py`,
  `viz_all_runs_history.ipynb`, `demo_exp_data.ipynb`,
  `viz_loopup_table.ipynb`).
- **`hpc/fix_initial_qtables.py`** — one-off repair of `init` Q-snapshots
  produced by a since-fixed bug; kept for reference.
- **exp03/exp05 analysis material** (`analysis/robust_exp03_K_action/`,
  `analysis/robust_exp05_k_1_scan/`): robustness results not in the current
  draft.
- **Legacy:** `experiments/exp01_initial_study.py`; dated backups under
  `data/archive/` and `analysis/figures/archive/`; superseded docs
  (old changelog-style README, old HPC operations log) under `docs/archive/`.
