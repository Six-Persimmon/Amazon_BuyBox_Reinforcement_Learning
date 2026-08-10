# HPC Guide

How to run `pricing_marl` experiments on the cluster, monitor them, and bring
results back. This replaces the old chronological log, which is preserved at
[docs/archive/HPC_PROGRESS_GUIDE.md](../docs/archive/HPC_PROGRESS_GUIDE.md)
(useful if you need the exact history of past reruns).

Everything in this folder is server-side tooling:

| File | Purpose |
|---|---|
| `heatmap.sbatch`, `heatmap_turingvm.sbatch` | exp02 full sweep (generic cluster / turingvm) |
| `heatmap_k30_qtable_turingvm.sbatch` | exp02 restricted to K=30 with Q-table snapshots → `data/result_K30_qtable` |
| `exp03_k_choice_{scrc,turingvm}.sbatch` | exp03 endogenous-K scan |
| `exp04_fix_k_choice_scrc.sbatch` | exp04 fixed heterogeneous-K |
| `exp05_k_1_scan_scrc.sbatch` | exp05 K=1 scan |
| `exp06_weighted_explore_scrc.sbatch` | exp06 weighted-exploration robustness → `data/results_exp06` |
| `exp07_no_collapse_scrc.sbatch` | exp07 no-state-collapse robustness → `data/results_exp07` |
| `exp08_async_poisson_scrc.sbatch` | exp08 async Poisson clocks → `data/results_exp08` |
| `exp09_calvano_ladder_scrc.sbatch` | exp09 Calvano ladder / mechanism decomposition → `data/results_exp09` (§8) |
| `progress_report.py` | progress monitor for exp02-style result trees |
| `exp0{3,4,5,6,7,8,9}_progress_report.py` | progress monitors for exp03–exp09 |
| `fix_initial_qtables.py` | one-off repair tool for old buggy `init` Q-snapshots (kept for reference) |

## 0. One-time migration to the new layout (as of 2026-08)

The repository was reorganized in 2026-08 (sbatch/progress scripts moved from
the project root into `hpc/`, debug analysis into `analysis/dev/`, backups
into `data/archive/`). The cluster copy at `~/bigdata/pricing_marl` still has
the **old flat layout**. All useful results have already been downloaded to
the laptop, so nothing on the cluster is precious. Before the next experiment
round, do a clean cutover:

```bash
# --- on the cluster ---
mv ~/bigdata/pricing_marl ~/bigdata/pricing_marl_old   # keep as safety net

# --- from the laptop: upload code + lookup tables (NOT results/archives) ---
cd ~/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning
rsync -av \
  --exclude '__pycache__' --exclude '.DS_Store' \
  --exclude 'data' --exclude 'analysis/figures' --exclude 'docs' \
  pricing_marl/ sl9818@login.scrc.nyu.edu:~/bigdata/pricing_marl/
rsync -av pricing_marl/data/lookup_tables/ \
  sl9818@login.scrc.nyu.edu:~/bigdata/pricing_marl/data/lookup_tables/
```

Notes:

- **Do not merge into the old folder** — stale root-level copies of
  `*.sbatch` / `progress_report.py` would risk being submitted by habit. A
  fresh folder guarantees only the `hpc/` versions exist.
- **Upload `data/lookup_tables/`** (currently ~127 MB): it saves recomputing
  every table on the first run. Everything else under `data/` stays local.
- The conda env lives in `~/.conda/envs/pricing_marl`, outside the project
  folder — it survives the move; no reinstall needed.
- Submission habit changes to: `cd ~/bigdata/pricing_marl && sbatch hpc/<file>.sbatch`
  (see §2). Progress checks become `python hpc/<...>_progress_report.py`.
- After the first successful new-layout run, delete the safety net:
  `rm -rf ~/bigdata/pricing_marl_old`.

## 1. One-time setup on the server

```bash
# project lives at ~/bigdata/pricing_marl (mirror of this folder)
rsync -av --exclude data --exclude '__pycache__' \
  pricing_marl/ sl9818@login.scrc.nyu.edu:~/bigdata/pricing_marl/

# create the conda env once
module load anaconda3/py3.9
conda create -n pricing_marl python=3.12
conda run -n pricing_marl pip install -r ~/bigdata/pricing_marl/requirements.txt
```

The sbatch scripts locate the env by probing a list of candidate paths
(`$HOME/.conda/envs/pricing_marl/bin/python`, etc.) — see the `ENV_PY` block
inside any `*_scrc.sbatch` if the env lives elsewhere.

## 2. Submitting a job

Always submit **from the project root** on the server:

```bash
cd ~/bigdata/pricing_marl
sbatch hpc/exp05_k_1_scan_scrc.sbatch
```

Notes:

- The `turingvm`/`scrc` scripts resolve the project root from
  `SLURM_SUBMIT_DIR` (they also tolerate being submitted from inside `hpc/`).
  `heatmap.sbatch` instead hard-codes `PROJECT_DIR="$HOME/bigdata/pricing_marl"`.
- Partitions in current scripts: `def` (SCRC, 32 CPUs) and `bigmem`
  (turingvm, 48 CPUs). Adjust `#SBATCH --partition/--cpus-per-task` per cluster.
- Worker count: the Python code reads `PRICING_MARL_N_JOBS` first, then
  `SLURM_CPUS_PER_TASK`. The sbatch scripts set this for you, plus
  `OMP_NUM_THREADS=1` etc. to prevent thread oversubscription.

### Targeted / partial runs (env-var filters)

Every experiment script accepts env-var filters so you can rerun a subset
without editing code. Naming pattern (see each script's header for the full
list):

```
PRICING_MARL_FILTER_N / _MU / _K                    # exp02
PRICING_MARL_EXP03_FILTER_N / _MU / _K              # exp03
PRICING_MARL_EXP04_FILTER_N / _MU / _K_PROFILE      # exp04
PRICING_MARL_EXP05_FILTER_N / _MU                   # exp05
..._EXPERIMENT_SET   (3strats | 4strats)
..._ROUNDS_PER_CONFIG, ..._MAX_EPISODES, ..._CONVERGE_PERIOD, ..._EVAL_H
..._RESULTS_DIR      (redirect output, e.g. for smoke tests)
```

### Smoke test before a big run

```bash
cd ~/bigdata/pricing_marl
export PRICING_MARL_EXP05_RESULTS_DIR="$HOME/bigdata/pricing_marl/data/results_exp05_smoke"
export PRICING_MARL_EXP05_FILTER_MU="0.04"
export PRICING_MARL_EXP05_ROUNDS_PER_CONFIG="1"
export PRICING_MARL_EXP05_MAX_EPISODES="100"
export PRICING_MARL_EXP05_CONVERGE_PERIOD="10"
export PRICING_MARL_EXP05_EVAL_H="60"
sbatch hpc/exp05_k_1_scan_scrc.sbatch
# then unset everything and submit the real run
```

## 3. Monitoring

```bash
squeue -u $USER
tail -f exp05_k1_<jobid>.out          # sbatch stdout log, written in submit dir

# progress by file counts ("paired" = run_<id>.parquet + run_<id>_qtable.parquet)
cd ~/bigdata/pricing_marl
python hpc/progress_report.py --rounds 100 --recent 20              # exp02 → data/results
python hpc/exp05_progress_report.py --rounds 30 --recent 20         # exp05 → data/results_exp05
# custom root (e.g. the K=30 qtable rerun):
python hpc/progress_report.py --root data/result_K30_qtable --rounds 100 \
  --grid-file experiments/exp02_heatmap_scan.py --n-count 5 --mu-count 10 --k-count 1
```

A run is complete only when **both** the eval parquet and the qtable parquet
exist; the runner skips finished run IDs on resubmission, so a crashed job can
simply be resubmitted.

## 4. Lookup-table reuse (saves hours)

Lookup tables (`data/lookup_tables/`) depend only on:
`N, num_grids, mu, a, c, a0, xi, K (base_K for exp03/04), strategy set, grid construction`.

They do **not** depend on: seeds/rounds, `converge_period`, `eval_H`, `beta`,
results directory, or (for exp03/04) the K-profile. So when rerunning with
different RL settings, keep the server's `data/lookup_tables/` in place and
only clear the results directory. Tables are file-locked, so parallel jobs on
a shared filesystem won't recompute them; grid-floor-affected configs get a
`_floorC` filename suffix automatically.

## 5. Getting results back

```bash
rsync -av sl9818@login.scrc.nyu.edu:~/bigdata/pricing_marl/data/results_exp05/ \
  ~/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning/pricing_marl/data/results_exp05/
```

Then run the analysis scripts/notebooks locally (see the main
[README](../README.md), §4).

## 6. Adding a new experiment (expNN checklist)

1. **Experiment script** `experiments/expNN_<name>.py` — copy
   `exp05_k_1_scan.py` as the template (it is the cleanest). Keep the
   conventions:
   - env-var filters named `PRICING_MARL_EXPNN_*`;
   - batch names `scan_<tag>_<label>_mu<mu>_k<K>` (analysis loaders parse this);
   - output root `data/results_expNN`;
   - call `run_experiment_batch(...)` from `src.runner` (or
     `src_ext_K_action.runner` for composite (rule, K) actions).
2. **sbatch file** `hpc/expNN_<name>_scrc.sbatch` — copy
   `exp05_k_1_scan_scrc.sbatch`, change the job name, exported env vars, and
   the final `python -u experiments/expNN_<name>.py` line.
3. **Progress monitor** `hpc/expNN_progress_report.py` — copy
   `exp05_progress_report.py` and adjust the expected grid / directory regex.
4. Smoke test with tiny overrides (§2), check output schema locally, then
   launch the full run.
5. Add the experiment and its figure/table mapping to the main README.

## 7. Current round: exp06 / exp07 / exp08 (reviewer response, 2026-08)

Full sequence, assuming the §0 migration is done:

```bash
# 1. login
ssh sl9818@login.scrc.nyu.edu

# 2. submit all three (independent jobs; each takes 32 CPUs on partition def)
cd ~/bigdata/pricing_marl
sbatch hpc/exp06_weighted_explore_scrc.sbatch
sbatch hpc/exp07_no_collapse_scrc.sbatch
sbatch hpc/exp08_async_poisson_scrc.sbatch

# 3. monitor
squeue -u sl9818
python hpc/exp06_progress_report.py --rounds 30    # expected 600 paired (N=3 + N=10, 4strats)
python hpc/exp07_progress_report.py --rounds 30    # expected 600 paired (300 per strategy set)
python hpc/exp08_progress_report.py --rounds 30    # expected 600 paired (300 per strategy set)
```

Then download from the laptop:

```bash
for e in exp06 exp07 exp08; do
  rsync -av sl9818@login.scrc.nyu.edu:~/bigdata/pricing_marl/data/results_$e/ \
    ~/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning/pricing_marl/data/results_$e/
done
```

and run the three `desc_exp0X_overview.ipynb` notebooks under
`analysis/robust_exp0X_*/`. If a job dies, just resubmit the same sbatch —
finished run IDs are skipped automatically.

## 8. exp09 — Calvano ladder (mechanism decomposition, 2026-08)

Five rungs (`cal_full`, `cal_smin`, `cal_arule`, `cal_both`, `cal_both_k30`)
× 10 mu × 30 seeds = **1,500 runs**, ~25 min on 32 cores. Self-contained: it
needs no lookup tables and does not read any other experiment's results.

```bash
cd ~/bigdata/pricing_marl
sbatch hpc/exp09_calvano_ladder_scrc.sbatch

# monitor (expected 1,500 paired: 300 per cell)
python hpc/exp09_progress_report.py --rounds 30
python hpc/exp09_progress_report.py --rounds 30 --cell cal_full   # one rung
```

Smoke test first, per §2:

```bash
PRICING_MARL_EXP09_RESULTS_DIR=$PWD/data/results_exp09_smoke \
PRICING_MARL_EXP09_FILTER_MU=0.25 \
PRICING_MARL_EXP09_ROUNDS_PER_CONFIG=1 \
PRICING_MARL_EXP09_MAX_EPISODES=20000 \
PRICING_MARL_EXP09_CONVERGE_PERIOD=2000 \
PRICING_MARL_EXP09_EVAL_H=600 \
sbatch hpc/exp09_calvano_ladder_scrc.sbatch
```

Engine validation (fast, no cluster needed): `python tests/test_exp09_engine.py`.

The optional low-β arm (Extention_Plan.md §4.5) is the same job with
`PRICING_MARL_EXP09_BETA=1e-6` and
`PRICING_MARL_EXP09_RESULTS_DIR=$PWD/data/results_exp09_lowbeta`; it is **not**
part of the scheduled round.
