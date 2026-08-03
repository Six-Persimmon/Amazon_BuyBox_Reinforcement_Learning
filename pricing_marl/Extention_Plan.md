# Extension & Robustness Plan (Reviewer Response)

Status: **planning document — no code written yet.** This file is the working
plan for three reviewer-driven robustness checks / extensions. All previously
open design choices have been decided (2026-08-03); they are marked
**Decision:** inline, with the full log in §5.
Numbering continues the existing experiment series: **exp06, exp07, exp08**.

| # | Reviewer concern | Our response | New experiment |
|---|---|---|---|
| 1 | 4th rule shifts the ε-exploration distribution toward price-falling actions (1/3 → 1/2); maybe that, not the Safety Net mechanism, drives the 3-vs-4-rule differences | Re-run the 4-rule row with **weighted exploration** that restores P(price-falling draw) = 1/3 | **exp06** (lookup-based, `src/` + small agent change) |
| 2 | "Collapse to lowest price" between episodes is a strong simplification | (a) micro-demo notebook: collapse vs carry-over trajectories are nearly identical; (b) full re-run **without collapse** | demo notebook + **exp07** (new atomic engine) |
| 3 | Synchronized K is restrictive | **Asynchronous revision clocks**: each agent gets stochastic rule-revision times with mean λ_K = 30 | **exp08** (same atomic engine) |

---

## 0. Shared infrastructure

### 0.1 New simulation engine `src_atomic/`

exp07 and exp08 both abandon the lookup table and simulate the market
**period by period at atomic time**, carrying the full price vector across
episodes. They share one new package (repo convention: like
`src_ext_K_action/`, a sibling of `src/` with the same module layout):

```
src_atomic/
├── __init__.py
├── config.py        # AtomicConfig: base fields + mode + async params (see below)
├── environment.py   # AtomicEnvironment: no lookup table (details below)
├── agent.py         # reuse/copy of QAgent (+ exploration weights, shared with exp06)
├── simulation.py    # run_simulation_sync_carryover(), run_simulation_async()
└── runner.py        # same batch/Parquet/skip-existing logic as src/runner.py
```

`environment.py` reuses `src.environment.get_demand_and_profit_static` and
`compute_nash_and_monopoly_static` (imports, not copies) and precomputes two
small arrays that make pure-Python atomic stepping fast:

- **Profit tensor** `PROFIT[i1,…,iN, seller]` over all grid-index combinations
  (N = 3, m = 10 → 1,000 combos; ~24 KB). One period's profits = one lookup.
- **Rule-transition table** `NEXT[rule, others_min_idx]` (|rules| × 10): all
  rules depend *only* on the competitors' minimum index, so one period's price
  update = compute each seller's others-min + one table lookup.

With these, one atomic period is a few array indexings; even the worst-case
training budget (≈ 60M atomic periods, see §3) is minutes per run.
Additionally, for exp07 (synchronous blocks) the K-period block dynamics are
deterministic given (price vector, rule profile), so an **in-run memo dict**
`(price_tuple, rule_tuple) → (avg rewards, next price_tuple)` recovers
lookup-table speed after warm-up with zero approximation.

Grid construction, cost floor, and Nash/monopoly benchmarks are identical to
`src/` (same code path), so Δ normalization is directly comparable.

### 0.2 Q-initialization

Baseline init is the heuristic `Q0(o,a) = E_{a_-i}[R]/(1−γ)` computed from the
lookup table. The needed lookup tables (N=3, K=30, both strategy sets; N=10,
K=30, 4strats) **already exist** in `data/lookup_tables/`. exp07/exp08 keep
using this same init (loading the existing table only for initialization) so
that any outcome difference is attributable to the dynamics, not the prior.

### 0.3 Conventions (same as exp03–05)

- Experiment scripts: `experiments/exp0X_<name>.py` with
  `PRICING_MARL_EXP0X_*` env-var filters/overrides.
- Batch naming: `scan_<tag>_<label>_mu<mu>_k<K>` so existing loaders parse it.
- Outputs: `data/results_exp0X/.../N_<N>/run_<id>.parquet` +
  `run_<id>_qtable.parquet` + `Config_N_<N>.json`; runner skips finished
  run_ids so crashed jobs are resubmittable.
- HPC: one `hpc/exp0X_*_scrc.sbatch` + one `hpc/exp0X_progress_report.py`
  per experiment (clone the exp05 pair; see `hpc/HPC_GUIDE.md` §6).
- Analysis: one `analysis/robust_exp0X_<name>/` folder per experiment with a
  `desc_exp0X_*_overview.ipynb` and its figures (exp04 convention).
- Seeds: **30 per cell**, `run_id = 0..29` (`np.random.seed(run_id)`) — the
  exp03–05 convention, and a subset of the baseline's `0..99`, so seed-matched
  comparisons are possible.

### 0.4 Common evaluation settings

`eval_H = 10,000` atomic periods (**Decision:** confirmed) — matches the
canonical `data/results` dataset (9,990-row eval files) rather than the 2,000
used in exp03–05; comparisons for exp07/exp08 are directly against
`data/results`, so matching its horizon avoids a needless asymmetry.
`converge_period` equivalent to baseline 10,000 episodes (see per-experiment
sections). All other parameters as baseline (`a=2, c=1, a0=0, γ=0.95,
α=0.15, β=1e-5, m=10`).

---

## 1. exp06 — Weighted exploration (sampling-distribution confound)

### 1.1 The concern, restated

During ε-exploration the agent draws uniformly from its action set
(`src/agent.py: choose_action → np.random.randint(num_actions)`). With 3 rules,
P(draw a price-falling rule) = P(Undercut) = 1/3; with 4 rules it is
P(Undercut) + P(Undercut+Reset) = 1/2. Because β = 1e-5 keeps ε high for a
long time (ε ≈ 0.37 even at t = 100k episodes), the learning path — and hence
which equilibrium gets selected — could plausibly be shaped by this mechanical
shift rather than by the Reset rule's strategic "safety net" effect.

### 1.2 Design

Everything identical to the baseline exp02 cell family, except the ε-draw
distribution:

- **Setting:** `N ∈ {3, 10}`, `K = 30`, `4strats`, all 10 mu values,
  **30 seeds**. N=10 is the paper's focal case for the Safety Net Paradox
  (Figure 8d); the N=3 row is cheap and lets the weighted-exploration control
  cover the market size used by exp07/exp08 as well.
- **Exploration weights** (order = `active_strategies` = [Undercut, Match,
  Above, Undercut+Reset] = IDs [0,1,2,3]):
  `[1/6, 1/3, 1/3, 1/6]` — total price-falling mass = 1/3, equal to the
  3-rule baseline; the Undercut mass is split evenly between the two
  undercut-style rules.
- Applied to **all agents** (all sellers are learners, so opponents' behavior
  during exploration is automatically reweighted too).
- Greedy selection, tie-breaking, Q-update, convergence check: unchanged.

Interpretation note (for the paper text): weighting equalizes the *marginal*
probability that a single ε-draw is price-falling. The joint distribution over
rule profiles still differs across the two action sets — that is unavoidable
when the action sets differ — but this is the natural first-order control for
the reviewer's stated concern.

**Decision: Q-init stays the uniform-based heuristic** (identical to
baseline). The init also encodes "uniform random opponents"
(`calculate_heuristic_init_values` weights lookup entries by permutation
multiplicity), but keeping it unchanged isolates the exploration channel —
exactly what the reviewer asked about. A variant with reweighted init remains
cheap to add later (config flag) if a reviewer pushes.

### 1.3 Implementation

Small, backward-compatible changes to `src/` (no new package):

1. `src/config.py`: add `exploration_weights: Optional[List[float]] = None`
   (None ⇒ uniform, i.e. exact current behavior; validated to length
   `num_actions` and sum 1).
2. `src/agent.py: choose_action`: on the ε-branch,
   `np.random.choice(num_actions, p=cfg.exploration_weights)` when weights are
   set, else the current `randint`.
3. `src/runner.py`: pass `exploration_weights` through `run_experiment_batch`
   into `Config` (it is then saved automatically in `Config_N_<N>.json`).
4. `experiments/exp06_weighted_explore.py`: exp05-style script; fixed
   `N_VALUES=[3, 10]`, `K=30`, `4strats` only, `MU_VALUES` = the 10 baseline
   values, `ROUNDS_PER_CONFIG=30`, batch names
   `scan_wexp_4strats_mu<mu>_k30` (with `N_3/` and `N_10/` subfolders as
   usual), output `data/results_exp06`.
5. `hpc/exp06_weighted_explore_scrc.sbatch`, `hpc/exp06_progress_report.py`.

Lookup tables: unchanged environment ⇒ **reuses the existing
`lookup_N10_G10_mu*_K30_strats0_1_2_3` and `lookup_N3_G10_mu*_K30_strats0_1_2_3`
tables**; no new table computation.

### 1.4 Analysis & expected output

`analysis/robust_exp06_weighted_explore/desc_exp06_overview.ipynb`:

- Load Δ per run: (i) baseline 3strats K=30 rows from `data/results`
  (100 runs/cell), (ii) baseline 4strats uniform rows from `data/results`,
  (iii) new weighted 4strats rows from `data/results_exp06` (30 runs/cell).
  All figures produced for both N=10 (headline) and N=3.
- **Figure 1**: line plot, x = mu, y = mean Δ with across-run SE bands, three
  lines (3-rule, 4-rule uniform, 4-rule weighted).
- **Figure 2**: two-row heatmap strip in the style of
  `heatmap_delta_diff_N10.png`: row 1 = Δ(4R uniform) − Δ(3R) (original), row
  2 = Δ(4R weighted) − Δ(3R). Same colormap/scale as the paper figure.
- **Figure 3** (mechanism check): rule-usage-share strip for the weighted run
  vs. the uniform run — does Undercut+Reset still take over at low/mid mu?
- Optional table: per-mu means, SDs, and a two-sample comparison
  (uniform-4R vs weighted-4R Δ per cell).

**Success criterion:** the negative Δ-difference dip at intermediate mu
(the Safety Net Paradox region, roughly mu ∈ [0.07, 0.16] for N=10) survives
under weighted exploration. Runtime: same order as baseline exp02 cells;
2 N × 10 mu × 30 seeds = 600 runs on 32 cores is a small job (the N=3 runs
are much cheaper than the N=10 ones).

---

## 2. exp07 — No state collapse (carry-over prices)

### 2.1 Why the assumption is weaker than it looks (goes in the paper text)

All rules map `others' min price index → own next index` and **ignore the
seller's own current price**. With synchronous updates this means the entire
price vector at inner step k+1 is a function of the per-seller others-min
values at step k. Consequently the initial condition of a block (collapsed vs
carried-over) can only influence the path through the *first* step's
others-min values; after that, both worlds run on the same one-step map. The
two trajectories provably coincide once their others-min vectors coincide,
which typically happens within 1–2 inner steps. The exceptions worth showing
honestly: (i) the seller holding the minimum sees others-min = second-lowest,
so heterogeneous starting prices can shift the path by one grid step / one
period; (ii) in reset-cycle dynamics a one-step phase shift can persist as a
phase offset (same cycle, shifted timing, ≈ same average profit).

Also worth noting: the learning question is whether outcomes survive when the
scalar observation (market min) is no longer a sufficient state — with
carry-over, transitions depend on the full price vector, so the environment is
only approximately Markov in the observation. exp07 tests exactly this.

### 2.2 Part A — Micro-demo notebook (no learning)

`analysis/robust_exp07_no_collapse/demo_collapse_vs_carryover.ipynb`, built
like `analysis/visualize_physics.ipynb` (direct micro-simulation, no lookup):

- `N = 3`, `mu = 0.25` (baseline demo value), `K = 30`, **2 consecutive
  blocks** (60 atomic periods).
- Scenarios: reuse the paper's Figure 5 trio — A: Undercut/Match/Above,
  B: Match/Above/Above, C: Undercut+Reset/Undercut+Reset/Match — plus
  (**Decision:** included) a fourth scenario **D** where the rule profile
  *changes* between block 1 and block 2 (undercutting block →
  Above/Above/Match recovery block), demonstrating that recovery dynamics —
  where a skeptic would most expect the collapse to matter — are also
  insensitive to it.
- For each scenario, two simulations from the same block-1 path:
  block 2 starts (i) collapsed to the block-1 terminal minimum (baseline
  assumption) vs (ii) from the sellers' actual terminal prices.
- **Plots:** per scenario, price trajectories overlaid (carry-over solid,
  collapse dashed; vertical line at the block boundary) + profit trajectories.
- **Metrics** printed per scenario: number of inner steps until the two paths
  coincide exactly (grid indices equal); per-seller block-2 average-profit
  difference (absolute and % of the collapse value); same for block-2 average
  min price.
- Output figures to `analysis/robust_exp07_no_collapse/figures/`
  (`demo_scenario_<x>_price.png`, …).

This notebook needs no new infrastructure (a ~50-line micro-simulator like
`simulate_inner_dynamics` in visualize_physics.ipynb) and can be built first.

### 2.3 Part B — Full no-collapse learning run

- **Setting:** `N = 3`, `K = 30`, both `3strats` and `4strats`, 10 mu values,
  30 seeds. Output `data/results_exp07`,
  batches `scan_nocollapse_<label>_mu<mu>_k30`.
- **Mechanics** (engine mode `sync_carryover`): agents still act
  synchronously every K periods; observation at a decision node = current
  market-min index (unchanged definition); reward = average profit over the
  block (unchanged). The only change: block k+1's initial price vector = block
  k's terminal price vector (no collapse). `t = 0`: each seller's price index
  drawn independently at random (this actually matches the paper's stated
  t = 0 randomization more literally than the current code, which collapses
  from the start).
- Convergence: identical rule (greedy policy matrix unchanged for 10,000
  consecutive episodes; cap `max_episodes = 2M`). Evaluation: greedy,
  `eval_H = 10,000` atomic periods, same Parquet schema as exp02 (analysis
  loaders work as-is).
- Q-table snapshots (init/final) saved as usual.

### 2.4 Analysis & expected output

`analysis/robust_exp07_no_collapse/desc_exp07_overview.ipynb`:

- Baseline comparator: `data/results` N=3, K=30 row (100 runs/cell).
- **Figure 1**: Δ vs mu, mean ± SE bands: baseline-collapse vs no-collapse,
  one panel per strategy set.
- **Figure 2**: same for average lowest price.
- **Figure 3**: rule-usage shares side by side (collapse vs carry-over).
- Table: per-(mu, ruleset) mean Δ, SD, |difference|, and overlap diagnostics.
- **Success criterion:** no-collapse Δ-vs-mu curves track the baseline within
  cross-run noise, and the 4R−3R gap keeps its sign pattern.

Runtime estimate: with the memo dict (§0.1) a run costs roughly what a
baseline run costs after warm-up; worst case (no memo hits, 2M episodes ×
30 steps) is still ≲ minutes/run in pure Python with the precomputed tensors.
2 sets × 10 mu × 30 seeds = 600 runs — small HPC job
(`hpc/exp07_no_collapse_scrc.sbatch`, 32 CPUs, partition `def`).

---

## 3. exp08 — Asynchronous Poisson revision clocks

### 3.1 Design

`N = 3`, `λ_K = 30`, both strategy sets, 10 mu values, 30 seeds. Engine mode
`async_poisson` (naturally carry-over; there is no meaningful "collapse" when
revisions are not aligned). Output `data/results_exp08`, batches
`scan_async_<label>_mu<mu>_lam30`.

Per-agent mechanics:

- Each agent i has a revision clock: at its revision period it (1) computes
  reward = **average per-period profit since its own last revision**
  (window-length normalized), (2) observes o = current market-min index,
  (3) Q-updates its previous (o, a) pair with this reward and the new o,
  (4) ε-greedily picks a new rule (which may equal the old one), (5) draws its
  next revision gap. Between revisions the agent's rule keeps executing every
  period.
- All sellers' prices update every atomic period by their current rules
  (identical one-step map as exp07).
- t = 0: random initial prices; all agents pick an initial rule at period 0;
  initial gaps drawn independently (this staggers agents immediately).

**Decision: gap distribution = Poisson-distributed gaps,
`gap ~ max(1, Poisson(λ_K))`.** Mean 30, SD ≈ 5.5 — agents stay "K ≈ 30
committers" but are de-synchronized and drift in phase. This isolates
*desynchronization* while holding commitment length essentially fixed, which
is precisely the reviewer's point. (The gap distribution is still implemented
as a config option, so a geometric-gap variant — the memoryless "Poisson
clock", mean 30, SD ≈ 29.5, which mixes desynchronization with
commitment-length heterogeneity — remains a one-flag follow-up if ever
wanted. It is *not* part of the planned runs.)

**Decision: discounting = γ = 0.95 per own revision event** (simple; with
Poisson gaps the windows are ≈ constant anyway, so the distortion relative to
elapsed-time discounting γ^(gap/λ_K) is negligible).

**Decision: ε-decay clock = ε = exp(−β·n_i)** where n_i = the agent's own
revision count (the natural analog of baseline episodes; scales identically
in expectation).

Convergence & caps (atomic-time analogs of the baseline):
- Converged when no agent's greedy policy has changed for
  **300,000 consecutive atomic periods** (≡ 10,000 baseline episodes × K=30);
  policy check performed at each revision event.
- Cap: **60M atomic periods** (≡ 2M episodes × 30).
- Evaluation: freeze all Q-tables (greedy, no updates), keep the stochastic
  revision clocks running (seeded), record `eval_H = 10,000` atomic periods.

Output schema: exp02 columns; `a_i` = seller i's *currently active* rule in
that period (changes mid-file at revisions). **Decision: include** per-seller
`rev_i` boolean columns marking revision periods (cheap, makes the analysis
of revision timing possible).

Q-init: same heuristic from the existing N=3 K=30 lookup tables (§0.2).

### 3.2 Analysis & expected output

`analysis/robust_exp08_async_poisson/desc_exp08_overview.ipynb`:

- **Figure 1**: Δ vs mu (mean ± SE): baseline sync (from `data/results`,
  N=3 K=30) vs async, per strategy set.
- **Figure 2**: 4R − 3R Δ-difference vs mu: baseline vs async — does the
  Safety Net dip survive desynchronization?
- **Figure 3**: rule-usage shares (async vs sync).
- **Figure 4** (illustrative): one representative price-history window with
  revision markers, showing staggered rule switching.
- Optional diagnostics: distribution of realized revision gaps; frequency of
  rule *changes* per revision (stickiness).
- **Success criterion:** qualitative preservation of (i) high Δ at high mu,
  (ii) the 4-rule destabilization at intermediate mu. Genuine deviations are
  themselves interesting and reportable (e.g. async may weaken coordination at
  the margin — that direction of difference still supports the paper's
  mechanism as long as the cross-(mu, ruleset) pattern holds).

Runtime: same engine as exp07 without the block memo (per-period stepping
always); worst case ≈ 60M cheap steps/run; 600 runs on 32 cores — one
overnight job at most (`hpc/exp08_async_poisson_scrc.sbatch`).

---

## 4. Execution order & effort

1. **Demo notebook (2.2)** — no new infra; fastest deliverable; also
   validates the one-step transition table reused by the engine.
2. **exp06** — ~30 lines of changes in `src/` + a script/sbatch clone; reuses
   existing lookup tables; small HPC job.
3. **`src_atomic/` engine + exp07** — the main new build. Validate by running
   engine in "collapse" debug mode against the lookup table for a few
   (state, profile) pairs (must match exactly), then launch.
4. **exp08** — adds the async loop on the validated engine.

Each experiment gets a smoke test first (tiny `MAX_EPISODES` /
`ROUNDS_PER_CONFIG=1` via env vars, results to `*_smoke` dirs) per
`hpc/HPC_GUIDE.md` §2.

## 5. Decision log (settled 2026-08-03)

1. **Seeds:** 30 per cell (`run_id = 0..29`) for exp06/07/08.
2. **`eval_H` = 10,000** atomic periods for all three experiments (matches
   the canonical `data/results` evaluation horizon).
3. **exp06 Q-init:** keep the uniform-based heuristic init (isolates the
   exploration channel); reweighted-init variant only if later requested.
4. **exp06 scope:** `N ∈ {3, 10}` (N=3 row added since it is cheap and links
   to the exp07/exp08 market size).
5. **Demo notebook:** includes the rule-profile-switch Scenario D
   (undercutting block → Above/Above/Match recovery block).
6. **exp08 gap distribution:** `gap ~ max(1, Poisson(λ_K))`, λ_K = 30
   (geometric variant implemented as config option but not run).
7. **exp08 discounting:** γ = 0.95 per own revision event.
8. **exp08 ε-decay:** ε = exp(−β·n_i), n_i = agent's own revision count.
9. **exp08 output:** per-seller `rev_i` revision-marker columns included.
