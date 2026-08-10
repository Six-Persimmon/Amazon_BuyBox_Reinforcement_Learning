# Extension & Robustness Plan (Reviewer Response)

Status: **exp06/07/08 implemented (2026-08-03)**, code smoke-tested and data
collected. **exp09 planned (2026-08-05), not yet implemented.** All design
choices are marked **Decision:** inline, with the full log in §6.
Numbering continues the existing experiment series: **exp06, exp07, exp08,
exp09**.

Implementation notes (deviations: none; validations performed):
- The `src_atomic` engine's block dynamics were verified to reproduce the
  baseline lookup tables **exactly** (all (state, profile) pairs for four
  (mu, ruleset) configurations, incl. a cost-floor case; reward diffs ≤ 3e-9,
  next-states identical, detailed trajectories identical).
- The `exploration_weights` change to `src/` is backward compatible: with
  weights unset, a stored paper run (`data/results/scan_3strats_mu0.25_k30/
  N_3/run_0.parquet`) is reproduced **bit-for-bit** (same convergence episode,
  zero numeric diff).
- The demo notebook (Part 2A) is already executed: in all four scenarios the
  collapse and carry-over paths coincide from inner step 0–1 of block 2;
  block-2 avg profit differences ≤ 0.45%, avg lowest price identical.
- Measured single-run times (laptop, mu=0.10, 4strats): exp07 ≈ 18 s,
  exp08 ≈ 54 s → full 600-run jobs each fit well within an hour on 32 cores.

| # | Reviewer concern | Our response | New experiment |
|---|---|---|---|
| 1 | 4th rule shifts the ε-exploration distribution toward price-falling actions (1/3 → 1/2); maybe that, not the Safety Net mechanism, drives the 3-vs-4-rule differences | Re-run the 4-rule row with **weighted exploration** that restores P(price-falling draw) = 1/3 | **exp06** (lookup-based, `src/` + small agent change) |
| 2 | "Collapse to lowest price" between episodes is a strong simplification | (a) micro-demo notebook: collapse vs carry-over trajectories are nearly identical; (b) full re-run **without collapse** | demo notebook + **exp07** (new atomic engine) |
| 3 | Synchronized K is restrictive | **Asynchronous revision clocks**: each agent gets stochastic rule-revision times with mean λ_K = 30 | **exp08** (same atomic engine) |
| 4 | *(self-initiated, anticipating "is this just Calvano-style Q-learning collusion relabelled? where does the gain actually come from?")* | **Decompose** the gain into state-space reduction, action-space reduction, and commitment, via a 5-rung ladder running from a Calvano (2020) replication to our full mechanism | **exp09** (new `src_calvano/` engine) |

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

## 4. exp09 — Where does the collusion gain come from? (mechanism decomposition)

### 4.1 The question

Our mechanism differs from a canonical Calvano et al. (2020) Q-learning
oligopoly along three dimensions at once:

1. **State reduction** — agents condition on a scalar (the lowest market
   price) instead of the full price vector.
2. **Action reduction** — agents choose among 3 repricer rules instead of
   all *m* prices.
3. **Commitment** — a chosen rule runs for `K = 30` periods instead of 1.

Each of these is a *coordination device*: an implicit common agreement across
repricer vendors about how to read the market and how to respond. exp09 turns
the three into an explicit **2×2 factorial plus a commitment rung**, so the
total gain can be additively attributed. This is a mechanism-decomposition
study, not a robustness check — it is the experiment that tells us which of
the paper's modelling choices is doing the economic work.

### 4.2 What "basic Calvano" actually is (verified against the paper)

Read off the AER accepted manuscript (Calvano, Calzolari, Denicolò &
Pastorello 2020, §2.2–2.5 and §3.1–3.2):

| Element | Calvano baseline | Our exp09 C1 |
|---|---|---|
| Firms | `n = 2` (they also run `n = 3, 4` in §5.1) | `N = 3` |
| Action space | `m = 15` prices, equally spaced on `[pN − ξ(pM − pN), pM + ξ(pM − pN)]`, `ξ = 0.1` | `m = 10`, our existing grid (implied `ξ = 1/(m−3) = 0.143`, plus the cost floor) — **unchanged from `src/`** |
| State | `s_t = {p_{t−1}, …, p_{t−k}}`, memory `k = 1`; `\|S\| = m^{nk}` — i.e. **all n firms' prices, own price included** | `s_t` = full price vector of all `N = 3` sellers, `\|S\| = 10^3 = 1000` |
| Exploration | ε-greedy, `ε_t = e^{−βt}`, focal `β = 4×10⁻⁶`, upper bound `β̄ = 2×10⁻⁵` | `ε_t = e^{−βt}`, two arms (§4.5) |
| Learning | `α = 0.15`, `δ = 0.95` | `α = 0.15`, `γ = 0.95` (already our baseline) |
| Q-init | Q₀ = discounted payoff if opponents randomize uniformly (their Eq. 8) | same formula, generalized (§4.4) |
| `s₀` | drawn at random each session | random initial price vector (as exp07) |
| Tie-break | lowest price | random among argmax (§4.6) |
| Convergence | argmax constant for **every** state, 100,000 consecutive periods; cap 1e9 | 100,000 consecutive periods; cap per arm (§4.5) |
| Sessions | 1,000 | 30 seeds (repo convention) |
| Outcome | `Δ = (π̄ − πN)/(πM − πN)` | identical definition, already our Δ |

Their §5.1 three-firm numbers are the **external validation target** for C1:
`Δ ≈ 0.64` with β held at the duopoly value, rising to `Δ ≈ 0.75` when β is
lowered to compensate for the larger Q-matrix. We run the first of these (see
§4.5); our grid also differs (`m = 10`, `ξ = 0.143`, plus cost floor), so we
expect the ballpark, not the decimal.

**Decision: every state definition in exp09 includes the seller's own price.**
For C1/C3 the state is the full N-vector of all sellers' previous-period
prices, own price included. For **C2/C4 the reduced state is the minimum over
all N sellers' previous-period prices — the seller's own previous price is one
of the N entries the minimum is taken over**, not just rivals' prices. Three
reasons: (i) the full-vector version is literally Calvano's `|S| = m^{nk}`;
(ii) `min` over all N is exactly the observation `min(price_vec)` already used
by `src/`, exp05 and exp07, so the ladder's bottom rungs are defined the same
way as the rest of the project; (iii) the map `(p_1,…,p_N) ↦ min_j p_j` is
then a clean coarsening of one and the same object, which is what makes the
"state reduction" arm an information-partition treatment rather than two
changes at once. (This is a deliberate departure from the "price vector of all
the *other* agents" phrasing in the original request.)

Note that the *rules* keep using the lowest **competitor** price
(`others_min`, excluding own) to set the next price, exactly as in `src/` and
`src_atomic/` — unchanged. Only the Q-learning observation is the all-N
minimum. The two are different objects and both are inherited unchanged from
the existing codebase.

### 4.3 The five cells

All cells: `N = 3`, all 10 baseline mu values, 30 seeds, carry-over prices
(no state collapse, no lookup table), same grid / cost floor / Δ normalization.

| Cell | Tag | State | Actions | K | `\|S\|` | `\|A\|` | Q-cells (reachable) |
|---|---|---|---|---|---|---|---|
| **C1** | `cal_full` | full price vector | all 10 prices | 1 | 1000 | 10 | 10,000 |
| **C2** | `cal_smin` | market min | all 10 prices | 1 | 10 | 10 | 100 |
| **C3** | `cal_arule` | full price vector | 3 rules | 1 | 1000 | 3 | **1,740** |
| **C4** | `cal_both` | market min | 3 rules | 1 | 10 | 3 | 30 |
| **C5** | `cal_both_k30` | market min | 3 rules | 30 | 10 | 3 | 30 |

C3's reachable count is measured, not nominal: after one rule step only
**580 of the 1000** price vectors are reachable under the 3-rule map (they
spread across all 10 market-min values, ~58 vectors per min value). So the
full-vector state really is ~58× richer than the market min even inside the
rule world — C3 vs C4 is a genuine contrast, not a degenerate one.

**Decomposition** (per mu, exact and additive):

```
D_state       = Δ(C2) − Δ(C1)
D_action      = Δ(C3) − Δ(C1)
D_interaction = Δ(C4) − Δ(C2) − Δ(C3) + Δ(C1)
D_commitment  = Δ(C5) − Δ(C4)
                                       -----------------------------
total gain    = Δ(C5) − Δ(C1) = D_state + D_action + D_interaction + D_commitment
```

**Decision: C5 is run inside exp09, self-contained.** exp07's
`3strats, N=3, K=30` arm is substantively the same setting, but it ran with
`converge_period = 10,000` where exp09 uses Calvano's 100,000 (§4.6). Rather
than carry that asymmetry into the decomposition, C5 is simply re-run here
(300 runs, ~2.5 core-hours — negligible). **exp09 neither reads nor cites
`data/results_exp07`**; the ladder stands on its own five cells, all produced
by the same engine under the same settings.

**C4 is *not* the same as exp05, and the difference is worth reporting.**
exp05's `K=1, 3strats, N=3` cell shares C4's state and action spaces, but it
runs on the lookup table with the **state-collapse** convention — every block
starts from all sellers sitting at the market minimum. C4 uses carry-over: the
price vector at the start of each decision is whatever the previous period's
rules actually produced, so sellers are generally at *different* prices. At
`K = 1` this is the sharpest possible version of the collapse assumption,
because there are no inner periods for the two worlds to re-converge in (the
exp07 argument in §2.1 — that collapsed and carried-over paths coincide after
1–2 inner steps — has no room to operate). C4 vs exp05 is therefore a clean
`K = 1` read on the collapse assumption and belongs in the analysis as a
side-comparison, not as a substitute for running C4.

The whole ladder can later be repeated with the 4-rule set to connect to the
Safety Net Paradox (optional, out of scope for now).

### 4.4 Q-initialization (single formula for every cell)

All five cells use **one** rule, which is simultaneously Calvano's Eq. (8) and
our own Eq. (6):

```
Q0_i(s, a) = (1 / (1 − γ)) · E_{a_-i ~ Uniform(A^{N−1})} [ R_i(s, a, a_-i) ]
```

where `R_i` is exactly the reward that decision earns: the one-period profit
for `K = 1` cells, the K-block average profit for C5. Instantiated:

- **C1 / C2 (price actions, K=1):** `R_i = π_i(a_i, a_-i)`, which does **not
  depend on s**. So `Q0(s,a) = (1/(1−γ)) · mean_{rival price combos}
  PROFIT[…]` — one mean over the rival axes of the existing profit tensor,
  broadcast across all states. This is Calvano Eq. (8) verbatim. C1 and C2
  therefore share the *same* per-action init values.
- **C3 (rule actions, full-vector state, K=1):** `R_i(s,a,a_-i) =
  π_i(f(s, (a, a_-i)))` where `f` is the one-step rule map. Enumerate
  1000 states × 3³ = 27,000 one-step evaluations at startup (well under a
  second) — no lookup table, no cache file needed.
  **The expectation must be accumulated per seller position** (config
  `init_pooling="per_seller"`, the default). C3's state is an *ordered* price
  vector and the rules key off the lowest *competitor* price, so at an
  asymmetric state the sellers face genuinely different decision problems: at
  `(0,5,9)` and mu=0.25 the seller holding the minimum sees others_min = 5 and
  prefers Undercut (0.073/0.060/0.049), while the other two see others_min = 0
  and prefer Above (0.132/0.132/0.142). Averaging the positions into one
  shared row — the original implementation — gives every seller a prior that
  is correct for none of them and flips the initial argmax for at least one
  seller at **582 of 1000 states**. C1/C2 are immune (the stage payoff depends
  on neither the state nor the position) and C4/C5 are immune (the collapsed
  representative vector is symmetric: measured max deviation 8.9e-16, zero
  argmax flips), so this is a C3-only issue. See §4.7 for the empirical effect.
- **C4 (rule actions, market-min state, K=1):** the reward depends on the full
  vector, which the state does not pin down. **Decision: use the collapsed
  representative vector** `(o, o, …, o)`, i.e. `R_i(o,a,a_-i) =
  π_i(f((o,…,o), (a,a_-i)))`. This reproduces
  `src.agent.calculate_heuristic_init_values` at `K = 1` **exactly**, so C4's
  prior is identical to exp05's and consistent with exp07's. (Alternative —
  averaging over the reachable vectors with `min = o` — goes in as a config
  flag `init_state_rep = "collapsed" | "reachable_avg"`, not run by default.)
- **C5:** the same rule-mode formula with `K = 30`, i.e. the block reward is
  averaged over 30 periods. Computed directly (10 states × 27 profiles × 30
  steps), **not** loaded from a lookup table — but verified numerically equal
  to `calculate_heuristic_init_values` on the cached `K = 30` table to 1.8e-15
  (`tests/test_exp09_engine.py`), so C5's prior is the one exp07 used. exp09
  therefore needs no lookup tables and writes no cache files at all.

Note the deliberate consequence: for C1/C2 the init is constant across states,
so at `t = 0` the greedy policy is the same price everywhere. That is
Calvano's design and is harmless (`ε = 1` at `t = 0`).

### 4.5 Exploration intensity: β, and why C1 is not a strict replication

**Decision: a single arm at `β = 1e-5`** — our repo-standard value, uniform
across all five cells. Equal learning technology everywhere, and directly
comparable to exp05 and the main sweep. This is what gets run.

The following caveat must be carried into the paper text; it is a real
limitation of C1, not a technicality.

> **Remark (C1 is a Calvano-*style* baseline, not a strict replication).**
> Two things differ from Calvano et al. (2020) beyond the obvious `N = 3`:
> the price grid (`m = 10`, implied `ξ = 0.143`, plus our cost floor, vs their
> `m = 15`, `ξ = 0.1`), and — more consequentially — the **effective amount of
> exploration per Q-matrix cell**. Calvano flag this mechanism themselves in
> §5.1: moving from `n = 2` to `n = 3` blows the Q-matrix up from 3,375 to
> ~50,000 cells, and *"since the parameter β is held constant, the increase in
> the size of the matrix makes the effective amount of exploration much
> lower"*; lowering β to compensate raises their three-firm Δ from ≈ 0.64 to
> ≈ 0.75. Writing `ν̄ = (1/β)/(|S|·|A|)` for the exploration budget per
> Q-cell, our ladder spans a **333× range in Q-matrix size** and therefore a
> 333× range in `ν̄` at fixed β:
>
> | Cell | Q-cells | `ν̄` at β = 1e-5 |
> |---|---|---|
> | C1 | 10,000 | **10** |
> | C2 | 100 | 1,000 |
> | C3 | 1,740 | 57 |
> | C4 / C5 | 30 | 3,333 |
>
> Calvano's focal duopoly point sits at `ν̄ ≈ 74` and their own upper-bound β̄
> at `ν̄ ≈ 15`, so **C1 at `ν̄ = 10` is under-explored by their standard**.
> Consequently: (i) C1 should be read as the analogue of their *β-held-fixed*
> three-firm number (`Δ ≈ 0.64`), not their compensated one; and (ii) part of
> any measured `D_state` / `D_action` is a **learning-tractability** channel
> — smaller table, easier to learn — rather than a purely strategic
> coordination channel. We regard the tractability channel as economically
> real and part of what adopting a common repricer actually does, so we do not
> attempt to net it out; but the decomposition should be described as
> *total* effect of dimension reduction, not as a pure coordination effect.

**Optional extension (not scheduled): a low-β arm at `β = 1e-6`.** Uniform
across cells as above, but 10× more exploration, which puts C1 at
`ν̄ = 100` — past Calvano's focal duopoly intensity — and maps to their
compensated `Δ ≈ 0.75`. The gap between the two arms would then measure the
learning-tractability channel directly, and the part of the ladder surviving
the low-β arm would be the coordination/focality channel. Cost ~2.6 h on 32 cores
(§4.8). The engine takes β as a config field and the experiment script exposes
`PRICING_MARL_EXP09_BETA` plus a separate results root, so running this later
requires no code change — only a launch.

Also rejected (kept as a config flag, not run): a per-cell **equal-ν̄** arm
with β scaled by `1/(|S||A|)`. It would need `β ≈ 3e-8` for C1 and `β ≈ 3e-4`
for C4 — a 10⁴ spread in ε-decay speed, which makes "the same algorithm,
different information" untrue in a different way.

Cap: `max_episodes = 5M`, ≥ 10× headroom over the `1/β = 100k` point where ε
becomes negligible.

### 4.6 Engine `src_calvano/` and mechanics

New sibling package (repo convention, like `src_ext_K_action/` and
`src_atomic/`); **do not modify `src_atomic/`**, whose outputs are already
canonical:

```
src_calvano/
├── __init__.py
├── config.py       # CalvanoConfig: + state_mode, action_mode, K, init_state_rep
├── environment.py  # thin layer over src_atomic.AtomicEnvironment (grid, PROFIT, NEXT)
├── agent.py        # QAgent over (num_states, num_actions) + the §4.4 inits
├── simulation.py   # one unified atomic loop
└── runner.py       # clone of src_atomic/runner.py
```

- `state_mode ∈ {"full_vector", "market_min"}`;
  `action_mode ∈ {"price", "rule"}`. Q-table is
  `(m**N or m, m or len(active_strategies))`. Full-vector encoding
  `s = Σ_i idx_i · m^(N−1−i)` (ordered, as Calvano).
- `environment.py` reuses `AtomicEnvironment` unchanged for the price grid,
  cost floor, `PROFIT` tensor, `NEXT` rule table and the Nash/monopoly
  benchmarks — so Δ is normalized identically to every other experiment.
- **Unified loop.** Every cell is "observe → choose → prices evolve for K
  periods → reward = average profit over those K periods → new state":
  `action_mode="price"` sets the next price vector directly (only sensible for
  `K = 1`); `action_mode="rule"` calls `step_one_period` / `run_block`. With
  `K = 1` + price actions this reduces to Calvano exactly. Timing note: our
  state is the *current* vector and the action determines the *next* one,
  which is Calvano's `s_t = p_{t−1}` convention re-indexed by one — identical
  structure, and it makes the reward for a rule (realized one period after the
  rule is chosen) line up with the reward for a price.
- **`t = 0`:** independent uniform random price index per seller (as exp07).
- **Convergence: `converge_period = 100,000` decision epochs** (Calvano's own
  criterion; the repo's 10,000 is too weak for a Q-matrix up to 100× larger).
  Cost is ~1 s/run, so this is nearly free insurance.
- **Required optimization:** check convergence *incrementally*. Only the one
  `(s, a)` cell each agent just updated can change that agent's argmax, so
  recompute `argmax` for that row only. Measured: the naive full-matrix
  `np.argmax(Q, axis=2)` check costs **34 µs/period** against a **10 µs**
  loop — a 4× slowdown — while the incremental check costs 2 µs.
- **Decision: random tie-breaking among argmax** (repo convention), not
  Calvano's lowest-price rule. Their rule would bias C1/C2 toward low prices
  and hence *understate* Δ in exactly the price-action cells, i.e. it would
  flatter our thesis; random is the conservative choice. Available as a config
  flag for a faithfulness check.
- **Evaluation:** greedy, frozen Q, `eval_H = 10,000` atomic periods, exp02
  Parquet schema (so existing loaders work). `a_i` holds the price grid index
  in C1/C2 and the strategy ID in C3/C4/C5; `state_mode` / `action_mode` in
  `Config_N_3.json` disambiguate.
- Batches `scan_<tag>_mu<mu>_k<K>` with the §4.3 tags, output
  `data/results_exp09/`; `hpc/exp09_calvano_ladder_scrc.sbatch` +
  `hpc/exp09_progress_report.py`. (β and the results root are env-var
  overridable, so the optional Arm B of §4.5 needs only a different launch.)

### 4.7 Analysis & expected output

`analysis/robust_exp09_calvano_ladder/desc_exp09_overview.ipynb`:

- **Figure 1** (headline): Δ vs mu, five lines C1–C5, mean ± across-run SE.
- **Figure 2** (headline): waterfall / stacked-bar decomposition per mu —
  C1 → `+D_state` → `+D_action` → `+D_interaction` → `+D_commitment` → C5.
- **Figure 3**: same two panels for average lowest price (level, not gain).
- **Figure 4**: converged-outcome composition — price distributions for
  C1/C2, rule-usage shares for C3/C4/C5.
- **Diagnostics**: converged fraction and mean stop episode per cell (C1 is
  the one at risk, per the §4.5 remark — report `ν̄` alongside); C4-vs-exp05
  overlay as the `K = 1` read on the state-collapse assumption (§4.3).
- **Table**: per-mu Δ mean/SD/SE for the five cells plus the four
  decomposition terms.

**What would count as a result either way.** If `D_state` and `D_action` are
large and positive, the paper's claim — that the collusive gain is
manufactured by the *common simplification* repricer vendors impose on the
market, not by Q-learning per se — is supported directly. If instead C1
already achieves Calvano-level Δ and the ladder is flat, the honest reading is
that our mechanism reproduces rather than creates algorithmic collusion, and
the paper's contribution shifts to the *interpretability and equilibrium
selection* results (which rules get selected, the Safety Net Paradox) rather
than the level of Δ. Both are publishable; the design should not be tuned to
produce the first.

### 4.8 Runtime & execution order

Measured on this laptop: **~10 µs per atomic period**, essentially independent
of `|S|` once the incremental convergence check is in (10.2 µs for C1 with
1000 states, 9.4 µs for C4 with 10).

Measured **single runs at full production settings** (`mu = 0.25`, seed 0,
`converge_period = 100,000`, `max_episodes = 5M`, `eval_H = 10,000`) — all
four converged well inside the cap:

| Cell | wall time | convergence episode | Δ (1 seed) |
|---|---|---|---|
| C1 `cal_full` | 37.5 s | 1,331,926 | +0.628 |
| C3 `cal_arule` | 29.3 s | 995,874 | +0.683 |
| C4 `cal_both` | 27.9 s | 959,529 | +0.758 |
| C5 `cal_both_k30` | 18.7 s | 737,961 | +1.000 |

Worth flagging: **C1's Δ = 0.628 sits essentially on Calvano's reported
three-firm `Δ ≈ 0.64`** (their β-held-fixed number, which per §4.5 is the one
C1 corresponds to). That is a single seed at a single mu and must not be
quoted as a result, but it is a strong sign the C1 rung is behaving as a
Calvano replication should.

| Arm | Runs | ≈ per run | Core-hours | On 32 cores |
|---|---|---|---|---|
| **β = 1e-5** (5 cells × 10 mu × 30 seeds) — the run | 1,500 | ~30 s | ~13 | **~25 min** |
| β = 1e-6 (optional Arm B, §4.5 — not scheduled) | 1,500 | ~200 s | ~83 | ~2.6 h |

Order: (1) `src_calvano/` engine + validation — **done**, see
`tests/test_exp09_engine.py`; (2) smoke test — **done**; (3) the β = 1e-5 run
on HPC; (4) notebook.

**Implementation status (2026-08-05): engine built and validated, run not yet
launched.** Files: `src_calvano/{config,environment,agent,simulation,runner}.py`,
`experiments/exp09_calvano_ladder.py`,
`hpc/exp09_calvano_ladder_scrc.sbatch`, `hpc/exp09_progress_report.py`,
`tests/test_exp09_engine.py`. Validation performed
(`python tests/test_exp09_engine.py`, all pass):

- full-vector state encoding is a bijection on all 1000 vectors and
  round-trips; `market_min` verified to include the seller's own price;
- C5 block dynamics reproduce `src_atomic` on **all 27,000 (price vector,
  rule profile) pairs** — next states identical, max reward diff 0;
- C4 (`K=1`) and C5 (`K=30`) Q-init reproduce
  `src.agent.calculate_heuristic_init_values` to **1.8e-15**, i.e. exp09's
  rule rungs carry exactly the exp05/exp07 prior;
- C1/C2 Q-init equals an independently computed Calvano Eq. (8) to 1.8e-15
  and is constant across states, as that formula requires;
- the 580/1000 reachable-vector count of §4.3 is reproduced;
- all five cells run end to end and emit the exp02 Parquet schema, with `a_i`
  in the price-grid range for C1/C2 and the strategy-ID range for C3/C4/C5.

---

## 5. Execution order & effort

1. **Demo notebook (2.2)** — no new infra; fastest deliverable; also
   validates the one-step transition table reused by the engine.
2. **exp06** — ~30 lines of changes in `src/` + a script/sbatch clone; reuses
   existing lookup tables; small HPC job.
3. **`src_atomic/` engine + exp07** — the main new build. Validate by running
   engine in "collapse" debug mode against the lookup table for a few
   (state, profile) pairs (must match exactly), then launch.
4. **exp08** — adds the async loop on the validated engine.
5. **`src_calvano/` engine + exp09** (2026-08-05 onward) — the mechanism
   decomposition; see §4.8 for its own ordering. Independent of exp06–08
   except that it reuses `src_atomic.AtomicEnvironment` and cross-checks
   against stored exp05/exp07 data.

Each experiment gets a smoke test first (tiny `MAX_EPISODES` /
`ROUNDS_PER_CONFIG=1` via env vars, results to `*_smoke` dirs) per
`hpc/HPC_GUIDE.md` §2.

## 6. Decision log

### exp06 / exp07 / exp08 (settled 2026-08-03)

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

### exp09 (settled 2026-08-05, pending implementation)

10. **State definition:** the C1/C3 state is the **full price vector of all N
    sellers, own price included** (`|S| = m^N = 1000`), matching Calvano's
    `|S| = m^{nk}`. The **C2/C4 reduced state is the minimum over all N
    sellers' previous prices, the seller's own previous price included** —
    the same observation `min(price_vec)` used by `src/`, exp05 and exp07.
    Departs from the "other agents' prices" phrasing in the original request.
    Unchanged and unrelated: the *rules* still key off `others_min` (lowest
    competitor price, excluding own).
11. **Design shape:** 2×2 factorial (state ∈ {full, min} × actions ∈ {prices,
    3 rules}) plus a fifth commitment rung at `K = 30`; the decomposition is
    exactly additive (§4.3).
12. **Q-init:** one formula for all cells,
    `Q0_i(s,a) = E_{a_-i ~ U}[R_i(s,a,a_-i)] / (1−γ)` — simultaneously
    Calvano Eq. (8) and our Eq. (6). Price cells get a state-independent init
    from the profit tensor; C3 enumerates 27,000 one-step outcomes; C4 uses
    the **collapsed representative vector** `(o,…,o)` so its prior is
    identical to exp05's.
12a. **(2026-08-06, after external review) The expectation is taken per seller
    position**, `init_pooling="per_seller"` (default). The first
    implementation pooled the positions into one shared row, which is exact
    for C1/C2/C4/C5 but wrong for C3 — see §4.4. C3 was regenerated; the
    pooled-init data is archived at
    `data/archive/results_exp09_c3_pooled_init_2026_08_06/`. Effect: C3 rises
    by +0.014 to +0.087 across mu. Because `Δ(C3)` enters `D_action` and
    `D_interaction` with opposite signs, their **sum is identically
    `Δ(C4) − Δ(C2)`** and is unaffected (verified to 6e-08); `D_state`,
    `D_commitment`, the net dimension-reduction effect and the total are
    numerically unchanged. Only the split between the action and interaction
    channels moves.
13. **One β arm, `β = 1e-5`, uniform across cells** (repo standard,
    exp05-comparable). C1 is therefore under-explored by Calvano's own
    standard (`ν̄ = 10` vs their focal 74), so **C1 is a Calvano-*style*
    baseline, not a strict replication** — it maps to their β-held-fixed
    three-firm `Δ ≈ 0.64`, and part of `D_state`/`D_action` is a
    learning-tractability channel that we report as such rather than net out.
    The mandatory paper-text remark is in §4.5. `β = 1e-6` (Arm B) and the
    per-cell equal-ν̄ scaling are **optional extensions, not scheduled**; β
    and the results root are env-var overridable so neither needs code.
14. **Convergence:** Calvano's `converge_period = 100,000` decision epochs for
    all exp09 cells (repo's 10,000 is too weak for a 100× larger Q-matrix);
    cap `max_episodes = 5M`.
15. **C5 is run inside exp09, self-contained.** exp09 neither reads nor cites
    `data/results_exp07`; all five rungs come from the same engine under the
    same settings, so the convergence criterion is uniform across the ladder.
    Kept as a reported side-comparison: **C4 vs exp05**, which is the `K = 1`
    read on the state-collapse assumption (see §4.3).
16. **Tie-breaking:** random among argmax (repo convention), not Calvano's
    lowest-price rule — the conservative choice, since theirs would depress Δ
    in exactly the price-action cells and flatter our thesis.
17. **Engine:** new sibling package `src_calvano/`; `src_atomic/` is not
    modified. Convergence must be checked **incrementally** (updated row
    only) — the naive full-matrix argmax check is a measured 4× slowdown.
18. **Scope:** `N = 3`, all 10 mu values, 30 seeds, 3-rule set. Repeating the
    ladder with the 4-rule set (Safety Net link) is an optional extension.
