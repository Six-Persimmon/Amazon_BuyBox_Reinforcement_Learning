# Investigation Report: Why 3-Strats and 4-Strats Show Different Equilibrium Patterns (N=10, K=40, mu=0.13)

Date: 2026-02-20  
Author: Codex investigation summary

## 1. Question

The observed phenomenon is:

- Two experiment settings use the same economic parameters:
  - `N=10`, `K=40`, `mu=0.13`
- Strategy sets:
  - `3-strats`: `[Undercut, Match, Above]` (IDs `[0,1,2]`)
  - `4-strats`: `[Undercut, Match, Above, Undercut+Reset]` (IDs `[0,1,2,3]`)
- In many `4-strats` runs, final equilibrium behavior appears to not use `Undercut+Reset` (or uses it very rarely), but the final pattern is still very different from `3-strats`.

Core question:

Why can the final equilibrium pattern differ so much even when the visible equilibrium rule set seems effectively the same?

## 2. Data and Scope

This investigation uses the existing batch outputs:

- `pricing_marl/data/results/scan_3strats_mu0.13_k40/N_10/run_*.parquet`
- `pricing_marl/data/results/scan_4strats_mu0.13_k40/N_10/run_*.parquet`

Each setting has 100 runs (`run_0` to `run_99`).

Config files used:

- `pricing_marl/data/results/scan_3strats_mu0.13_k40/Config_N_10.json`
- `pricing_marl/data/results/scan_4strats_mu0.13_k40/Config_N_10.json`

Common important hyperparameters (both sets):

- `max_episodes=2_000_000`
- `converge_period=10_000`
- `beta=1e-5`
- `eval_H=10_000`
- `K=40`

## 3. Method

## 3.1 Population-level comparison over all 100 runs

For each parquet file, compute:

- whether training converged (`converged`)
- training stop episode (`episode`)
- equilibrium type (`H`/`C` etc.) using:
  - `analysis/viz_heatmap_eq_diversity_k_mu_by_n.py:90` (`classify_equilibrium`)
- final action signature (sorted actions at episode starts)
- price and collusion metrics from tail windows
  - tail-200 mean: `price_mean`, `delta`
  - tail-1000 volatility/amplitude:
    - `std(price_mean)`
    - `max(price_min)-min(price_min)`

## 3.2 Check whether `Undercut+Reset` is truly absent

In `4-strats`, scan all `a_i` columns for action ID `3`.

## 3.3 Same-seed paired comparison

Pair runs by the same `run_id` between 3-strats and 4-strats and compare:

- final signature equality
- mean differences in price, delta, and cycle amplitude

## 3.4 Code-path mechanism tracing

Read key implementation paths:

- exploration policy: `pricing_marl/src/agent.py:57`
- random action sampling over `num_actions`: `pricing_marl/src/agent.py:61`
- heuristic Q initialization from lookup table: `pricing_marl/src/agent.py:4`
- lookup table action-space enumeration: `pricing_marl/src/environment.py:180`
- convergence check: `pricing_marl/src/simulation.py:58`
- convergence stop condition: `pricing_marl/src/simulation.py:65`

## 3.5 Controlled counterfactual on convergence period

Re-run same seed(s) with changed `converge_period` (10k vs 100k) to check if outcome changes under longer training.

## 4. Main Results

## 4.1 Equilibrium type distribution (100 runs each)

| Setting | H | C |
|---|---:|---:|
| 3-strats | 95 | 5 |
| 4-strats | 16 | 84 |

Interpretation:

- `3-strats` is mostly high/static-like (`H`).
- `4-strats` is mostly cycle (`C`), dominated by large oscillation behavior.

## 4.2 Is `Undercut+Reset` actually used in 4-strats?

| Metric | Value |
|---|---:|
| Runs with any action `3` appearing | 3 / 100 |
| Run IDs | 21, 39, 51 |

Even excluding those 3 runs, `4-strats` still has overwhelmingly cycle outcomes (`C` remains dominant).

## 4.3 Price/collusion statistics

Tail metrics by run, then averaged across runs:

| Metric | 3-strats | 4-strats |
|---|---:|---:|
| Mean tail-200 `price_mean` | 2.140999 | 1.618815 |
| Mean tail-200 `delta` | 0.885118 | 0.423824 |
| Mean tail-1000 `std(price_mean)` | 0.026740 | 0.444892 |
| Mean tail-1000 `amp(price_min)` | 0.057931 | 0.973235 |
| Median tail-1000 `amp(price_min)` | 0.000000 | 1.158613 |
| Runs with `amp(price_min) > 0.05` | 5 | 84 |

Interpretation:

- `4-strats` is much more volatile and much less collusive on average in this cell.
- The cycle pattern is not subtle; it is large-amplitude.

## 4.4 Same-seed paired comparison (run_id aligned)

| Metric | Value |
|---|---:|
| Paired runs | 100 |
| Same final signature count | 7 |
| Mean(`price_mean_4 - price_mean_3`) | -0.522183 |
| Mean(`delta_4 - delta_3`) | -0.461294 |
| Mean(`amp_4 - amp_3`) | +0.915304 |

Interpretation:

- With the same seed index, outcomes are usually different across action spaces.
- This is a true equilibrium-selection shift, not a plotting artifact.

## 4.5 Early-stop correlation with cycle outcomes

Training stop episode (`episode`) split:

### 3-strats

- `episode <= 20k`: 5 runs, all `C`
- `episode > 20k`: 95 runs, all `H`

### 4-strats

- `episode <= 20k`: 77 runs, all `C`
- `episode > 20k`: 23 runs, `H=16`, `C=7`

Interpretation:

- In this cell, early stop strongly correlates with cycle outcomes.
- `4-strats` triggers early stop much more often.

### Training-stop distribution details

| Setting | Mean stop episode | Median | Min | Max | `<=20k` | `<=100k` |
|---|---:|---:|---:|---:|---:|---:|
| 3-strats | 758,915.4 | 796,076 | 10,000 | 923,693 | 5 | 5 |
| 4-strats | 172,554.5 | 10,000 | 10,000 | 863,782 | 77 | 78 |

### Epsilon-at-stop summary (`epsilon = exp(-beta * t)`, `beta=1e-5`)

| Setting | Mean epsilon at stop | Median epsilon | Runs `epsilon>0.5` | Runs `epsilon>0.9` |
|---|---:|---:|---:|---:|
| 3-strats | 0.045598 | 0.000349 | 5 | 5 |
| 4-strats | 0.699532 | 0.904837 | 78 | 62 |

Interpretation:

- Most 4-strats runs stop when exploration is still very high.
- Most 3-strats runs stop only after exploration has decayed a lot.

## 4.6 Representative same-seed example: `run_1`

Files:

- `pricing_marl/data/results/scan_3strats_mu0.13_k40/N_10/run_1.parquet`
- `pricing_marl/data/results/scan_4strats_mu0.13_k40/N_10/run_1.parquet`

### 3-strats run_1

- training stop episode: `763825`
- eval action signature (250/250 episodes): `(1,1,1,1,1,2,2,2,2,2)`
- `price_min` start/end of each K-period stays at high extreme (`2.174308...`)

### 4-strats run_1

- training stop episode: `10000`
- eval action signatures split:
  - 125 episodes: all `Above` (`2`)
  - 125 episodes: all `Undercut` (`0`)
- `price_min` alternates between lowest and highest grid extremes:
  - `1.015695... <-> 2.174308...`

Interpretation:

- This is the exact "big cycle driven by undercut+above" pattern.

### Quick visual count (ASCII bars)

Equilibrium counts in this cell:

- 3-strats: `H=95`, `C=5`  
  `H: ############################################################### (95)`  
  `C: ### (5)`
- 4-strats: `H=16`, `C=84`  
  `H: ########## (16)`  
  `C: ######################################################## (84)`

## 5. Why This Happens (Mechanism)

## 5.1 Adding an action changes training dynamics even if final equilibrium barely uses it

In this implementation, adding action `3` changes:

1. Exploration probability mass across actions
2. Heuristic Q initialization values (including shared actions 0/1/2)
3. Lookup-table payoff statistics used by initialization

So `4-strats` is not equivalent to `3-strats` + an ignored action.

## 5.2 Exploration effect

`choose_action` is epsilon-greedy:

- epsilon: `exp(-beta * t)` at `pricing_marl/src/agent.py:57`
- random action sampled uniformly over all actions at `pricing_marl/src/agent.py:61`

With `beta=1e-5`, exploration is still high early.  
Before episode 10k, expected random picks are very large, and in 4-strats a quarter of random picks go to action `3`, perturbing updates.

Numerical scale (first 10k episodes):

- `sum_{t=0}^{9999} epsilon_t = 9516.3058`
- per agent expected random picks in first 10k episodes: `9516.3058`
- for 10 agents: `95163.0578`
- in 4-strats, expected random picks of action `3` (uniform among 4 actions): `23790.7644`

## 5.3 Heuristic initialization effect

Initialization comes from lookup-table averages:

- `pricing_marl/src/agent.py:4`

Lookup table enumerates action profiles with `range(num_actions)^N`:

- `pricing_marl/src/environment.py:180`

So moving from 3 to 4 actions changes value estimates even for actions 0/1/2.

Observed initial argmax differences (same economic params):

- 3-strats initial argmax by state: `[2,2,2,2,2,2,0,0,0,0]`
- 4-strats initial argmax by state: `[2,2,2,2,2,2,2,0,0,0]`

At high states (e.g., state 9), 4-strats initialization lowers `Match` significantly and keeps strong `Undercut` pressure.

Selected initial Q values (states 6-9):

| State | 3-strats `[U,M,A]` | 4-strats `[U,M,A,UR]` |
|---:|---|---|
| 6 | `[0.2267, 0.2102, 0.2066]` | `[0.2009, 0.1399, 0.2159, 0.1576]` |
| 7 | `[0.2986, 0.2550, 0.2227]` | `[0.2591, 0.1683, 0.2239, 0.2170]` |
| 8 | `[0.3775, 0.2977, 0.2373]` | `[0.3236, 0.1987, 0.2327, 0.2828]` |
| 9 | `[0.4474, 0.3142, 0.2820]` | `[0.3840, 0.2246, 0.2569, 0.3445]` |

## 5.4 Convergence criterion amplifies path dependence

Convergence check in training:

- compare full greedy policy matrix each episode: `pricing_marl/src/simulation.py:58`
- stop when unchanged for `converge_period`: `pricing_marl/src/simulation.py:65`

With `converge_period=10_000`, many 4-strats runs stop while still in a high-exploration regime (very early), locking in cycle attractors.

## 6. Counterfactual: Longer Convergence Period

Same-seed controlled test (`4-strats`, same `N/K/mu`):

### Seed 1

- `converge_period=10k`:
  - converged at `10000`
  - final high-state policy `state9={0:10}` (all undercut-like, cycle attractor)
- `converge_period=100k`:
  - converged at `1209320`
  - final high-state policy `state9={1:6, 2:4}` (match/above mix, high-state stabilization)

### Seed 2

- `converge_period=10k`:
  - converged at `10000`
  - `state9={0:10}`
- `converge_period=100k`:
  - converged at `1034956`
  - still `state9={0:10}`

Conclusion:

- Increasing convergence period can materially change outcomes and reduce premature cycle lock-in.
- But it is not guaranteed for every seed.

## 7. Final Diagnosis

The observed discrepancy is explained by equilibrium selection under different learning dynamics:

1. `4-strats` changes exploration and initialization, even if action `3` is rarely visible at the end.
2. With current `converge_period=10k`, 4-strats frequently stops early and locks into cycle attractors.
3. 3-strats typically trains much longer in this cell, allowing migration toward high-price (`H`) attractors.

So the two settings are behaviorally different because training trajectories are different, not because plotted equilibria are misread.

## 8. Reproducibility Notes

Key scripts/files touched for analysis:

- `pricing_marl/src/agent.py`
- `pricing_marl/src/environment.py`
- `pricing_marl/src/simulation.py`
- `pricing_marl/analysis/viz_heatmap_eq_diversity_k_mu_by_n.py`
- parquet outputs under:
  - `pricing_marl/data/results/scan_3strats_mu0.13_k40/N_10/`
  - `pricing_marl/data/results/scan_4strats_mu0.13_k40/N_10/`

If rerunning, keep the same parameter cell:

- `N=10`, `K=40`, `mu=0.13`
- compare strategy sets `[0,1,2]` vs `[0,1,2,3]`
- run count `100`
