# Amazon_BuyBox_Reinforcement_Learning
Multi-agent price competition in Amazon Buy Box environment. This project aims to answer the following questions

1. How does the Amazon marketplace with the Buy Box (or Amazon Featured Offer) mechanism differ from the classic Bertrand competition scenario in Industrial Organization?
2. Given the existing repricer services in the market, can we observe evidence of supracompetitive equilibrium prices (i.e. as described in Asker et al. 2022)?
3. Given the heterogeneity of sellers (e.g. size, history, comments), what are the best dynamic pricing strategies for the sellers to win the Buy Box and win over sales in each ASIN market?

To answer these questions:

1. We use deep learning models to learn and mimic the Buy Box assignment mechanism given the state of the market. This step learns the model from observational data from Amazon.
2. Given the estimated model, we simulate multi-agent scenarios where each agent can choose the specific pricing rules (i.e. match the buy box price, match the lowest price, undercut Amazon's price etc.) in each period. Each agent also has heterogeneity in their marginal cost, and whether they are fulfilled by Amazon (FBA).
3. We assume there is a simple demand-side model due to the lack of observed transaction data.
4. We observe and report the scenarios most suitable for price collusion, and the best strategy for different sellers to survive longer and win the buy box.

## N-player MARL Analysis Pipeline

- `experiments/n_player_experiments.py` centralizes experiment orchestration: houses the default N=2-5 analysis pipeline, config-driven batch runners, and validation utilities.
- `run_n_player_analysis.py` now wraps the experiment module so legacy entry points still launch the default analysis.
- `simulations/n_player_simulation.py` coordinates each batch: it builds an `NPlayerLogitDemandPricingEnv`, loops through periods, and returns price, rule, and profit histories for every run.
- `env/NPlayerLogitDemandPricingEnv.py` implements the repeated logit-demand Bertrand game that maps discrete price indices to actual prices, logit shares, and per-period profits while exposing monopoly and symmetric Nash calculators.
- `agents/n_player_rule_agent.py` defines the rule-based Q-learning agents that choose among five pricing heuristics (match, above, below-with-cost-floor, hold, conditional raise) based on a reduced state `(own_price_idx, min_competitor_idx)`.
- `analysis/plotting.py` provides smoothing and visualization utilities that the analysis harness can call when generating price trajectories or uncertainty bands.
- `utils/data_utils.py` persists the simulation panels, metadata, and LaTeX-ready tables to `./data/n_player_analysis`.

### MARL formulation

We cast the pricing competition as a stochastic game $$(S, {A_i}_{i=1}^N, P, {r_i}_{i=1}^N, \gamma)$$ with discount factor $$\gamma = 0.9$$ inside the agent update and $$\beta = 0.95$$ in the environment. The joint state $$s_t$$ is the tuple of last-period price indices $$(p^1_{t-1}, ..., p^N_{t-1})$$, while each agent observes the reduced state $$s^i_t = (p^i_{t-1}, \min_{j \neq i} p^j_{t-1})$$. Every agent selects a rule $$u^i_t \in \{0,1,2,3,4\}$$ that maps to an executable price index $$a^i_t$$ through rule heuristics (match lowest, step above, cost-bounded step below, hold, conditional raise).

Given a realized price profile $$a_t$$, the environment converts indices to prices $$\pi_i(a_t)$$ and delivers profits

$$
r_i(s_t, a_t) = (\pi_i(a_t) - c) \cdot \frac{\exp((a_{12} - \pi_i(a_t))/\mu)}{\exp(a_0/\mu) + \sum_{j=1}^N \exp((a_{12} - \pi_j(a_t))/\mu)}
$$

where $$c$$ is marginal cost. Transitions are deterministic on price indices, so $$s_{t+1}$$ equals the action tuple. Agents update their rule-value estimates with tabular Q-learning over the reduced state:

$$
Q_i(s^i_t, u^i_t) \leftarrow Q_i(s^i_t, u^i_t) + \alpha \Big[r_i(s_t, a_t) + \gamma \max_{u} Q_i(s^i_{t+1}, u) - Q_i(s^i_t, u^i_t)\Big]
$$

and follow an exponentially decaying $$\varepsilon$$-greedy policy $$\varepsilon_t = \exp(-\omega t)$$ to balance exploration and exploitation. The orchestrator aggregates the final slices of `prices`, `actions`, and `profits` across runs, produces equilibrium comparisons, heatmaps, and moving-average trajectories, and saves outputs to `./figure/n_player_analysis` for visualization.

## Contextual Bandit Repricer Pipeline

- `env/NPlayerPoissonDemandEnv.py` runs the inner $T$-period competition with Poisson arrivals, mixed logit/naive demand, and executes the selected meta repricing rules.
- `agents/repricer_meta_actions.py` enumerates the meta-action library (no repricer plus every rule/reset/raise combination) and provides helpers for feature building and per-player executors.
- `agents/linucb.py` implements the shared LinUCB learner with per-action statistics and model persistence.
- `agents/neural_ucb.py` offers a lightweight neural contextual bandit alternative with epsilon-greedy exploration.
- `agents/contextual_bandit_agent.py` stitches contexts, feature engineering, action selection, and learner updates, exposing a common interface for both learning backends.
- `pipelines/run_contextual_bandit_experiment.py` orchestrates end-to-end training across outer rounds. Each round samples fresh market contexts `(N, \lambda, \rho)` and marginal costs, trains the shared bandit, stores timestamped checkpoints, and records round-level outcomes under `data/contextual_bandit_runs`.
- `experiments/contextual_bandit_scenarios.py` loads a saved bandit and evaluates it in user-specified scenarios (fixed `(N, \lambda, \rho, \text{MC})`, custom horizons). It emits detailed logs—including price histories and interaction networks—for post-training analysis.
- `analysis/contextual_bandit_reporting.ipynb` is a notebook workspace for loading saved runs/models, computing summaries, and plotting counterfactuals or post-training diagnostics.

Saved models are stored in `data/contextual_bandit_models` with filenames such as `contextual_bandit_linucb_20240101T120000Z.npz`, enabling quick retrieval for counterfactual evaluation or continuing training.





I am trying to build a MARL problem about amazon seller repricer competition. The goal is to a) understand how sellers learn to choose to use different repricer rules during the evolution. b) record how market outcomes changes during the learning and at the equilibrium e.g. average lowest price, average price c) vary the parameters in the demand setting, as well as the number of sellers and their marginal costs to see how these affects the market outcomes d) network analysis on what kind of network structure results in higher/lower/more stablized/less stablized prices (I will explain more about the newtork analysis later).

For each simulation study, I start with a fixed N (number of sellers), fixed $\rho$ and $\lambda$ and other demand environment parameters, the meta action rule library (all the repricer rules available to all the sellers, which includes an action to use no rule), the list of margininal costs for each seller

I start with the DEMAND function, which is in "env/NPlayerPoissonDemandEnv.py". You can read it and get a sense of how the demand is generated under this specific logit demand system.

Next, here is a sudo code of how the MARL game works:

```
for each outer loop k from 1 to K:
	Each seller observes obs={his chosen rule in the previous period, }
```

