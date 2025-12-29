# MARL Pricing Simulation with Rule-Based Strategies

## 1. Overview
This script implements a Multi-Agent Reinforcement Learning (MARL) framework to simulate price competition among $N$ sellers in an oligopoly setting. Unlike traditional continuous control models, this simulation restricts agents to a discrete set of **rule-based pricing strategies** (e.g., undercutting, matching, or reseting prices).

The objective is to study the emergence of collusive behavior (e.g., Edgeworth Cycles) and equilibrium prices under specific market parameters. The agents utilize **Independent Q-Learning** to learn the optimal policy over a discretized price grid, where their observation is limited to the lowest price currently in the market.

## 2. MARL Configuration & Mathematical Model

### A. Demand and Profit System
The market is modeled using a Logit demand system. For a seller $i$ setting price $p_i$, the demand $D_i$ is given by:

$$
D_i(p) = \frac{\exp(\frac{a - p_i}{\mu})}{\sum_{j=1}^{N} \exp(\frac{a - p_j}{\mu}) + \exp(\frac{a_0}{\mu})}
$$

Where:
* $a$: Quality/intercept parameter.
* $\mu$: Horizontal differentiation (smoothing parameter).
* $a_0$: Outside option utility.

**Profit:** $\pi_i = (p_i - c) \cdot D_i$, where $c$ is the marginal cost.

### B. Price Grid Construction ($\xi$)
Prices are discretized into a grid of size $M$ (`num_grids`). The grid boundaries are determined by the stage-game Nash equilibrium ($p^{Nash}$) and the Monopoly price ($p^{Mon}$), extended by a parameter $\xi$:

$$
p_{grid} \in [p^{Nash} - \xi(p^{Mon} - p^{Nash}), \quad p^{Mon} + \xi(p^{Mon} - p^{Nash})]
$$

* $\xi > 0$ allows the grid to extend slightly beyond the theoretical benchmarks, facilitating study of boundary behaviors (e.g., Calvano et al., 2025).

### C. State Space, Observations, and Actions

1.  **True State ($S$):** The global state is the tuple of price indices for all sellers:
    $$S_t = (idx_1, idx_2, \dots, idx_N)$$
2.  **Observation ($O_i$):** Agents operate under partial observability (state reduction). An agent $i$ only observes the **lowest price index** in the market:
    $$O_{i,t} = \min(S_t)$$
3.  **Action Space ($A$):** Agents choose a high-level pricing rule rather than a raw price. The next price index $p'_{i}$ is a function of the agent's current price and the minimum competitor price ($p_{-i}^{min}$):
    * **Undercut:** $p'_{i} = \max(0, p_{-i}^{min} - 1)$
    * **Match:** $p'_{i} = p_{-i}^{min}$
    * **Price Above:** $p'_{i} = \min(M-1, p_{-i}^{min} + 1)$
    * **Undercut + Reset:** Undercut, but if $p_{-i}^{min} = 0$, jump to $p^{Mon}$ (index $M-1$).
    * **Match + Reset:** Match, but if $p_{-i}^{min} = 0$, jump to $p^{Mon}$.

### D. Q-Learning Algorithm
Agents maximize the discounted future profit using Q-learning. The update rule for seller $i$ is:

$$
Q_i(o, a) \leftarrow (1-\alpha)Q_i(o, a) + \alpha \left[ r_i + \gamma \max_{a'} Q_i(o', a') \right]
$$

* $\alpha$: Learning rate (0.15).
* $\gamma$: Discount factor (0.95).
* $\epsilon$-Greedy Policy: Exploration decays according to $\epsilon_t = e^{-\beta t}$.

## 3. Code Walkthrough

The script `1d_state.py` is structured as follows:

### 1. System Configuration
Sets global parameters including the number of sellers ($N$), simulation length ($T$), grid size, and economic parameters ($\mu, \alpha, \gamma$). It also defines $\beta$ for epsilon decay.

### 2. Profit & Grid Functions
* **`get_demand_and_profit`**: Implements the Logit demand model. Includes the "Max-Subtraction Trick" to prevent numerical overflow/underflow when calculating exponentials.
* **Grid Construction**: Solves for $p^{Nash}$ and $p^{Mon}$ using `scipy.optimize`, then generates the linear price grid based on $\xi$.

### 3. Flexible Rule Definitions
Defines the transition logic for the pricing strategies (e.g., `rule_undercut`, `rule_match_with_reset`). These functions determine the next price index based on the current market state.

### 4. N-Agent Lookup Table
To speed up the simulation, the script pre-calculates the environment dynamics.
* **Canonicalization**: Exploits player symmetry. The state-action pair `(State=[0,2], Actions=[Match, Undercut])` is mathematically equivalent to `(State=[2,0], Actions=[Undercut, Match])`.
* **Simulation**: For every unique canonical scenario, the script simulates $K$ periods to calculate the average immediate reward and the deterministic next state transition.

### 5. Initialization (Value Iteration)
Instead of starting Q-tables at zero, the script performs Value Iteration on the reduced state space (lowest price index).
* Iteratively updates a value function $V(s)$ until convergence.
* Populates the initial $Q(s,a)$ tables based on these converged values to accelerate learning during the main loop.

### 6. Main Q-Learning Loop
The core training loop runs for $T$ periods:
1.  **Action Selection**: Agents choose strategies using $\epsilon$-greedy logic based on their observation (market minimum price).
2.  **Environment Step**: Retrieves payoffs and next states from the pre-computed **Lookup Table**.
3.  **Q-Update**: Updates Q-values based on the observed reward and the max Q-value of the next state.
4.  **Logging**: Periodically prints the current state and strategies.

### 7. Final Learned Policy
After training, the script outputs the deterministic policy (best action) for each agent at every possible observation state (lowest price grid index). This allows for analysis of the equilibrium behavior (e.g., whether agents learned to coordinate on a "reset" strategy).
