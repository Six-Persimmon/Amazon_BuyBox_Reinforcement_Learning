"""
Atomic-time environment: no lookup table, no state collapse.

Precomputes two small arrays so pure-Python period-by-period stepping is fast:
  - PROFIT tensor over all grid-index combinations: profit_table[idx_1, ...,
    idx_N, seller] (N=3, m=10 -> 1000 combos).
  - Rule-transition table next_idx_table[action_local, others_min_idx]: every
    pricing rule depends only on the competitors' minimum index, so one
    period's price update is a table lookup.

Grid construction (incl. the cost floor) and the Nash/monopoly benchmarks are
identical to src.environment.PricingEnvironment.
"""

import itertools

import numpy as np

from src.environment import (
    compute_nash_and_monopoly_static,
    get_demand_and_profit_static,
)
from src.strategies import get_strategy_function
from src_atomic.config import AtomicConfig


class AtomicEnvironment:
    def __init__(self, config: AtomicConfig):
        self.cfg = config

        # 1. Benchmarks (same solver as baseline)
        self.p_nash, self.p_monopoly = compute_nash_and_monopoly_static(
            config.num_sellers, config.a_val, config.mu, config.a0, config.c_val
        )

        # 2. Grid (identical construction to src.environment._build_grid)
        step = (self.p_monopoly - self.p_nash) / (config.num_grids - 3)
        price_grid = np.linspace(
            self.p_nash - step, self.p_monopoly + step, config.num_grids
        )
        self.grid_min_before_floor = float(price_grid[0])
        self.grid_floor_applied = bool(price_grid[0] < config.c_val)
        if self.grid_floor_applied:
            price_grid[0] = config.c_val
        self.grid_min_after_floor = float(price_grid[0])
        self.nash_idx = 1
        self.monopoly_idx = config.num_grids - 2
        self.price_grid = price_grid

        # 3. Profit tensor over all grid-index combinations
        n, m = config.num_sellers, config.num_grids
        self.profit_table = np.zeros((m,) * n + (n,))
        for combo in itertools.product(range(m), repeat=n):
            prices = [price_grid[i] for i in combo]
            self.profit_table[combo] = get_demand_and_profit_static(
                prices, config.a_val, config.mu, config.a0, config.c_val
            )

        # 4. Rule-transition table: next own index given others' min index.
        #    All rules ignore the seller's own current index (my_idx unused).
        self.next_idx_table = np.zeros((config.num_actions, m), dtype=np.int64)
        for a_local, strat_id in enumerate(config.active_strategies):
            func = get_strategy_function(strat_id)
            for om in range(m):
                self.next_idx_table[a_local, om] = func(0, om, m)

        # 5. Static benchmark profits for the delta normalization
        prices_nash = [self.p_nash] * n
        self.pi_nash = get_demand_and_profit_static(
            prices_nash, config.a_val, config.mu, config.a0, config.c_val
        )[0]
        prices_mon = [self.p_monopoly] * n
        self.pi_mon = get_demand_and_profit_static(
            prices_mon, config.a_val, config.mu, config.a0, config.c_val
        )[0]
        denom = self.pi_mon - self.pi_nash
        self.delta_denominator = denom if denom > 1e-9 else 1.0

    # -----------------------------------------------------------------
    # Atomic-time primitives
    # -----------------------------------------------------------------

    @staticmethod
    def _others_min(idx_vec):
        """others_min[i] = min over j != i of idx_vec[j] (synchronous move)."""
        m1 = min(idx_vec)
        cnt = idx_vec.count(m1) if isinstance(idx_vec, list) else int(np.sum(np.asarray(idx_vec) == m1))
        if cnt > 1:
            return [m1] * len(idx_vec)
        m2 = min(v for v in idx_vec if v != m1)
        return [m2 if v == m1 else m1 for v in idx_vec]

    def step_one_period(self, idx_vec, actions_local):
        """
        One synchronous atomic period: every seller reprices via its current
        rule as a function of its competitors' minimum index.

        Args:
            idx_vec: list of current price indices (len N)
            actions_local: list of local action indices (len N)
        Returns:
            list of next price indices (len N)
        """
        om = self._others_min(idx_vec)
        table = self.next_idx_table
        return [int(table[actions_local[i], om[i]]) for i in range(len(idx_vec))]

    def profits(self, idx_vec):
        """Per-seller profits at the given price-index vector (np array, len N)."""
        return self.profit_table[tuple(idx_vec)]

    def prices(self, idx_vec):
        return [float(self.price_grid[i]) for i in idx_vec]

    def run_block(self, idx_vec, actions_local, K):
        """
        Simulate K synchronous periods under a fixed rule profile, starting
        from an arbitrary price vector (the carry-over generalization of the
        lookup-table block).

        Reward accounting matches src.environment._compute_lookup_table:
        profits are summed over the K post-move states (the initial state's
        profit is NOT included), and the returned next vector is the state
        after K moves.

        Returns:
            avg_rewards (np array len N), final_idx_vec (list)
        """
        cur = list(idx_vec)
        cum = np.zeros(self.cfg.num_sellers)
        for _ in range(K):
            cur = self.step_one_period(cur, actions_local)
            cum += self.profit_table[tuple(cur)]
        return cum / K, cur

    def run_block_detailed(self, idx_vec, actions_local, K):
        """
        Evaluation unroll of one K-block, matching
        src.environment._simulate_k_steps_detailed: records the price/profit
        of the CURRENT state at each of the K steps (so the initial state is
        included and the final post-move state is not), then returns the
        post-move final vector for carry-over.

        Returns:
            trajectory dict {"prices": [(N,)]*K, "profits": [(N,)]*K},
            final_idx_vec (list, state after K moves)
        """
        cur = list(idx_vec)
        trajectory = {"prices": [], "profits": []}
        for _ in range(K):
            trajectory["prices"].append(self.prices(cur))
            trajectory["profits"].append(np.array(self.profit_table[tuple(cur)]))
            cur = self.step_one_period(cur, actions_local)
        return trajectory, cur
