"""
Environment for the exp09 Calvano ladder.

Thin layer over src_atomic.AtomicEnvironment: the price grid (incl. the cost
floor), the profit tensor, the rule-transition table and the Nash/monopoly
benchmarks are reused unchanged, so Delta is normalized identically to every
other experiment in the project. What this module adds is

  1. state encoding for the two state_modes, and
  2. a block transition that dispatches on action_mode.

No lookup table and no cache files: every quantity exp09 needs is derived from
the profit tensor and the one-step rule map at startup.
"""

import numpy as np

from src_atomic.config import AtomicConfig
from src_atomic.environment import AtomicEnvironment
from src_calvano.config import CalvanoConfig


class CalvanoEnvironment:
    def __init__(self, config: CalvanoConfig):
        self.cfg = config

        # AtomicEnvironment builds next_idx_table from active_strategies, so
        # hand it the rule list regardless of action_mode (in price mode the
        # table is simply unused).
        atomic_cfg = AtomicConfig(
            num_sellers=config.num_sellers,
            num_grids=config.num_grids,
            active_strategies=list(config.active_strategies),
            a_val=config.a_val,
            c_val=config.c_val,
            a0=config.a0,
            mu=config.mu,
            xi=config.xi,
            alpha=config.alpha,
            gamma=config.gamma,
            K=config.K,
        )
        self.atomic = AtomicEnvironment(atomic_cfg)

        # Re-export the pieces the simulation and analysis need.
        self.price_grid = self.atomic.price_grid
        self.profit_table = self.atomic.profit_table
        self.next_idx_table = self.atomic.next_idx_table
        self.p_nash = self.atomic.p_nash
        self.p_monopoly = self.atomic.p_monopoly
        self.pi_nash = self.atomic.pi_nash
        self.pi_mon = self.atomic.pi_mon
        self.delta_denominator = self.atomic.delta_denominator
        self.grid_floor_applied = self.atomic.grid_floor_applied
        self.nash_idx = self.atomic.nash_idx
        self.monopoly_idx = self.atomic.monopoly_idx

        # Positional weights for the full-vector state encoding.
        m, n = config.num_grids, config.num_sellers
        self._pow = np.array([m ** (n - 1 - i) for i in range(n)], dtype=np.int64)

    # -----------------------------------------------------------------
    # State encoding
    # -----------------------------------------------------------------

    def encode(self, idx_vec) -> int:
        """
        Map a price-index vector to the Q-table row.

        full_vector: the ordered N-vector, s = sum_i idx_i * m^(N-1-i)
                     (Calvano's |S| = m^(nk) with k = 1).
        market_min:  the minimum over ALL N sellers, own price included --
                     the same observation used by src/, exp05 and exp07.
        """
        if self.cfg.state_mode == "full_vector":
            return int(np.dot(np.asarray(idx_vec, dtype=np.int64), self._pow))
        return int(min(idx_vec))

    def decode(self, state: int):
        """Inverse of encode() for full_vector; for market_min returns the
        collapsed representative vector (o, ..., o)."""
        m, n = self.cfg.num_grids, self.cfg.num_sellers
        if self.cfg.state_mode == "full_vector":
            out = [0] * n
            for i in range(n - 1, -1, -1):
                out[i] = state % m
                state //= m
            return out
        return [int(state)] * n

    # -----------------------------------------------------------------
    # Dynamics
    # -----------------------------------------------------------------

    def prices(self, idx_vec):
        return [float(self.price_grid[i]) for i in idx_vec]

    def step_one_period(self, idx_vec, actions_local):
        """One atomic period under the current actions."""
        if self.cfg.action_mode == "price":
            # The action IS the next price index.
            return [int(a) for a in actions_local]
        return self.atomic.step_one_period(idx_vec, actions_local)

    def run_block(self, idx_vec, actions_local, K):
        """
        Simulate K periods under a fixed action profile.

        Reward accounting matches src/ and src_atomic/: profits are summed
        over the K POST-move states (the initial state's profit is excluded),
        and the returned vector is the state after K moves.

        Returns: (avg_rewards (N,), final_idx_vec (list))
        """
        if self.cfg.action_mode == "price":
            # Prices are pinned at the chosen actions for the whole block, so
            # all K post-move states coincide.
            nxt = [int(a) for a in actions_local]
            return np.array(self.profit_table[tuple(nxt)]), nxt
        return self.atomic.run_block(idx_vec, actions_local, K)

    def run_block_detailed(self, idx_vec, actions_local, K):
        """
        Evaluation unroll of one block. Records the CURRENT (pre-move) state
        at each of the K steps and returns the post-move final vector, exactly
        as src.environment._simulate_k_steps_detailed and
        src_atomic.run_block_detailed do.

        Consequence (identical in every cell of the ladder, so differencing
        Delta across cells stays clean): at eval row t the recorded prices are
        the vector the agents *observed*, while `a_i` is the action they chose
        there, which determines the prices recorded at row t+1.

        Returns: (trajectory {"prices": [...], "profits": [...]}, final_vec)
        """
        if self.cfg.action_mode == "price":
            trajectory = {"prices": [], "profits": []}
            cur = list(idx_vec)
            for _ in range(K):
                trajectory["prices"].append(self.prices(cur))
                trajectory["profits"].append(np.array(self.profit_table[tuple(cur)]))
                cur = [int(a) for a in actions_local]
            return trajectory, cur
        return self.atomic.run_block_detailed(idx_vec, actions_local, K)

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    def reachable_vectors(self):
        """
        Price vectors reachable after at least one rule step, i.e. the image
        of the one-step map over all start vectors and action profiles. Used
        by the "reachable_avg" Q-init variant and reported as a diagnostic
        (Extention_Plan.md 4.3: 580 of 1000 under the 3-rule set).

        Only meaningful in rule mode; in price mode every vector is reachable.
        """
        import itertools

        m, n = self.cfg.num_grids, self.cfg.num_sellers
        if self.cfg.action_mode == "price":
            return set(itertools.product(range(m), repeat=n))

        image = set()
        for vec in itertools.product(range(m), repeat=n):
            for profile in itertools.product(range(self.cfg.num_actions), repeat=n):
                image.add(tuple(self.atomic.step_one_period(list(vec), list(profile))))
        return image
