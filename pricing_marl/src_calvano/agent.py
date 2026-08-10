"""
Agent and Q-initialization for the exp09 Calvano ladder.

The learning agent is src.agent.QAgent (epsilon = exp(-beta t), the standard
Q-update, optional exploration weights), subclassed only to (a) size the
Q-table by `num_states` rather than `num_grids` and (b) offer Calvano's
lowest-price tie-break as an option.

Q-INITIALIZATION -- one formula for every rung of the ladder, which is
simultaneously Calvano et al. (2020) Eq. (8) and this project's Eq. (6):

    Q0_i(s, a) = (1 / (1 - gamma)) * E_{a_-i ~ Uniform(A^(N-1))} [ R_i(s, a, a_-i) ]

where R_i is exactly the reward that decision earns -- the one-period profit
when K = 1, the K-block average profit when K > 1. Instantiated per cell:

  price actions   R_i = pi_i(a_i, a_-i), independent of s, so Q0 is constant
                  across states: literally Calvano Eq. (8).
  rule actions +
  full_vector     R_i = block reward starting from the decoded price vector.
  rule actions +
  market_min      the reward depends on the full vector, which the observation
                  does not pin down; `init_state_rep="collapsed"` evaluates the
                  representative vector (o, ..., o), reproducing
                  src.agent.calculate_heuristic_init_values exactly (verified
                  in tests/test_exp09_engine.py).
"""

import itertools

import numpy as np

from src.agent import QAgent
from src_calvano.config import CalvanoConfig


class CalvanoQAgent(QAgent):
    """QAgent sized by num_states, with an optional Calvano tie-break."""

    def __init__(self, agent_id, config: CalvanoConfig, initial_Q_table=None):
        if initial_Q_table is None:
            initial_Q_table = np.zeros((config.num_states, config.num_actions))
        super().__init__(agent_id, config, initial_Q_table=initial_Q_table)
        self.tie_break = getattr(config, "tie_break", "random")

    def choose_action(self, observation, t_step):
        if self.tie_break == "random":
            return super().choose_action(observation, t_step)

        # Calvano's rule: on ties pick the lowest action index, which in price
        # mode is the lowest price.
        if self.forced_action is not None:
            return self.forced_action
        epsilon = np.exp(-self.cfg.beta * t_step)
        if np.random.rand() < epsilon:
            if self.explore_probs is not None:
                return int(np.random.choice(self.cfg.num_actions, p=self.explore_probs))
            return np.random.randint(self.cfg.num_actions)
        return int(np.argmax(self.Q[observation, :]))


# =====================================================================
# Heuristic Q-initialization
# =====================================================================

def _init_price_mode(config: CalvanoConfig, env) -> np.ndarray:
    """
    Calvano Eq. (8) verbatim: the discounted payoff to seller i from playing
    each price while opponents randomize uniformly over the grid. The stage
    payoff does not depend on the state, and by symmetry of the demand system
    it does not depend on the seller's position either, so the same row is
    broadcast across every state and every seller.
    """
    m = config.num_grids
    # profit_table[i_1, ..., i_N, seller]; axis 0 is seller 0's own index.
    own_profits = env.profit_table[..., 0]              # shape (m,) * N
    per_action = own_profits.reshape(m, -1).mean(axis=1)  # mean over rival combos
    row = per_action / (1.0 - config.gamma)
    shared = np.tile(row, (config.num_states, 1))
    return np.broadcast_to(shared, (config.num_sellers,) + shared.shape).copy()


def _representative_vectors(config: CalvanoConfig, env, state: int):
    """Price vectors used to evaluate the reward at a given observation."""
    if config.state_mode == "full_vector":
        return [env.decode(state)]
    if config.init_state_rep == "collapsed":
        return [[state] * config.num_sellers]
    # reachable_avg: every reachable vector whose market minimum is `state`
    reps = [list(v) for v in sorted(env.reachable_vectors()) if min(v) == state]
    return reps or [[state] * config.num_sellers]


def _init_rule_mode(config: CalvanoConfig, env) -> np.ndarray:
    """
    Same formula, with R_i = the K-block reward from the representative price
    vector, and opponents marginalized by enumerating every rule profile in
    A^N.

    The expectation is accumulated SEPARATELY FOR EACH SELLER POSITION. This
    matters under state_mode="full_vector": the rules respond to the lowest
    COMPETITOR price, so at an asymmetric state such as (0, 5, 9) the seller
    holding the minimum faces others_min = 5 while the other two face
    others_min = 0, and their expected rule payoffs genuinely differ. Pooling
    those positions into one row (config.init_pooling="pooled") gives every
    seller a prior that is correct for none of them; with 3 sellers and a
    10-point grid it flips the initial argmax for at least one seller at 58%
    of states. Under state_mode="market_min" the representative vector
    (o,...,o) is symmetric, so the two settings coincide exactly.
    """
    n, n_act = config.num_sellers, config.num_actions
    sums = np.zeros((n, config.num_states, n_act))
    counts = np.zeros((n, config.num_states, n_act))
    profiles = list(itertools.product(range(n_act), repeat=n))

    for state in range(config.num_states):
        for vec in _representative_vectors(config, env, state):
            for profile in profiles:
                rewards, _ = env.run_block(vec, list(profile), config.K)
                for i in range(n):
                    sums[i, state, profile[i]] += rewards[i]
                    counts[i, state, profile[i]] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        if config.init_pooling == "pooled":
            pooled = np.nan_to_num(sums.sum(axis=0) / counts.sum(axis=0))
            avg = np.broadcast_to(pooled, (n,) + pooled.shape).copy()
        else:
            avg = np.nan_to_num(sums / counts)
    return avg / (1.0 - config.gamma)


def heuristic_init(config: CalvanoConfig, env) -> np.ndarray:
    """
    Initial Q-tables, shape (num_sellers, num_states, num_actions).

    Seller i is initialized with slice i. The slices are identical except
    under state_mode="full_vector" with rule actions and
    init_pooling="per_seller"; see _init_rule_mode.
    """
    if config.action_mode == "price":
        return _init_price_mode(config, env)
    return _init_rule_mode(config, env)
