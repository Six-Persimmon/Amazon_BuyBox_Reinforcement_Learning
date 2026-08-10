"""
Agents for the atomic-time engine.

The Q-learning agent is identical to the baseline (including the optional
exploration_weights added for exp06), so we re-export it rather than fork it.
The heuristic Q-initialization reuses the baseline lookup table for the
matching (N, mu, strategy set, block_K) configuration — see
Extention_Plan.md §0.2: keeping the init identical to baseline makes outcome
differences attributable to the dynamics, not the prior.
"""

from src.agent import QAgent, calculate_heuristic_init_values  # noqa: F401
from src.config import Config
from src.environment import PricingEnvironment

from src_atomic.config import AtomicConfig


def heuristic_init_from_baseline_lookup(config: AtomicConfig):
    """
    Compute the baseline heuristic init Q-table by loading (or creating) the
    baseline lookup table for the matching configuration.

    Uses K = config.block_K (K for sync mode, lambda_K for async mode).
    """
    base_cfg = Config(
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
        K=config.block_K,
    )
    base_env = PricingEnvironment(base_cfg)
    return calculate_heuristic_init_values(base_env, base_cfg)
