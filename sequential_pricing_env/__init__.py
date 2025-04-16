"""
Top‑level package for the sequential‑pricing project.

Importing this module automatically registers the environment with
`gymnasium`, so you can call  gym.make("SequentialPricing-v0").
"""

from gymnasium.envs.registration import register

# Re‑export the actual Env class at package level (optional convenience)
from .envs.sequential_pricing_env import SequentialPricingEnv

__all__ = ["SequentialPricingEnv"]

# -----------------------------------------------------------------------------
# Register the env with gymnasium the first time the package is imported
# -----------------------------------------------------------------------------
register(
    id="SequentialPricing-v0",
    entry_point="sequential_pricing_env.envs:SequentialPricingEnv",
    max_episode_steps=None,          # infinite‑horizon
)