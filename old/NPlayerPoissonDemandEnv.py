#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Inner environment used by the contextual bandit training pipeline.

The environment executes a repeated Bertrand-style pricing game in which
N sellers choose prices from a discrete grid.  Demand in each period is a
convex combination of logit demand ("sophisticated" buyers) and an
outside option that always purchases from the current lowest-priced
seller ("naive" buyers).  The environment exposes a single helper that
runs a full inner episode given meta-actions supplied by the bandit.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, List, Optional, Tuple

from agents.repricer_meta_actions import MetaAction, create_action_executor


class NPlayerPoissonDemandEnv:
    """Discrete-time price competition environment with mixed demand."""

    def __init__(
        self,
        n_players: int,
        price_min: float = 0.01,
        price_max: float = 10.0,
        grid_size: int = 25,
        a0: float = 0.0,
        a12: float = 10.0,
        mu: float = 0.25,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Build an environment with the supplied price grid and logit params."""
        self.n_players = n_players
        self.price_min = price_min
        self.price_max = price_max
        self.grid_size = grid_size
        self.prices = np.linspace(price_min, price_max, grid_size)
        self.a0 = a0
        self.a12 = a12
        self.mu = mu
        self.rng = rng or np.random.default_rng()

    def _sample_poisson(self, lam: float, size: int) -> np.ndarray:
        """Draw Poisson-distributed arrivals for each period."""
        lam = max(lam, 0.0)
        return self.rng.poisson(lam, size=size)

    def _logit_shares(self, price_vector: np.ndarray) -> np.ndarray:
        """Return logit purchase probabilities for all players."""
        logits = np.exp((self.a12 - price_vector) / self.mu)
        denom = np.exp(self.a0 / self.mu) + np.sum(logits)
        return logits / denom

    def run_episode(
        self,
        meta_actions: Iterable[MetaAction],
        marginal_costs: Iterable[float],
        lam: float,
        rho: float,
        periods: int,
        return_histories: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Execute a full inner episode and return mean profits and optional traces."""

        meta_actions = list(meta_actions)
        if len(meta_actions) != self.n_players:
            raise ValueError("meta_actions length must equal number of players")
        marginal_costs = np.array(list(marginal_costs), dtype=float)
        if marginal_costs.shape[0] != self.n_players:
            raise ValueError("marginal_costs length must equal number of players")

        episode_rng = rng or self.rng
        demand_draws = self._sample_poisson(lam, periods)

        executors = [
            create_action_executor(action, self.prices, mc, episode_rng)
            for action, mc in zip(meta_actions, marginal_costs)
        ]

        price_indices = np.array([executor.current_index for executor in executors], dtype=int)
        prices = self.prices[price_indices]

        price_history = []
        demand_history = []
        profit_history = []

        for t in range(periods):
            prices = self.prices[price_indices]
            price_history.append(prices.copy())

            shares_logit = self._logit_shares(prices)
            min_price = np.min(prices)
            tie_mask = np.isclose(prices, min_price)
            tie_count = np.count_nonzero(tie_mask)
            naive_share = np.zeros(self.n_players)
            if tie_count > 0:
                naive_share[tie_mask] = 1.0 / tie_count

            q_t = demand_draws[t]
            demand = rho * q_t * shares_logit + (1.0 - rho) * q_t * naive_share
            demand_history.append(demand.copy())

            profits = (prices - marginal_costs) * demand
            profit_history.append(profits.copy())

            next_indices = np.empty_like(price_indices)
            for idx, executor in enumerate(executors):
                competitors = np.delete(price_indices, idx)
                if competitors.size == 0:
                    competitor_min_idx = price_indices[idx]
                else:
                    competitor_min_idx = int(np.min(competitors))
                next_idx = executor.next_price_index(
                    competitor_min_idx=competitor_min_idx,
                )
                next_indices[idx] = next_idx

            price_indices = next_indices

        price_history = np.asarray(price_history)
        demand_history = np.asarray(demand_history)
        profit_history = np.asarray(profit_history)

        avg_rewards = profit_history.mean(axis=0)

        result = {
            "average_profits": avg_rewards,
            "final_prices": self.prices[price_indices],
        }

        if return_histories:
            result.update(
                {
                    "price_history": price_history,
                    "demand_history": demand_history,
                    "profit_history": profit_history,
                }
            )

        return result


__all__ = ["NPlayerPoissonDemandEnv"]
