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

from collections import deque

import numpy as np
from typing import Iterable, List, Optional, Tuple

from agents.repricer_meta_actions import MetaAction, create_action_executor


class NPlayerPoissonDemandEnv:
    """Discrete-time price competition environment with mixed demand."""

    def __init__(
        self,
        n_players: int,
        price_min: float = 0.5,
        price_max: float = 10.0,
        grid_size: int = 20,
        a0: float = 0.0,
        a12: float = 10.0, # 2025-12-03. This is too high and can envourage very high prices, so I built a new version.
        mu: float = 0.25,
        lam: float = 1.0,
        rho: float = 0.5, # proportion of sophisticated buyers
        repricer_cost: float = 0.0, # cost of using algorithmic repricer
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
        self.lam = lam
        self.rho = rho # proportion of sophisticated buyers
        self.repricer_cost = repricer_cost

    def snap_price_to_grid(self, price) -> float:
        """Round an arbitrary price (scalar or array) to the closest grid value."""

        arr = np.asarray(price, dtype=float)
        # Broadcast against grid: grid is (grid_size,), arr can be any shape
        diffs = np.abs(self.prices.reshape(-1, 1) - arr.reshape(1, -1))
        nearest_idx = np.argmin(diffs, axis=0)
        snapped = self.prices[nearest_idx]
        return float(snapped) if snapped.shape == () else snapped.reshape(arr.shape)

    def _sample_poisson(self, lam: float, size: int) -> np.ndarray:
        """Draw Poisson-distributed arrivals for each period."""
        lam = max(lam, 0.0)
        return np.random.poisson(lam, size=size)

    def _logit_shares(self, price_vector: np.ndarray) -> np.ndarray:
        """Return logit purchase probabilities for all players."""
        logits = np.exp((self.a12 - price_vector) / self.mu)
        denom = np.exp(self.a0 / self.mu) + np.sum(logits)
        return logits / denom
    
    def monopoly_price_mixed(self, mc: float = 1.0) -> float:
        prices = self.prices  # 同一个 grid
        A = np.exp((self.a12 - prices) / self.mu)
        C = np.exp(self.a0 / self.mu)
        s_logit = A / (A + C)  # monopoly 下只有一个 seller 的 logit 份额
        # 期望销量（忽略 lambda 常数）
        q = self.rho * s_logit + (1.0 - self.rho) * 1.0
        profits = (prices - mc) * q
        return prices[np.argmax(profits)]

    def profit_given_others_price(self, my_price, others_price, mc: float = 1.0):
        prices = np.array([my_price] + [others_price] * (self.n_players - 1))
        # logit share（只给各 seller 的）
        shares_logit = self._logit_shares(prices)
        # naive share allocation
        min_price = prices.min()
        tie_mask = np.isclose(prices, min_price)
        tie_count = tie_mask.sum()
        naive_share = np.zeros_like(prices, dtype=float)
        naive_share[tie_mask] = 1.0 / tie_count

        # 看 firm 0（就是我们）的份额
        s_logit_0 = shares_logit[0]
        s_naive_0 = naive_share[0]

        # 期望销量（忽略 lambda）
        q = self.rho * s_logit_0 + (1.0 - self.rho) * s_naive_0
        return (my_price - mc) * q
    
    def nash_price_mixed_discrete(self, mc: float = 1.0):
        prices = self.prices
        n = len(prices)
        best_response = np.empty(n, dtype=int)

        for j, p_sym in enumerate(prices):
            # 我方在 grid 上找最优
            profits = [
                self.profit_given_others_price(my_p, p_sym, mc=mc)
                for my_p in prices
            ]
            best_idx = int(np.argmax(profits))
            best_response[j] = best_idx

        # 对称 Nash: 对称价 p_sym 的 index j 自己就是 best response
        # 即 best_response[j] == j
        candidates = np.where(best_response == np.arange(n))[0]
        if candidates.size == 0:
            return None  # 没有离散对称NE
        # 如果有多个，挑一个，比如中间的
        return prices[candidates[len(candidates)//2]]

    def run_episode(
        self,
        meta_actions: Iterable[MetaAction],
        marginal_costs: Iterable[float],
        periods: int, # number of inner loop periods
        return_histories: bool = False, # whether to return full traces
        initial_price_indices: Optional[Iterable[int]] = None,
    ) -> dict:
        """Execute a full inner episode and return mean profits and optional traces."""

        meta_actions = list(meta_actions)
        repricer_usage_dummy = np.asarray([not a.is_static for a in meta_actions], dtype=float)
        repricer_flag_list = [a for a in meta_actions]

        if len(meta_actions) != self.n_players:
            raise ValueError("meta_actions length must equal number of players")
        marginal_costs = np.array(list(marginal_costs), dtype=float)
        if marginal_costs.shape[0] != self.n_players:
            raise ValueError("marginal_costs length must equal number of players")

        demand_draws = self._sample_poisson(self.lam, periods) # shape (periods,), draw from Poisson distribution
        if initial_price_indices is not None:
            initial_price_indices = np.array(list(initial_price_indices), dtype=int)
            if initial_price_indices.shape[0] != self.n_players:
                raise ValueError("initial_price_indices length must equal number of players")
        else:
            initial_price_indices = None

        executors = []
        for idx, (action, mc) in enumerate(zip(meta_actions, marginal_costs)):
            current_index = -1
            if initial_price_indices is not None and not action.is_static: # If a seller picks 'no_repricer', he will ignore the initial price index provided and pick his own initial price as planned, which is sampled above marginal cost.
                current_index = int(initial_price_indices[idx])
            executors.append(
                create_action_executor(
                    action,
                    self.prices,
                    mc,
                    current_index=current_index,
                )
            )

        price_indices = np.array([executor.current_index for executor in executors], dtype=int) # current price indices for all players, shape (n_players,)
        prices = self.prices[price_indices] # current prices for all players, shape (n_players,)
        price_history = []
        demand_history = []
        profit_history = []

        for t in range(periods):
            # record current prices
            prices = self.prices[price_indices] # index to price mapping. Shape (n_players,)
            price_history.append(prices.copy())

            shares_logit = self._logit_shares(prices)
            min_price = np.min(prices)
            tie_mask = np.isclose(prices, min_price)
            tie_count = np.count_nonzero(tie_mask)
            naive_share = np.zeros(self.n_players)
            if tie_count > 0: # always true since at least one player has the min price
                naive_share[tie_mask] = 1.0 / tie_count

            q_t = demand_draws[t]
            demand = self.rho * q_t * shares_logit + (1.0 - self.rho) * q_t * naive_share # shape (n_players,) sophisticated + naive buyers
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

            price_indices = next_indices # update the next period price indices

        price_history = np.asarray(price_history) # shape (periods, n_players)
        demand_history = np.asarray(demand_history)
        profit_history = np.asarray(profit_history)

        avg_rewards = profit_history.mean(axis=0)
        # consider repricer cost
        avg_rewards -= self.repricer_cost * repricer_usage_dummy

        avg_prices = price_history.mean(axis=0) # shape (n_players,) This is worth a debate: in the end we use the average price for oneself as one of the states. Is it a good choice?
        avg_lowest_prices = price_history.min(axis=1).mean() # proxy for buy box price

        result = {
            "average_profits": avg_rewards,
            "final_prices_idx": price_indices,
            "final_prices": self.prices[price_indices],
            "average_prices": avg_prices,
            "average_lowest_price": avg_lowest_prices,
            "repricer_usage": repricer_usage_dummy,
            "meta_actions": repricer_flag_list,
        }

        if return_histories:
            result.update(
                {
                    "price_history": price_history, # shape (periods, n_players)
                    "demand_history": demand_history,
                    "profit_history": profit_history,
                }
            )

        return result


__all__ = ["NPlayerPoissonDemandEnv"]

if __name__ == "__main__":
    # show Nash equilibrium price for 5 players and monopoly price
    env = NPlayerPoissonDemandEnv(n_players=5, a12=2.0)
    nash_price = env.nash_price_mixed_discrete(mc=1.0)
    monopoly_price = env.monopoly_price_mixed(mc=1.0)
    print(f"Nash equilibrium price (5 players, mc=1.0): {nash_price:.4f}")
    print(f"Monopoly price (mc=1.0): {monopoly_price:.4f}")
