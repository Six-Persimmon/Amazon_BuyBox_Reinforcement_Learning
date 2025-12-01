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
        a12: float = 10.0,
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

    def _sample_poisson(self, lam: float, size: int) -> np.ndarray:
        """Draw Poisson-distributed arrivals for each period."""
        lam = max(lam, 0.0)
        return np.random.poisson(lam, size=size)

    def _logit_shares(self, price_vector: np.ndarray) -> np.ndarray:
        """Return logit purchase probabilities for all players."""
        logits = np.exp((self.a12 - price_vector) / self.mu)
        denom = np.exp(self.a0 / self.mu) + np.sum(logits)
        return logits / denom
    
    def _get_monopoly_price(self) -> float:
        """Compute the monopoly price as if there were only one seller facing only sophisticated buyers."""
        from scipy.optimize import minimize_scalar
        
        def monopoly_profit(p):
            d = np.exp((self.a12 - p) / self.mu) / (np.exp((self.a12 - p) / self.mu) + np.exp(self.a0 / self.mu))
            return (p - self.repricer_cost) * d

        
        result = minimize_scalar(lambda p: -monopoly_profit(p), 
                               bounds=(self.price_min, self.price_max), 
                               method='bounded')
        return result.x

    def _get_nash_equilibrium_price(self, mc: float=2.0) -> float:
        """Calculate symmetric Nash equilibrium price for N players with same marginal cost facing only sophisticated buyers."""
        from scipy.optimize import fsolve
        
        def nash_condition(p):
            # For symmetric equilibrium, all players charge same price p
            # First-order condition: derivative of profit w.r.t. own price = 0
            
            # Demand when all charge price p
            deno = self.n_players * np.exp((self.a12 - p) / self.mu) + np.exp(self.a0 / self.mu)
            d = np.exp((self.a12 - p) / self.mu) / deno

            # First-order condition:
            foc = d - (p - mc) * np.exp(self.a0 / self.mu)
            return foc
        
        # Solve for Nash equilibrium price
        p_nash = fsolve(nash_condition, mc + 0.2)[0]
        return max(self.price_min, min(self.price_max, p_nash))

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
