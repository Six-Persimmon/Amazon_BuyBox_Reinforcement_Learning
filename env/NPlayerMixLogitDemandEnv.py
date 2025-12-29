#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
N-player pricing environment with canonical logit demand (Calvano et al. 2020; Wang et al. 2022).

Key features vs. the original NPlayerPoissonDemandEnv:
- Demand: canonical logit demand with a unit mass of consumers each period
  (no Poisson arrivals, no naive \rho component).
- Parameters: logit parameters match Wang et al. (2022) baseline by default:
    a0 = 0.0, a12 = 2.0, mu = 0.25, base_mc = 1.0
- Price grid ("Wang-style"):
    1. Use a fine continuous grid [mc, mc + 4] with 200 points.
    2. On this fine grid, compute:
       - symmetric static Bertrand-Nash price of the stage game;
       - symmetric monopoly price (joint profit maximization).
    3. Construct the actual discrete action grid as K = grid_size equally spaced
       points between p_Nash and p_Monopoly, inclusive.
- The public interface (class name, run_episode signature and outputs, and
  snap_price_to_grid) is kept compatible with the previous version so that
  downstream code continues to work.

Demand in each period t:
    q_i(p_t) = exp((a12 - p_{i,t}) / mu) /
               ( sum_j exp((a12 - p_{j,t}) / mu) + exp(a0 / mu) )

Profit:
    π_i,t = (p_{i,t} - c_i) * q_i(p_t)

We normalize the "number of consumers" to 1, exactly as in Wang et al. (2022).
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, List, Optional, Tuple

# from agents.repricer_meta_actions import MetaAction, create_action_executor


class NPlayerLogitDemandEnv:
    """Discrete-time price competition environment with canonical logit demand.

    NOTE: Name kept for backward compatibility. This version no longer uses a
    Poisson arrival process or a rho-mixing with naive buyers. Demand is purely
    canonical logit with a unit mass of consumers each period.
    """

    def __init__(
        self,
        n_players: int,
        grid_size: int = 10,
        a0: float = 0.0,
        a12: float = 2.0,
        mu: float = 0.25,
        repricer_cost: float = 0.0,
        # New args controlling the "Wang-style" grid generation:
        base_mc: float = 1.0,
        fine_grid_points: int = 200,
        fine_grid_span: float = 4.0,
    ) -> None:
        """
        Build an environment with canonical logit demand and Wang-style price grid.

        Parameters
        ----------
        n_players : int
            Number of sellers (products) in the market.
        grid_size : int
            Number of discrete price points in the action grid (K in Wang et al.).
        a0 : float
            Outside option utility parameter.
        a12 : float
            Common product quality index a_i for all i (vertical differentiation).
        mu : float
            Horizontal differentiation parameter.
        repricer_cost : float
            Per-period cost for using a repricer (charged in avg reward).
        base_mc : float
            Symmetric marginal cost used to compute the fine-grid Nash and monopoly
            prices and hence the action grid.
        fine_grid_points : int
            Number of points in the temporary fine grid [mc, mc + fine_grid_span]
            used to compute static Nash and monopoly prices.
        fine_grid_span : float
            Length of the temporary fine grid above mc, default [mc, mc + 4].
        """
        self.n_players = n_players

        # Store logit parameters (matching Wang et al. baseline by default).
        self.a0 = a0
        self.a12 = a12
        self.mu = mu

        # Repricer cost:
        self.repricer_cost = repricer_cost

        # Grid construction parameters:
        self.grid_size = grid_size
        self.base_mc = base_mc
        self.fine_grid_points = fine_grid_points
        self.fine_grid_span = fine_grid_span

        # Step 1: build a fine grid [mc, mc + span] for the static stage game
        fine_low = base_mc
        fine_high = base_mc + fine_grid_span
        self._fine_prices = np.linspace(fine_low, fine_high, fine_grid_points)

        # Step 2: compute symmetric Nash and monopoly prices on the fine grid
        self.p_nash = self._compute_symmetric_nash_price_discrete(mc=base_mc)
        self.p_monopoly = self._compute_monopoly_price_discrete(mc=base_mc)

        # Step 3: construct the actual price grid as in Wang et al.:
        #         K equally spaced points between p_Nash and p_Monopoly.
        if self.p_nash is None or self.p_monopoly is None:
            raise RuntimeError("Failed to compute Nash/monopoly prices on fine grid.")

        self.prices = np.linspace(self.p_nash, self.p_monopoly, grid_size)

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def snap_price_to_grid(self, price) -> float:
        """Round an arbitrary price (scalar or array) to the closest grid value."""

        arr = np.asarray(price, dtype=float)
        diffs = np.abs(self.prices.reshape(-1, 1) - arr.reshape(1, -1))
        nearest_idx = np.argmin(diffs, axis=0)
        snapped = self.prices[nearest_idx]
        return float(snapped) if snapped.shape == () else snapped.reshape(arr.shape)

    def _logit_shares(self, price_vector: np.ndarray) -> np.ndarray:
        """Return canonical logit purchase probabilities for all players.

        qi(p) = exp((a12 - p_i) / mu) /
                (sum_j exp((a12 - p_j) / mu) + exp(a0 / mu))
        """
        logits = np.exp((self.a12 - price_vector) / self.mu)
        denom = np.exp(self.a0 / self.mu) + np.sum(logits)
        return logits / denom

    # ------------------------------------------------------------------
    # Static stage-game analysis on the fine grid
    # ------------------------------------------------------------------
    def _profit_given_others_price_logit(
        self,
        my_price: float,
        others_price: float,
        mc: float,
    ) -> float:
        """Static (one-shot) profit for firm 0 under canonical logit demand.

        - There are n_players firms.
        - Firm 0 charges my_price, all others charge others_price (symmetric opponents).
        """
        prices = np.array([my_price] + [others_price] * (self.n_players - 1))
        shares = self._logit_shares(prices)
        q_i = shares[0]  # demand share for firm 0, total market size normalized to 1
        return (my_price - mc) * q_i

    def _compute_symmetric_nash_price_discrete(self, mc: float) -> Optional[float]:
        """Approximate symmetric static Bertrand-Nash price on the fine grid.

        We search over the temporary fine grid (self._fine_prices). For each
        candidate symmetric price p_sym, we compute the best response of firm 0
        when all others charge p_sym. A discrete symmetric Nash price is any
        p_sym for which the best response equals p_sym itself (on the grid).
        """
        prices = self._fine_prices
        n = len(prices)
        best_response_indices = np.empty(n, dtype=int)

        for j, p_sym in enumerate(prices):
            # firm 0's best response given all others at price p_sym
            profits = [
                self._profit_given_others_price_logit(my_p, p_sym, mc=mc)
                for my_p in prices
            ]
            best_idx = int(np.argmax(profits))
            best_response_indices[j] = best_idx

        # Symmetric NE candidates: indices where best_response_indices[j] == j
        candidates = np.where(best_response_indices == np.arange(n))[0]
        if candidates.size > 0:
            # If multiple, choose the "middle" one
            idx = candidates[candidates.size // 2]
            return float(prices[idx])

        # Fallback: choose the price whose best response is closest on the grid
        deviations = np.abs(best_response_indices - np.arange(n))
        idx = int(np.argmin(deviations))
        return float(prices[idx])

    def _compute_monopoly_price_discrete(self, mc: float) -> Optional[float]:
        """Approximate symmetric monopoly price on the fine grid.

        Monopoly: a single decision-maker controls all n_players products and
        sets a common price p to maximize total profit:
            Π(p) = sum_i (p - mc) * q_i(p)
                 = n_players * (p - mc) * q_i(p)   (by symmetry).
        """
        prices = self._fine_prices
        best_profit = -np.inf
        best_price = None

        for p in prices:
            price_vec = np.full(self.n_players, p)
            shares = self._logit_shares(price_vec)
            q_i = shares[0]
            total_profit = self.n_players * (p - mc) * q_i
            if total_profit > best_profit:
                best_profit = total_profit
                best_price = p

        return float(best_price) if best_price is not None else None

    def nash_price_discrete(self) -> float:
        """Return the symmetric Nash price used to construct the grid."""
        return self.p_nash

    def monopoly_price_discrete(self) -> float:
        """Return the symmetric monopoly price used to construct the grid."""
        return self.p_monopoly

    # ------------------------------------------------------------------
    # Dynamic episode (unchanged interface vs. old version)
    # ------------------------------------------------------------------
    def run_episode(
        self,
        meta_actions: Iterable[MetaAction],
        marginal_costs: Iterable[float],
        periods: int,
        return_histories: bool = False,
        initial_price_indices: Optional[Iterable[int]] = None,
    ) -> dict:
        """Execute a full episode and return mean profits and optional traces.

        This keeps the same interface and output keys as the original
        NPlayerPoissonDemandEnv.run_episode, but demand is now purely
        deterministic canonical logit with unit market size each period.
        """

        meta_actions = list(meta_actions)
        repricer_usage_dummy = np.asarray(
            [not a.is_static for a in meta_actions],
            dtype=float,
        )
        repricer_flag_list = [a for a in meta_actions]

        if len(meta_actions) != self.n_players:
            raise ValueError("meta_actions length must equal number of players")

        marginal_costs = np.array(list(marginal_costs), dtype=float)
        if marginal_costs.shape[0] != self.n_players:
            raise ValueError("marginal_costs length must equal number of players")

        # initial_price_indices handling (unchanged)
        if initial_price_indices is not None:
            initial_price_indices = np.array(list(initial_price_indices), dtype=int)
            if initial_price_indices.shape[0] != self.n_players:
                raise ValueError(
                    "initial_price_indices length must equal number of players"
                )
        else:
            initial_price_indices = None

        # Build action executors for each seller
        executors: List = []
        for idx, (action, mc) in enumerate(zip(meta_actions, marginal_costs)):
            current_index = -1
            # As before: a static seller ignores the provided initial index
            if initial_price_indices is not None and not action.is_static:
                current_index = int(initial_price_indices[idx])

            executors.append(
                create_action_executor(
                    action,
                    self.prices,
                    current_index=current_index,
                )
            )

        # Initialize price indices and histories
        price_indices = np.array(
            [executor.current_index for executor in executors],
            dtype=int,
        )
        prices = self.prices[price_indices]
        price_history: List[np.ndarray] = []
        demand_history: List[np.ndarray] = []
        profit_history: List[np.ndarray] = []

        for t in range(periods):
            # Current prices from indices
            prices = self.prices[price_indices]
            price_history.append(prices.copy())

            # Canonical logit demand shares (unit mass of consumers)
            shares_logit = self._logit_shares(prices)
            demand = shares_logit  # deterministic unit-demand allocation
            demand_history.append(demand.copy())

            # Profits
            profits = (prices - marginal_costs) * demand
            profit_history.append(profits.copy())

            # Next-period price indices chosen by each agent
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

        # Convert histories to arrays
        price_history = np.asarray(price_history)  # (periods, n_players)
        demand_history = np.asarray(demand_history)
        profit_history = np.asarray(profit_history)

        # Average rewards per seller
        avg_rewards = profit_history.mean(axis=0)
        # Subtract repricer cost if applicable
        avg_rewards -= self.repricer_cost * repricer_usage_dummy

        # Summary statistics, same keys as before
        avg_prices = price_history.mean(axis=0)
        avg_lowest_prices = price_history.min(axis=1).mean()

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
                    "price_history": price_history,
                    "demand_history": demand_history,
                    "profit_history": profit_history,
                }
            )

        return result


__all__ = ["NPlayerLogitDemandEnv"]


if __name__ == "__main__":
    # Simple sanity check: symmetric 5-player environment
    env = NPlayerLogitDemandEnv(
        n_players=5,
        grid_size=10,
        a0=0.0,
        a12=2.0,
        mu=0.25,
        base_mc=1.0,
    )
    print(f"Fine-grid Nash price (approx): {env.nash_price_discrete():.4f}")
    print(f"Fine-grid Monopoly price (approx): {env.monopoly_price_discrete():.4f}")
    print("Action grid (10 points between Nash and Monopoly):")
    print(env.prices)
