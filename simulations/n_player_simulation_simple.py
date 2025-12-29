#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Lightweight MARL simulation using a tabular Q-learning rule agent."""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import networkx as nx

from agents.n_player_rule_agent_simple_q import NPlayerSimpleQLearningRuleAgent
from agents.repricer_meta_actions import MetaAction, MetaActionLibrary
from env.NPlayerLogitDemandEnv import NPlayerLogitDemandEnv


@dataclass
class EnvironmentConfig:
    """Parameters used to build :class:`NPlayerLogitDemandEnv`."""

    grid_size: int = 10
    a0: float = 0.0
    a12: float = 2.0
    mu: float = 0.25
    repricer_cost: float = 0.0
    base_mc: float = 1.0
    fine_grid_points: int = 200
    fine_grid_span: float = 4.0
    max_price: Optional[float] = None  # Optional override for the maximum price in the grid.


@dataclass
class SimulationConfig:
    """Top-level knobs for the training loop (tabular Q-learning version)."""

    n_players: int
    outer_episodes: int = 50
    inner_periods: int = 25
    share_parameters: bool = True
    marginal_costs: Optional[Sequence[float]] = None
    seed: Optional[int] = None
    verbose: bool = False
    log_interval: int = 10

    # Tabular Q-learning hyperparameters
    learning_rate: float = 1e-1
    discount_rate: float = 0.95
    epsilon_omega: float = 1.5e-5  # decay rate, approximately 10 / N. N = 120,000 in our baseline case. epsilon = exp(-t * epsilon_omega)

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    scenario_label: Optional[str] = None
    carry_over_prices: bool = True
    allowed_action_ids: Optional[Sequence[int]] = None


def _build_environment(n_players: int, cfg: EnvironmentConfig) -> NPlayerLogitDemandEnv:
    """Factory for the pricing environment."""

    return NPlayerLogitDemandEnv(
        n_players=n_players,
        grid_size=cfg.grid_size,
        a0=cfg.a0,
        a12=cfg.a12,
        mu=cfg.mu,
        repricer_cost=cfg.repricer_cost,
        base_mc=cfg.base_mc,
        fine_grid_points=cfg.fine_grid_points,
        fine_grid_span=cfg.fine_grid_span,
        max_price=cfg.max_price,
    )


def _make_agent(
    config: SimulationConfig,
    action_dim: int,
    state_dim: int,
    price_grid: np.ndarray,
) -> NPlayerSimpleQLearningRuleAgent:
    """Instantiate a tabular Q-learning repricer selector with shared defaults."""

    return NPlayerSimpleQLearningRuleAgent(
        action_dim=action_dim,
        state_dim=state_dim,
        learning_rate=config.learning_rate,
        discount_rate=config.discount_rate,
        epsilon_omega=config.epsilon_omega,
        price_grid=price_grid,
        allowed_action_ids=config.allowed_action_ids,
    )


def _bin_gap(value: float, edges: np.ndarray) -> float:
    """Discretize a gap value into a small integer bin."""

    # np.digitize returns indices in 1..len(edges); subtract 1 to get 0-based bin.
    return float(np.digitize([value], edges)[0] - 1)


def _create_networkx_graph(matrix: np.ndarray) -> "nx.DiGraph":
    """Create a NetworkX directed graph from a weighted adjacency matrix."""

    G = nx.DiGraph()
    n_players = matrix.shape[0]
    for i in range(n_players):
        G.add_node(i)
    for i in range(n_players):
        for j in range(n_players):
            if i != j and matrix[i, j] > 0:
                G.add_edge(i, j, weight=matrix[i, j])
    return G


def _compute_network_metrics(
    price_history: np.ndarray,
    meta_actions: Sequence[MetaAction],
) -> Dict[str, float]:
    """Derive network metrics from inferred targeting behaviour."""

    base_metrics = {
        "network_density": 0.0,
        "weighted_density": 0.0,
        "in_degree_centralization": 0.0,
    }

    if price_history.size == 0:
        return base_metrics.copy()

    periods, n_players = price_history.shape
    if n_players <= 1 or periods == 0:
        return base_metrics.copy()

    if len(meta_actions) != n_players:
        raise ValueError("meta_actions length must equal number of players")

    targeting_rules = {"match", "undercut", "above"}
    targeting_mask = [action.base_rule in targeting_rules for action in meta_actions]
    price_tolerance = 1e-9

    anchors: List[int] = []
    anchor_id: Optional[int] = None
    # Identify the anchor (lowest-priced seller) for each period
    for prices in price_history:
        min_price = float(np.min(prices))
        if anchor_id is None or not np.isclose(prices[anchor_id], min_price, atol=price_tolerance):
            tie_indices = np.where(np.isclose(prices, min_price, atol=price_tolerance))[0]
            anchor_id = int(tie_indices[0]) if tie_indices.size else 0
        anchors.append(anchor_id)

    weights = np.zeros((n_players, n_players), dtype=float)  # weights[seller, target_seller]
    for anchor in anchors:
        for seller_id in range(n_players):
            if seller_id == anchor or not targeting_mask[seller_id]: # should not be 1. anchor itself or 2. not using a targeting rule
                continue
            weights[seller_id, anchor] += 1.0

    max_weight = np.max(weights)
    weights_norm = weights / max_weight if max_weight > 0 else weights
    possible_edges = n_players * (n_players - 1)

    G = _create_networkx_graph(weights_norm)
    network_density = nx.density(G)
    weighted_density = (np.sum(weights_norm) / possible_edges) if possible_edges else 0.0
    centralities = nx.in_degree_centrality(G)
    max_centrality = max(centralities.values()) if centralities else 0.0
    numerator = sum(max_centrality - c for c in centralities.values())
    in_deg_centralization = numerator / (n_players - 1) if n_players > 2 else 0.0

    return {
        "network_density": float(network_density),
        "weighted_density": float(weighted_density),
        "in_degree_centralization": float(in_deg_centralization),
    }


def run_simulation(config: SimulationConfig) -> Dict[str, List[Dict[str, float]]]:
    """Execute the outer/inner loop training procedure with tabular Q-learning."""

    if config.seed is not None:
        np.random.seed(config.seed)

    env = _build_environment(config.n_players, config.environment)
    library = MetaActionLibrary()
    action_dim = len(library.list_actions())
    # state_dim = 3  # [own avg price, avg lowest price, last lowest price]
    state_dim = 1  # [last lowest price]
    # state_dim = 2  # [last lowest price, is_lowest flag]
    # state_dim = 2  # [lowest price of others in last period, own price in last period]  # previous version
    # state_dim = 1  # [lowest price of others in last period]
    price_grid = env.prices
    profit_nash = env.per_seller_profit_nash
    profit_monopoly = env.per_seller_profit_monopoly
    norm_profit_denom = profit_monopoly - profit_nash

    if config.share_parameters:
        shared_agent = _make_agent(config, action_dim, state_dim, price_grid)
        agents = [shared_agent] * config.n_players
    else:
        agents = [_make_agent(config, action_dim, state_dim, price_grid) for _ in range(config.n_players)]

    if config.marginal_costs is not None:
        marginal_costs = np.asarray(config.marginal_costs, dtype=float)
        if marginal_costs.shape[0] != config.n_players:
            raise ValueError("marginal_costs length must match n_players")
    else:
        marginal_costs = np.full(config.n_players, float(env.base_mc), dtype=float)

    price_floor = float(env.prices[0])
    prev_other_last_lowest = [price_floor] * config.n_players
    # prev_own_last_price = [price_floor] * config.n_players  # previous 2-feature state
    prev_price_indices: Optional[List[int]] = None
    last_lowest_price = price_floor

    summaries: List[Dict[str, float]] = []
    td_trace: List[float] = []

    for episode in range(config.outer_episodes):
        states: List[np.ndarray] = []
        chosen_actions: List[MetaAction] = []
        chosen_ids: List[int] = []

        for player_id in range(config.n_players):
            # Previous 3-feature state:
            # state = np.asarray(
            #     [
            #         prev_avg_prices[player_id],
            #         prev_avg_lowest_price,
            #         prev_last_lowest_price,
            #     ],
            #     dtype=np.float32,
            # )
            # Previous 1-feature state:
            # state = np.asarray(
            #     [
            #         rounded_prev_last_lowest,
            #     ],
            #     dtype=np.float32,
            # )
            # Previous 2-feature state with last-lowest/is_lowest flag:
            # state = np.asarray(
            #     [
            #         rounded_prev_last_lowest,
            #         1.0 if np.isclose(prev_last_prices[player_id], prev_last_lowest_price) else 0.0,
            #     ],
            #     dtype=np.float32,
            # )
            # Previous 2-feature state including own last price:
            # state = np.asarray(
            #     [
            #         prev_other_last_lowest[player_id],
            #         prev_own_last_price[player_id],
            #     ],
            #     dtype=np.float32,
            # )
            state = np.asarray(
                [
                    # prev_other_last_lowest[player_id],
                    last_lowest_price, # 1-feature state, lowest price among others at the end of the last inner episode

                ],
                dtype=np.float32,
            )
            states.append(state)
            action = agents[player_id].take_action(state)
            chosen_actions.append(action)
            chosen_ids.append(action.action_id)

        result = env.run_episode(
            meta_actions=chosen_actions,
            marginal_costs=marginal_costs,
            periods=config.inner_periods,
            return_histories=True,
            initial_price_indices=prev_price_indices if config.carry_over_prices else None,
        )

        rewards = np.asarray(result["average_profits"], dtype=float) # shape (n_players,)
        avg_prices = np.asarray(result["average_prices"], dtype=float) # shape (n_players,)
        avg_lowest = float(result["average_lowest_price"]) # scalar
        price_history = np.asarray(result["price_history"], dtype=float) # shape (periods, n_players)
        repricer_share = float(np.mean(result["repricer_usage"])) # scalar
        metrics = _compute_network_metrics(price_history, chosen_actions)
        done_flag = episode == (config.outer_episodes - 1)
        # Compute last-period prices
        if price_history.size:
            last_prices = price_history[-1]  # shape (n_players,)
            last_lowest_price = float(np.min(last_prices))
            other_last_lowest = []
            for player_id in range(config.n_players):
                others = np.delete(last_prices, player_id)
                other_last_lowest.append(float(np.min(others)) if others.size else float(last_prices[player_id]))
        else:
            last_prices = np.full(config.n_players, price_floor, dtype=float)
            other_last_lowest = [price_floor] * config.n_players
            last_lowest_price = price_floor

        for player_id in range(config.n_players):
            # Previous 3-feature next_state:
            # next_state = np.asarray(
            #     [
            #         avg_prices[player_id],
            #         avg_lowest,
            #         last_lowest,
            #     ],
            #     dtype=np.float32,
            # )
            # Previous 1-feature next_state:
            # next_state = np.asarray(
            #     [
            #         rounded_last_lowest,
            #     ],
            #     dtype=np.float32,
            # )
            # Previous 2-feature next_state with last-lowest/is_lowest flag:
            # next_state = np.asarray(
            #     [
            #         rounded_last_lowest,
            #         1.0 if np.isclose(last_prices[player_id], last_lowest) else 0.0,
            #     ],
            #     dtype=np.float32,
            # )
            # Previous 2-feature next_state including own last price:
            # next_state = np.asarray(
            #     [
            #         other_last_lowest[player_id],
            #         last_prices[player_id],
            #     ],
            #     dtype=np.float32,
            # )
            next_state = np.asarray(
                [
                    # other_last_lowest[player_id],
                    last_lowest_price, # 1-feature state, lowest price among others at the end of the last inner episode
                ],
                dtype=np.float32,
            )
            td_error = agents[player_id].store_transition(
                states[player_id],
                chosen_ids[player_id],
                rewards[player_id],
                next_state,
                done_flag,
            )
            td_trace.append(float(abs(td_error)))

        prev_other_last_lowest = other_last_lowest
        # prev_own_last_price = last_prices.tolist()  # previous 2-feature state
        if config.carry_over_prices:
            final_indices = result.get("final_prices_idx")
            if final_indices is not None:
                prev_price_indices = [int(idx) for idx in np.asarray(final_indices).tolist()]
            else:
                prev_price_indices = None
        else:
            prev_price_indices = None

        summaries.append(
            {
                "episode": float(episode),
                "mean_profit": float(np.mean(rewards)),
                "norm_profit": float(
                    (np.mean(rewards) - profit_nash) / norm_profit_denom
                )
                if norm_profit_denom != 0
                else float("nan"),
                "avg_price": float(np.mean(avg_prices)),
                "avg_lowest_price": avg_lowest,
                "repricer_share": repricer_share,
                "network_density": metrics["network_density"],
                "weighted_density": metrics["weighted_density"],
                "in_degree_centralization": metrics["in_degree_centralization"],
            }
        )

        action_counts = Counter(chosen_ids)
        for action in library.list_actions():
            share_key = f"action_share_{action.name}"
            summaries[-1][share_key] = float(action_counts.get(action.action_id, 0) / config.n_players)

        if config.verbose and ((episode + 1) % max(1, config.log_interval) == 0 or episode == config.outer_episodes - 1):
            print(
                f"[Simulation-SimpleQ] episodes completed: {episode + 1}/{config.outer_episodes} "
                f"(share_parameters={config.share_parameters})"
            )

    return {"summaries": summaries, "losses": td_trace}
