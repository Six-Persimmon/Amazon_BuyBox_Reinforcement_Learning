#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Lightweight MARL simulation harness for repricer selection."""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# Ensure repository root is on the Python path before local imports.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import networkx as nx

from agents.n_player_rule_agent import NPlayerQLearningRuleAgent
from agents.repricer_meta_actions import MetaAction, MetaActionLibrary
from env.NPlayerPoissonDemandEnv import NPlayerPoissonDemandEnv


@dataclass
class EnvironmentConfig:
    """Parameters used to build :class:`NPlayerPoissonDemandEnv`."""

    price_min: float = 0.5
    price_max: float = 10.0
    grid_size: int = 20
    a0: float = 0.0
    a12: float = 10.0
    mu: float = 0.25
    lam: float = 1.0
    rho: float = 0.5
    repricer_cost: float = 0.0


@dataclass
class SimulationConfig:
    """Top-level knobs for the training loop."""

    n_players: int
    outer_episodes: int = 50
    inner_periods: int = 25
    share_parameters: bool = True
    marginal_costs: Optional[Sequence[float]] = None
    seed: Optional[int] = None
    verbose: bool = False
    log_interval: int = 10

    # DQN hyperparameters (kept intentionally small for demos)
    hidden_dim: int = 32
    learning_rate: float = 1e-3
    discount_rate: float = 0.99
    epsilon_omega: float = 5e-4
    target_update: int = 100
    replay_capacity: int = 10_000
    batch_size: int = 64
    device: str = "cpu"

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    scenario_label: Optional[str] = None
    carry_over_prices: bool = True
    allowed_action_ids: Optional[Sequence[int]] = None


def _build_environment(n_players: int, cfg: EnvironmentConfig) -> NPlayerPoissonDemandEnv:
    """Factory for the pricing environment."""

    return NPlayerPoissonDemandEnv(
        n_players=n_players,
        price_min=cfg.price_min,
        price_max=cfg.price_max,
        grid_size=cfg.grid_size,
        a0=cfg.a0,
        a12=cfg.a12,
        mu=cfg.mu,
        lam=cfg.lam,
        rho=cfg.rho,
        repricer_cost=cfg.repricer_cost,
    )


def _make_agent(config: SimulationConfig, action_dim: int, state_dim: int) -> NPlayerQLearningRuleAgent:
    """Instantiate a DQN repricer selector with shared defaults."""

    return NPlayerQLearningRuleAgent(
        action_dim=action_dim,
        hidden_dim=config.hidden_dim,
        state_dim=state_dim,
        learning_rate=config.learning_rate,
        discount_rate=config.discount_rate,
        epsilon_omega=config.epsilon_omega,
        target_update=config.target_update,
        device=config.device,
        replay_capacity=config.replay_capacity,
        batch_size=config.batch_size,
        allowed_action_ids=config.allowed_action_ids,
    )


def _normalised_observation(
    avg_price: float,
    avg_lowest_price: float,
    last_lowest_price: float,
    price_scale: float,
) -> np.ndarray:
    """Map episode-level price stats onto the DQN state vector."""

    price_scale = price_scale or 1.0

    return np.array(
        [
            avg_price / price_scale,
            avg_lowest_price / price_scale,
            last_lowest_price / price_scale,
        ],
        dtype=np.float32,
    )

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
        "weighted_in_degree_centralization": 0.0,
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

    anchors: List[int] = [] # List to store anchor sellers per period
    anchor_id: Optional[int] = None
    for prices in price_history:
        min_price = float(np.min(prices))
        # Determine anchor seller (lowest price seller from previous period)
        if anchor_id is None or not np.isclose(prices[anchor_id], min_price, atol=price_tolerance):
            tie_indices = np.where(np.isclose(prices, min_price, atol=price_tolerance))[0]
            anchor_id = int(tie_indices[0]) if tie_indices.size else 0
        anchors.append(anchor_id)

    weights = np.zeros((n_players, n_players), dtype=float) # weights[seller, target_seller]
    current_targets: List[Optional[int]] = [None] * n_players   # current target seller per seller
    current_streaks: List[int] = [0] * n_players # current targeting streak lengths per seller

    def _finalize_streak(seller_idx: int) -> None:
        # Finalize the current targeting streak for a given seller, which means given the current target and streak length, update the weights matrix accordingly.
        target_idx = current_targets[seller_idx]
        streak = current_streaks[seller_idx]
        if target_idx is not None and streak > 0:
            weights[seller_idx, target_idx] = max(weights[seller_idx, target_idx], streak)
        current_targets[seller_idx] = None
        current_streaks[seller_idx] = 0

    for anchor_id in anchors: # notice len(anchors) == periods
        for seller_id in range(n_players):
            if seller_id == anchor_id or not targeting_mask[seller_id]: # skip self-targeting and non-targeting sellers
                _finalize_streak(seller_id)
                continue
            if current_targets[seller_id] == anchor_id:
                current_streaks[seller_id] += 1
            else:
                _finalize_streak(seller_id)
                current_targets[seller_id] = anchor_id
                current_streaks[seller_id] = 1

    for seller_idx in range(n_players):
        _finalize_streak(seller_idx)

    max_weight = np.max(weights)
    weights_norm = weights / max_weight if max_weight > 0 else weights
    possible_edges = n_players * (n_players - 1)
    # create directed graph using the normalized weights
    G = _create_networkx_graph(weights_norm) # weighted directed graph
    # Density
    network_density = nx.density(G)
    # weighted density
    weighted_density = (np.sum(weights_norm) / possible_edges) if possible_edges else 0.0

    # In degree centralization of the network
    centralities = nx.in_degree_centrality(G)
    max_centrality = max(centralities.values()) if centralities else 0.0
    numerator = sum(max_centrality - c for c in centralities.values())
    in_deg_centralization = numerator / (n_players - 1) if n_players > 2 else 0.0  # normalize by max possible value. notice networkx already normalized the centralizties for each node by (n_players-1)

    # weighted in-degree centralization
    weighted_in_degrees = dict(G.in_degree(weight="weight"))
    max_weighted_in_degree = max(weighted_in_degrees.values()) if weighted_in_degrees else 0.0
    sum_diff = sum(max_weighted_in_degree - deg for deg in weighted_in_degrees.values())
    max_possible_sum = (n_players - 1) * max_weighted_in_degree
    weighted_in_deg_centralization = sum_diff / max_possible_sum if max_possible_sum > 0 else 0.0

    return {
        "network_density": float(network_density),
        "weighted_density": float(weighted_density),
        "in_degree_centralization": float(in_deg_centralization),
        "weighted_in_degree_centralization": float(weighted_in_deg_centralization),
    }


def run_simulation(config: SimulationConfig) -> Dict[str, List[Dict[str, float]]]:
    """Execute the outer/inner loop training procedure.

    The function returns a dictionary with per-episode summaries that can be
    consumed by plotting utilities or test harnesses.
    """

    if config.seed is not None:
        np.random.seed(config.seed)

    env = _build_environment(config.n_players, config.environment)
    library = MetaActionLibrary()
    action_dim = len(library.list_actions())
    state_dim = 3  # [avg price, avg lowest price, last lowest price]
    # state_dim = 2 #[avg lowest price, last lowest price]

    if config.share_parameters:
        shared_agent = _make_agent(config, action_dim, state_dim)
        # All agents are references to the same instance; learning is shared and actions affect the same model.
        agents = [shared_agent] * config.n_players
    else:
        agents = [_make_agent(config, action_dim, state_dim) for _ in range(config.n_players)]

    if config.marginal_costs is not None:
        marginal_costs = np.asarray(config.marginal_costs, dtype=float)
        if marginal_costs.shape[0] != config.n_players:
            raise ValueError("marginal_costs length must match n_players")
    else:
        marginal_costs = np.zeros(config.n_players, dtype=float)

    price_scale = float(env.price_max)

    prev_avg_prices = [env.price_min] * config.n_players
    prev_avg_lowest_price = env.price_min
    prev_last_lowest_price = env.price_min
    prev_price_indices: Optional[List[int]] = None

    summaries: List[Dict[str, float]] = []
    loss_trace: List[float] = []

    for episode in range(config.outer_episodes):
        states: List[np.ndarray] = []
        chosen_actions: List[MetaAction] = []
        chosen_ids: List[int] = []

        for player_id in range(config.n_players):
            state = _normalised_observation(
                prev_avg_prices[player_id],
                prev_avg_lowest_price,
                prev_last_lowest_price,
                price_scale,
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

        rewards = np.asarray(result["average_profits"], dtype=float)
        avg_prices = np.asarray(result["average_prices"], dtype=float)
        avg_lowest = float(result["average_lowest_price"])
        price_history = np.asarray(result["price_history"], dtype=float)
        last_lowest = float(np.min(price_history[-1])) if price_history.size else avg_lowest # last period's lowest price
        repricer_share = float(np.mean(result["repricer_usage"]))
        metrics = _compute_network_metrics(price_history, chosen_actions)
        done_flag = episode == (config.outer_episodes - 1)

        for player_id in range(config.n_players):
            # The state variables we use: own avg price, avg lowest price, last inner loop period lowest price
            next_state = _normalised_observation(
                avg_prices[player_id],
                avg_lowest,
                last_lowest,
                price_scale,
            )
            agents[player_id].store_transition(
                states[player_id],
                chosen_ids[player_id],
                rewards[player_id],
                next_state,
                done_flag,
            )
            loss = agents[player_id].train_step()
            if loss is not None:
                loss_trace.append(loss)

        prev_avg_prices = avg_prices.tolist()
        prev_avg_lowest_price = avg_lowest
        prev_last_lowest_price = last_lowest
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
                'avg_price': float(np.mean(avg_prices)),
                "avg_lowest_price": avg_lowest,
                "repricer_share": repricer_share,
                "network_density": metrics["network_density"],
                "weighted_density": metrics["weighted_density"],
                "in_degree_centralization": metrics["in_degree_centralization"],
                "weighted_in_degree_centralization": metrics["weighted_in_degree_centralization"],
            }
        )

        action_counts = Counter(chosen_ids)
        for action in library.list_actions():
            share_key = f"action_share_{action.name}"
            summaries[-1][share_key] = float(action_counts.get(action.action_id, 0) / config.n_players)

        if config.verbose and ((episode + 1) % max(1, config.log_interval) == 0 or episode == config.outer_episodes - 1):
            print(
                f"[Simulation] episodes completed: {episode + 1}/{config.outer_episodes} "
                f"(share_parameters={config.share_parameters})"
            )

    return {"summaries": summaries, "losses": loss_trace}


def _print_demo_summary(title: str, result: Dict[str, List[Dict[str, float]]]) -> None:
    """Pretty-print the last episode summary from a demo run."""

    last_summary = result["summaries"][-1]
    print("=" * 60)
    print(title)
    print(f"Final episode: {int(last_summary['episode'])}")
    print(f"Mean profit:   {last_summary['mean_profit']:.4f}")
    print(f"Avg buy box:   {last_summary['avg_lowest_price']:.4f}")
    print(f"Repricer use:  {last_summary['repricer_share']:.3f}")
    print(f"Recorded losses: {len(result['losses'])}")


def demo_parameter_sharing() -> None:
    """Quick smoke test with a single shared agent."""

    config = SimulationConfig(
        n_players=3,
        outer_episodes=10,
        inner_periods=20,
        share_parameters=True,
        seed=123,
    )
    result = run_simulation(config)
    _print_demo_summary("Demo: parameter sharing", result)


if __name__ == "__main__":
    demo_parameter_sharing()
