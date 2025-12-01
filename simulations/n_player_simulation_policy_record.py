#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Simulation variant that records policy probes and optional Q-net snapshots."""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Ensure repository root is on the Python path before local imports.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

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
class PolicyRecordSimulationConfig:
    """Configuration for simulations that probe learned policies."""

    n_players: int
    outer_episodes: int = 50
    inner_periods: int = 25
    share_parameters: bool = True
    marginal_costs: Optional[Sequence[float]] = None
    seed: Optional[int] = None
    verbose: bool = False
    log_interval: int = 10

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
    carry_over_prices: bool = False

    # Policy probe options
    policy_probe_states: Optional[Sequence[Sequence[float]]] = None
    policy_probe_labels: Optional[Sequence[str]] = None
    policy_probe_states_normalised: bool = False
    policy_probe_use_greedy: bool = True

    # Q-net snapshot options
    policy_qnet_snapshot_episodes: Sequence[int] = field(default_factory=list)
    policy_save_final_qnets: bool = False
    policy_output_dir: Optional[str] = None
    policy_file_prefix: Optional[str] = None


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


def _make_agent(config: PolicyRecordSimulationConfig, action_dim: int, state_dim: int) -> NPlayerQLearningRuleAgent:
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


def _prepare_policy_probe_states(
    config: PolicyRecordSimulationConfig,
    price_scale: float,
) -> List[Dict[str, object]]:
    """Normalise and label policy probe states."""

    states = config.policy_probe_states
    if not states:
        return []

    labels = list(config.policy_probe_labels or [])
    if labels and len(labels) != len(states):
        raise ValueError("policy_probe_labels length must match policy_probe_states")

    probes: List[Dict[str, object]] = []
    for idx, state_values in enumerate(states):
        if len(state_values) != 3:
            raise ValueError("Each policy probe state must provide three values")
        if config.policy_probe_states_normalised:
            norm = np.asarray(state_values, dtype=np.float32)
            raw = (norm * price_scale).tolist()
        else:
            raw_vals = [float(v) for v in state_values]
            norm = _normalised_observation(raw_vals[0], raw_vals[1], raw_vals[2], price_scale)
            raw = raw_vals
        probes.append(
            {
                "label": labels[idx] if labels else f"probe_{idx}",
                "state_raw": raw,
                "state_vector": norm,
            }
        )
    return probes


def _select_policy_action(
    agent: NPlayerQLearningRuleAgent,
    state_vector: np.ndarray,
    greedy: bool,
) -> Tuple[int, MetaAction]:
    """Pick an action for a probe state without touching replay buffers or epsilon timers."""

    state = np.asarray(state_vector, dtype=np.float32)
    if state.ndim == 1:
        state = state.reshape(1, -1)
    elif state.shape[0] != 1:
        raise ValueError("Probe states must be 1D or single-row 2D arrays")

    with torch.no_grad():
        tensor_state = torch.from_numpy(state).to(agent.device)
        q_values = agent.q_net(tensor_state)[0].detach().cpu().numpy()

    action_ids = agent._valid_action_ids
    masked_q = np.full(agent.action_dim, -np.inf, dtype=np.float32)
    masked_q[action_ids] = q_values[action_ids]
    greedy_id = int(np.argmax(masked_q))

    if greedy:
        chosen_id = greedy_id
    else:
        epsilon = agent.epsilon
        if np.random.rand() < epsilon:
            chosen_id = int(np.random.choice(action_ids))
        else:
            chosen_id = greedy_id

    return chosen_id, agent._id_to_action[chosen_id]


def _evaluate_policy_probes(
    agents: Sequence[NPlayerQLearningRuleAgent],
    probes: Sequence[Dict[str, object]],
    greedy: bool,
) -> List[Dict[str, object]]:
    """Record each agent's response to the supplied probe states."""

    if not probes:
        return []

    results: List[Dict[str, object]] = []
    for probe in probes:
        state_vec = np.asarray(probe["state_vector"], dtype=np.float32)
        actions: List[Dict[str, object]] = []
        counter: Counter[str] = Counter()
        for player_id, agent in enumerate(agents):
            action_id, action = _select_policy_action(agent, state_vec, greedy=greedy)
            actions.append(
                {
                    "player": int(player_id),
                    "action_id": int(action_id),
                    "action_name": action.name,
                }
            )
            counter[action.name] += 1
        total_players = len(agents)
        proportions = {name: count / total_players for name, count in counter.items()}
        results.append(
            {
                "label": probe["label"],
                "state_raw": list(probe["state_raw"]),
                "state_normalized": state_vec.tolist(),
                "player_actions": actions,
                "action_counts": {name: int(count) for name, count in counter.items()},
                "action_proportions": proportions,
            }
        )
    return results


def _snapshot_qnets(
    agents: Sequence[NPlayerQLearningRuleAgent],
    output_dir: Path,
    prefix: str,
    seed: Optional[int],
    episode_label: str,
) -> List[Dict[str, object]]:
    """Persist each agent's Q-network parameters to disk."""

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_text = str(seed) if seed is not None else "NA"
    records: List[Dict[str, object]] = []
    for player_id, agent in enumerate(agents):
        filename = f"{prefix}_policy_record_seed{seed_text}_episode{episode_label}_player{player_id}.pt"
        path = output_dir / filename
        torch.save(agent.q_net.state_dict(), path)
        try:
            rel_path = path.relative_to(ROOT)
            path_text = str(rel_path)
        except ValueError:
            path_text = str(path)
        records.append({"player": int(player_id), "path": path_text})
    return records


def run_simulation_policy_record(config: PolicyRecordSimulationConfig) -> Dict[str, object]:
    """Execute the simulation while probing policy behaviour after each episode."""

    if config.seed is not None:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    env = _build_environment(config.n_players, config.environment)
    library = MetaActionLibrary()
    library_actions = library.list_actions()
    action_dim = len(library_actions)
    state_dim = 3
    action_names = [action.name for action in library_actions]

    if config.share_parameters:
        shared_agent = _make_agent(config, action_dim, state_dim)
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
    probe_states = _prepare_policy_probe_states(config, price_scale)
    snapshot_episodes = {int(ep) for ep in config.policy_qnet_snapshot_episodes}
    snapshot_dir = Path(config.policy_output_dir) if config.policy_output_dir else ROOT / "analysis" / "results"
    snapshot_prefix = config.policy_file_prefix or (config.scenario_label or "simulation")

    prev_avg_prices = [env.price_min] * config.n_players
    prev_avg_lowest_price = env.price_min
    prev_last_lowest_price = env.price_min
    prev_price_indices: Optional[List[int]] = None

    summaries: List[Dict[str, float]] = []
    loss_trace: List[float] = []
    qnet_snapshots: List[Dict[str, object]] = []

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
        last_lowest = float(np.min(price_history[-1])) if price_history.size else avg_lowest
        repricer_share = float(np.mean(result["repricer_usage"]))
        done_flag = episode == (config.outer_episodes - 1)

        for player_id in range(config.n_players):
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

        summary: Dict[str, float] = {
            "episode": float(episode),
            "mean_profit": float(np.mean(rewards)),
            "avg_price": float(np.mean(avg_prices)),
            "avg_lowest_price": avg_lowest,
            "repricer_share": repricer_share,
        }
        action_counts = Counter(chosen_ids)
        for action in library.list_actions():
            share_key = f"action_share_{action.name}"
            summary[share_key] = float(action_counts.get(action.action_id, 0) / config.n_players)
        summaries.append(summary)

        probe_result = _evaluate_policy_probes(agents, probe_states, greedy=config.policy_probe_use_greedy)
        if probe_result:
            for probe in probe_result:
                label = probe["label"]
                label_slug = label.lower().replace(" ", "_")
                proportions = probe.get("action_proportions", {})
                for action_name in action_names:
                    share = float(proportions.get(action_name, 0.0))
                    key = f"policy_probe_{label_slug}_share_{action_name}"
                    summary[key] = share

        if episode in snapshot_episodes:
            episode_label = f"{episode:06d}"
            snapshot_files = _snapshot_qnets(agents, snapshot_dir, snapshot_prefix, config.seed, episode_label)
            qnet_snapshots.append({"episode": float(episode), "files": snapshot_files})

        if config.verbose and ((episode + 1) % max(1, config.log_interval) == 0 or done_flag):
            print(
                f"[PolicyRecordSimulation] episodes completed: {episode + 1}/{config.outer_episodes} "
                f"(share_parameters={config.share_parameters})"
            )

    if config.policy_save_final_qnets:
        snapshot_files = _snapshot_qnets(agents, snapshot_dir, snapshot_prefix, config.seed, "final")
        qnet_snapshots.append({"episode": float(config.outer_episodes - 1), "label": "final", "files": snapshot_files})

    return {
        "summaries": summaries,
        "losses": loss_trace,
        "qnet_snapshots": qnet_snapshots,
    }


__all__ = [
    "EnvironmentConfig",
    "PolicyRecordSimulationConfig",
    "run_simulation_policy_record",
]
