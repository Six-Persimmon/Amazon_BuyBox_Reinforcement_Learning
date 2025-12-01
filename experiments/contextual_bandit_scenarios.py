#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Evaluate trained contextual bandits under custom post-training scenarios."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from agents.contextual_bandit_agent import AgentConfig, ContextualBanditAgent
from agents.repricer_meta_actions import MetaActionLibrary
from env.NPlayerPoissonDemandEnv import NPlayerPoissonDemandEnv


@dataclass
class ScenarioConfig:
    """Lightweight container describing a scenario to evaluate."""
    model_path: str
    learner_type: str
    n_players: int
    price_min: float = 0.01
    price_max: float = 10.0
    grid_size: int = 25
    a0: float = 0.0
    a12: float = 10.0
    mu: float = 0.25
    inner_periods: int = 200
    eval_rounds: int = 1
    mc_min: Optional[float] = None
    mc_max: Optional[float] = None
    marginal_costs: Optional[List[float]] = None
    lam: float = 5.0
    rho: float = 0.5
    learn_during_eval: bool = False
    seed: Optional[int] = None


class ScenarioRunner:
    """Utility for running saved bandits in user-specified environments."""

    def __init__(self, config: ScenarioConfig, agent: Optional[ContextualBanditAgent] = None) -> None:
        """Create a runner and optionally reuse an already loaded bandit."""
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.model_path = Path(config.model_path)
        if not self.model_path.is_absolute():
            self.model_path = (ROOT / self.model_path).resolve()

        if agent is None:
            agent_cfg = AgentConfig(learner_type=config.learner_type)
            self.agent = ContextualBanditAgent.from_saved_model(
                model_path=str(self.model_path),
                config=agent_cfg,
                n_players=config.n_players,
            )
        else:
            self.agent = agent
        self.env_kwargs = dict(
            price_min=config.price_min,
            price_max=config.price_max,
            grid_size=config.grid_size,
            a0=config.a0,
            a12=config.a12,
            mu=config.mu,
        )
        self.logs: List[Dict] = []

    def _draw_marginal_costs(self) -> np.ndarray:
        """Sample a marginal-cost vector for all sellers."""
        cfg = self.config
        if cfg.marginal_costs is not None:
            costs = np.array(cfg.marginal_costs, dtype=float)
            if costs.size != cfg.n_players:
                raise ValueError("marginal_costs length must match n_players")
            return costs
        if cfg.mc_min is None or cfg.mc_max is None:
            raise ValueError("Either marginal_costs or mc_min/mc_max must be provided")
        return self.rng.uniform(cfg.mc_min, cfg.mc_max, size=cfg.n_players)

    def _build_contexts(self, mc_vector: np.ndarray) -> List[Dict[str, float]]:
        """Construct per-seller context dictionaries consumed by the agent."""
        cfg = self.config
        return [
            {
                "N": cfg.n_players,
                "MC": float(mc),
                "lambda": cfg.lam,
                "rho": cfg.rho,
            }
            for mc in mc_vector
        ]

    def evaluate(self) -> Dict:
        """Run the scenario with all sellers controlled by the trained bandit."""
        cfg = self.config
        scenario_env = NPlayerPoissonDemandEnv(
            n_players=cfg.n_players,
            rng=self.rng,
            **self.env_kwargs,
        )

        for round_idx in range(cfg.eval_rounds):
            mc_vector = self._draw_marginal_costs()
            contexts = self._build_contexts(mc_vector)
            actions, action_ids, features = self.agent.select_actions(contexts)
            episode_result = scenario_env.run_episode(
                meta_actions=actions,
                marginal_costs=mc_vector,
                lam=cfg.lam,
                rho=cfg.rho,
                periods=cfg.inner_periods,
                rng=self.rng,
                return_histories=True,
            )
            rewards = episode_result["average_profits"]
            if cfg.learn_during_eval:
                self.agent.update(rewards)

            log_entry = {
                "round": round_idx,
                "actions": action_ids,
                "features": [feat.tolist() for feat in features],
                "marginal_costs": mc_vector.tolist(),
                "rewards": rewards.tolist(),
                "price_history": episode_result["price_history"].tolist(),
                "demand_history": episode_result["demand_history"].tolist(),
                "profit_history": episode_result["profit_history"].tolist(),
            }
            log_entry["network_stats"] = self._compute_network_stats(
                episode_result["price_history"]
            )
            self.logs.append(log_entry)

        return {
            "scenario": asdict(cfg),
            "logs": self.logs,
        }

    def evaluate_with_fixed_meta_action(
        self,
        meta_action_id: int,
        seller_index: int = 0,
    ) -> Dict:
        """Run the scenario while forcing one seller to use a specific meta action."""
        cfg = self.config
        if seller_index < 0 or seller_index >= cfg.n_players:
            raise ValueError("seller_index out of range")

        scenario_env = NPlayerPoissonDemandEnv(
            n_players=cfg.n_players,
            rng=self.rng,
            **self.env_kwargs,
        )
        meta_library = MetaActionLibrary()
        fixed_action = meta_library.get_action(meta_action_id)
        baseline_logs: List[Dict] = []

        for round_idx in range(cfg.eval_rounds):
            mc_vector = self._draw_marginal_costs()
            contexts = self._build_contexts(mc_vector)
            actions, action_ids, features = self.agent.select_actions(contexts)
            actions[seller_index] = fixed_action
            action_ids[seller_index] = meta_action_id

            episode_result = scenario_env.run_episode(
                meta_actions=actions,
                marginal_costs=mc_vector,
                lam=cfg.lam,
                rho=cfg.rho,
                periods=cfg.inner_periods,
                rng=self.rng,
                return_histories=False,
            )
            rewards = episode_result["average_profits"]

            baseline_logs.append(
                {
                    "round": round_idx,
                    "actions": action_ids,
                    "features": [feat.tolist() for feat in features],
                    "marginal_costs": mc_vector.tolist(),
                    "rewards": rewards.tolist(),
                }
            )

        return {
            "scenario": asdict(cfg),
            "meta_action_id": meta_action_id,
            "seller_index": seller_index,
            "logs": baseline_logs,
        }

    def _compute_network_stats(self, price_history: np.ndarray) -> Dict[str, List[List[int]]]:
        """Extract directed follower networks from the recorded price history."""
        tolerance = 1e-8
        n_players = price_history.shape[1]
        matrices = {
            "undercut": np.zeros((n_players, n_players), dtype=int),
            "match": np.zeros((n_players, n_players), dtype=int),
            "above": np.zeros((n_players, n_players), dtype=int),
        }

        for prices in price_history:
            for i in range(n_players):
                for j in range(n_players):
                    if i == j:
                        continue
                    if prices[i] + tolerance < prices[j]:
                        matrices["undercut"][i, j] += 1
                    elif abs(prices[i] - prices[j]) <= tolerance:
                        matrices["match"][i, j] += 1
                    else:
                        matrices["above"][i, j] += 1

        return {key: mat.tolist() for key, mat in matrices.items()}


def compute_baseline_reward_histories(
    model_path: str,
    learner_type: str,
    training_logs: List[Dict],
    env_settings: Dict,
    inner_periods: int,
    action_ids: List[int],
    seller_index: int = 0,
    max_samples: int = 200,
    seed: int = 1234,
) -> Dict[str, Dict[int, List[float]]]:
    """Compare the trained bandit against fixed meta-action baselines."""
    if not training_logs:
        raise ValueError("training_logs must not be empty")

    indices = list(range(len(training_logs)))
    if max_samples and len(indices) > max_samples:
        indices = indices[-max_samples:]

    model_path = Path(model_path)
    if not model_path.is_absolute():
        model_path = (ROOT / model_path).resolve()

    agent_cfg = AgentConfig(learner_type=learner_type)
    agent = ContextualBanditAgent.from_saved_model(
        model_path=str(model_path),
        config=agent_cfg,
        n_players=None,
    )

    price_min = env_settings.get("price_min", 0.01)
    price_max = env_settings.get("price_max", 10.0)
    grid_size = env_settings.get("grid_size", 25)
    a0 = env_settings.get("a0", 0.0)
    a12 = env_settings.get("a12", 10.0)
    mu = env_settings.get("mu", 0.25)

    bandit_rewards: List[float] = []
    baseline_rewards: Dict[int, List[float]] = {aid: [] for aid in action_ids}

    for sample_idx, log_idx in enumerate(indices):
        entry = training_logs[log_idx]
        n_players = entry.get("n_players", len(entry["marginal_costs"]))
        lam_val = entry["lambda"]
        rho_val = entry["rho"]
        mc_vector = entry["marginal_costs"]

        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = (ROOT / model_path).resolve()
        scenario_cfg = ScenarioConfig(
            model_path=str(model_path),
            learner_type=learner_type,
            n_players=n_players,
            price_min=price_min,
            price_max=price_max,
            grid_size=grid_size,
            a0=a0,
            a12=a12,
            mu=mu,
            inner_periods=inner_periods,
            eval_rounds=1,
            marginal_costs=mc_vector,
            lam=lam_val,
            rho=rho_val,
            learn_during_eval=False,
            seed=seed + sample_idx,
        )

        bandit_runner = ScenarioRunner(scenario_cfg, agent=agent)
        bandit_result = bandit_runner.evaluate()
        bandit_reward = float(np.mean(bandit_result["logs"][0]["rewards"]))
        bandit_rewards.append(bandit_reward)

        for aid in action_ids:
            baseline_runner = ScenarioRunner(scenario_cfg, agent=agent)
            baseline_result = baseline_runner.evaluate_with_fixed_meta_action(
                meta_action_id=aid,
                seller_index=seller_index,
            )
            reward = float(np.mean(baseline_result["logs"][0]["rewards"]))
            baseline_rewards[aid].append(reward)

    return {
        "indices": indices,
        "bandit_rewards": bandit_rewards,
        "baseline_rewards": baseline_rewards,
    }


def save_scenario_output(payload: Dict, directory: str = "data/contextual_bandit_scenarios") -> str:
    """Persist scenario logs to disk and return the output path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = (
        np.datetime64("now", "ms").astype(str)
        .replace("-", "")
        .replace(":", "")
        .replace(".", "")
    )
    filename = f"scenario_{timestamp}.json"
    path = directory / filename
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return str(path)


def main() -> None:
    """Run a small example when the module is executed directly."""
    example_scenario = ScenarioConfig(
        model_path="data/contextual_bandit_models/contextual_bandit_linucb_example.npz",
        learner_type="linucb",
        n_players=5,
        lam=1.2,
        rho=0.6,
        mc_min=1.5,
        mc_max=4.0,
        inner_periods=500,
        eval_rounds=1,
        seed=123,
    )
    runner = ScenarioRunner(example_scenario)
    result = runner.evaluate()
    output_path = save_scenario_output(result)
    print(f"Scenario results saved to {output_path}")


if __name__ == "__main__":
    main()
