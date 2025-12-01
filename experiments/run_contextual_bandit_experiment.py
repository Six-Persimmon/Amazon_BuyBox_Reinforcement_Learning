#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Train the contextual bandit on synthetic market states (outer loop)."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from agents.contextual_bandit_agent import AgentConfig, ContextualBanditAgent
from env.NPlayerPoissonDemandEnv import NPlayerPoissonDemandEnv


@dataclass
class EnvironmentConfig:
    """Parameter ranges used to randomise contexts during training."""
    price_min: float = 0.01
    price_max: float = 10.0
    grid_size: int = 100
    a0: float = 0.0
    a12: float = 10.0
    mu: float = 0.25
    inner_periods: int = 100
    n_players_values: List[int] = field(default_factory=list)
    lambda_values: List[float] = field(default_factory=list)
    rho_values: List[float] = field(default_factory=list)
    mc_min: float = 1.0
    mc_max: float = 4.0


@dataclass
class ExperimentConfig:
    """High-level configuration for a single training run."""
    env: EnvironmentConfig
    agent: AgentConfig
    outer_rounds: int = 200
    random_seed: Optional[int] = None
    model_dir: str = "data/contextual_bandit_models"
    run_dir: str = "data/contextual_bandit_runs"
    learner_prefix: str = "contextual_bandit"


class ContextualBanditExperiment:
    """Wrapper coordinating context sampling, environment rollouts, and updates."""

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialise the experiment and validate the supplied ranges."""
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        env_config = config.env
        if not env_config.n_players_values:
            raise ValueError("EnvironmentConfig.n_players_values must not be empty")
        if not env_config.lambda_values:
            raise ValueError("EnvironmentConfig.lambda_values must not be empty")
        if not env_config.rho_values:
            raise ValueError("EnvironmentConfig.rho_values must not be empty")

        self.base_env_kwargs = dict(
            price_min=env_config.price_min,
            price_max=env_config.price_max,
            grid_size=env_config.grid_size,
            a0=env_config.a0,
            a12=env_config.a12,
            mu=env_config.mu,
        )

        self.agent = ContextualBanditAgent(config=config.agent)
        self.logs: List[dict] = []

    def _sample_round_parameters(self) -> tuple:
        """Draw a single (N, λ, ρ, MC vector) tuple for the next outer round."""
        env_cfg = self.config.env
        n_players = int(self.rng.choice(env_cfg.n_players_values))
        lam = float(self.rng.choice(env_cfg.lambda_values))
        rho = float(self.rng.choice(env_cfg.rho_values))
        price_grids = np.linspace(env_cfg.price_min, env_cfg.price_max, env_cfg.grid_size)
        mc_grids = price_grids[(price_grids <= env_cfg.mc_max) & (price_grids >= env_cfg.mc_min)]
        mc_vector = self.rng.choice(mc_grids, size=n_players, replace=True)
        return n_players, lam, rho, mc_vector

    def run(self) -> dict:
        """Execute the full training loop and return a summary dictionary."""
        env_cfg = self.config.env
        for round_idx in range(self.config.outer_rounds):
            n_players, lam_val, rho_val, mc_vector = self._sample_round_parameters()

            env = NPlayerPoissonDemandEnv(
                n_players=n_players,
                rng=self.rng,
                **self.base_env_kwargs,
            )

            contexts = [
                {
                    "N": n_players,
                    "MC": float(mc),
                    "lambda": lam_val,
                    "rho": rho_val,
                }
                for mc in mc_vector
            ]

            actions, action_ids, features = self.agent.select_actions(contexts)
            episode_result = env.run_episode(
                meta_actions=actions,
                marginal_costs=mc_vector,
                lam=lam_val,
                rho=rho_val,
                periods=env_cfg.inner_periods,
                rng=self.rng,
                return_histories=False,
            )
            rewards = episode_result["average_profits"]
            self.agent.update(rewards) # We use shared UCB, so a single update call using all the rewards.
            self.logs.append(
                {
                    "round": round_idx,
                    "n_players": n_players,
                    "lambda": lam_val,
                    "rho": rho_val,
                    "marginal_costs": mc_vector.tolist(),
                    "actions": action_ids,
                    "features": [feat.tolist() for feat in features],
                    "rewards": rewards.tolist(),
                }
            )

            if (round_idx + 1) % 25 == 0 or round_idx == 0:
                avg_reward = float(np.mean(rewards))
                print(
                    f"Round {round_idx + 1:>4}/{self.config.outer_rounds} "
                    f"| N={n_players:>2} λ={lam_val:.3f} ρ={rho_val:.3f} "
                    f"| avg reward={avg_reward:.4f}"
                )

        metadata = {
            "outer_rounds": self.config.outer_rounds,
            "inner_periods": env_cfg.inner_periods,
            "lambda_values": env_cfg.lambda_values,
            "rho_values": env_cfg.rho_values,
            "n_players_values": env_cfg.n_players_values,
            "mc_min": env_cfg.mc_min,
            "mc_max": env_cfg.mc_max,
            "learner": self.config.agent.learner_type,
        }
        model_path = self.agent.save_model(
            directory=self.config.model_dir,
            prefix=self.config.learner_prefix,
            metadata=metadata,
        )
        print(f"Training complete. Model saved to {model_path}")
        run_summary = {
            "model_path": model_path,
            "metadata": metadata,
            "config": {
                "env": asdict(self.config.env),
                "agent": asdict(self.config.agent),
                "outer_rounds": self.config.outer_rounds,
            },
        }
        self._persist_logs(run_summary)
        return run_summary

    def _persist_logs(self, summary: dict) -> None:
        """Write the detailed run log to disk next to the saved model."""
        os.makedirs(self.config.run_dir, exist_ok=True)
        timestamp = summary["model_path"].split("_")[-1].split(".")[0]
        log_path = os.path.join(self.config.run_dir, f"run_{timestamp}.json")
        payload = {
            "summary": summary,
            "logs": self.logs,
        }
        with open(log_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def main() -> None:
    """Train a LinUCB bandit using the default configuration."""
    env_cfg = EnvironmentConfig(
        inner_periods=1000,
        n_players_values=list(np.linspace(2, 25, 24, dtype=int)),
        lambda_values=list(np.linspace(0.5, 2.0, 4)),
        rho_values=list(np.linspace(0.1, 0.9, 5)),
        mc_min=1.0,
        mc_max=4.0,
    )
    agent_cfg = AgentConfig(
        learner_type="linucb",
        alpha_ucb=1.0,
        regularization=1.0,
    )
    exp_cfg = ExperimentConfig(
        env=env_cfg,
        agent=agent_cfg,
        outer_rounds=10000,
        random_seed=42,
    )
    experiment = ContextualBanditExperiment(exp_cfg)
    summary = experiment.run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
