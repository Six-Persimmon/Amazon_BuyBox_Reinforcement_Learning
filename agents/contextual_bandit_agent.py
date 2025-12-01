#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Shared contextual bandit agent for repricer selection."""

from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from agents.linucb import LinUCB, LinUCBConfig
from agents.neural_ucb import NeuralBandit, NeuralBanditConfig
from agents.repricer_meta_actions import (
    MetaAction,
    MetaActionLibrary,
    build_feature_vector,
)


@dataclass
class AgentConfig:
    learner_type: str = "linucb"
    alpha_ucb: float = 1.0
    regularization: float = 1.0
    hidden_dim: int = 32
    learning_rate: float = 1e-2
    epsilon: float = 0.1
    l2: float = 1e-4


class ContextualBanditAgent:
    """Coordinates meta-action selection and learner updates."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.library = MetaActionLibrary()
        self.feature_dim = 10
        self.learner_type = config.learner_type.lower()
        action_ids = [action.action_id for action in self.library.actions]
        if self.learner_type == "linucb":
            learner_config = LinUCBConfig(
                feature_dim=self.feature_dim,
                alpha_ucb=config.alpha_ucb,
                regularization=config.regularization,
            )
            self.learner = LinUCB(action_ids, learner_config)
        elif self.learner_type == "neural":
            learner_config = NeuralBanditConfig(
                feature_dim=self.feature_dim,
                hidden_dim=config.hidden_dim,
                learning_rate=config.learning_rate,
                epsilon=config.epsilon,
                l2=config.l2,
            )
            self.learner = NeuralBandit(action_ids, learner_config)
        else:
            raise ValueError(f"Unknown learner type: {config.learner_type}")
        self._latest_round: List[Tuple[int, np.ndarray]] = []

    def list_available_actions(self) -> List[MetaAction]:
        return self.library.list_actions()

    def select_actions(
        self,
        contexts: Iterable[Dict[str, float]],
    ) -> Tuple[List[MetaAction], List[int], List[np.ndarray]]:
        contexts = list(contexts)
        if not contexts:
            raise ValueError("No contexts provided to select_actions")
        chosen_actions: List[MetaAction] = []
        chosen_ids: List[int] = []
        features: List[np.ndarray] = []
        self._latest_round = []

        for context in contexts:
            candidates = []
            for action in self.library.actions:
                feature_vec = build_feature_vector(context, action)
                candidates.append((action.action_id, feature_vec))
            action_id = self.learner.select_action(candidates)
            action = self.library.get_action(action_id)
            feature = dict(candidates)[action_id]
            chosen_actions.append(action)
            chosen_ids.append(action_id)
            features.append(feature)
            self._latest_round.append((action_id, feature))

        return chosen_actions, chosen_ids, features

    def update(self, rewards: Iterable[float]) -> None:
        rewards = list(rewards)
        if len(rewards) != len(self._latest_round):
            raise ValueError("Reward length mismatch on update")
        for (action_id, feature), reward in zip(self._latest_round, rewards):
            self.learner.update(action_id, feature, float(reward))

    def save_model(
        self,
        directory: str,
        prefix: str,
        metadata: Dict[str, str],
    ) -> str:
        timestamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%SZ")
        filename = f"{prefix}_{self.learner_type}_{timestamp}"
        os.makedirs(directory, exist_ok=True)
        path = self.learner.save(directory, filename, metadata)
        return path

    @classmethod
    def from_saved_model(
        cls,
        model_path: str,
        config: AgentConfig,
        n_players: Optional[int] = None,
    ) -> "ContextualBanditAgent":
        agent = cls(n_players=n_players, config=config)
        if agent.learner_type == "linucb":
            agent.learner = LinUCB.load(model_path)
        elif agent.learner_type == "neural":
            action_ids = [action.action_id for action in agent.library.actions]
            agent.learner = NeuralBandit.load(model_path, action_ids)
        else:
            raise ValueError(f"Unknown learner type: {config.learner_type}")
        return agent


__all__ = ["ContextualBanditAgent", "AgentConfig"]
