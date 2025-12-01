#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Simple neural contextual bandit with epsilon-greedy exploration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass
class NeuralBanditConfig:
    feature_dim: int
    hidden_dim: int = 32
    learning_rate: float = 1e-2
    epsilon: float = 0.1
    l2: float = 1e-4


class NeuralBandit:
    """ReLU MLP with mean-squared updates and epsilon-greedy policy."""

    def __init__(self, action_ids: Iterable[int], config: NeuralBanditConfig) -> None:
        self.config = config
        self.feature_dim = config.feature_dim
        self.hidden_dim = config.hidden_dim
        self.lr = config.learning_rate
        self.epsilon = config.epsilon
        self.l2 = config.l2
        self.rng = np.random.default_rng()
        self._init_parameters()
        self.action_ids = list(action_ids)

    def _init_parameters(self) -> None:
        limit1 = np.sqrt(6.0 / (self.feature_dim + self.hidden_dim))
        limit2 = np.sqrt(6.0 / (self.hidden_dim + 1))
        self.W1 = self.rng.uniform(-limit1, limit1, size=(self.hidden_dim, self.feature_dim))
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = self.rng.uniform(-limit2, limit2, size=(self.hidden_dim,))
        self.b2 = 0.0

    def _forward(self, feature: np.ndarray) -> Tuple[np.ndarray, float]:
        z1 = self.W1 @ feature + self.b1
        h1 = np.maximum(z1, 0.0)
        output = float(self.W2 @ h1 + self.b2)
        return h1, output

    def predict(self, feature: np.ndarray) -> float:
        _, out = self._forward(feature)
        return out

    def select_action(self, candidates: Iterable[Tuple[int, np.ndarray]]) -> int:
        candidate_list = list(candidates)
        if not candidate_list:
            raise ValueError("No actions provided to select_action")
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice([aid for aid, _ in candidate_list]))
        best_action = None
        best_value = -np.inf
        for action_id, feature in candidate_list:
            value = self.predict(feature)
            if value > best_value:
                best_value = value
                best_action = action_id
        if best_action is None:
            raise ValueError("No actions provided to select_action")
        return best_action

    def update(self, action_id: int, feature: np.ndarray, reward: float) -> None:
        h1, pred = self._forward(feature)
        error = pred - reward
        grad_w2 = error * h1 + self.l2 * self.W2
        grad_b2 = error
        relu_grad = (h1 > 0).astype(float)
        grad_hidden = error * self.W2 * relu_grad
        grad_w1 = np.outer(grad_hidden, feature) + self.l2 * self.W1
        grad_b1 = grad_hidden

        self.W1 -= self.lr * grad_w1
        self.b1 -= self.lr * grad_b1
        self.W2 -= self.lr * grad_w2
        self.b2 -= self.lr * grad_b2

    def save(self, directory: str, filename: str, metadata: Dict[str, str]) -> str:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{filename}.npz")
        meta_path = os.path.join(directory, f"{filename}.json")
        np.savez(
            path,
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            learning_rate=self.lr,
            epsilon=self.epsilon,
            l2=self.l2,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
        )
        with open(meta_path, "w", encoding="utf-8") as meta_file:
            json.dump({"metadata": metadata}, meta_file, indent=2)
        return path

    @classmethod
    def load(cls, filepath: str, action_ids: Iterable[int]) -> "NeuralBandit":
        data = np.load(filepath, allow_pickle=False)
        config = NeuralBanditConfig(
            feature_dim=int(data["feature_dim"]),
            hidden_dim=int(data["hidden_dim"]),
            learning_rate=float(data["learning_rate"]),
            epsilon=float(data["epsilon"]),
            l2=float(data["l2"]),
        )
        learner = cls(action_ids, config)
        learner.W1 = data["W1"]
        learner.b1 = data["b1"]
        learner.W2 = data["W2"]
        learner.b2 = float(data["b2"])
        return learner


__all__ = ["NeuralBandit", "NeuralBanditConfig"]
