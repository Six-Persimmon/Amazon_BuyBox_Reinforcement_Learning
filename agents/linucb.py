#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Linear upper confidence bound learner."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass
class LinUCBConfig:
    feature_dim: int
    alpha_ucb: float = 1.0
    regularization: float = 1.0


class LinUCB:
    """Per-action LinUCB with shared feature space."""

    def __init__(self, action_ids: Iterable[int], config: LinUCBConfig) -> None:
        self.config = config
        self.feature_dim = config.feature_dim
        self.alpha = config.alpha_ucb
        self.regularization = config.regularization
        self._init_action_matrices(action_ids)

    def _init_action_matrices(self, action_ids: Iterable[int]) -> None:
        self.A: Dict[int, np.ndarray] = {}
        self.A_inv: Dict[int, np.ndarray] = {}
        self.b: Dict[int, np.ndarray] = {}
        identity = np.eye(self.feature_dim) * self.regularization
        for aid in action_ids:
            self.A[aid] = identity.copy()
            self.A_inv[aid] = np.linalg.inv(self.A[aid])
            self.b[aid] = np.zeros(self.feature_dim)

    def _ensure_action(self, action_id: int) -> None:
        if action_id in self.A:
            return
        identity = np.eye(self.feature_dim) * self.regularization
        self.A[action_id] = identity.copy()
        self.A_inv[action_id] = np.linalg.inv(self.A[action_id])
        self.b[action_id] = np.zeros(self.feature_dim)

    def score(self, action_id: int, feature: np.ndarray) -> Tuple[float, float, float]:
        self._ensure_action(action_id)
        theta = self.A_inv[action_id] @ self.b[action_id]
        mean = float(feature @ theta)
        exploration = self.alpha * float(np.sqrt(feature @ self.A_inv[action_id] @ feature))
        ucb = mean + exploration
        return mean, exploration, ucb

    def select_action(self, candidates: Iterable[Tuple[int, np.ndarray]]) -> int:
        best_action = None
        best_value = -np.inf
        for action_id, feature in candidates:
            _, _, value = self.score(action_id, feature)
            if value > best_value:
                best_value = value
                best_action = action_id
        if best_action is None:
            raise ValueError("No actions provided to select_action")
        return best_action

    def update(self, action_id: int, feature: np.ndarray, reward: float) -> None:
        self._ensure_action(action_id)
        x = feature.reshape(-1, 1)
        self.A[action_id] += x @ x.T
        self.b[action_id] += reward * feature
        self.A_inv[action_id] = np.linalg.inv(self.A[action_id])

    def save(self, directory: str, filename: str, metadata: Dict[str, str]) -> str:
        os.makedirs(directory, exist_ok=True)
        matrix_path = os.path.join(directory, f"{filename}.npz")
        meta_path = os.path.join(directory, f"{filename}.json")

        np.savez(
            matrix_path,
            feature_dim=self.feature_dim,
            alpha=self.alpha,
            regularization=self.regularization,
            action_ids=np.array(list(self.A.keys()), dtype=int),
            A=np.array([self.A[aid] for aid in self.A], dtype=float),
            b=np.array([self.b[aid] for aid in self.A], dtype=float),
        )

        with open(meta_path, "w", encoding="utf-8") as meta_file:
            payload = {"metadata": metadata}
            json.dump(payload, meta_file, indent=2)

        return matrix_path

    @classmethod
    def load(cls, filepath: str) -> "LinUCB":
        data = np.load(filepath, allow_pickle=False)
        feature_dim = int(data["feature_dim"]) if np.ndim(data["feature_dim"]) == 0 else int(data["feature_dim"][0])
        config = LinUCBConfig(
            feature_dim=feature_dim,
            alpha_ucb=float(data["alpha"]),
            regularization=float(data["regularization"]),
        )
        action_ids = [int(x) for x in data["action_ids"]]
        learner = cls(action_ids, config)
        matrices = data["A"]
        vectors = data["b"]
        for idx, action_id in enumerate(action_ids):
            learner.A[action_id] = matrices[idx]
            learner.A_inv[action_id] = np.linalg.inv(learner.A[action_id])
            learner.b[action_id] = vectors[idx]
        return learner


__all__ = ["LinUCB", "LinUCBConfig"]
