#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Minimal tabular Q-learning agent for repricer rule selection."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from agents.repricer_meta_actions import MetaAction, MetaActionLibrary


class NPlayerSimpleQLearningRuleAgent:
    """Lightweight tabular Q-learner (no replay, no batching).

    The agent keeps a simple dictionary-backed Q-table keyed by a rounded state
    vector. Each call to ``store_transition`` performs a single Q-learning
    update with the provided transition.
    """

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        learning_rate: float,
        discount_rate: float,
        epsilon_omega: float,
        time: int = 0,
        *,
        price_grid: Sequence[float],
        action_library: Optional[MetaActionLibrary] = None,
        allowed_action_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self.alpha = float(learning_rate)
        self.discount = float(discount_rate)
        self.epsilon_omega = float(epsilon_omega)
        self.state_dim = int(state_dim)
        self.timestep = int(time)

        self.library = action_library or MetaActionLibrary()
        library_actions = self.library.list_actions()
        library_ids = sorted(action.action_id for action in library_actions)
        self.action_dim = int(action_dim)
        if self.action_dim != len(library_actions):
            raise ValueError(
                "action_dim does not match number of meta actions in library: "
                f"{self.action_dim} vs {len(library_actions)}"
            )
        if library_ids != list(range(self.action_dim)):
            raise ValueError(
                "MetaActionLibrary must expose consecutive identifiers starting from 0"
        )

        self._id_to_action = {action.action_id: action for action in library_actions}
        grid_arr = np.asarray(price_grid, dtype=np.float32).flatten()
        if grid_arr.size == 0:
            raise ValueError("price_grid must contain at least one value")
        self.price_grid = grid_arr
        if allowed_action_ids is None:
            valid_ids = np.arange(self.action_dim, dtype=int)
        else:
            arr_ids = np.asarray(allowed_action_ids, dtype=int)
            if arr_ids.size == 0:
                raise ValueError("allowed_action_ids must contain at least one action id")
            if np.any(arr_ids < 0) or np.any(arr_ids >= self.action_dim):
                raise ValueError("allowed_action_ids contain out-of-range identifiers")
            valid_ids = np.unique(arr_ids)
        self._valid_action_ids = valid_ids
        self._valid_action_mask = np.zeros(self.action_dim, dtype=bool)
        self._valid_action_mask[self._valid_action_ids] = True

        self.q_table: Dict[Tuple[float, ...], np.ndarray] = {}
        self.training_steps = 0
        self.last_action_id: Optional[int] = None
        self.latest_error: Optional[float] = None

    @property
    def epsilon(self) -> float:
        """Current exploration probability following an exponential schedule."""

        return float(np.exp(-self.timestep * self.epsilon_omega))

    def _snap_to_grid(self, arr: np.ndarray) -> np.ndarray:
        """Snap a state vector to the closest values on the provided price grid."""

        diff = np.abs(arr.reshape(-1, 1) - self.price_grid.reshape(1, -1))
        nearest_idx = np.argmin(diff, axis=1)
        snapped = self.price_grid[nearest_idx]
        return snapped.reshape(arr.shape)

    def _state_to_key(self, state: Sequence[float]) -> Tuple[float, ...]:
        """Map a state onto the nearest grid points and flatten to a hashable key."""

        arr = np.asarray(state, dtype=np.float32).flatten()
        if arr.size != self.state_dim:
            raise ValueError(f"Expected state_dim={self.state_dim}, got {arr.size}")
        arr = self._snap_to_grid(arr)
        return tuple(arr.tolist())

    def _row(self, key: Tuple[float, ...]) -> np.ndarray:
        """Return (and create) the Q-vector for a given state key."""

        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_dim, dtype=np.float32)
        return self.q_table[key]

    def take_action(
        self,
        observation: Sequence[float],
        available_action_ids: Optional[Sequence[int]] = None,
        *,
        greedy: bool = False,
    ) -> MetaAction:
        """Select a meta-action using an epsilon-greedy policy."""

        state = np.asarray(observation, dtype=np.float32)
        if state.ndim == 1:
            state = state.reshape(1, -1)
        if state.shape[0] != 1:
            raise ValueError(
                "Observation must be a 1D array or a single-row 2D array; "
                f"received shape {state.shape}"
            )

        action_ids = (
            np.asarray(available_action_ids, dtype=int)
            if available_action_ids is not None
            else self._valid_action_ids
        )

        epsilon = 0.0 if greedy else self.epsilon

        state_key = self._state_to_key(state[0])
        q_values = self._row(state_key)

        if np.random.rand() < epsilon:
            action_id = int(np.random.choice(action_ids))
        else:
            masked_q = np.full(self.action_dim, -np.inf, dtype=np.float32)
            masked_q[action_ids] = q_values[action_ids]
            action_id = int(np.argmax(masked_q))

        self.last_action_id = action_id
        self.timestep += 1
        return self._id_to_action[action_id]

    def _update_single(
        self,
        state_key: Tuple[float, ...],
        action_id: int,
        reward: float,
        next_key: Tuple[float, ...],
        done: bool,
    ) -> float:
        """Run one tabular Q-learning update and return the TD error."""

        current_row = self._row(state_key)
        next_row = self._row(next_key)
        masked_next = np.where(self._valid_action_mask, next_row, -np.inf)
        best_next = np.max(masked_next) if np.any(self._valid_action_mask) else 0.0
        if not np.isfinite(best_next):
            best_next = 0.0
        target = reward if done else reward + self.discount * best_next
        td_error = target - current_row[action_id]
        current_row[action_id] += self.alpha * td_error
        return float(td_error)

    def store_transition(
        self,
        state: Sequence[float],
        action_id: int,
        reward: float,
        next_state: Sequence[float],
        done: bool,
    ) -> float:
        """Perform an immediate Q-learning update using the supplied transition."""

        if action_id not in self._valid_action_ids:
            raise ValueError(f"Attempted to store transition with disallowed action id {action_id}")
        state_key = self._state_to_key(state)
        next_key = self._state_to_key(next_state)
        td_error = self._update_single(
            state_key,
            int(action_id),
            float(reward),
            next_key,
            bool(done),
        )
        self.training_steps += 1
        self.latest_error = abs(td_error)
        return td_error

    def train_step(self) -> None:
        """Kept for API compatibility; updates happen in store_transition."""
        return None

    def reset_time(self, timestep: int = 0) -> None:
        """Reset the internal timestep counter controlling the epsilon schedule."""

        self.timestep = int(timestep)


__all__ = ["NPlayerSimpleQLearningRuleAgent"]
