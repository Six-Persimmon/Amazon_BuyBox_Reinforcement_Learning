#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Deep Q-learning agent that selects repricer meta-actions for N-player games."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from agents.dqn import Qnet, ReplayBuffer
from agents.repricer_meta_actions import MetaAction, MetaActionLibrary


@dataclass
class Transition:
    """Lightweight container used when exporting replay contents."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class NPlayerQLearningRuleAgent:
    """DQN-based agent that treats repricer rules as discrete actions.

    The agent consumes low-dimensional market statistics (e.g. previous episode
    averages) and outputs the identifier of a meta-action from
    :class:`MetaActionLibrary`.  Experience is stored in an internal replay
    buffer and the target network is periodically synchronised with the online
    network.
    """

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int,
        state_dim: int,
        learning_rate: float,
        discount_rate: float,
        epsilon_omega: float,
        target_update: int,
        device: str,
        cost: float = 0.0,
        time: int = 0,
        *,
        replay_capacity: int = 100_000,
        batch_size: int = 64,
        action_library: Optional[MetaActionLibrary] = None,
        allowed_action_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self.device = torch.device(device)
        self.discount = float(discount_rate)
        self.learning_rate = float(learning_rate)
        self.epsilon_omega = float(epsilon_omega)
        self.target_update = int(target_update)
        self.batch_size = int(batch_size)
        self.cost = float(cost)
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
                "MetaActionLibrary must expose consecutive identifiers "
                "starting from 0 for DQN compatibility"
            )

        self._id_to_action = {action.action_id: action for action in library_actions}
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

        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(self.device)
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()  # target network is not trained directly
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

        self.replay_buffer = ReplayBuffer(replay_capacity)

        self.training_steps = 0
        self.last_action_id: Optional[int] = None
        self.latest_loss: Optional[float] = None

    @property
    def epsilon(self) -> float:
        """Current exploration probability following an exponential schedule."""

        return float(np.exp(-self.timestep * self.epsilon_omega))

    def take_action(
        self,
        observation: Sequence[float],
        available_action_ids: Optional[Sequence[int]] = None,
        *,
        greedy: bool = False,
        epsilon_override: Optional[float] = None,
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
        if np.any(~np.isin(action_ids, self._valid_action_ids)):
            raise ValueError("available_action_ids must be a subset of allowed action ids")
        if action_ids.size == 0:
            raise ValueError("No available actions provided to take_action")

        epsilon = 0.0 if greedy else self.epsilon
        if epsilon_override is not None:
            epsilon = float(epsilon_override)

        # epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action_id = int(np.random.choice(action_ids))
        else:
            with torch.no_grad():
                tensor_state = torch.from_numpy(state).to(self.device)
                q_values = self.q_net(tensor_state)[0].detach().cpu().numpy()
            masked_q = np.full(self.action_dim, -np.inf, dtype=np.float32)
            masked_q[action_ids] = q_values[action_ids] # mask out unavailable actions
            action_id = int(np.argmax(masked_q))

        self.last_action_id = action_id
        self.timestep += 1
        return self._id_to_action[action_id]

    def store_transition(
        self,
        state: Sequence[float],
        action_id: int,
        reward: float,
        next_state: Sequence[float],
        done: bool,
    ) -> None:
        """Append a transition to the replay buffer."""

        state_arr = np.asarray(state, dtype=np.float32)
        next_state_arr = np.asarray(next_state, dtype=np.float32)
        if action_id not in self._valid_action_ids:
            raise ValueError(f"Attempted to store transition with disallowed action id {action_id}")
        self.replay_buffer.add(state_arr, int(action_id), float(reward), next_state_arr, bool(done))

    def sample_transitions(self, batch_size: Optional[int] = None) -> Iterable[Transition]:
        """Yield a batch of transitions without applying a gradient step."""

        batch_size = batch_size or self.batch_size
        if self.replay_buffer.size() < batch_size:
            return []
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        transitions = []
        for i in range(batch_size):
            transitions.append(
                Transition(
                    state=np.asarray(states[i], dtype=np.float32),
                    action=int(actions[i]),
                    reward=float(rewards[i]),
                    next_state=np.asarray(next_states[i], dtype=np.float32),
                    done=bool(dones[i]),
                )
            )
        return transitions

    def train_step(self, batch_size: Optional[int] = None) -> Optional[float]:
        """Run one optimisation step if enough replay samples are available."""

        batch_size = batch_size or self.batch_size
        if self.replay_buffer.size() < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device).view(-1, 1)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).view(-1, 1)
        next_states_tensor = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device).view(-1, 1)
        valid_mask = torch.as_tensor(self._valid_action_mask, dtype=torch.bool, device=self.device)
        q_values = self.q_net(states_tensor).gather(1, actions_tensor)
        with torch.no_grad():
            next_q_raw = self.target_q_net(next_states_tensor)
            masked_next_q = next_q_raw.masked_fill(~valid_mask, float("-inf"))
            next_q_values = masked_next_q.max(dim=1, keepdim=True)[0]
            targets = rewards_tensor + self.discount * (1.0 - dones_tensor) * next_q_values

        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=5.0) # gradient clipping for stable training. Not sure if necessary but I put it here.
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.target_update == 0:
            self.sync_target_network()

        loss_value = float(loss.item())
        self.latest_loss = loss_value
        return loss_value

    def sync_target_network(self) -> None:
        """Copy parameters from the online network to the target network."""

        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def reset_time(self, timestep: int = 0) -> None:
        """Reset the internal timestep counter controlling the epsilon schedule."""

        self.timestep = int(timestep)

    def to(self, device: str) -> None:
        """Move the underlying networks to a new device in-place."""

        self.device = torch.device(device)
        self.q_net.to(self.device)
        self.target_q_net.to(self.device)


__all__ = [
    "NPlayerQLearningRuleAgent",
    "Transition",
    "reduced_state_to_index",
    "index_to_reduced_state",
    "extract_reduced_state",
]
