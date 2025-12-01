#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Meta-action definitions for contextual bandit repricers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

BASE_RULES = ("undercut", "match", "above")


@dataclass(frozen=True)
class MetaAction:
    action_id: int
    name: str
    base_rule: str
    reset_when_below_cost: bool
    raise_if_below_min: bool
    is_static: bool

    def feature_flags(self) -> Dict[str, int]:
        flags = {
            "no_repricer": int(self.is_static),
            "undercut": int(self.base_rule == "undercut"),
            "match": int(self.base_rule == "match"),
            "above": int(self.base_rule == "above"),
            "reset": int(self.reset_when_below_cost),
            "raise": int(self.raise_if_below_min),
        }
        return flags


class MetaActionExecutor:
    """Maintains per-agent state for a selected meta action.
    Input: MetaAction, price grid, marginal cost of the current agent
    Output: next price index given the competitor minimum price index
    """

    def __init__(
        self,
        action: MetaAction,
        price_grid: np.ndarray,
        marginal_cost: float,
        current_index: int = -1,
    ) -> None:
        self.action = action
        self.price_grid = price_grid
        self.marginal_cost = marginal_cost
        self.marginal_cost_idx = int(np.argmin(np.abs(price_grid - marginal_cost)))
        self.current_index = current_index if current_index >= 0 else self._initial_index()
        self.raise_on = 0 # counters to ensure raise/reset only trigger in the second consecutive time condition is met. i.e. raise_on = 2
        self.reset_on = 0

    def _initial_index(self) -> int:
        "Pick a random initial price above marginal cost from uniform distribution in the price grid."
        lower = max(self.marginal_cost, float(self.price_grid[0]))
        price = np.random.uniform(lower, float(self.price_grid[-1]))
        idx = int(np.argmin(np.abs(self.price_grid - price)))
        return max(idx, self.marginal_cost_idx)

    def next_price_index(self, competitor_min_idx: int) -> int:
        if self.action.is_static:
            # Static sellers keep the initially sampled price for the full episode.
            return self.current_index

        next_idx = self.current_index
        grid_last = len(self.price_grid) - 1

        if self.action.base_rule == "undercut": # undercut will not go below marginal cost
            next_idx = max(competitor_min_idx - 1, self.marginal_cost_idx)
        elif self.action.base_rule == "match":
            next_idx = competitor_min_idx
        elif self.action.base_rule == "above":
            next_idx = min(competitor_min_idx + 1, grid_last)

        # Initialize raise_idx and reset_idx
        raise_idx = None
        reset_idx = None

        if self.action.raise_if_below_min:
            if self.current_index < competitor_min_idx:
                raise_idx = min(next_idx + 1, grid_last)
                self.raise_on += 1
            else:
                self.raise_on = 0

        if self.action.reset_when_below_cost:
            if self.price_grid[next_idx] <= self.marginal_cost: # should be marginal cost index
                reset_idx = grid_last
                self.reset_on += 1
            else:
                self.reset_on = 0
        
        # Reset and raise are mutually exclusive. Reset has priority.
        # Reset/Raise does not apply the first time the price goes below cost. It applies to the consecutive second time.
        if self.reset_on > 1 and reset_idx is not None:
            next_idx = reset_idx
        elif self.raise_on > 1 and raise_idx is not None:
            next_idx = raise_idx

        self.current_index = next_idx
        return self.current_index


class MetaActionLibrary:
    """Enumerates available meta actions and helper lookups.

    Action identifiers are allocated as follows:

	•	0: no_repricer
	•	1: undercut
	•	2: undercut_raise
	•	3: undercut_reset
	•	4: undercut_reset_raise
	•	5: match
	•	6: match_raise
	•	7: match_reset
	•	8: match_reset_raise
	•	9: above
	•	10: above_raise
	•	11: above_reset
	•	12: above_reset_raise

    The ordering is deterministic, so persisted bandit parameters can be interpreted back
    to the corresponding rule combination without additional metadata.
    """

    def __init__(self) -> None:
        self.actions: List[MetaAction] = self._build_actions()
        self._id_to_action = {a.action_id: a for a in self.actions}

    @staticmethod
    def _build_actions() -> List[MetaAction]:
        actions: List[MetaAction] = []
        action_id = 0

        actions.append(
            MetaAction(
                action_id=action_id,
                name="no_repricer",
                base_rule="static",
                reset_when_below_cost=False,
                raise_if_below_min=False,
                is_static=True,
            )
        )
        action_id += 1

        for base_rule in BASE_RULES:
            for reset_flag in (False, True):
                for raise_flag in (False, True):
                    name_parts = [base_rule]
                    if reset_flag:
                        name_parts.append("reset")
                    if raise_flag:
                        name_parts.append("raise")
                    actions.append(
                        MetaAction(
                            action_id=action_id,
                            name="_".join(name_parts),
                            base_rule=base_rule,
                            reset_when_below_cost=reset_flag,
                            raise_if_below_min=raise_flag,
                            is_static=False,
                        )
                    )
                    action_id += 1
        return actions

    def get_action(self, action_id: int) -> MetaAction:
        return self._id_to_action[action_id]

    def list_actions(self) -> List[MetaAction]:
        return list(self.actions)

    # def feature_dim(self) -> int:
    #     return 10


def create_action_executor(
    action: MetaAction,
    price_grid: np.ndarray,
    marginal_cost: float,
    current_index: int = -1,
) -> MetaActionExecutor:
    return MetaActionExecutor(action, price_grid, marginal_cost, current_index)


def build_observation_vector(
    context: Dict[str, float],
    action: MetaAction,
) -> np.ndarray:
    values: List[float] = [
        float(context["N"]),
        float(context["MC"]),
        float(context["lambda"]),
        float(context["rho"]),
    ]
    flags = action.feature_flags()
    values.extend(
        [
            float(flags["no_repricer"]),
            float(flags["undercut"]),
            float(flags["match"]),
            float(flags["above"]),
            float(flags["reset"]),
            float(flags["raise"]),
        ]
    )
    return np.asarray(values, dtype=float)


__all__ = [
    "MetaAction",
    "MetaActionExecutor",
    "MetaActionLibrary",
    "create_action_executor",
    # "build_feature_vector",
]
