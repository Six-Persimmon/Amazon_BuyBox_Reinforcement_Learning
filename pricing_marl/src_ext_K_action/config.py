from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .strategies import (
    ACT_ABOVE,
    ACT_MATCH,
    ACT_UNDER_RESET,
    ACT_UNDERCUT,
)


@dataclass
class KActionConfig:
    """
    Configuration for exp03 endogenous-K simulations.

    The market transition always uses base_K. Composite actions choose both a
    pricing rule and a commitment length K, expressed as an integer multiple of
    base_K.
    """

    # System dimensions
    num_sellers: int = 3
    num_grids: int = 10
    active_strategies: List[int] = field(
        default_factory=lambda: [
            ACT_UNDERCUT,
            ACT_MATCH,
            ACT_ABOVE,
            ACT_UNDER_RESET,
        ]
    )

    # Endogenous-K action settings
    base_K: int = 10
    k_choices: List[int] = field(default_factory=lambda: [10, 30, 60])

    # Economic parameters
    a_val: float = 2.0
    c_val: float = 1.0
    a0: float = 0.0
    mu: float = 0.02
    xi: float = 0.1

    # Reinforcement learning
    alpha: float = 0.15
    gamma: float = 0.95
    max_episodes: int = 2_000_000
    beta: Optional[float] = None
    converge_period: int = 10_000

    # Evaluation
    eval_H: int = 2_000
    save_training_data: bool = False

    # Paths
    base_dir: Path = field(init=False)
    data_dir: Path = field(init=False)
    lookup_dir: Path = field(init=False)
    results_dir: Path = field(init=False)

    # Composite action metadata
    action_map: List[Dict[str, int]] = field(init=False)

    def __post_init__(self):
        self.active_strategies = [int(x) for x in self.active_strategies]
        self.k_choices = [int(x) for x in self.k_choices]

        if self.num_sellers <= 1:
            raise ValueError("num_sellers must be at least 2.")
        if self.num_grids <= 1:
            raise ValueError("num_grids must be at least 2.")
        if not self.active_strategies:
            raise ValueError("active_strategies must not be empty.")
        if self.base_K <= 0:
            raise ValueError("base_K must be positive.")
        if not self.k_choices:
            raise ValueError("k_choices must not be empty.")
        for k_val in self.k_choices:
            if k_val <= 0:
                raise ValueError(f"k_choices must be positive, got {k_val}.")
            if k_val % self.base_K != 0:
                raise ValueError(
                    f"k_choice={k_val} is not a multiple of base_K={self.base_K}."
                )

        self.k_choices = list(dict.fromkeys(self.k_choices))
        self.action_map = self._build_action_map()

        self.base_dir = Path(__file__).resolve().parent.parent
        self.data_dir = self.base_dir / "data"
        self.lookup_dir = self.data_dir / "lookup_tables"
        self.results_dir = self.data_dir / "results_exp03"

        self.lookup_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        if self.beta is None:
            self.beta = 1e-5

    def _build_action_map(self) -> List[Dict[str, int]]:
        action_map: List[Dict[str, int]] = []
        action_idx = 0
        for rule_action_idx, action_id in enumerate(self.active_strategies):
            for k_choice in self.k_choices:
                action_map.append(
                    {
                        "composite_action_idx": action_idx,
                        "rule_action_idx": int(rule_action_idx),
                        "action_id": int(action_id),
                        "k_choice": int(k_choice),
                        "duration_blocks": int(k_choice // self.base_K),
                    }
                )
                action_idx += 1
        return action_map

    @property
    def num_rule_actions(self) -> int:
        return len(self.active_strategies)

    @property
    def num_actions(self) -> int:
        return len(self.action_map)

    def get_strategy_string(self) -> str:
        return "_".join(map(str, self.active_strategies))

    def get_composite_action(self, action_idx: int) -> Dict[str, int]:
        if action_idx < 0 or action_idx >= self.num_actions:
            raise IndexError(f"Composite action index out of range: {action_idx}")
        return self.action_map[action_idx]
