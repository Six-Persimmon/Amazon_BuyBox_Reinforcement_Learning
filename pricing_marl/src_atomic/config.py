"""
Configuration for the atomic-time simulation engine (exp07 / exp08).

Unlike src.config.Config, experiments built on this engine do NOT use the
inner-loop lookup table and do NOT collapse sellers to the lowest price
between episodes. The full price vector carries over across decision points.

Two modes:
  - "sync_carryover": all sellers still revise rules synchronously every K
    periods (exp07). Only difference from baseline: no state collapse.
  - "async_poisson": each seller has its own stochastic revision clock with
    mean gap lambda_K (exp08). Naturally carry-over.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.strategies import ACT_UNDERCUT, ACT_MATCH, ACT_ABOVE, ACT_UNDER_RESET


@dataclass
class AtomicConfig:
    # --- System Dimensions ---
    num_sellers: int = 3
    num_grids: int = 10

    active_strategies: List[int] = field(default_factory=lambda: [
        ACT_UNDERCUT,
        ACT_MATCH,
        ACT_ABOVE,
        ACT_UNDER_RESET,
    ])

    # --- Economic Parameters (identical to baseline Config) ---
    a_val: float = 2.0
    c_val: float = 1.0
    a0: float = 0.0
    mu: float = 0.25
    xi: float = 0.1

    # --- Reinforcement Learning ---
    alpha: float = 0.15
    gamma: float = 0.95
    beta: Optional[float] = None
    exploration_weights: Optional[List[float]] = None

    # --- Engine mode ---
    mode: str = "sync_carryover"  # "sync_carryover" | "async_poisson"

    # --- Sync mode (exp07): fixed synchronous block length ---
    K: int = 30
    max_episodes: int = 2_000_000
    converge_period: int = 10_000  # consecutive episodes with unchanged policy

    # --- Async mode (exp08): per-agent revision clocks ---
    lambda_K: int = 30
    gap_distribution: str = "poisson"  # "poisson": gap ~ max(1, Poisson(lambda_K))
                                       # "geometric": revise each period w.p. 1/lambda_K
    # Atomic-time analogs of the baseline caps (baseline episodes x K=30):
    max_atomic_periods: int = 60_000_000       # ~ 2M episodes * 30
    converge_atomic_periods: int = 300_000     # ~ 10k episodes * 30

    # --- Evaluation ---
    eval_H: int = 10_000  # atomic periods

    save_training_data: bool = False

    # --- Paths ---
    base_dir: Path = field(init=False)
    data_dir: Path = field(init=False)
    lookup_dir: Path = field(init=False)
    results_dir: Path = field(init=False)

    def __post_init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.data_dir = self.base_dir / "data"
        self.lookup_dir = self.data_dir / "lookup_tables"
        self.results_dir = self.data_dir / "results"

        self.lookup_dir.mkdir(parents=True, exist_ok=True)

        if self.beta is None:
            self.beta = 1e-5

        if self.mode not in ("sync_carryover", "async_poisson"):
            raise ValueError(f"Unknown mode: {self.mode}")
        if self.gap_distribution not in ("poisson", "geometric"):
            raise ValueError(f"Unknown gap_distribution: {self.gap_distribution}")

    @property
    def num_actions(self) -> int:
        return len(self.active_strategies)

    def get_strategy_string(self) -> str:
        return "_".join(map(str, self.active_strategies))

    @property
    def block_K(self) -> int:
        """Block length used for the heuristic Q-init lookup table."""
        return self.K if self.mode == "sync_carryover" else self.lambda_K
