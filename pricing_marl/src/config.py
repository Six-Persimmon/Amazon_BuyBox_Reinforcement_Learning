from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
# Import constants
from src.strategies import ACT_UNDERCUT, ACT_MATCH, ACT_ABOVE, ACT_UNDER_RESET, ACT_MATCH_RESET

@dataclass
class Config:
    """
    Configuration object.
    """
    # --- System Dimensions ---
    num_sellers: int = 4
    num_grids: int = 10
    
    # Active Strategies (List of Integers)
    active_strategies: List[int] = field(default_factory=lambda: [
        ACT_UNDERCUT, 
        ACT_MATCH, 
        ACT_ABOVE, 
        ACT_UNDER_RESET, 
        ACT_MATCH_RESET
    ])

    # --- Economic Parameters ---
    a_val: float = 2.0
    c_val: float = 1.0
    a0: float = 0.0
    mu: float = 0.02
    xi: float = 0.1

    # --- Reinforcement Learning ---
    alpha: float = 0.15
    gamma: float = 0.95
    # T: int = 1_000_000
    max_episodes: int = 5_000_000
    beta: Optional[float] = None

    # Exploration sampling distribution over active_strategies (same order).
    # None = uniform (baseline behavior). Used by exp06 robustness check.
    exploration_weights: Optional[List[float]] = None
    
    # Convergence Threshold. Number of consecutive periods with unchanged policy.
    converge_period: int = 10_000

    # --- Lookup Table ---
    K: int = 50

    # --- Evaluation Phase Parameters ---
    eval_H : int = 2_000

    # --- whether to save training data. If False, only save Evaluation ---
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
        self.results_dir.mkdir(parents=True, exist_ok=True)

        if self.beta is None:
            self.beta = 1e-5  # Default small decay rate 10 / 1,000,000

    @property
    def num_actions(self) -> int:
        return len(self.active_strategies)

    def get_strategy_string(self) -> str: 
        """
        Returns a string representation of active strategy IDs in EXACT order.
        Example: "0_1_3" vs "1_0_3" will be different files.
        """
        # [FIX] Do NOT sort. Keep the order to ensure Action Index 0 maps to the same strategy 
        # as when the table was computed.
        return "_".join(map(str, self.active_strategies))