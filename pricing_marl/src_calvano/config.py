"""
Configuration for the exp09 Calvano-ladder engine.

Two orthogonal switches define the four K=1 rungs of the ladder; K adds the
fifth:

    state_mode  = "full_vector" | "market_min"
    action_mode = "price"       | "rule"

    C1 cal_full     full_vector + price   K=1   (Calvano-style baseline)
    C2 cal_smin     market_min  + price   K=1   (state reduction only)
    C3 cal_arule    full_vector + rule    K=1   (action reduction only)
    C4 cal_both     market_min  + rule    K=1   (both reductions)
    C5 cal_both_k30 market_min  + rule    K=30  (both + commitment)

Conventions inherited unchanged from src/ and src_atomic/:
  - the Q-learning observation "market minimum" is the minimum over ALL N
    sellers' previous-period prices, the seller's own price included
    (Extention_Plan.md decision 10);
  - the pricing rules key off `others_min`, the lowest COMPETITOR price
    (excluding own) -- a different object, and not affected by state_mode.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.strategies import ACT_ABOVE, ACT_MATCH, ACT_UNDERCUT

STATE_MODES = ("full_vector", "market_min")
ACTION_MODES = ("price", "rule")


@dataclass
class CalvanoConfig:
    # --- System Dimensions ---
    num_sellers: int = 3
    num_grids: int = 10

    # Only consulted when action_mode == "rule".
    active_strategies: List[int] = field(default_factory=lambda: [
        ACT_UNDERCUT,
        ACT_MATCH,
        ACT_ABOVE,
    ])

    # --- Ladder switches ---
    state_mode: str = "full_vector"
    action_mode: str = "price"

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

    # Greedy tie-breaking. "random" is the repo convention (and the
    # conservative choice here); "lowest" is Calvano's own rule and exists for
    # a faithfulness check. See Extention_Plan.md decision 16.
    tie_break: str = "random"

    # --- Commitment ---
    K: int = 1

    # --- Training budget (Calvano's own convergence criterion) ---
    max_episodes: int = 5_000_000
    converge_period: int = 100_000

    # --- Q-initialization ---
    # For state_mode="market_min" the reward depends on the full price vector,
    # which the observation does not pin down. "collapsed" evaluates the
    # representative vector (o, ..., o), reproducing
    # src.agent.calculate_heuristic_init_values exactly; "reachable_avg"
    # averages over the reachable vectors whose market minimum is o.
    init_state_rep: str = "collapsed"

    # How the opponent-marginalized reward is attributed across seller
    # positions when building the prior.
    #   "per_seller": Q0_i(s,a) = E_{a_-i}[R_i(s,a,a_-i)]/(1-gamma), evaluated
    #       separately for each seller position i. This is the formula as
    #       stated, and it is the correct one whenever the state can be
    #       asymmetric across sellers -- which happens only under
    #       state_mode="full_vector" with rule actions, where sellers at
    #       different positions in the price vector face different lowest
    #       COMPETITOR prices and hence different rule payoffs.
    #   "pooled": average the seller-specific priors into one row given to
    #       every seller (a deliberately symmetrized prior).
    # The two coincide exactly for price actions (the stage payoff does not
    # depend on the state) and for state_mode="market_min" with the collapsed
    # representative vector (o,...,o) (all sellers symmetric); they differ
    # only for the full_vector + rule combination.
    init_pooling: str = "per_seller"

    # --- Evaluation ---
    eval_H: int = 10_000  # atomic periods

    save_training_data: bool = False

    # --- Paths ---
    base_dir: Path = field(init=False)
    data_dir: Path = field(init=False)
    results_dir: Path = field(init=False)

    def __post_init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.data_dir = self.base_dir / "data"
        self.results_dir = self.data_dir / "results_exp09"

        if self.beta is None:
            self.beta = 1e-5

        if self.state_mode not in STATE_MODES:
            raise ValueError(
                f"Unknown state_mode: {self.state_mode!r} (expected one of {STATE_MODES})"
            )
        if self.action_mode not in ACTION_MODES:
            raise ValueError(
                f"Unknown action_mode: {self.action_mode!r} (expected one of {ACTION_MODES})"
            )
        if self.init_state_rep not in ("collapsed", "reachable_avg"):
            raise ValueError(f"Unknown init_state_rep: {self.init_state_rep!r}")
        if self.init_pooling not in ("per_seller", "pooled"):
            raise ValueError(f"Unknown init_pooling: {self.init_pooling!r}")
        if self.tie_break not in ("random", "lowest"):
            raise ValueError(f"Unknown tie_break: {self.tie_break!r}")
        if self.K < 1:
            raise ValueError(f"K must be >= 1, got {self.K}")
        if self.action_mode == "price" and self.K > 1:
            # Not an error -- a chosen price simply persists for K periods --
            # but it is degenerate (all K periods are identical), so flag it.
            print(
                f"[CalvanoConfig] Warning: action_mode='price' with K={self.K}; "
                "the K inner periods are identical by construction."
            )

    # ------------------------------------------------------------------
    # Derived dimensions
    # ------------------------------------------------------------------

    @property
    def num_actions(self) -> int:
        if self.action_mode == "price":
            return self.num_grids
        return len(self.active_strategies)

    @property
    def num_states(self) -> int:
        if self.state_mode == "full_vector":
            return self.num_grids ** self.num_sellers
        return self.num_grids

    def action_id(self, action_idx: int) -> int:
        """
        Externally reported action id, written to the `a_i` eval columns and
        the Q-snapshot `action_id` column: the price grid index in price mode,
        the strategy ID in rule mode.
        """
        if self.action_mode == "price":
            return int(action_idx)
        return int(self.active_strategies[action_idx])

    def get_strategy_string(self) -> str:
        return "_".join(map(str, self.active_strategies))

    @property
    def cell_tag(self) -> str:
        """Short tag used in batch directory names (see Extention_Plan.md 4.3)."""
        if self.action_mode == "price":
            return "cal_full" if self.state_mode == "full_vector" else "cal_smin"
        if self.state_mode == "full_vector":
            return "cal_arule"
        return "cal_both_k30" if self.K > 1 else "cal_both"
