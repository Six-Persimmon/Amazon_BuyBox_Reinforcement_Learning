#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Compare average profits for different meta-action profiles in NPlayerLogitDemandEnv.

Scenarios (default in main):
1) All sellers: undercut + reset
2) All sellers: match + reset, starting from a high-price grid index (e.g. -2)

You can easily tweak:
- n_players
- inner periods
- which meta-action name to use
- initial price indices
"""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List

try:
    NOTEBOOK_DIR = Path(__file__).resolve().parent
except NameError:
    NOTEBOOK_DIR = Path.cwd()

ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == 'analysis' else NOTEBOOK_DIR
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from typing import Optional, Sequence, Dict, Any

import numpy as np

from env.NPlayerLogitDemandEnv import NPlayerLogitDemandEnv
from agents.repricer_meta_actions import MetaActionLibrary, MetaAction


def build_uniform_meta_actions(
    action_name: str,
    n_players: int,
    library: MetaActionLibrary,
) -> Sequence[MetaAction]:
    """
    Return a list of length n_players where every entry is the same MetaAction
    identified by `action_name` (e.g., 'undercut_reset', 'match_reset').
    """
    actions_by_name = {a.name: a for a in library.list_actions()}
    if action_name not in actions_by_name:
        raise ValueError(
            f"Unknown action_name={action_name!r}. "
            f"Available: {sorted(actions_by_name.keys())}"
        )
    chosen = actions_by_name[action_name]
    return [chosen] * n_players


def run_single_episode_profile(
    env: NPlayerLogitDemandEnv,
    action_name: str,
    periods: int = 50,
    initial_index: Optional[int] = None,
    marginal_costs: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Run a single episode in `env` where all sellers use the same meta-action
    (identified by `action_name`), optionally starting from a given price-grid index.

    Parameters
    ----------
    env : NPlayerLogitDemandEnv
        Environment instance (must already be constructed with desired params).
    action_name : str
        Name of the meta-action, e.g. 'undercut_reset', 'match_reset'.
    periods : int
        Number of periods in the inner loop (episode length).
    initial_index : Optional[int]
        If provided, all non-static sellers start from this price-grid index.
        You can pass negative indices (e.g. -2 for the second highest price).
    marginal_costs : Optional[Sequence[float]]
        Per-seller marginal costs. If None, uses env.base_mc for all sellers.

    Returns
    -------
    dict
        The full result dict returned by env.run_episode, plus an 'mean_profit'
        key giving average profit across sellers.
    """
    library = MetaActionLibrary()
    meta_actions = build_uniform_meta_actions(action_name, env.n_players, library)

    if marginal_costs is None:
        mc = np.full(env.n_players, float(env.base_mc), dtype=float)
    else:
        mc = np.asarray(marginal_costs, dtype=float)
        if mc.shape[0] != env.n_players:
            raise ValueError("marginal_costs length must equal env.n_players")

    if initial_index is not None:
        # Broadcast a single starting index to all sellers
        init_indices = [int(initial_index)] * env.n_players
    else:
        init_indices = None

    result = env.run_episode(
        meta_actions=meta_actions,
        marginal_costs=mc,
        periods=periods,
        return_histories=False,
        initial_price_indices=init_indices,
    )

    mean_profit = float(np.mean(result["average_profits"]))
    result["mean_profit"] = mean_profit
    return result


def run_deviation_profile(
    env: NPlayerLogitDemandEnv,
    others_action_name: str,
    dev_action_name: str,
    dev_player_idx: int = 0,
    periods: int = 50,
    initial_index: Optional[int] = None,
    marginal_costs: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Run a single episode where all sellers except one use `others_action_name`,
    and a single deviating seller (indexed by dev_player_idx) uses `dev_action_name`.

    Parameters
    ----------
    env : NPlayerLogitDemandEnv
        Environment instance (already constructed with desired params).
    others_action_name : str
        Name of the meta-action used by all non-deviating sellers.
    dev_action_name : str
        Name of the meta-action used by the deviating seller.
    dev_player_idx : int
        Index of the deviating seller (0-based).
    periods : int
        Number of periods in the inner loop.
    initial_index : Optional[int]
        If provided, all sellers start from this price-grid index.
        You can pass negative indices (e.g. -1 for the highest price).
    marginal_costs : Optional[Sequence[float]]
        Per-seller marginal costs. If None, uses env.base_mc for all sellers.

    Returns
    -------
    dict
        The full result dict returned by env.run_episode, plus:
        - 'mean_profit': mean profit across all sellers
        - 'deviator_profit': average profit of the deviating seller
    """
    library = MetaActionLibrary()
    actions_by_name = {a.name: a for a in library.list_actions()}

    if others_action_name not in actions_by_name:
        raise ValueError(
            f"Unknown others_action_name={others_action_name!r}. "
            f"Available: {sorted(actions_by_name.keys())}"
        )
    if dev_action_name not in actions_by_name:
        raise ValueError(
            f"Unknown dev_action_name={dev_action_name!r}. "
            f"Available: {sorted(actions_by_name.keys())}"
        )

    if not (0 <= dev_player_idx < env.n_players):
        raise ValueError(
            f"dev_player_idx={dev_player_idx} is out of range for n_players={env.n_players}"
        )

    others_action = actions_by_name[others_action_name]
    dev_action = actions_by_name[dev_action_name]

    meta_actions: List[MetaAction] = [others_action] * env.n_players
    meta_actions[dev_player_idx] = dev_action

    if marginal_costs is None:
        mc = np.full(env.n_players, float(env.base_mc), dtype=float)
    else:
        mc = np.asarray(marginal_costs, dtype=float)
        if mc.shape[0] != env.n_players:
            raise ValueError("marginal_costs length must equal env.n_players")

    if initial_index is not None:
        init_indices = [int(initial_index)] * env.n_players
    else:
        init_indices = None

    result = env.run_episode(
        meta_actions=meta_actions,
        marginal_costs=mc,
        periods=periods,
        return_histories=False,
        initial_price_indices=init_indices,
    )

    mean_profit = float(np.mean(result["average_profits"]))
    deviator_profit = float(result["average_profits"][dev_player_idx])

    result["mean_profit"] = mean_profit
    result["deviator_profit"] = deviator_profit
    return result


def main() -> None:
    # ------------------------------------------------------------------
    # Environment configuration
    # ------------------------------------------------------------------
    n_players = 5
    inner_periods = 50

    # Use your current canonical logit environment settings
    env = NPlayerLogitDemandEnv(
        n_players=n_players,
        grid_size=10,        # you can change this
        a0=0.0,
        a12=2.0,
        mu=0.04,
        repricer_cost=0.0,
        base_mc=1.0,
        fine_grid_points=200,
        fine_grid_span=5.0,
        # max_price=4,   # or set manually if you want to override the top grid price
    )

    print("Price grid:", env.prices)
    print("Nash price (approx):     ", env.nash_price_discrete())
    print("Monopoly price (approx): ", env.monopoly_price_discrete())
    print()

    # For convenience, compute grid index for "near monopoly", e.g. -1
    high_index = 9  # the highest index in a grid_size=10 setup
    high_price = env.prices[high_index]
    print(f"Using index {high_index} as 'near monopoly' starting grid: price={high_price:.4f}")
    print()

    # ------------------------------------------------------------------
    # Scenario 1: all sellers use undercut + reset
    # ------------------------------------------------------------------
    scenario1_name = "undercut_reset"
    res_undercut = run_single_episode_profile(
        env=env,
        action_name=scenario1_name,
        periods=inner_periods,
        initial_index=high_index,  # start from each executor's own default
        marginal_costs=None,  # uses env.base_mc for all
    )

    print("=== Scenario 1: all sellers use undercut + reset ===")
    print("Average profits per seller:", res_undercut["average_profits"])
    print("Mean profit across sellers:", res_undercut["mean_profit"])
    print()

    # ------------------------------------------------------------------
    # Scenario 2: all sellers use match + reset,
    # starting from a high-price grid index (e.g., -2)
    # ------------------------------------------------------------------
    scenario2_name = "match_reset"
    res_match = run_single_episode_profile(
        env=env,
        action_name=scenario2_name,
        periods=inner_periods,
        initial_index=high_index,  # start from near-monopoly price
        marginal_costs=None,
    )

    print("=== Scenario 2: all sellers use match + reset (start from high index) ===")
    print("Average profits per seller:", res_match["average_profits"])
    print("Mean profit across sellers:", res_match["mean_profit"])
    print()

    # ------------------------------------------------------------------
    # Deviation scenarios: one seller deviates while others coordinate
    # ------------------------------------------------------------------
    print("=== Deviation analysis ===")
    dev_player = 0  # index of the deviating seller

    # Case A: Others use match + reset, deviator uses undercut + reset
    dev_case_A = run_deviation_profile(
        env=env,
        others_action_name="match_reset",
        dev_action_name="undercut_reset",
        dev_player_idx=dev_player,
        periods=inner_periods,
        initial_index=high_index,
        marginal_costs=None,
    )

    print("Case A: others = match_reset, deviator = undercut_reset")
    print("  All sellers' average profits:", dev_case_A["average_profits"])
    print("  Mean profit across sellers:  ", dev_case_A["mean_profit"])
    print("  Deviator's average profit:   ", dev_case_A["deviator_profit"])
    print()

    # Case B: Others use match + reset, deviator also uses match + reset
    dev_case_B = run_deviation_profile(
        env=env,
        others_action_name="match_reset",
        dev_action_name="match_reset",
        dev_player_idx=dev_player,
        periods=inner_periods,
        initial_index=high_index,
        marginal_costs=None,
    )

    print("Case B: others = match_reset, deviator = match_reset")
    print("  All sellers' average profits:", dev_case_B["average_profits"])
    print("  Mean profit across sellers:  ", dev_case_B["mean_profit"])
    print("  Deviator's average profit:   ", dev_case_B["deviator_profit"])
    print()

    # You can add more cases here, e.g. others=undercut_reset, deviator=match_reset, etc.


if __name__ == "__main__":
    main()