"""
Pricing rule definitions for exp03.

This is intentionally copied into src_ext_K_action so the exp03 extension does
not depend on edits to the original exp02 source package.
"""


def rule_undercut(my_idx, others_min_idx, n_grids):
    return max(0, others_min_idx - 1)


def rule_match(my_idx, others_min_idx, n_grids):
    return others_min_idx


def rule_price_above(my_idx, others_min_idx, n_grids):
    return min(n_grids - 1, others_min_idx + 1)


def rule_undercut_with_reset(my_idx, others_min_idx, n_grids):
    if others_min_idx == 0:
        return n_grids - 1
    return max(0, others_min_idx - 1)


def rule_match_with_reset(my_idx, others_min_idx, n_grids):
    if others_min_idx == 0:
        return n_grids - 1
    return others_min_idx


ACT_UNDERCUT = 0
ACT_MATCH = 1
ACT_ABOVE = 2
ACT_UNDER_RESET = 3
ACT_MATCH_RESET = 4


ID_TO_FUNC = {
    ACT_UNDERCUT: rule_undercut,
    ACT_MATCH: rule_match,
    ACT_ABOVE: rule_price_above,
    ACT_UNDER_RESET: rule_undercut_with_reset,
    ACT_MATCH_RESET: rule_match_with_reset,
}


ID_TO_NAME = {
    ACT_UNDERCUT: "Undercut",
    ACT_MATCH: "Match",
    ACT_ABOVE: "Above",
    ACT_UNDER_RESET: "Under+Reset",
    ACT_MATCH_RESET: "Match+Reset",
}


def get_strategy_function(action_id: int):
    if action_id not in ID_TO_FUNC:
        raise ValueError(f"Action ID {action_id} not defined.")
    return ID_TO_FUNC[action_id]


def get_strategy_name(action_id: int):
    return ID_TO_NAME.get(action_id, f"Unknown({action_id})")

