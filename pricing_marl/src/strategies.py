"""
strategies.py

Defines the rule-based strategies agents can choose from.
Each function takes:
    - my_idx: The agent's current price index
    - others_min_idx: The lowest price index among competitors
    - n_grids: Total size of the price grid
Returns:
    - The next price index
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
    else:
        return max(0, others_min_idx - 1)

def rule_match_with_reset(my_idx, others_min_idx, n_grids):
    if others_min_idx == 0:
        return n_grids - 1
    else:
        return others_min_idx


# --- Define Integer IDs ---
ACT_UNDERCUT      = 0
ACT_MATCH         = 1
ACT_ABOVE         = 2
ACT_UNDER_RESET   = 3
ACT_MATCH_RESET   = 4

# ID -> Function (Used by Environment/Simulation)
ID_TO_FUNC = {
    ACT_UNDERCUT:    rule_undercut,
    ACT_MATCH:       rule_match,
    ACT_ABOVE:       rule_price_above,
    ACT_UNDER_RESET: rule_undercut_with_reset,
    ACT_MATCH_RESET: rule_match_with_reset
}

# ID -> Readable Name (Used for Plotting/Logs)
ID_TO_NAME = {
    ACT_UNDERCUT:    "Undercut",
    ACT_MATCH:       "Match",
    ACT_ABOVE:       "Above",
    ACT_UNDER_RESET: "Under+Reset",
    ACT_MATCH_RESET: "Match+Reset"
}

def get_strategy_function(action_id: int):
    """Retrieve function by Integer ID."""
    if action_id not in ID_TO_FUNC:
        raise ValueError(f"Action ID {action_id} not defined.")
    return ID_TO_FUNC[action_id]

def get_strategy_name(action_id: int):
    """Retrieve name by Integer ID."""
    return ID_TO_NAME.get(action_id, f"Unknown({action_id})")