import numpy as np
import itertools
from scipy.optimize import minimize

# ==========================================
# --- 1. SYSTEM CONFIGURATION ---
# ==========================================
num_sellers = 4  # <--- CHANGE THIS (2, 3, 4, 5, etc.)
num_grids = 10
T = 100
gamma = 0.95
# N_big_periods = 1000000
N_big_periods = 1000
alpha = 0.15

# CHANGE 1: Updated Beta as requested
beta = 10 / N_big_periods

# Parameters (Symmetric)
a_val = 2
c_val = 1
mu = 0.025
# mu = 0.02
a0 = 0

# xi = 0.1 # for price grid, follow Calvano et al. 2025

xi=0

'''
If xi = 0.1, use the rules below:
rule_undercut_with_reset
rule_match_with_reset
rule_price_above
We can sustain circle pricing with mu = 0.02 and T = 100 for 4 sellers. But not for mu = 0.025

Condition on we see circle: (N, mu, T, grid size)
'''

# ==========================================
# --- 2. PROFIT & GRID FUNCTIONS ---
# ==========================================
def get_demand_and_profit(prices):
    """
    Input: list or array of prices [p1, p2, ..., pN]
    Output: list of profits [pi1, pi2, ..., piN]
    """
    prices = np.array(prices)
    exps = np.exp((a_val - prices) / mu)
    exp0 = np.exp(a0 / mu)
    denom = np.sum(exps) + exp0

    demands = exps / denom
    profits = (prices - c_val) * demands
    return profits


# --- Grid Construction Helpers ---
def neg_joint_profit_Np(p_scalar_list):
    '''negative joint profit for N players with same price'''
    p = p_scalar_list[0]
    prices = [p] * num_sellers
    prof = get_demand_and_profit(prices)
    return -np.sum(prof)


def neg_own_profit_Np(p_own_list, p_comp):
    '''negative own profit for N players with same p_comp as competitors' price'''
    p_own = p_own_list[0]
    prices = [p_own] + [p_comp] * (num_sellers - 1)
    prof = get_demand_and_profit(prices)
    return -prof[0]


# 1. Find Monopoly Price: all sellers set same price to maximize joint profit
res_monopoly = minimize(neg_joint_profit_Np, x0=[1.5], bounds=[(1.0, 5.0)])
p_monopoly = res_monopoly.x[0]

# 2. Find Nash Price: p_nash is best response given n-1 competitors also set p_nash
p_nash = 1.1
for _ in range(10):
    res_br = minimize(neg_own_profit_Np, x0=[p_nash], args=(p_nash,), bounds=[(1.0, 5.0)])
    p_nash = res_br.x[0]

# 3. Create Grid
grid_low = p_nash - xi * (p_monopoly - p_nash)
grid_high = p_monopoly + xi * (p_monopoly - p_nash)
price_grid = np.linspace(grid_low, grid_high, num_grids)

print(f"Simulation Setup: {num_sellers} Sellers")
print(f"Price Grid: {price_grid[0]:.4f}, {price_grid[1]:.4f} ...{price_grid[-2]:.4f}, {price_grid[-1]:.4f}")
print(f"Nash Price: {p_nash:.4f}, Monopoly Price: {p_monopoly:.4f}")


# ==========================================
# --- 3. FLEXIBLE RULE DEFINITIONS ---
# ==========================================

def rule_undercut(my_idx, others_min_idx, n_grids):
    """Undercut the lowest competitor by 1 grid."""
    return max(0, others_min_idx - 1)


def rule_match(my_idx, others_min_idx, n_grids):
    """Match the lowest competitor."""
    return others_min_idx


def rule_price_above(my_idx, others_min_idx, n_grids):
    """Price 1 grid above the lowest competitor."""
    return min(n_grids - 1, others_min_idx + 1)

def rule_undercut_with_reset(my_idx, others_min_idx, n_grids):
    """Undercut, but if market is at bottom (0), jump to Monopoly (Max)."""
    if others_min_idx == 0:
        return n_grids - 1
    else:
        return max(0, others_min_idx - 1)
    
def rule_match_with_reset(my_idx, others_min_idx, n_grids):
    """Match, but if market is at bottom (0), jump to Monopoly (Max)."""
    if others_min_idx == 0:
        return n_grids - 1
    else:
        return others_min_idx


# Registry
STRATEGIES = [
    rule_undercut,
    rule_match,
    rule_price_above,
    rule_undercut_with_reset,  
    rule_match_with_reset,      # <--- Added new strategy here
]

# Create shorter names for cleaner logging
STRATEGY_NAMES = [func.__name__.replace("rule_", "") for func in STRATEGIES]
# Specifically shorten the new ones for display
STRATEGY_NAMES = [n.replace("undercut_with_reset", "Under+Reset") for n in STRATEGY_NAMES]
STRATEGY_NAMES = [n.replace("match_with_reset", "Match+Reset") for n in STRATEGY_NAMES]

num_actions = len(STRATEGIES)
print(f"Strategies Loaded: {STRATEGY_NAMES}")

# Helper: canonicalize state/action by sorting (price_idx, action_id) pairs.
def canonicalize_state_actions(state, actions):
    pairs = list(zip(state, actions))
    perm = sorted(range(len(pairs)), key=lambda i: (pairs[i][0], pairs[i][1]))
    canon_state = tuple(pairs[i][0] for i in perm)
    canon_actions = tuple(pairs[i][1] for i in perm)
    inv_perm = [0] * len(pairs)
    for canon_pos, orig_idx in enumerate(perm):
        inv_perm[orig_idx] = canon_pos
    return canon_state, canon_actions, perm, inv_perm

def canonicalize_state_only(state):
    return tuple(sorted(state))

# ==========================================
# --- 4. N-AGENT LOOKUP TABLE ---
# ==========================================
lookup_table = {}
action_combinations = list(itertools.product(range(num_actions), repeat=num_sellers)) # e.g. for 2 sellers and 3 actions: (0,0), (0,1), (0,2), (1,0), ...

state_space = list(itertools.combinations_with_replacement(range(num_grids), num_sellers)) # e.g. for 2 sellers and 3 grids: (0,0), (0,1), (0,2), (1,1), (1,2), (2,2). Notice (1, 2) and (2, 1) are the same canonical state.
print(f"Building Lookup Table with symmetry compression ({len(action_combinations)} actions, {len(state_space)} canonical states)...")

for start_state in state_space:
    for actions in action_combinations:
        canon_state, canon_actions, perm, inv_perm = canonicalize_state_actions(start_state, actions)
        key = tuple(list(canon_state) + list(canon_actions))
        if key in lookup_table:
            continue

        # --- Inner Simulation (T periods) ---
        current_price_indices = list(canon_state)
        cumulative_profits = np.zeros(num_sellers)

        for t in range(T):
            next_price_indices = []
            for i in range(num_sellers):
                my_prev = current_price_indices[i]
                others_indices = current_price_indices[:i] + current_price_indices[i + 1:]
                others_min = min(others_indices)

                # Apply strategy
                rule_func = STRATEGIES[canon_actions[i]]
                p_next = rule_func(my_prev, others_min, num_grids)
                next_price_indices.append(p_next)

            current_price_indices = next_price_indices
            current_prices_val = [price_grid[idx] for idx in current_price_indices]
            step_profits = get_demand_and_profit(current_prices_val)
            cumulative_profits += step_profits

        # Use average profit over T periods; keep final per-seller price index in canonical (sorted) order
        avg_profits = cumulative_profits / T
        end_state = canonicalize_state_only(tuple(current_price_indices))
        lookup_table[key] = (avg_profits, end_state)

print("Lookup Table Built.")

# Precompute groups of canonical states by their lowest price (for value iteration on reduced state)
low_to_states = {g: [] for g in range(num_grids)}
for st in state_space:
    low_to_states[min(st)].append(st)

# ==========================================
# --- 5. INITIALIZATION (Value Iteration) ---
# ==========================================
print("Initializing Q-tables via Value Iteration (Random Policy)...")

# Agent state is only the lowest price index (0 .. num_grids-1)
V = np.zeros(num_grids)
delta = float('inf')
theta = 1e-4

opponent_combinations = list(itertools.product(range(num_actions), repeat=num_sellers - 1))

while delta > theta:
    delta = 0
    V_new = np.zeros(num_grids)

    for low_idx in range(num_grids):
        candidate_states = low_to_states.get(low_idx, [])
        if not candidate_states:
            continue
        q_sum = 0
        for my_action in range(num_actions):
            total_reward = 0
            total_next_v = 0
            count = 0

            for state in candidate_states:
                for opp_actions in opponent_combinations:
                    actions_list = [my_action] + list(opp_actions)
                    canon_state, canon_actions, perm, inv_perm = canonicalize_state_actions(state, actions_list)
                    key = tuple(list(canon_state) + list(canon_actions))
                    payoffs_canon, next_state = lookup_table[key]
                    payoffs = [payoffs_canon[inv_perm[i]] for i in range(num_sellers)]

                    total_reward += payoffs[0]
                    total_next_v += V[min(next_state)]
                    count += 1

            avg_reward = total_reward / count
            avg_next_v = total_next_v / count
            q_val = avg_reward + gamma * avg_next_v
            q_sum += q_val

        V_new[low_idx] = q_sum / num_actions
        delta = max(delta, abs(V_new[low_idx] - V[low_idx]))
    V = V_new

Q_tables = [np.zeros((num_grids, num_actions)) for _ in range(num_sellers)]
opponent_combinations = list(itertools.product(range(num_actions), repeat=num_sellers - 1))

for low_idx in range(num_grids):
    candidate_states = low_to_states.get(low_idx, []) # get all canonical states with this lowest price index
    if not candidate_states:
        continue
    for a in range(num_actions):
        total_q_val = 0.0
        count = 0
        # loop over all canonical states with this lowest price index
        for state in candidate_states:
            for opp_actions in opponent_combinations:
                actions_list = [a] + list(opp_actions)

                # Canonicalize, so that we can lookup in the table
                canon_state, canon_actions, perm, inv_perm = canonicalize_state_actions(state, actions_list)
                key = tuple(list(canon_state) + list(canon_actions))
                # lookup the table and get payoffs + next canonical state
                payoffs_canon, next_canon_state = lookup_table[key]
                payoffs = [payoffs_canon[inv_perm[i]] for i in range(num_sellers)]
                r = payoffs[0]              # payoff to seller 1 (index 0)
                # use the V table to initialize Q value
                next_low_idx = min(next_canon_state)
                v_next = V[next_low_idx]

                total_q_val += r + gamma * v_next # Q(s,a) = E(r + gamma * V(s'))
                count += 1
        q_init = total_q_val / count               # expected T-period profit vs uniform opponents
        for i in range(num_sellers):
            Q_tables[i][low_idx, a] = q_init

print("Initialization Complete.")

# ==========================================
# --- 6. MAIN Q-LEARNING LOOP ---
# ==========================================

state = tuple([0] * num_sellers)  # Start with all sellers at Nash price index
log_interval = int(N_big_periods * 0.1)

print(f"\nStarting Simulation ({N_big_periods} periods)...")
h_strats = " | ".join([f"S{i + 1} Rule" for i in range(num_sellers)])
print(f"{'Period':<10} | {'Epsilon':<8} | {'State':<5} | {h_strats}")
print("-" * (35 + 12 * num_sellers))

for t in range(1, N_big_periods + 1):

    # Epsilon Decay (Updated Beta)
    epsilon = np.exp(-beta * t)

    # Choose Actions
    chosen_actions = []
    for i in range(num_sellers):
        if np.random.rand() < epsilon:
            act = np.random.randint(num_actions)
        else:
            obs_state = min(state)  # agent observes only the lowest price index
            q_vals = Q_tables[i][obs_state]
            max_val = np.max(q_vals)
            candidates = np.where(q_vals == max_val)[0]
            act = np.random.choice(candidates)
        chosen_actions.append(act)

    # Logging
    if t % log_interval == 0:
        strat_str = " | ".join([f"{STRATEGY_NAMES[a]:<10}" for a in chosen_actions])
        print(f"{t:<10} | {epsilon:.5f}  | {state} | {strat_str}")

    # Lookup & Update
    canon_state, canon_actions, perm, inv_perm = canonicalize_state_actions(state, chosen_actions)
    key = tuple(list(canon_state) + list(canon_actions))
    payoffs_canon, next_state = lookup_table[key]
    payoffs = [payoffs_canon[inv_perm[i]] for i in range(num_sellers)]
    obs_next = min(next_state)

    for i in range(num_sellers):
        a = chosen_actions[i]
        r = payoffs[i]
        obs_state = min(state)
        best_next_q = np.max(Q_tables[i][obs_next])

        old_q = Q_tables[i][obs_state, a]
        new_q = (1 - alpha) * old_q + alpha * (r + gamma * best_next_q)
        Q_tables[i][obs_state, a] = new_q

    state = next_state

# ==========================================
# --- 7. FINAL LEARNED POLICY ---
# ==========================================
print("-" * (35 + 12 * num_sellers))
print("\nFinal Learned Policies (Greedy Actions per State):")
print(f"{'Lowest Price State':<18} | " + " | ".join([f"Seller {i + 1} Strategy" for i in range(num_sellers)]))
print("-" * (18 + 20 * num_sellers))

for s in range(num_grids):
    seller_strats = []
    for i in range(num_sellers):
        q_vals = Q_tables[i][s]
        best_idx = np.argmax(q_vals)
        seller_strats.append(STRATEGY_NAMES[best_idx])

    strat_output = " | ".join([f"{st:<17}" for st in seller_strats])
    print(f"{s:<18} | {strat_output}")