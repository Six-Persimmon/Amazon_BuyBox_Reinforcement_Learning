import numpy as np
import itertools
from scipy.optimize import minimize

# ==========================================
# --- 1. SYSTEM CONFIGURATION ---
# ==========================================
num_sellers = 4  # <--- CHANGE THIS (2, 3, 4, 5, etc.)
num_grids = 10
K = 8  # inner loop number of periods
gamma = 0.95
# T = 1000000
T = 1000000 # outer loop number of periods
alpha = 0.15

# CHANGE 1: Updated Beta as requested
beta = 10 / T

# Parameters (Symmetric)
a_val = 2
c_val = 1
mu = 0.25
# mu = 0.005
a0 = 0

# xi = 0.1 # for price grid, follow Calvano et al. 2025

# xi=0

xi = -1  # for price grid, extend one step below/above

'''
Dec 17th: It seems small mu + small K(smaller than grid size) can trigger circle better
----
If xi = 0.1, use the rules below:
rule_undercut_with_reset
rule_match_with_reset
rule_price_above
We can sustain circle pricing with mu = 0.02 and T = 50 for 4 sellers. But not for mu = 0.025

------------------------------
If xi = 0, use the rules below:
rule_undercut_with_reset
rule_match_with_reset
rule_price_above
1. mu = 0.02 and T = 50 for 3 sellers. price end at grid 9 with match+reset/above
2. mu = 0.01/0.02 and T = 50 for 4 sellers. cannot sustain circle pricing, match/above at price grid 9
3. mu = 0.02 and T = 50 for 5 sellers. price end at grid 9 with match+reset/above

4. mu = 0.25 and T = 50 for 3 sellers. price end at grid 9/8, 2 picks above, 1 picks undercut+reset
5. mu = 0.25 and T = 50 for 4 sellers. price end at grid 9/8, 3 picks above, 1 picks undercut+reset
6. mu = 0.25 and T = 50 for 5 sellers. 
------------------------------
If xi = 0, use the rules below:
undercut
match
price_above
1. mu = 0.02 and T = 50 for 3 sellers: price end at grid 9 with match/above
2. mu = 0.02 and T = 50 for 4 sellers. price end at grid 9 with match/above
3. mu = 0.02 and T = 50 for 5 sellers: price end at grid 9 with match/above

4. mu = 0.25 and T = 50 for 3 sellers: price end at grid 9. mixture of above and match
5. mu = 0.25 and T = 50 for 4 sellers. price end at grid 9. mixture of above and match
------------------------------
If xi = 0.1, use the rules below:
undercut
match
price_above
undercut_with_reset
1. mu = 0.02 and T = 50 for 3 sellers: price end at grid 8 with match
2. mu = 0.02 and T = 50 for 4 sellers: price circle, undercut+reset
3. mu = 0.02 and T = 50 for 5 sellers: price circle, undercut+reset
4. mu = 0.25 and T = 50 for 3 sellers: price end at grid 8 with match
5. mu = 0.25 and T = 50 for 4 sellers: price end at grid 8 with match


We can sustain circle pricing with mu = 0.02 and T = 50 for 4 sellers. The displayed price depends on grid size & T.

------------------------------
If xi = 0.1, use the rules below:
match
price_above
undercut_with_reset
1. mu = 0.25, T = 50 for 3 sellers. price grid at 8, match for all
2. mu = 0.25, T = 50 for 4 sellers. price grid at 8, match for all
3. mu = 0.25, T = 50 for 5 sellers.
4. mu = 0.02, T = 50 for 3 sellers. price grid at 8. Start with undercut+reset for all, then match for all
5. mu = 0.02, T = 50 for 4 sellers. price grid at 6. Start with undercut+reset for all, then match for all
6. mu = 0.02, T = 50 for 5 sellers. price grid at 8. Start with undercut+reset for all, then match for all

Condition on we see circle: (N, mu, T, grid size)
'''

# ==========================================
# --- 2. PROFIT & GRID FUNCTIONS ---
# ==========================================
# def get_demand_and_profit(prices):
#     """
#     Input: list or array of prices [p1, p2, ..., pN]
#     Output: list of profits [pi1, pi2, ..., piN]
#     """
#     prices = np.array(prices)
#     exps = np.exp((a_val - prices) / mu)
#     exp0 = np.exp(a0 / mu)
#     denom = np.sum(exps) + exp0

#     demands = exps / denom
#     profits = (prices - c_val) * demands
#     return profits

def get_demand_and_profit(prices):
    """
    Input: list or array of prices [p1, p2, ..., pN]
    Output: list of profits [pi1, pi2, ..., piN]
    Numerical stable version using Max-Subtraction Trick
    """
    prices = np.array(prices)
    
    logits = (a_val - prices) / mu
    logit0 = a0 / mu
    
    max_logit = np.max(logits)
    max_logit = max(max_logit, logit0)
    
    exps = np.exp(logits - max_logit)
    exp0 = np.exp(logit0 - max_logit)
    
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
# # old: use xi
# grid_low = p_nash - xi * (p_monopoly - p_nash)
# grid_high = p_monopoly + xi * (p_monopoly - p_nash)
# price_grid = np.linspace(grid_low, grid_high, num_grids)

# # NEW:
# 3. Create Grid (extend one step below/above)
step = (p_monopoly - p_nash) / (num_grids - 3)
price_grid = np.linspace(p_nash - step, p_monopoly + step, num_grids)

print(f"Simulation Setup: {num_sellers} Sellers")
print(f"Parameters: a_val={a_val}, c_val={c_val}, mu={mu}, a0={a0}, xi={xi}")
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


# Registry. Every action is a function: def func(my_idx, others_min_idx, n_grids) -> next_price_idx
STRATEGIES = [
    # rule_undercut,
    rule_match,
    rule_price_above,
    rule_undercut_with_reset,  
    # rule_match_with_reset,    # <--- Added new strategy here
]

# Create shorter names for cleaner logging
STRATEGY_NAMES = [func.__name__.replace("rule_", "") for func in STRATEGIES]
# Specifically shorten the new ones for display
STRATEGY_NAMES = [n.replace("undercut_with_reset", "Under+Reset") for n in STRATEGY_NAMES]
STRATEGY_NAMES = [n.replace("match_with_reset", "Match+Reset") for n in STRATEGY_NAMES]

num_actions = len(STRATEGIES)
print(f"Strategies Loaded: {STRATEGY_NAMES}")

'''
# Canonicalize flow (example: state=(3,0,5,2), actions=(2,1,0,2))
# ┌──────────────────────────────────────────────────────────────┐
# │ 1) Pair up: zip(state, actions)                              │
# │    [(3,2), (0,1), (5,0), (2,2)]                              │
# │                                                              │
# │ 2) Sort by price_idx, then action_id                         │
# │    perm = [1, 3, 0, 2] (reordering indices)                  │
# │    sorted: [(0,1), (2,2), (3,2), (5,0)]                      │
# │                                                              │
# │ 3) Build canonical key                                       │
# │    canon_state   = (0, 2, 3, 5)                              │
# │    canon_actions = (1, 2, 2, 0)                              │
# │    key = (canon_state, canon_actions)                        │
# │                                                              │
# │ 4) Inverse permutation to restore per-seller order           │
# │    inv_perm = [2, 0, 3, 1]                                   │
# │    Use inv_perm to map payoffs/next_state back to each seller│
# └──────────────────────────────────────────────────────────────┘
# In short: inv_perm[seller_id] gives the index of seller_id's data in the canonical ordering
'''

# ==========================================
# --- HELPER: CANONICALIZATION ---
# ==========================================
def canonicalize_state_actions(state, actions):
    """
    Sorts (state, action) pairs to enforce symmetry, enabling lookup table compression.
    
    Inputs:
        state: tuple/list of price indices (e.g., (2, 0, 5, 2))
        actions: tuple/list of action indices (e.g., (1, 1, 0, 2))
    
    Outputs:
        key: tuple, flattened canonical key (s_sorted..., a_sorted...) for lookup
        inv_perm: list, indices to map canonical results back to specific agents
        canon_state: tuple, sorted state tuple (used for next-state transition)
    """
    pairs = list(zip(state, actions))

    # Get sorted indices based on (state, action) pairs. First sort by state, then by action
    perm = sorted(range(len(pairs)), key=lambda i: pairs[i])
    sorted_pairs = [pairs[i] for i in perm] # e.g. [(0,1), (2,2), (3,2), (5,0)]
    
    # Split back into sorted states and actions
    canon_state = tuple(p[0] for p in sorted_pairs)
    canon_actions = tuple(p[1] for p in sorted_pairs)
    
    # Flatten into the format used by lookup_table keys: (s1, s2..., a1, a2...)
    key = tuple(list(canon_state) + list(canon_actions))
    
    # Compute Inverse Permutation
    #    Used to route rewards from the sorted (anonymous) entry back to the specific agent ID
    inv_perm = [0] * len(pairs)
    for sorted_pos, original_pos in enumerate(perm):
        inv_perm[original_pos] = sorted_pos
        
    return key, inv_perm, canon_state

# ==========================================
# --- 4. N-AGENT LOOKUP TABLE ---
# ==========================================
lookup_table = {}

# 1. All possible (Price_Index, Action_Index) pairs
#    Total atomic pairs = Num_Grids * Num_Actions
atomic_pairs = list(itertools.product(range(num_grids), range(num_actions)))

# 2. all possible combinations of atomic pairs for num_sellers
canonical_scenarios = list(itertools.combinations_with_replacement(atomic_pairs, num_sellers))

print(f"Building Lookup Table via Atomic Pairs ({len(atomic_pairs)} pairs -> {len(canonical_scenarios)} unique scenarios)...")

for scenario in canonical_scenarios:
    # scenario is a list of tuples: [(p1, a1), (p2, a2), (p3, a3), (p4, a4)] (already sorted)
    
    # build key as canon_state + canon_actions
    s_list = [item[0] for item in scenario] # e.g. [0, 2, 3, 5]
    a_list = [item[1] for item in scenario] # e.g. [1, 2, 2, 0]
    key = tuple(s_list + a_list) # e.g. (0, 2, 3, 5, 1, 2, 2, 0)
    
    # --- Simulation ---
    current_price_indices = list(s_list)
    cumulative_profits = np.zeros(num_sellers)
    current_actions = list(a_list)

    for t in range(K):
        next_price_indices = []
        for i in range(num_sellers): # for each seller, get their next price index
            my_prev = current_price_indices[i]
            others_indices = current_price_indices[:i] + current_price_indices[i + 1:]
            others_min = min(others_indices)

            rule_func = STRATEGIES[current_actions[i]]
            p_next = rule_func(my_prev, others_min, num_grids)
            next_price_indices.append(p_next)

        # use next prices to compute profits
        current_price_indices = next_price_indices
        current_prices_val = [price_grid[idx] for idx in current_price_indices]
        step_profits = get_demand_and_profit(current_prices_val)
        cumulative_profits += step_profits

    avg_profits = cumulative_profits / K
    
    # save to Table
    # Note: Profit and End_State are saved as the order of the key (canonical order)
    # Note: the end state is also in canonical order, which corresponds to teh order in the key
    end_state = tuple(current_price_indices)
    
    lookup_table[key] = (avg_profits, end_state)

print("Lookup Table Built.")

# Precompute groups of canonical states by their lowest price (for value iteration on reduced state)
low_to_states = {g: [] for g in range(num_grids)}
state_space = list(itertools.combinations_with_replacement(range(num_grids), num_sellers))
for st in state_space:
    low_to_states[min(st)].append(st) # res: dict: lowest_price_idx -> list of states(sorted and canonical) with that lowest price

# # ==========================================
# # --- 5. INITIALIZATION (Value Iteration) ---
# # ==========================================
# print("Initializing Q-tables via Value Iteration (Optimized)...")

# V = np.zeros(num_grids)
# delta = float('inf')
# theta = 1e-4

# opponent_combinations = list(itertools.product(range(num_actions), repeat=num_sellers - 1))

# # --- Value Iteration ---
# while delta > theta:
#     delta = 0
#     V_new = np.zeros(num_grids)

#     for low_idx in range(num_grids):
#         candidate_states = low_to_states.get(low_idx, []) # all the canonical states with this lowest price
#         if not candidate_states:
#             continue
        
#         q_sum = 0
#         for my_action in range(num_actions):
#             total_val = 0
#             count = 0

#             for state in candidate_states:
#                 for opp_actions in opponent_combinations:
#                     # assume we are Seller 0
#                     actions_list = [my_action] + list(opp_actions)
                    
#                     # get key (real price states, actions) and inverse permutation
#                     key, inv_perm, _ = canonicalize_state_actions(state, actions_list)
                    
#                     # lookup for payoffs and next states for all sellers (canonical order, not real order)
#                     payoffs_canon, next_state_canon = lookup_table[key]
                    
#                     # recove own reward using inv_perm. inv_perm[0] tells us where Seller 0's data is in the canonical ordering
#                     r = payoffs_canon[inv_perm[0]]
                    
#                     # calculate V_next
#                     v_next = V[min(next_state_canon)]
#                     total_val += (r + gamma * v_next)
#                     count += 1

#             # calculate the average Q(low_idx, my_action)
#             q_val = total_val / count # average over all candidate states and opponent actions
#             q_sum += q_val

#         # update V(low_idx). average over all my actions as we assume uniform random policy initially
#         V_new[low_idx] = q_sum / num_actions
#         delta = max(delta, abs(V_new[low_idx] - V[low_idx]))
#     V = V_new

# print("V-Values Converged.")

# --- Q-Tables Initialization ---
Q_tables = [np.zeros((num_grids, num_actions)) for _ in range(num_sellers)] # size: num_grids x num_actions, one table per seller

# for low_idx in range(num_grids):
#     candidate_states = low_to_states.get(low_idx, [])
#     if not candidate_states:
#         continue
#     for a in range(num_actions):
#         total_q = 0.0
#         count = 0
#         for state in candidate_states:
#             for opp_actions in opponent_combinations:
#                 # get full actions with me as seller 0
#                 actions_list = [a] + list(opp_actions)
                
#                 # call canonicalization
#                 key, inv_perm, _ = canonicalize_state_actions(state, actions_list)
                
#                 # lookup for payoffs and next states for all sellers (canonical order, not real order)
#                 payoffs_canon, next_state_canon = lookup_table[key]
                
#                 # recove own reward using inv_perm. inv_perm[0] tells us where Seller 0's data is in the canonical ordering
#                 r = payoffs_canon[inv_perm[0]]
                
#                 # calculate V_next. Use Q(obs,a) = E_s(r(s,obs,a)) + gamma * E_s(V(obs')) to get initial Q
#                 v_next = V[min(next_state_canon)]
#                 total_q += (r + gamma * v_next)
#                 count += 1
        
#         q_init = total_q / count
#         for i in range(num_sellers):
#             Q_tables[i][low_idx, a] = q_init

# print("Initialization Complete.")

# ==========================================
# --- 6. MAIN Q-LEARNING LOOP ---
# ==========================================

state = tuple([0] * num_sellers)  # initial state: all sellers at lowest price grid
log_interval = int(T * 0.1)  # log

print(f"\nStarting Simulation ({T} periods)...")
h_strats = " | ".join([f"S{i + 1} Rule" for i in range(num_sellers)])
print(f"{'Period':<10} | {'Epsilon':<8} | {'State (Sorted)':<20} | {h_strats}")
print("-" * (45 + 12 * num_sellers))

for t in range(1, T + 1):
    # epsilon decay, same as Calvano et al. 2020
    epsilon = np.exp(-beta * t)

    # choose action
    chosen_actions = []
    for i in range(num_sellers):
        if np.random.rand() < epsilon:
            act = np.random.randint(num_actions)
        else:
            obs_state = min(state)  # notice sellers only obs lowest price index. They don't see full state, which is price grid indices of all sellers
            q_vals = Q_tables[i][obs_state]
            # if tie, randomly choose among best actions
            candidates = np.where(q_vals == np.max(q_vals))[0]
            act = np.random.choice(candidates)
        chosen_actions.append(act)

    # logging: print very seller's strategy
    if t % log_interval == 0:
        strat_str = " | ".join([f"{STRATEGY_NAMES[a]:<10}" for a in chosen_actions])
        # print state, which are the true state grids for each seller, not sorted
        print(f"{t:<10} | {epsilon:.5f}  | {str(state):<20} | {strat_str}")

    # A. Canonicalize state and actions to get lookup table key
    key, inv_perm, _ = canonicalize_state_actions(state, chosen_actions)
    
    # B. obtain payoffs and next state from lookup table. Note: both are in canonical order
    payoffs_canon, next_state_canon = lookup_table[key]
    
    # C. Recover order for real payoffs and next state for each seller using inv_perm
    real_payoffs = [payoffs_canon[inv_perm[i]] for i in range(num_sellers)]
    real_next_state = [next_state_canon[inv_perm[i]] for i in range(num_sellers)]
    real_next_state = tuple(real_next_state)
    
    # Obs for Q-learning Agents. They only obs the lowest price index among all sellers
    obs_next = min(real_next_state)
    obs_current = min(state)

    # D. Q-Learning Update for each seller
    for i in range(num_sellers):
        a = chosen_actions[i]
        r = real_payoffs[i] 

        # Standard Q-Learning update
        best_next_q = np.max(Q_tables[i][obs_next])
        old_q = Q_tables[i][obs_current, a]
        # Q(s,a) = (1-alpha)*Q(s,a) + alpha*(r + gamma*max_a' Q(s',a')), standard Q-learning update
        new_q = (1 - alpha) * old_q + alpha * (r + gamma * best_next_q)
        Q_tables[i][obs_current, a] = new_q

    # E. Transition to next state: price grid idx for each seller
    state = real_next_state 

print("\nSimulation Finished.")
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
