import numpy as np
import itertools
from scipy.optimize import minimize

# ==========================================
# --- 1. SYSTEM CONFIGURATION ---
# ==========================================
num_sellers = 5  # <--- CHANGE THIS (2, 3, 4, 5, etc.)
num_grids = 10
T = 50
gamma = 0.95
N_big_periods = 1000000
# N_big_periods = 600000
alpha = 0.15

# CHANGE 1: Updated Beta as requested
beta = 10 / N_big_periods

# Parameters (Symmetric)
a_val = 2
c_val = 1
# mu = 0.25
mu = 0.01
a0 = 0


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
    p = p_scalar_list[0]
    prices = [p] * num_sellers
    prof = get_demand_and_profit(prices)
    return -np.sum(prof)


def neg_own_profit_Np(p_own_list, p_comp):
    p_own = p_own_list[0]
    prices = [p_own] + [p_comp] * (num_sellers - 1)
    prof = get_demand_and_profit(prices)
    return -prof[0]


# 1. Find Monopoly Price
res_monopoly = minimize(neg_joint_profit_Np, x0=[1.5], bounds=[(1.0, 5.0)])
p_monopoly = res_monopoly.x[0]

# 2. Find Nash Price
p_nash = 1.1
for _ in range(10):
    res_br = minimize(neg_own_profit_Np, x0=[p_nash], args=(p_nash,), bounds=[(1.0, 5.0)])
    p_nash = res_br.x[0]

# 3. Create Grid
# price_grid = np.linspace(p_nash, p_monopoly, num_grids)

# # NEW:
# 3. Create Grid (extend one step below/above)
step = (p_monopoly - p_nash) / (num_grids - 3)
price_grid = np.linspace(p_nash - step, p_monopoly + step, num_grids)

print(f"Simulation Setup: {num_sellers} Sellers")
print(f"Parameters: a_val={a_val}, c_val={c_val}, mu={mu}, a0={a0}")
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


# CHANGE 2: Added "Undercut with Reset"
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
    # rule_match_with_reset,      # <--- Added new strategy here
]

# Create shorter names for cleaner logging
STRATEGY_NAMES = [func.__name__.replace("rule_", "") for func in STRATEGIES]
# Specifically shorten the new one for display
STRATEGY_NAMES = [n.replace("undercut_with_reset", "Under+Reset") for n in STRATEGY_NAMES]

num_actions = len(STRATEGIES)
print(f"Strategies Loaded: {STRATEGY_NAMES}")

# ==========================================
# --- 4. N-AGENT LOOKUP TABLE ---
# ==========================================
lookup_table = {}
action_combinations = list(itertools.product(range(num_actions), repeat=num_sellers))

print(f"Building Lookup Table ({len(action_combinations) * num_grids} entries)...")

for s_idx in range(num_grids):
    for actions in action_combinations:

        # --- Inner Simulation (T periods) ---
        current_price_indices = [s_idx] * num_sellers
        cumulative_profits = np.zeros(num_sellers)

        for t in range(T):
            next_price_indices = []
            for i in range(num_sellers):
                my_prev = current_price_indices[i]
                others_indices = current_price_indices[:i] + current_price_indices[i + 1:]
                others_min = min(others_indices)

                # Apply strategy
                rule_func = STRATEGIES[actions[i]]
                p_next = rule_func(my_prev, others_min, num_grids)
                next_price_indices.append(p_next)

            current_price_indices = next_price_indices
            current_prices_val = [price_grid[idx] for idx in current_price_indices]
            step_profits = get_demand_and_profit(current_prices_val)
            cumulative_profits += step_profits

        end_state = min(current_price_indices)
        key = tuple([s_idx] + list(actions))
        lookup_table[key] = (cumulative_profits, end_state)

print("Lookup Table Built.")

# ==========================================
# --- 5. INITIALIZATION (Value Iteration) ---
# ==========================================
print("Initializing Q-tables via Value Iteration (Random Policy)...")

V = np.zeros(num_grids)
delta = float('inf')
theta = 1e-4

opponent_combinations = list(itertools.product(range(num_actions), repeat=num_sellers - 1)) # action combos for opponents. E.g. if num_sellers=3, this is [(0,0),(0,1),(1,0),(1,1)] for 2 actions.

while delta > theta:
    delta = 0
    V_new = np.zeros(num_grids)

    for s in range(num_grids):
        q_sum = 0
        for my_action in range(num_actions):
            total_reward = 0
            total_next_v = 0
            count = 0

            for opp_actions in opponent_combinations:
                full_actions = tuple([s] + [my_action] + list(opp_actions))
                payoffs, next_s = lookup_table[full_actions] # lookup table: (state, my action + others actions) -> (payoffs, next state)

                total_reward += payoffs[0]  # took my own payoff
                total_next_v += V[next_s]
                count += 1

            avg_reward = total_reward / count
            avg_next_v = total_next_v / count
            q_val = avg_reward + gamma * avg_next_v
            q_sum += q_val

        V_new[s] = q_sum / num_actions
        delta = max(delta, abs(V_new[s] - V[s]))
    V = V_new   # update V. Finally the V contains the average discounted reward at state s given all players play uniformly random actions.

Q_tables = [np.zeros((num_grids, num_actions)) for _ in range(num_sellers)] # Q-tables for each seller
opponent_combinations = list(itertools.product(range(num_actions), repeat=num_sellers - 1)) # action combos for opponents. E.g. if num_sellers=3, this is [(0,0),(0,1),(1,0),(1,1)] for 2 actions.

for s in range(num_grids):
    for a in range(num_actions):
        total_value = 0.0
        count = 0
        for opp_actions in opponent_combinations:
            # Assemble actions with me as seller 0 taking 'a'
            full_actions = tuple([s] + [a] + list(opp_actions))
            payoffs, next_s = lookup_table[full_actions] # output is (payoffs, next_state)
            r = payoffs[0]  # payoff to seller 0
            v_next = V[next_s] # value of next state, pre-computed
            # Notice Q(s,a) = E[r + gamma * V(s')]
            total_value += r + gamma * v_next
            count += 1
        q_init = total_value / count               # expected T-period profit vs uniform opponents
        for i in range(num_sellers):
            Q_tables[i][s, a] = q_init

print("Initialization Complete.")

# ==========================================
# --- 6. MAIN Q-LEARNING LOOP ---
# ==========================================

state = 0  # Start at Nash
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
            q_vals = Q_tables[i][state]
            max_val = np.max(q_vals)
            candidates = np.where(q_vals == max_val)[0]
            act = np.random.choice(candidates)
        chosen_actions.append(act)

    # Logging
    if t % log_interval == 0:
        strat_str = " | ".join([f"{STRATEGY_NAMES[a]:<10}" for a in chosen_actions])
        print(f"{t:<10} | {epsilon:.5f}  | {state:<5} | {strat_str}")

    # Lookup & Update
    key = tuple([state] + chosen_actions)
    payoffs, next_state = lookup_table[key]

    for i in range(num_sellers):
        a = chosen_actions[i]
        r = payoffs[i]
        best_next_q = np.max(Q_tables[i][next_state])

        old_q = Q_tables[i][state, a]
        # Q(s,a) = (1-alpha)*Q(s,a) + alpha*(r + gamma*max_a' Q(s',a')), standard Q-learning update
        new_q = (1 - alpha) * old_q + alpha * (r + gamma * best_next_q)
        Q_tables[i][state, a] = new_q

    state = next_state

# ==========================================
# --- 7. FINAL LEARNED POLICY ---
# ==========================================
print("-" * (35 + 12 * num_sellers))
print("\nFinal Learned Policies (Greedy Actions per State):")
print(f"{'State':<6} | {'Price':<8} | " + " | ".join([f"Seller {i + 1} Strategy" for i in range(num_sellers)]))
print("-" * (20 + 20 * num_sellers))

for s in range(num_grids):
    price_val = price_grid[s]

    seller_strats = []
    for i in range(num_sellers):
        q_vals = Q_tables[i][s]
        best_idx = np.argmax(q_vals)
        seller_strats.append(STRATEGY_NAMES[best_idx])

    strat_output = " | ".join([f"{st:<17}" for st in seller_strats])
    print(f"{s:<6} | {price_val:<8.3f} | {strat_output}")