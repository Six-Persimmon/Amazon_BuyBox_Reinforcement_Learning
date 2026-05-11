import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import Config
from src.environment import PricingEnvironment
from src.agent import QAgent, calculate_heuristic_init_values

def run_simulation(config: Config, run_id: int = 0, disable_tqdm: bool = False):
    """
    Executes a single full simulation run.
    Two-Phase Simulation:
    1. Training Phase: Run until convergence or max_episodes. (No saving)
    2. Evaluation Phase: Run for eval_H steps with greedy policy. (Save detailed atomic data)
    
    Args:
        config: Experiment configuration.
        run_id: Unique identifier for the run.
        disable_tqdm: If True, suppresses the progress bar (essential for parallel jobs).
    """
    
    # 1. Setup Environment
    env = PricingEnvironment(config)
    
    # 2. Initialize Agents (Heuristic Init)
    init_q = calculate_heuristic_init_values(env, config)
    agents = [QAgent(i, config, initial_Q_table=init_q) for i in range(config.num_sellers)]
    
    # 3. Initial State (Random Start - single lowest price index for all sellers)
    state = int(np.random.randint(0, config.num_grids))
    
    # 4. Data Storage & Constants
    history = []
    
    # Pre-calculate constants for Delta
    prices_nash = [env.p_nash] * config.num_sellers
    pi_nash = env._get_demand_and_profit(prices_nash)[0]
    prices_mon = [env.p_monopoly] * config.num_sellers
    pi_mon = env._get_demand_and_profit(prices_mon)[0]
    
    denominator = pi_mon - pi_nash
    if denominator == 0: denominator = 1e-9
    
    # [NEW] Convergence Tracking
    prev_policy_matrix = np.zeros((config.num_sellers, config.num_grids), dtype=int)
    stability_counter = 0
    is_converged_flag = False

    # [MODIFIED] Progress Bar Logic
    # Only show tqdm if not disabled AND T is large enough
    iterator = range(config.T)
    if config.T > 10000 and not disable_tqdm:
        iterator = tqdm(range(config.T), desc=f"Run {run_id}", mininterval=5.0)

    for t in iterator:
        obs = state
        
        # --- B. Action Selection ---
        actions_indices = [] 
        for agent in agents:
            act_idx = agent.choose_action(obs, t)
            actions_indices.append(act_idx)
        actions_indices = tuple(actions_indices)
        
        # --- C. Environment Step ---
        canon_actions, _, inv_perm = env.canonicalize_actions(actions_indices)
        key = tuple([state] + list(canon_actions))
        if key not in env.lookup_table:
            raise KeyError(f"Missing Key: {key}")

        canonical_profits, next_lowest_state, avg_price_k, avg_lowest_price_k = env.lookup_table[key]

        rewards = [0.0] * config.num_sellers
        for original_idx in range(config.num_sellers):
            canonical_idx = inv_perm[original_idx]
            rewards[original_idx] = canonical_profits[canonical_idx]
            
        next_state = next_lowest_state
        next_obs = next_state
        
        # --- D. Update ---
        for i, agent in enumerate(agents):
            agent.update(obs, actions_indices[i], rewards[i], next_obs)
            
        # --- Check Convergence ---
        current_policy_matrix = np.array([ag.get_greedy_policy() for ag in agents])
        
        if np.array_equal(current_policy_matrix, prev_policy_matrix):
            stability_counter += 1
        else:
            stability_counter = 0 
            prev_policy_matrix = current_policy_matrix
            
        if stability_counter >= config.converge_period:
            is_converged_flag = True

        # --- E. Record ---
        avg_profit = np.mean(rewards)
        delta = (avg_profit - pi_nash) / denominator
        
        record = {
            "run_id": run_id,
            "t": t,
            "lowest_price": env.price_grid[obs],
            "average_price": env.price_grid[obs],
            "average_K_price": avg_price_k,
            "average_lowest_K_price": avg_lowest_price_k,
            "delta": delta,
            "is_converged": is_converged_flag, 
        }


        # Add individual Price and Action (all sellers share the same starting price index)
        for i in range(config.num_sellers):
            record[f"p_{i}"] = env.price_grid[state]

            # Map action index to global Action ID
            act_idx = actions_indices[i]
            global_act_id = config.active_strategies[act_idx]
            record[f"a_{i}"] = global_act_id
            record[f"pi_{i}"] = rewards[i] # individual profit
        history.append(record)
        
        state = next_state

    df = pd.DataFrame(history)
    # save memory
    df['average_price'] = df['average_price'].astype(np.float32)
    df['average_K_price'] = df['average_K_price'].astype(np.float32)
    df['average_lowest_K_price'] = df['average_lowest_K_price'].astype(np.float32)
    df['delta'] = df['delta'].astype(np.float32)
    return df
