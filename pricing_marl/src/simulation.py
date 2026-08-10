import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import Config
from src.environment import PricingEnvironment
from src.agent import QAgent, calculate_heuristic_init_values

def _build_q_snapshot_df(
    agents,
    config: Config,
    run_id: int,
    training_stop_episode: int,
    snapshot_type: str,
):
    """Flatten all sellers' Q-tables into a tidy DataFrame."""
    rows = []
    for agent in agents:
        seller_id = int(agent.id)
        for state_idx in range(config.num_grids):
            for action_idx in range(config.num_actions):
                rows.append(
                    {
                        "run_id": int(run_id),
                        "seller_id": seller_id,
                        "state_idx": int(state_idx),
                        "action_idx": int(action_idx),
                        "action_id": int(config.active_strategies[action_idx]),
                        "q_value": float(agent.Q[state_idx, action_idx]),
                        "training_stop_episode": int(training_stop_episode),
                        "qtable_type": snapshot_type,
                    }
                )

    q_df = pd.DataFrame(rows)
    if not q_df.empty:
        q_df["run_id"] = q_df["run_id"].astype("int32")
        q_df["seller_id"] = q_df["seller_id"].astype("int16")
        q_df["state_idx"] = q_df["state_idx"].astype("int16")
        q_df["action_idx"] = q_df["action_idx"].astype("int16")
        q_df["action_id"] = q_df["action_id"].astype("int16")
        q_df["training_stop_episode"] = q_df["training_stop_episode"].astype("int32")
        q_df["q_value"] = q_df["q_value"].astype("float32")

    return q_df


def run_simulation(
    config: Config,
    run_id: int = 0,
    disable_tqdm: bool = False,
    return_q_snapshot: bool = False,
):
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

    q_snapshot_init_df = None
    if return_q_snapshot:
        # Capture true initialization snapshot before any training update.
        q_snapshot_init_df = _build_q_snapshot_df(
            agents=agents,
            config=config,
            run_id=run_id,
            training_stop_episode=-1,
            snapshot_type="init",
        )
    
    # 3. Initial State (Random Start - single lowest price index for all sellers)
    state = int(np.random.randint(0, config.num_grids))
    
    # --- PHASE 1: TRAINING ---
    # 不记录详细数据，只关注 Q-Table 更新和收敛
    
    is_converged = False
    policy_stable_counter = 0
    prev_policy = np.zeros((config.num_sellers, config.num_grids), dtype=int)
    
    # 进度条只针对 Training
    iterator = range(config.max_episodes)
    if not disable_tqdm:
        iterator = tqdm(iterator, desc=f"Run {run_id} [Train]")

    training_stop_episode = -1

    for t in iterator:
        # 1. Get Actions
        actions_indices = [agent.choose_action(state, t) for agent in agents]
        
        # 2. Env Step (Fast Lookup)
        # NOTE (verified 2026-08-03): env.step() un-permutes the canonical
        # lookup profits back to the original agent order via inv_perm; the
        # old TODO about a missing inv perm is stale. Checked empirically:
        # permuting the action profile permutes profits identically, and the
        # lookup path matches a direct original-order K-step simulation for
        # all (state, profile) pairs.
        # return: rewards, next_state_idx, avg_p, avg_low_p
        rewards, next_state, _, _ = env.step(state, actions_indices, return_details=False)
        
        # 3. Update Agents
        for i, agent in enumerate(agents):
            agent.update(state, actions_indices[i], rewards[i], next_state)
            
        # 4. Convergence Check
        current_policy = np.array([agent.get_greedy_policy() for agent in agents])
        if np.array_equal(current_policy, prev_policy):
            policy_stable_counter += 1
        else:
            policy_stable_counter = 0
            prev_policy = current_policy
            
        if policy_stable_counter >= config.converge_period:
            is_converged = True
            training_stop_episode = int(t)
            if not disable_tqdm:
                print(f"  Run {run_id}: Converged at episode {t}")
            break
            
        state = next_state

    if training_stop_episode < 0:
        training_stop_episode = int(t) if config.max_episodes > 0 else 0

    # --- PHASE 2: EVALUATION (H-Step) ---
    # 冻结 Agent (Epsilon=0, Learning=0)
    # 展开 K-step 记录每一“天”的数据
    
    # 计算需要跑多少个 Outer Episodes
    num_eval_episodes = max(1, config.eval_H // config.K)
    
    eval_history = []

    # Pre-calculate constants for Delta
    prices_nash = [env.p_nash] * config.num_sellers
    pi_nash = env._get_demand_and_profit(prices_nash)[0]
    prices_mon = [env.p_monopoly] * config.num_sellers
    pi_mon = env._get_demand_and_profit(prices_mon)[0]
    
    denominator = pi_mon - pi_nash if (pi_mon - pi_nash) > 1e-9 else 1.0

    current_t_atomic = 0 # 全局原子时间步计数
    
    # 确保 Eval 从当前 State 继续 (or we reset state, but it's ok to keep moving)
    
    q_snapshot_df = None
    if return_q_snapshot:
        q_snapshot_final_df = _build_q_snapshot_df(
            agents=agents,
            config=config,
            run_id=run_id,
            training_stop_episode=training_stop_episode,
            snapshot_type="final",
        )
        q_snapshot_df = pd.concat([q_snapshot_init_df, q_snapshot_final_df], ignore_index=True)

    for _ in range(num_eval_episodes):
        # 1. Greedy Actions
        # 强制 epsilon=0 (greedy)
        # 注意：这里我们直接用 choose_action，但我们可以传入一个很大的 t 让 epsilon 归零，
        # 或者在 agent 里加个 force_greedy flag。最简单是传 t=1e9。
        actions_indices = [agent.choose_action(state, t_step=1e9) for agent in agents]
        
        # 2. Env Step with Details
        # returns: ..., trajectory dict
        rewards_avg, next_state, _, _, trajectory = env.step(state, actions_indices, return_details=True)
        
        # 3. Unroll Trajectory & Record
        # trajectory['prices'] is List of (N,) arrays, length K
        # trajectory['profits'] is List of (N,) arrays, length K
        
        prices_seq = trajectory['prices']
        profits_seq = trajectory['profits']
        
        for k in range(config.K):
            p_t = prices_seq[k]   # [p0, p1, ..., pN]
            pi_t = profits_seq[k] # [pi0, pi1, ..., piN]
            
            # Global Stats
            min_p = float(np.min(p_t))
            mean_p = float(np.mean(p_t))
            mean_pi = float(np.mean(pi_t))
            
            delta = (mean_pi - pi_nash) / denominator
            
            record = {
                "run_id": run_id,
                "t_global": current_t_atomic,
                "episode": training_stop_episode, # Training 结束时的 outer episode
                "step_in_k": k,
                "price_min": min_p,
                "price_mean": mean_p,
                "delta": delta,
                "converged": is_converged, # 标记这个 Run 是否收敛了
                "is_cycle": False # 这是一个 placeholder，后续分析脚本算
            }
            
            # Record individual actions (注意：这一个 K 周期内 Action 都是一样的)
            for i in range(config.num_sellers):
                act_id = config.active_strategies[actions_indices[i]]
                record[f"a_{i}"] = act_id
                record[f"p_{i}"] = float(p_t[i])
                record[f"pi_{i}"] = float(pi_t[i]) # 利润如果需要也可以存
                
            eval_history.append(record)
            current_t_atomic += 1
            
        state = next_state

    # 转为 DataFrame
    df = pd.DataFrame(eval_history)
    
    # 类型压缩
    f_cols = df.select_dtypes(include=['float64']).columns
    df[f_cols] = df[f_cols].astype('float32')
    i_cols = df.select_dtypes(include=['int64']).columns
    df[i_cols] = df[i_cols].astype('int32')

    if return_q_snapshot:
        return df, q_snapshot_df
    return df
