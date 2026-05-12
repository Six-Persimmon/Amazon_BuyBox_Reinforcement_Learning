import numpy as np
import pandas as pd
from tqdm import tqdm

from .agent import KActionQAgent, calculate_composite_heuristic_init_values
from .config import KActionConfig
from .environment import BaseKLookupEnvironment


def _build_q_snapshot_df(
    agents,
    config: KActionConfig,
    run_id: int,
    training_stop_tick: int,
    snapshot_type: str,
):
    rows = []
    for agent in agents:
        seller_id = int(agent.id)
        for state_idx in range(config.num_grids):
            for meta in config.action_map:
                action_idx = meta["composite_action_idx"]
                rows.append(
                    {
                        "run_id": int(run_id),
                        "seller_id": seller_id,
                        "state_idx": int(state_idx),
                        "composite_action_idx": int(action_idx),
                        "rule_action_idx": int(meta["rule_action_idx"]),
                        "action_id": int(meta["action_id"]),
                        "k_choice": int(meta["k_choice"]),
                        "duration_blocks": int(meta["duration_blocks"]),
                        "q_value": float(agent.Q[state_idx, action_idx]),
                        "training_stop_tick": int(training_stop_tick),
                        "qtable_type": snapshot_type,
                    }
                )

    q_df = pd.DataFrame(rows)
    if not q_df.empty:
        int_cols = [
            "run_id",
            "seller_id",
            "state_idx",
            "composite_action_idx",
            "rule_action_idx",
            "action_id",
            "k_choice",
            "duration_blocks",
            "training_stop_tick",
        ]
        for col in int_cols:
            q_df[col] = q_df[col].astype("int32")
        q_df["seller_id"] = q_df["seller_id"].astype("int16")
        q_df["state_idx"] = q_df["state_idx"].astype("int16")
        q_df["composite_action_idx"] = q_df["composite_action_idx"].astype("int16")
        q_df["rule_action_idx"] = q_df["rule_action_idx"].astype("int16")
        q_df["action_id"] = q_df["action_id"].astype("int16")
        q_df["duration_blocks"] = q_df["duration_blocks"].astype("int16")
        q_df["q_value"] = q_df["q_value"].astype("float32")

    return q_df


def _current_policy_matrix(agents):
    return np.array([agent.get_greedy_policy() for agent in agents])


def _select_new_action(agent, state, t_block, config: KActionConfig, greedy=False):
    if greedy:
        action_idx = agent.choose_greedy_action(state)
    else:
        action_idx = agent.choose_action(state, t_block)
    meta = config.get_composite_action(action_idx)
    return action_idx, meta


def run_simulation(
    config: KActionConfig,
    run_id: int = 0,
    disable_tqdm: bool = False,
    return_q_snapshot: bool = False,
):
    env = BaseKLookupEnvironment(config)

    init_q = calculate_composite_heuristic_init_values(env, config)
    agents = [
        KActionQAgent(i, config, initial_Q_table=init_q)
        for i in range(config.num_sellers)
    ]

    q_snapshot_init_df = None
    if return_q_snapshot:
        q_snapshot_init_df = _build_q_snapshot_df(
            agents=agents,
            config=config,
            run_id=run_id,
            training_stop_tick=-1,
            snapshot_type="init",
        )

    state = int(np.random.randint(0, config.num_grids))

    active_action_indices = np.full(config.num_sellers, -1, dtype=int)
    active_rule_indices = np.zeros(config.num_sellers, dtype=int)
    active_k_choices = np.zeros(config.num_sellers, dtype=int)
    active_duration_blocks = np.zeros(config.num_sellers, dtype=int)
    remaining_blocks = np.zeros(config.num_sellers, dtype=int)
    action_start_states = np.zeros(config.num_sellers, dtype=int)
    reward_sums = np.zeros(config.num_sellers, dtype=float)
    elapsed_blocks = np.zeros(config.num_sellers, dtype=int)

    is_converged = False
    policy_stable_counter = 0
    prev_policy = None
    training_stop_tick = -1

    iterator = range(config.max_episodes)
    if not disable_tqdm:
        iterator = tqdm(iterator, desc=f"Run {run_id} [exp03 train]")

    last_t = -1
    for t_block in iterator:
        last_t = int(t_block)

        for i, agent in enumerate(agents):
            if remaining_blocks[i] <= 0: # an agent can only update and choose a new action when its previous action's duration (how many K blocks) has elapsed
                action_idx, meta = _select_new_action(
                    agent=agent,
                    state=state,
                    t_block=t_block,
                    config=config,
                    greedy=False,
                )
                active_action_indices[i] = action_idx
                active_rule_indices[i] = meta["rule_action_idx"]
                active_k_choices[i] = meta["k_choice"]
                active_duration_blocks[i] = meta["duration_blocks"]
                remaining_blocks[i] = meta["duration_blocks"]
                action_start_states[i] = state
                reward_sums[i] = 0.0
                elapsed_blocks[i] = 0

        rewards, next_state, _, _ = env.step(
            state,
            active_rule_indices.tolist(),
            return_details=False,
        )

        reward_sums += rewards
        elapsed_blocks += 1
        remaining_blocks -= 1

        for i, agent in enumerate(agents):
            if remaining_blocks[i] == 0:
                avg_reward = reward_sums[i] / max(1, elapsed_blocks[i])
                agent.update(
                    int(action_start_states[i]),
                    int(active_action_indices[i]),
                    float(avg_reward),
                    int(next_state),
                )
                reward_sums[i] = 0.0
                elapsed_blocks[i] = 0

        current_policy = _current_policy_matrix(agents)
        if prev_policy is not None and np.array_equal(current_policy, prev_policy):
            policy_stable_counter += 1
        else:
            policy_stable_counter = 0
            prev_policy = current_policy

        if policy_stable_counter >= config.converge_period:
            is_converged = True
            training_stop_tick = int(t_block)
            if not disable_tqdm:
                print(f"  Run {run_id}: converged at base block {t_block}")
            state = int(next_state)
            break

        state = int(next_state)

    if training_stop_tick < 0:
        training_stop_tick = last_t if config.max_episodes > 0 else 0

    q_snapshot_df = None
    if return_q_snapshot:
        q_snapshot_final_df = _build_q_snapshot_df(
            agents=agents,
            config=config,
            run_id=run_id,
            training_stop_tick=training_stop_tick,
            snapshot_type="final",
        )
        q_snapshot_df = pd.concat(
            [q_snapshot_init_df, q_snapshot_final_df],
            ignore_index=True,
        )

    eval_df = _run_evaluation(
        env=env,
        agents=agents,
        config=config,
        run_id=run_id,
        initial_state=state,
        training_stop_tick=training_stop_tick,
        is_converged=is_converged,
    )

    if return_q_snapshot:
        return eval_df, q_snapshot_df
    return eval_df


def _run_evaluation(
    env: BaseKLookupEnvironment,
    agents,
    config: KActionConfig,
    run_id: int,
    initial_state: int,
    training_stop_tick: int,
    is_converged: bool,
):
    state = int(initial_state)
    num_eval_blocks = max(1, config.eval_H // config.base_K)

    active_action_indices = np.full(config.num_sellers, -1, dtype=int)
    active_rule_indices = np.zeros(config.num_sellers, dtype=int)
    active_k_choices = np.zeros(config.num_sellers, dtype=int)
    active_duration_blocks = np.zeros(config.num_sellers, dtype=int)
    remaining_blocks = np.zeros(config.num_sellers, dtype=int)

    prices_nash = [env.p_nash] * config.num_sellers
    pi_nash = env._get_demand_and_profit(prices_nash)[0]
    prices_mon = [env.p_monopoly] * config.num_sellers
    pi_mon = env._get_demand_and_profit(prices_mon)[0]
    denominator = pi_mon - pi_nash if (pi_mon - pi_nash) > 1e-9 else 1.0

    eval_history = []
    current_t_atomic = 0

    for base_block in range(num_eval_blocks):
        decision_flags = np.zeros(config.num_sellers, dtype=bool)

        for i, agent in enumerate(agents):
            if remaining_blocks[i] <= 0:
                action_idx, meta = _select_new_action(
                    agent=agent,
                    state=state,
                    t_block=base_block,
                    config=config,
                    greedy=True,
                )
                active_action_indices[i] = action_idx
                active_rule_indices[i] = meta["rule_action_idx"]
                active_k_choices[i] = meta["k_choice"]
                active_duration_blocks[i] = meta["duration_blocks"]
                remaining_blocks[i] = meta["duration_blocks"]
                decision_flags[i] = True

        _, next_state, _, _, trajectory = env.step(
            state,
            active_rule_indices.tolist(),
            return_details=True,
        )

        prices_seq = trajectory["prices"]
        profits_seq = trajectory["profits"]

        for step_in_base_k in range(config.base_K):
            p_t = prices_seq[step_in_base_k]
            pi_t = profits_seq[step_in_base_k]

            min_p = float(np.min(p_t))
            mean_p = float(np.mean(p_t))
            mean_pi = float(np.mean(pi_t))
            delta = (mean_pi - pi_nash) / denominator

            record = {
                "run_id": int(run_id),
                "t_global": int(current_t_atomic),
                "episode": int(training_stop_tick), # num base blocks until convergence (or end of training if not converged)
                "base_block": int(base_block), # index of base block in evaluation phase. Value from 0 to num_eval_blocks-1
                "step_in_k": int(step_in_base_k),# if base block is K = 10, then step_in_k goes from 0 to 9
                "step_in_base_K": int(step_in_base_k), # same as above, just for clearer naming in the dataframe
                "price_min": min_p,
                "price_mean": mean_p,
                "delta": float(delta),
                "converged": bool(is_converged),
                "is_cycle": False,
            }

            for i in range(config.num_sellers):
                record[f"composite_a_{i}"] = int(active_action_indices[i])
                record[f"a_{i}"] = int(config.active_strategies[active_rule_indices[i]])
                record[f"k_{i}"] = int(active_k_choices[i])
                record[f"decision_{i}"] = bool(decision_flags[i])
                record[f"remaining_blocks_{i}"] = int(remaining_blocks[i])
                record[f"p_{i}"] = float(p_t[i])
                record[f"pi_{i}"] = float(pi_t[i])

            eval_history.append(record)
            current_t_atomic += 1

        remaining_blocks -= 1
        state = int(next_state)

    df = pd.DataFrame(eval_history)
    if not df.empty:
        f_cols = df.select_dtypes(include=["float64"]).columns
        df[f_cols] = df[f_cols].astype("float32")
        i_cols = df.select_dtypes(include=["int64"]).columns
        df[i_cols] = df[i_cols].astype("int32")

    return df

