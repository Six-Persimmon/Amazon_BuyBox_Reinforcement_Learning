"""
Training + evaluation for one exp09 run.

One unified atomic loop serves all five rungs of the ladder: observe -> choose
-> prices evolve for K periods -> reward = average profit over those K periods
-> new state. With K = 1 and price actions this is Calvano et al. (2020)
exactly; with market_min + rules it is the mechanism used in the main paper.

Timing note. Our state is the CURRENT price vector and the action determines
the NEXT one, which is Calvano's `s_t = p_{t-1}` convention re-indexed by one.
Identical structure, and it makes the reward for a rule (realized one period
after the rule is chosen) line up with the reward for a price.

Returns the same (eval_df, q_snapshot_df) pair as src.simulation.run_simulation
so the existing analysis loaders work unchanged.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from src_calvano.agent import CalvanoQAgent, heuristic_init
from src_calvano.config import CalvanoConfig
from src_calvano.environment import CalvanoEnvironment


def _build_q_snapshot_df(agents, config: CalvanoConfig, run_id, training_stop_episode, snapshot_type):
    """
    Flatten all sellers' Q-tables into a tidy DataFrame.

    Same schema as src.simulation._build_q_snapshot_df, but iterates
    `num_states` (up to 1000 here) instead of `num_grids`, and `action_id`
    holds the price grid index in price mode / the strategy ID in rule mode.
    """
    n_states, n_act = config.num_states, config.num_actions
    n_agents = len(agents)
    total = n_agents * n_states * n_act

    seller_ids = np.empty(total, dtype=np.int16)
    state_idx = np.empty(total, dtype=np.int16 if n_states <= 32767 else np.int32)
    action_idx = np.empty(total, dtype=np.int16)
    action_id = np.empty(total, dtype=np.int16)
    q_values = np.empty(total, dtype=np.float32)

    action_ids_row = np.array([config.action_id(a) for a in range(n_act)], dtype=np.int16)
    states_col = np.repeat(np.arange(n_states), n_act)
    actions_col = np.tile(np.arange(n_act), n_states)

    block = n_states * n_act
    for k, agent in enumerate(agents):
        sl = slice(k * block, (k + 1) * block)
        seller_ids[sl] = int(agent.id)
        state_idx[sl] = states_col
        action_idx[sl] = actions_col
        action_id[sl] = np.tile(action_ids_row, n_states)
        q_values[sl] = np.asarray(agent.Q, dtype=np.float32).ravel()

    return pd.DataFrame({
        "run_id": np.full(total, int(run_id), dtype=np.int32),
        "seller_id": seller_ids,
        "state_idx": state_idx,
        "action_idx": action_idx,
        "action_id": action_id,
        "q_value": q_values,
        "training_stop_episode": np.full(total, int(training_stop_episode), dtype=np.int32),
        "qtable_type": snapshot_type,
    })


def run_simulation(
    config: CalvanoConfig,
    run_id: int = 0,
    disable_tqdm: bool = False,
    return_q_snapshot: bool = False,
):
    env = CalvanoEnvironment(config)

    # heuristic_init returns one Q-table per seller; the slices differ only
    # for full_vector + rule actions under init_pooling="per_seller".
    init_q = heuristic_init(config, env)
    agents = [CalvanoQAgent(i, config, initial_Q_table=init_q[i])
              for i in range(config.num_sellers)]

    q_snapshot_init_df = None
    if return_q_snapshot:
        q_snapshot_init_df = _build_q_snapshot_df(
            agents=agents, config=config, run_id=run_id,
            training_stop_episode=-1, snapshot_type="init",
        )

    # t = 0: each seller's price index drawn independently at random
    # (Calvano's "s_0 is drawn randomly", and the exp07 convention).
    price_vec = [int(v) for v in np.random.randint(0, config.num_grids, size=config.num_sellers)]

    # --- PHASE 1: TRAINING ---
    is_converged = False
    policy_stable_counter = 0
    training_stop_episode = -1

    # Incremental convergence tracking. Only the single (s, a) cell an agent
    # just updated can change that agent's argmax, so we keep the greedy policy
    # matrix and refresh one row per agent per episode. This is exactly
    # equivalent to recomputing np.argmax(Q, axis=1) every episode, but the
    # full recompute measures ~34 us/period against a ~10 us/period loop.
    policy = np.array([agent.get_greedy_policy() for agent in agents])

    # Block dynamics are deterministic given (price vector, action profile),
    # so memoize them when a block spans several periods (exp07's trick).
    block_memo = {} if config.K > 1 else None

    iterator = range(config.max_episodes)
    if not disable_tqdm:
        iterator = tqdm(iterator, desc=f"Run {run_id} [Train {config.cell_tag}]")

    t = 0
    for t in iterator:
        obs = env.encode(price_vec)
        actions_indices = [agent.choose_action(obs, t) for agent in agents]

        if block_memo is not None:
            key = (tuple(price_vec), tuple(actions_indices))
            cached = block_memo.get(key)
            if cached is None:
                rewards, next_vec = env.run_block(price_vec, actions_indices, config.K)
                block_memo[key] = (rewards, next_vec)
            else:
                rewards, next_vec = cached
        else:
            rewards, next_vec = env.run_block(price_vec, actions_indices, config.K)

        next_obs = env.encode(next_vec)

        policy_changed = False
        for i, agent in enumerate(agents):
            agent.update(obs, actions_indices[i], rewards[i], next_obs)
            new_best = int(np.argmax(agent.Q[obs, :]))
            if new_best != policy[i, obs]:
                policy[i, obs] = new_best
                policy_changed = True

        if policy_changed:
            policy_stable_counter = 0
        else:
            policy_stable_counter += 1
            if policy_stable_counter >= config.converge_period:
                is_converged = True
                training_stop_episode = int(t)
                if not disable_tqdm:
                    print(f"  Run {run_id}: Converged at episode {t}")
                break

        price_vec = list(next_vec)

    if training_stop_episode < 0:
        training_stop_episode = int(t) if config.max_episodes > 0 else 0

    q_snapshot_df = None
    if return_q_snapshot:
        q_snapshot_final_df = _build_q_snapshot_df(
            agents=agents, config=config, run_id=run_id,
            training_stop_episode=training_stop_episode, snapshot_type="final",
        )
        q_snapshot_df = pd.concat([q_snapshot_init_df, q_snapshot_final_df], ignore_index=True)

    # --- PHASE 2: EVALUATION (greedy, frozen Q) ---
    num_eval_episodes = max(1, config.eval_H // config.K)
    eval_history = []
    current_t_atomic = 0

    for _ in range(num_eval_episodes):
        obs = env.encode(price_vec)
        actions_indices = [agent.choose_action(obs, t_step=1e9) for agent in agents]

        trajectory, final_vec = env.run_block_detailed(price_vec, actions_indices, config.K)

        for k in range(config.K):
            p_t = trajectory["prices"][k]
            pi_t = trajectory["profits"][k]

            record = {
                "run_id": run_id,
                "t_global": current_t_atomic,
                "episode": training_stop_episode,
                "step_in_k": k,
                "price_min": float(np.min(p_t)),
                "price_mean": float(np.mean(p_t)),
                "delta": (float(np.mean(pi_t)) - env.pi_nash) / env.delta_denominator,
                "converged": is_converged,
                "is_cycle": False,
            }
            for i in range(config.num_sellers):
                record[f"a_{i}"] = config.action_id(actions_indices[i])
                record[f"p_{i}"] = float(p_t[i])
                record[f"pi_{i}"] = float(pi_t[i])

            eval_history.append(record)
            current_t_atomic += 1

        price_vec = list(final_vec)

    df = pd.DataFrame(eval_history)
    f_cols = df.select_dtypes(include=["float64"]).columns
    df[f_cols] = df[f_cols].astype("float32")
    i_cols = df.select_dtypes(include=["int64"]).columns
    df[i_cols] = df[i_cols].astype("int32")

    if return_q_snapshot:
        return df, q_snapshot_df
    return df
