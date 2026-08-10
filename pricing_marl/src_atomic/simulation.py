"""
Atomic-time simulation runs (no lookup table, no state collapse).

- run_simulation_sync_carryover (exp07): synchronous K-blocks, but the price
  vector carries over across episodes instead of collapsing to the market min.
- run_simulation_async (exp08): per-agent stochastic revision clocks with
  mean gap lambda_K; sellers revise (Q-update + rule pick) only at their own
  clock events, prices evolve every atomic period.

Both return the same (eval_df, q_snapshot_df) structure as
src.simulation.run_simulation so src-style runners/analysis work unchanged.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.simulation import _build_q_snapshot_df
from src_atomic.agent import QAgent, heuristic_init_from_baseline_lookup
from src_atomic.config import AtomicConfig
from src_atomic.environment import AtomicEnvironment


def _draw_gap(config: AtomicConfig) -> int:
    """Draw one revision gap (>= 1 atomic period)."""
    if config.gap_distribution == "poisson":
        return max(1, int(np.random.poisson(config.lambda_K)))
    # geometric: revise each period w.p. 1/lambda_K => gap ~ Geometric
    return int(np.random.geometric(1.0 / config.lambda_K))


# =====================================================================
# Mode 1: synchronous blocks with carry-over prices (exp07)
# =====================================================================

def run_simulation_sync_carryover(
    config: AtomicConfig,
    run_id: int = 0,
    disable_tqdm: bool = False,
    return_q_snapshot: bool = False,
):
    env = AtomicEnvironment(config)

    init_q = heuristic_init_from_baseline_lookup(config)
    agents = [QAgent(i, config, initial_Q_table=init_q) for i in range(config.num_sellers)]

    q_snapshot_init_df = None
    if return_q_snapshot:
        q_snapshot_init_df = _build_q_snapshot_df(
            agents=agents, config=config, run_id=run_id,
            training_stop_episode=-1, snapshot_type="init",
        )

    # Initial condition: each seller's price index randomized independently
    price_vec = [int(v) for v in np.random.randint(0, config.num_grids, size=config.num_sellers)]

    # --- PHASE 1: TRAINING ---
    is_converged = False
    policy_stable_counter = 0
    prev_policy = np.zeros((config.num_sellers, config.num_grids), dtype=int)
    training_stop_episode = -1

    # Deterministic block dynamics => memoize (price vector, rule profile)
    block_memo = {}

    iterator = range(config.max_episodes)
    if not disable_tqdm:
        iterator = tqdm(iterator, desc=f"Run {run_id} [Train nocollapse]")

    for t in iterator:
        obs = min(price_vec)
        actions_indices = [agent.choose_action(obs, t) for agent in agents]

        key = (tuple(price_vec), tuple(actions_indices))
        cached = block_memo.get(key)
        if cached is None:
            rewards, next_vec = env.run_block(price_vec, actions_indices, config.K)
            block_memo[key] = (rewards, next_vec)
        else:
            rewards, next_vec = cached

        next_obs = min(next_vec)
        for i, agent in enumerate(agents):
            agent.update(obs, actions_indices[i], rewards[i], next_obs)

        current_policy = np.array([agent.get_greedy_policy() for agent in agents])
        if np.array_equal(current_policy, prev_policy):
            policy_stable_counter += 1
        else:
            policy_stable_counter = 0
            prev_policy = current_policy

        if policy_stable_counter >= config.converge_period:
            is_converged = True
            training_stop_episode = int(t)
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

    # --- PHASE 2: EVALUATION ---
    num_eval_episodes = max(1, config.eval_H // config.K)
    eval_history = []
    current_t_atomic = 0

    for _ in range(num_eval_episodes):
        obs = min(price_vec)
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
                record[f"a_{i}"] = config.active_strategies[actions_indices[i]]
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


# =====================================================================
# Mode 2: asynchronous Poisson revision clocks (exp08)
# =====================================================================

def run_simulation_async(
    config: AtomicConfig,
    run_id: int = 0,
    disable_tqdm: bool = False,
    return_q_snapshot: bool = False,
):
    env = AtomicEnvironment(config)

    init_q = heuristic_init_from_baseline_lookup(config)
    agents = [QAgent(i, config, initial_Q_table=init_q) for i in range(config.num_sellers)]
    n = config.num_sellers

    q_snapshot_init_df = None
    if return_q_snapshot:
        q_snapshot_init_df = _build_q_snapshot_df(
            agents=agents, config=config, run_id=run_id,
            training_stop_episode=-1, snapshot_type="init",
        )

    price_vec = [int(v) for v in np.random.randint(0, config.num_grids, size=n)]

    # Per-agent revision state
    obs0 = min(price_vec)
    current_actions = [agent.choose_action(obs0, t_step=0) for agent in agents]  # eps=1
    prev_obs = [obs0] * n
    prev_act = list(current_actions)
    accum = [0.0] * n
    window = [0] * n
    n_picks = [1] * n  # each agent has made its initial pick
    next_rev = [1 + _draw_gap(config) for _ in range(n)]  # first move happens at tau=1

    # Convergence tracking in atomic time
    prev_policy = np.array([agent.get_greedy_policy() for agent in agents])
    last_policy_change_tau = 0
    is_converged = False
    training_stop_tau = -1

    max_tau = config.max_atomic_periods
    progress = None
    if not disable_tqdm:
        progress = tqdm(total=max_tau, desc=f"Run {run_id} [Train async]", mininterval=5.0)

    tau = 0
    while tau < max_tau:
        tau += 1

        # 1. Revision events (before this period's move; observe current state)
        revised_any = False
        for i in range(n):
            if next_rev[i] == tau:
                new_obs = min(price_vec)
                if window[i] > 0:
                    reward = accum[i] / window[i]
                    agents[i].update(prev_obs[i], prev_act[i], reward, new_obs)
                    revised_any = True
                # epsilon decays in the agent's own revision count
                action = agents[i].choose_action(new_obs, t_step=n_picks[i])
                n_picks[i] += 1
                current_actions[i] = action
                prev_obs[i] = new_obs
                prev_act[i] = action
                accum[i] = 0.0
                window[i] = 0
                next_rev[i] = tau + _draw_gap(config)

        if revised_any:
            current_policy = np.array([agent.get_greedy_policy() for agent in agents])
            if not np.array_equal(current_policy, prev_policy):
                prev_policy = current_policy
                last_policy_change_tau = tau

        # 2. Synchronous price move under current rules; accrue profits
        price_vec = env.step_one_period(price_vec, current_actions)
        profits = env.profit_table[tuple(price_vec)]
        for i in range(n):
            accum[i] += float(profits[i])
            window[i] += 1

        # 3. Convergence check (atomic-time analog of baseline)
        if tau - last_policy_change_tau >= config.converge_atomic_periods:
            is_converged = True
            training_stop_tau = tau
            break

        if progress is not None and tau % 100_000 == 0:
            progress.update(100_000)

    if progress is not None:
        progress.close()
    if training_stop_tau < 0:
        training_stop_tau = tau

    q_snapshot_df = None
    if return_q_snapshot:
        q_snapshot_final_df = _build_q_snapshot_df(
            agents=agents, config=config, run_id=run_id,
            training_stop_episode=training_stop_tau, snapshot_type="final",
        )
        q_snapshot_df = pd.concat([q_snapshot_init_df, q_snapshot_final_df], ignore_index=True)

    # --- PHASE 2: EVALUATION (greedy, frozen Q, clocks keep running) ---
    eval_history = []
    for t_eval in range(config.eval_H):
        tau += 1
        revised_flags = [False] * n

        for i in range(n):
            if next_rev[i] == tau:
                new_obs = min(price_vec)
                current_actions[i] = agents[i].choose_action(new_obs, t_step=1e9)  # greedy
                revised_flags[i] = True
                next_rev[i] = tau + _draw_gap(config)

        price_vec = env.step_one_period(price_vec, current_actions)
        profits = env.profit_table[tuple(price_vec)]
        p_t = env.prices(price_vec)

        record = {
            "run_id": run_id,
            "t_global": t_eval,
            "episode": training_stop_tau,  # atomic period at training stop
            "step_in_k": -1,               # no synchronous K blocks in async mode
            "price_min": float(np.min(p_t)),
            "price_mean": float(np.mean(p_t)),
            "delta": (float(np.mean(profits)) - env.pi_nash) / env.delta_denominator,
            "converged": is_converged,
            "is_cycle": False,
        }
        for i in range(n):
            record[f"a_{i}"] = config.active_strategies[current_actions[i]]
            record[f"p_{i}"] = float(p_t[i])
            record[f"pi_{i}"] = float(profits[i])
            record[f"rev_{i}"] = revised_flags[i]

        eval_history.append(record)

    df = pd.DataFrame(eval_history)
    f_cols = df.select_dtypes(include=["float64"]).columns
    df[f_cols] = df[f_cols].astype("float32")
    i_cols = df.select_dtypes(include=["int64"]).columns
    df[i_cols] = df[i_cols].astype("int32")

    if return_q_snapshot:
        return df, q_snapshot_df
    return df


# =====================================================================
# Dispatcher
# =====================================================================

def run_simulation(
    config: AtomicConfig,
    run_id: int = 0,
    disable_tqdm: bool = False,
    return_q_snapshot: bool = False,
):
    if config.mode == "sync_carryover":
        return run_simulation_sync_carryover(
            config, run_id=run_id, disable_tqdm=disable_tqdm,
            return_q_snapshot=return_q_snapshot,
        )
    if config.mode == "async_poisson":
        return run_simulation_async(
            config, run_id=run_id, disable_tqdm=disable_tqdm,
            return_q_snapshot=return_q_snapshot,
        )
    raise ValueError(f"Unknown mode: {config.mode}")
