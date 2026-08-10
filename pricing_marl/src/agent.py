import numpy as np
from src.config import Config

def calculate_heuristic_init_values(env, config: Config):
    """
    Heuristic initialization.
    Iterates through lookup table to find average profit per Action ID.
    With action canonicalization, weight each canonical entry by the number of
    permutations it represents to mimic uniform random opponents.
    """
    profit_sums = np.zeros((config.num_grids, config.num_actions))
    counts = np.zeros((config.num_grids, config.num_actions))

    for key, val in env.lookup_table.items():
        start_state = key[0]
        canon_actions_indices = key[1:]  # indices into active_strategies list
        profits = val[0]                 # profits are aligned with canonical action order
        multiplicity = env.action_profile_multiplicity(canon_actions_indices)

        for i, act_idx in enumerate(canon_actions_indices):
            profit_sums[start_state, act_idx] += profits[i] * multiplicity
            counts[start_state, act_idx] += multiplicity
            
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_profits = profit_sums / counts
        avg_profits = np.nan_to_num(avg_profits)
        
    return avg_profits / (1 - config.gamma)


class QAgent:
    """
    Independent Q-Learning Agent.
    """
    def __init__(self, agent_id, config: Config, initial_Q_table=None):
        self.id = agent_id
        self.cfg = config

        # Q-Table: Shape [Num_Grids, Num_Actions]
        if initial_Q_table is not None:
            self.Q = np.copy(initial_Q_table)
        else:
            self.Q = np.zeros((config.num_grids, config.num_actions))

        # Exploration distribution: None = uniform (baseline)
        weights = getattr(config, "exploration_weights", None)
        if weights is None:
            self.explore_probs = None
        else:
            probs = np.asarray(weights, dtype=float)
            if probs.shape != (config.num_actions,):
                raise ValueError(
                    f"exploration_weights must have length {config.num_actions}, "
                    f"got {probs.shape}"
                )
            if np.any(probs < 0) or not np.isclose(probs.sum(), 1.0):
                raise ValueError("exploration_weights must be non-negative and sum to 1.")
            self.explore_probs = probs / probs.sum()

        # Deviation hook
        self.forced_action = None

    def choose_action(self, observation, t_step):
        """
        Selects an action using epsilon-greedy policy.
        """
        # 1. Forced deviation
        if self.forced_action is not None:
            return self.forced_action

        # 2. Epsilon Calculation
        epsilon = np.exp(-self.cfg.beta * t_step)
        
        # 3. Epsilon-Greedy
        if np.random.rand() < epsilon:
            if self.explore_probs is not None:
                return int(np.random.choice(self.cfg.num_actions, p=self.explore_probs))
            return np.random.randint(self.cfg.num_actions)
        else:
            # Greedy with random tie-breaking
            q_values = self.Q[observation, :]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)

    def update(self, obs, action_idx, reward, next_obs):
        """Standard Q-Learning Update"""
        current_q = self.Q[obs, action_idx]
        max_next_q = np.max(self.Q[next_obs, :])
        
        new_q = (1 - self.cfg.alpha) * current_q + \
                self.cfg.alpha * (reward + self.cfg.gamma * max_next_q)
        
        self.Q[obs, action_idx] = new_q

    # Helper for Convergence Checking
    def get_greedy_policy(self):
        """
        Returns the current optimal action index for every state.
        Shape: (num_grids,)
        Deterministic Argmax.
        """
        return np.argmax(self.Q, axis=1)

    def set_forced_action(self, action_idx):
        self.forced_action = action_idx

    def clear_forced_action(self):
        self.forced_action = None
