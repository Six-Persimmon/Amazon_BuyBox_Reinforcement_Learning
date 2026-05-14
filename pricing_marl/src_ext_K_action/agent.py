import numpy as np

from .config import KActionConfig


def calculate_rule_heuristic_init_values(env, config: KActionConfig):
    """
    Reproduce the exp02 rule-level heuristic initialization from a base-K lookup.
    """
    profit_sums = np.zeros((config.num_grids, config.num_rule_actions))
    counts = np.zeros((config.num_grids, config.num_rule_actions))

    for key, val in env.lookup_table.items():
        start_state = key[0]
        canon_rule_indices = key[1:]
        profits = val[0]
        multiplicity = env.action_profile_multiplicity(canon_rule_indices)

        for i, rule_action_idx in enumerate(canon_rule_indices):
            profit_sums[start_state, rule_action_idx] += profits[i] * multiplicity
            counts[start_state, rule_action_idx] += multiplicity

    with np.errstate(divide="ignore", invalid="ignore"):
        avg_profits = profit_sums / counts
        avg_profits = np.nan_to_num(avg_profits)

    return avg_profits / (1 - config.gamma)


def calculate_composite_heuristic_init_values(env, config: KActionConfig):
    """
    Broadcast rule-level heuristic Q values to all composite actions sharing a rule.
    This keeps initialization neutral over K choices.
    First, calculate (G, Rule_Action) heuristic values, then broadcast to ( G, Rule_action X K_choice ) according to the action_map.
    """
    rule_q = calculate_rule_heuristic_init_values(env, config)
    composite_q = np.zeros((config.num_grids, config.num_actions))

    for meta in config.action_map:
        composite_q[:, meta["composite_action_idx"]] = rule_q[:, meta["rule_action_idx"]]

    return composite_q


class KActionQAgent:
    def __init__(self, agent_id, config: KActionConfig, initial_Q_table=None):
        self.id = agent_id
        self.cfg = config
        self.allowed_action_indices = np.array(
            config.allowed_action_indices_by_agent[agent_id],
            dtype=int,
        )

        if initial_Q_table is not None:
            # each row is a state, each column is a composite action in the form of (rule_action_idx, k_choice)
            self.Q = np.copy(initial_Q_table)
        else:
            self.Q = np.zeros((config.num_grids, config.num_actions))

        self.forced_action = None

    def choose_action(self, observation, t_step):
        if self.forced_action is not None:
            return self.forced_action

        epsilon = np.exp(-self.cfg.beta * t_step)
        if np.random.rand() < epsilon:
            return int(np.random.choice(self.allowed_action_indices))

        return self.choose_greedy_action(observation)

    def choose_greedy_action(self, observation):
        allowed_q_values = self.Q[observation, self.allowed_action_indices]
        max_q = np.max(allowed_q_values)
        best_positions = np.where(allowed_q_values == max_q)[0]
        best_actions = self.allowed_action_indices[best_positions]
        return int(np.random.choice(best_actions))

    def update(self, obs, action_idx, reward, next_obs):
        current_q = self.Q[obs, action_idx]
        max_next_q = np.max(self.Q[next_obs, self.allowed_action_indices])
        new_q = (1 - self.cfg.alpha) * current_q + self.cfg.alpha * (
            reward + self.cfg.gamma * max_next_q
        )
        self.Q[obs, action_idx] = new_q

    def get_greedy_policy(self):
        allowed_q_values = self.Q[:, self.allowed_action_indices]
        best_positions = np.argmax(allowed_q_values, axis=1)
        return self.allowed_action_indices[best_positions]

    def set_forced_action(self, action_idx):
        self.forced_action = action_idx

    def clear_forced_action(self):
        self.forced_action = None
