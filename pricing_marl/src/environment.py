import numpy as np
import itertools
import pickle
from math import factorial
from scipy.optimize import minimize
from src.config import Config
from src.strategies import get_strategy_function

def get_demand_and_profit_static(prices, a_val, mu, a0, c_val):
    """
    Static version of demand calculation, no class dependency.
    """
    prices = np.array(prices)
    # Logit calculation
    logits = (a_val - prices) / mu
    logit0 = a0 / mu
    
    # Stability trick
    max_logit = np.max(logits)
    max_logit = max(max_logit, logit0)
    
    exps = np.exp(logits - max_logit)
    exp0 = np.exp(logit0 - max_logit)
    
    denom = np.sum(exps) + exp0
    demands = exps / denom
    
    profits = (prices - c_val) * demands
    return profits

def compute_nash_and_monopoly_static(num_sellers, a_val, mu, a0, c_val):
    """
    Calculates Stage-Game Nash and Monopoly prices using Scipy.
    Pure math, extremely fast.
    """
    # 1. Helper: Negative Joint Profit (for Monopoly)
    def neg_joint_profit(p_list):
        # Monopoly implies symmetric prices
        p = p_list[0]
        prices = [p] * num_sellers
        prof = get_demand_and_profit_static(prices, a_val, mu, a0, c_val)
        return -np.sum(prof)

    # 2. Helper: Negative Own Profit (for Nash Best Response)
    def neg_own_profit(p_own_list, p_comp):
        p_own = p_own_list[0]
        prices = [p_own] + [p_comp] * (num_sellers - 1)
        prof = get_demand_and_profit_static(prices, a_val, mu, a0, c_val)
        return -prof[0]

    # --- Solve Monopoly ---
    res_mon = minimize(neg_joint_profit, x0=[1.5], bounds=[(1.0, 5.0)])
    p_mon = res_mon.x[0]

    # --- Solve Nash ---
    p_nash = 1.1 # Initial guess
    for _ in range(10):
        res_br = minimize(neg_own_profit, x0=[p_nash], args=(p_nash,), bounds=[(1.0, 5.0)])
        p_nash = res_br.x[0]
        
    return p_nash, p_mon


# ==========================================
# Main Environment Class
# ==========================================

class PricingEnvironment:
    def __init__(self, config: Config):
        self.cfg = config
        
        # 1. Benchmarks (Now uses static helper)
        self.p_nash, self.p_monopoly = compute_nash_and_monopoly_static(
            self.cfg.num_sellers,
            self.cfg.a_val,
            self.cfg.mu,
            self.cfg.a0,
            self.cfg.c_val
        )

        
        # 2. Grid
        self.price_grid = self._build_grid()
        # Only print in main process to avoid spam
        # print(f"[Env] Grid: [{self.price_grid[0]:.3f} ... {self.price_grid[-1]:.3f}]")

        # 3. Lookup Table
        self.lookup_table = self._get_or_create_lookup_table()
        
    def _get_demand_and_profit(self, prices):
        return get_demand_and_profit_static(
            prices, self.cfg.a_val, self.cfg.mu, self.cfg.a0, self.cfg.c_val
        )

    def _build_grid(self):
        step = (self.p_monopoly - self.p_nash) / (self.cfg.num_grids - 3)
        price_grid = np.linspace(self.p_nash - step, self.p_monopoly + step, self.cfg.num_grids)
        # by construction, nash_idx is always 1, monopoly_idx is always num_grids - 2
        self.nash_idx = 1
        self.monopoly_idx = self.cfg.num_grids - 2
        return price_grid

    def canonicalize_actions(self, actions):
        """
        Canonicalize only the action profile (state is now a single lowest price index).
        Returns the sorted actions, a permutation to sorted order, and an inverse permutation
        to recover original agent ordering.
        """
        perm = sorted(range(len(actions)), key=lambda i: actions[i])
        canon_actions = tuple(actions[i] for i in perm)
        inv_perm = [0] * len(actions)
        for canon_pos, original_pos in enumerate(perm):
            inv_perm[original_pos] = canon_pos
        return canon_actions, perm, inv_perm

    def action_profile_multiplicity(self, canon_actions):
        """
        Number of distinct permutations represented by a canonical action multiset.
        Used for weighting heuristic Q initialization to mimic uniform random opponents.
        """
        counts = {}
        for a in canon_actions:
            counts[a] = counts.get(a, 0) + 1
        denom = 1
        for freq in counts.values():
            denom *= factorial(freq)
        return factorial(len(canon_actions)) // denom

    # ==========================================
    # --- LOOKUP TABLE ---
    # ==========================================

    def _get_or_create_lookup_table(self):
        # [MODIFIED] Use readable strategy string instead of hash
        strats_str = self.cfg.get_strategy_string()
        
        filename = (
            f"lookup_N{self.cfg.num_sellers}_G{self.cfg.num_grids}_"
            f"mu{self.cfg.mu}_a{self.cfg.a_val}_c{self.cfg.c_val}_a0{self.cfg.a0}_"
            f"xi{self.cfg.xi}_K{self.cfg.K}_"
            f"strats{strats_str}_stateLow.pkl"  # <--- explicitly named e.g. strats0_1_3.pkl
        )
        file_path = self.cfg.lookup_dir / filename

        if file_path.exists():
            print(f"[Env] Loading cached table: {filename}")
            with open(file_path, "rb") as f:
                return pickle.load(f)
        else:
            print(f"[Env] Computing new table: {filename}...")
            return self._compute_lookup_table(file_path)

    def _compute_lookup_table(self, file_path):
        '''
        Lookup Table contains:
        Key: (start_state, canonical_action_1, ..., canonical_action_N)
        Value: (array of profits per agent, next_lowest_state, avg_price_over_K, avg_lowest_price_over_K)
        '''
        table = {}

        action_combinations = list(itertools.product(
            range(self.cfg.num_actions), repeat=self.cfg.num_sellers
        ))

        # total = self.cfg.num_grids * len(action_combinations)
        # print(f"[Env] Simulating {total} scenarios for strategies: {self.cfg.active_strategies}")

        for s_idx in range(self.cfg.num_grids):
            for actions in action_combinations:
                # Canonicalize actions to collapse symmetric permutations
                canon_actions, _, _ = self.canonicalize_actions(actions)
                key = tuple([s_idx] + list(canon_actions))

                if key in table:
                    continue

                current_indices = [s_idx] * self.cfg.num_sellers
                cum_profits = np.zeros(self.cfg.num_sellers)
                sum_avg_price = 0.0
                sum_lowest_price = 0.0

                current_funcs = [
                    get_strategy_function(self.cfg.active_strategies[a_idx])
                    for a_idx in canon_actions
                ]

                # 实际运行中，K步迭代里是按照“其他seller的最低价格”、自己当前的price index、和自己选择的策略函数来决定下一步价格索引的
                for _ in range(self.cfg.K):
                    next_indices = []
                    for i in range(self.cfg.num_sellers):
                        others = current_indices[:i] + current_indices[i+1:]
                        others_min = min(others)
                        p_next = current_funcs[i](current_indices[i], others_min, self.cfg.num_grids)
                        next_indices.append(p_next)

                    current_indices = next_indices
                    prices = [self.price_grid[idx] for idx in current_indices]
                    sum_avg_price += float(np.mean(prices))
                    sum_lowest_price += float(np.min(prices))
                    cum_profits += self._get_demand_and_profit(prices)

                avg_price_over_k = sum_avg_price / self.cfg.K
                avg_lowest_over_k = sum_lowest_price / self.cfg.K
                table[key] = (
                    cum_profits / self.cfg.K, # shape: (num_sellers,)
                    min(current_indices),
                    avg_price_over_k,
                    avg_lowest_over_k
                )

        with open(file_path, "wb") as f:
            pickle.dump(table, f)
        return table
    
    def step(self, state_idx, actions_indices, return_details=False):
        """
        Look up the next state and rewards.
        Args:
            return_details: 如果为 True (评估模式)，则实时重新计算 K 步的路径，而不是只查表返回平均值。
        """
        # Canonicalize actions for lookup
        canon_actions = tuple(sorted(actions_indices))
        key = (state_idx, *canon_actions)

        if not return_details:
            # 训练阶段：极速查表
            canon_actions, perm, inv_perm = self.canonicalize_actions(actions_indices)
            key = tuple([state_idx] + list(canon_actions))

            if key not in self.lookup_table:
                # 理论上应该都在表里，除非 K 或 Grid 变了没清空缓存
                raise ValueError(f"Key {key} not found in lookup table.")
            
            res = self.lookup_table[key]
            # Un-permute profits to original agent order
            profits_canon = res[0]
            profits = np.zeros(self.cfg.num_sellers)
            for i in range(self.cfg.num_sellers):
                profits[i] = profits_canon[inv_perm[i]]
            return profits, res[1], res[2], res[3]
        else:
            # 评估阶段：我们需要每一小步的数据 (K-step unroll)
            # 这里我们需要重新运行 K 步模拟来获得轨迹
            return self._simulate_k_steps_detailed(state_idx, actions_indices)

    def _simulate_k_steps_detailed(self, start_idx, actions_indices):
        """
        Evaluation helper: Re-runs the K-step logic to return full trajectory.
        """
        current_indices = [start_idx] * self.cfg.num_sellers
        
        # 映射 Strategy ID 到函数
        current_funcs = [
            get_strategy_function(self.cfg.active_strategies[a_idx])
            for a_idx in actions_indices
        ]
        
        # 存储轨迹
        trajectory = {
            "prices": [],  # Shape: (K, N)
            "profits": []  # Shape: (K, N)
        }

        cum_profits = np.zeros(self.cfg.num_sellers)
        
        for _ in range(self.cfg.K):
            # 1. 记录当前价格
            current_prices = [self.price_grid[idx] for idx in current_indices]
            trajectory["prices"].append(current_prices)
            
            # 2. 计算当前利润
            profits = self._get_demand_and_profit(current_prices)
            trajectory["profits"].append(profits)
            cum_profits += profits
            
            # 3. 演化到下一步
            next_indices = []
            for i in range(self.cfg.num_sellers):
                # 每个人看到的是 excludes i 的 min price
                others = current_indices[:i] + current_indices[i+1:]
                others_min = min(others)
                p_next_idx = current_funcs[i](current_indices[i], others_min, self.cfg.num_grids)
                next_indices.append(p_next_idx)
            current_indices = next_indices

        # 返回格式要和查表对齐，但增加 info
        # info 包含 detailed trajectory
        avg_profits = cum_profits / self.cfg.K
        next_state_min = min(current_indices)
        
        # 计算查表里的那些平均值用于兼容（虽然 evaluation 此时可能不看这个）
        flat_prices = np.array(trajectory["prices"])
        avg_price = np.mean(flat_prices)
        avg_lowest = np.mean(np.min(flat_prices, axis=1))

        return avg_profits, next_state_min, avg_price, avg_lowest, trajectory
    
    
