import itertools
import pickle
import time
from math import factorial
from pathlib import Path

import numpy as np
from filelock import FileLock, Timeout
from scipy.optimize import minimize

from .config import KActionConfig
from .strategies import get_strategy_function


def get_demand_and_profit_static(prices, a_val, mu, a0, c_val):
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


def compute_nash_and_monopoly_static(num_sellers, a_val, mu, a0, c_val):
    def neg_joint_profit(p_list):
        p = p_list[0]
        prices = [p] * num_sellers
        prof = get_demand_and_profit_static(prices, a_val, mu, a0, c_val)
        return -np.sum(prof)

    def neg_own_profit(p_own_list, p_comp):
        p_own = p_own_list[0]
        prices = [p_own] + [p_comp] * (num_sellers - 1)
        prof = get_demand_and_profit_static(prices, a_val, mu, a0, c_val)
        return -prof[0]

    res_mon = minimize(neg_joint_profit, x0=[1.5], bounds=[(1.0, 5.0)])
    p_mon = res_mon.x[0]

    p_nash = 1.1
    for _ in range(10):
        res_br = minimize(
            neg_own_profit,
            x0=[p_nash],
            args=(p_nash,),
            bounds=[(1.0, 5.0)],
        )
        p_nash = res_br.x[0]

    return p_nash, p_mon


class BaseKLookupEnvironment:
    """
    Lowest-state environment for exp03.

    Each step advances exactly config.base_K atomic periods and then compresses
    the state to the resulting lowest price index. Lookup filenames match the
    exp02 convention, with K replaced by base_K, so existing K=10 stateLow
    tables can be reused.
    """

    def __init__(self, config: KActionConfig):
        self.cfg = config
        self.p_nash, self.p_monopoly = compute_nash_and_monopoly_static(
            self.cfg.num_sellers,
            self.cfg.a_val,
            self.cfg.mu,
            self.cfg.a0,
            self.cfg.c_val,
        )
        self.price_grid = self._build_grid()
        self.lookup_table = self._get_or_create_lookup_table()

    def _get_demand_and_profit(self, prices):
        return get_demand_and_profit_static(
            prices,
            self.cfg.a_val,
            self.cfg.mu,
            self.cfg.a0,
            self.cfg.c_val,
        )

    def _build_grid(self):
        step = (self.p_monopoly - self.p_nash) / (self.cfg.num_grids - 3)
        price_grid = np.linspace(
            self.p_nash - step,
            self.p_monopoly + step,
            self.cfg.num_grids,
        )
        self.grid_min_before_floor = float(price_grid[0])
        self.grid_floor_applied = bool(price_grid[0] < self.cfg.c_val)
        if self.grid_floor_applied:
            price_grid[0] = self.cfg.c_val
        self.grid_min_after_floor = float(price_grid[0])
        self.nash_idx = 1
        self.monopoly_idx = self.cfg.num_grids - 2
        return price_grid

    def canonicalize_actions(self, rule_action_indices):
        perm = sorted(range(len(rule_action_indices)), key=lambda i: rule_action_indices[i])
        canon_actions = tuple(rule_action_indices[i] for i in perm)
        inv_perm = [0] * len(rule_action_indices)
        for canon_pos, original_pos in enumerate(perm):
            inv_perm[original_pos] = canon_pos
        return canon_actions, perm, inv_perm

    def action_profile_multiplicity(self, canon_actions):
        counts = {}
        for action in canon_actions:
            counts[action] = counts.get(action, 0) + 1
        denom = 1
        for freq in counts.values():
            denom *= factorial(freq)
        return factorial(len(canon_actions)) // denom

    def _get_lookup_path(self):
        strats_str = self.cfg.get_strategy_string()
        filename = (
            f"lookup_N{self.cfg.num_sellers}_G{self.cfg.num_grids}_"
            f"mu{self.cfg.mu}_a{self.cfg.a_val}_c{self.cfg.c_val}_a0{self.cfg.a0}_"
            f"xi{self.cfg.xi}_K{self.cfg.base_K}_"
            f"strats{strats_str}_stateLow"
            f"{'_floorC' if self.grid_floor_applied else ''}.pkl"
        )
        return self.cfg.lookup_dir / filename

    def _get_or_create_lookup_table(self):
        file_path = self._get_lookup_path()
        lock = FileLock(Path(f"{file_path}.lock"))
        lock_timeout = 3 * 60 * 60
        max_retries = 5
        retry_backoff = 5

        for attempt in range(max_retries):
            try:
                with lock.acquire(timeout=lock_timeout):
                    if file_path.exists():
                        print(f"[Env exp03] Loading cached base-K table: {file_path.name}")
                        with open(file_path, "rb") as f:
                            return pickle.load(f)

                    print(f"[Env exp03] Computing new base-K table: {file_path.name}...")
                    return self._compute_lookup_table(file_path)
            except Timeout:
                if attempt == max_retries - 1:
                    raise TimeoutError(
                        f"Timeout waiting for lookup table lock after {lock_timeout}s"
                    )
                time.sleep(retry_backoff)

    def _compute_lookup_table(self, file_path):
        table = {}
        action_combinations = list(
            itertools.product(
                range(self.cfg.num_rule_actions),
                repeat=self.cfg.num_sellers,
            )
        )

        for s_idx in range(self.cfg.num_grids):
            for actions in action_combinations:
                canon_actions, _, _ = self.canonicalize_actions(actions)
                key = tuple([s_idx] + list(canon_actions))

                if key in table:
                    continue

                current_indices = [s_idx] * self.cfg.num_sellers
                cum_profits = np.zeros(self.cfg.num_sellers)
                sum_avg_price = 0.0
                sum_lowest_price = 0.0

                current_funcs = [
                    get_strategy_function(self.cfg.active_strategies[action_idx])
                    for action_idx in canon_actions
                ]

                for _ in range(self.cfg.base_K):
                    next_indices = []
                    for i in range(self.cfg.num_sellers):
                        others = current_indices[:i] + current_indices[i + 1 :]
                        others_min = min(others)
                        p_next = current_funcs[i](
                            current_indices[i],
                            others_min,
                            self.cfg.num_grids,
                        )
                        next_indices.append(p_next)

                    current_indices = next_indices
                    prices = [self.price_grid[idx] for idx in current_indices]
                    sum_avg_price += float(np.mean(prices))
                    sum_lowest_price += float(np.min(prices))
                    cum_profits += self._get_demand_and_profit(prices)

                table[key] = (
                    cum_profits / self.cfg.base_K,
                    min(current_indices),
                    sum_avg_price / self.cfg.base_K,
                    sum_lowest_price / self.cfg.base_K,
                )

        tmp_path = Path(f"{file_path}.tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(table, f)
        tmp_path.replace(file_path)
        return table

    def step(self, state_idx, rule_action_indices, return_details=False):
        if not return_details:
            canon_actions, _, inv_perm = self.canonicalize_actions(rule_action_indices)
            key = tuple([state_idx] + list(canon_actions))
            if key not in self.lookup_table:
                raise ValueError(f"Key {key} not found in lookup table.")

            res = self.lookup_table[key]
            profits_canon = res[0]
            profits = np.zeros(self.cfg.num_sellers)
            for i in range(self.cfg.num_sellers):
                profits[i] = profits_canon[inv_perm[i]]
            return profits, res[1], res[2], res[3]

        return self._simulate_base_k_steps_detailed(state_idx, rule_action_indices)

    def _simulate_base_k_steps_detailed(self, start_idx, rule_action_indices):
        current_indices = [start_idx] * self.cfg.num_sellers
        current_funcs = [
            get_strategy_function(self.cfg.active_strategies[action_idx])
            for action_idx in rule_action_indices
        ]

        trajectory = {
            "prices": [],
            "profits": [],
        }
        cum_profits = np.zeros(self.cfg.num_sellers)

        for _ in range(self.cfg.base_K):
            current_prices = [self.price_grid[idx] for idx in current_indices]
            trajectory["prices"].append(current_prices)

            profits = self._get_demand_and_profit(current_prices)
            trajectory["profits"].append(profits)
            cum_profits += profits

            next_indices = []
            for i in range(self.cfg.num_sellers):
                others = current_indices[:i] + current_indices[i + 1 :]
                others_min = min(others)
                p_next_idx = current_funcs[i](
                    current_indices[i],
                    others_min,
                    self.cfg.num_grids,
                )
                next_indices.append(p_next_idx)
            current_indices = next_indices

        avg_profits = cum_profits / self.cfg.base_K
        next_state_min = min(current_indices)
        flat_prices = np.array(trajectory["prices"])
        avg_price = np.mean(flat_prices)
        avg_lowest = np.mean(np.min(flat_prices, axis=1))

        return avg_profits, next_state_min, avg_price, avg_lowest, trajectory

