#!/usr/bin/env python

# -*-coding:utf-8 -*-
'''
@File    :   bertrand_env.py
@Time    :   2025/04/16 16:05:33
@Author  :   Shijian Liu
@Version :   1.0
@Contact :   lshijian405@gmail.com
@Desc    :   
    A two-firm repeated Bertrand pricing environment.
    Actions: tuple (a_i, a_j) with each in {0,...,M-1}, indexing price grid.
    Observation: dummy constant (state is singleton).
    Rewards: profits for each firm given marginal cost and demand rule.

'''


import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BertrandPricingEnv(gym.Env):
    """
    Two‑firm discrete Bertrand pricing game.
    - Prices on a grid from price_min to price_max (inclusive).
    - Demand: firm with lower price gets full demand = 1;
      if equal price, split demand = 0.5 each; otherwise zero.
    - Profit = (price – cost) * demand.
    State is a singleton; no terminal condition.
    """
    metadata = {"render_modes": []}

    def __init__(self,
                 price_min: float = 0.01,
                 price_max: float = 10.0,
                 grid_size: int = 100,
                 marginal_cost: float = 2.0):
        super().__init__()
        self.prices = np.linspace(price_min, price_max, grid_size)
        self.cost = marginal_cost

        # each firm’s action is picking an index in {0,…,grid_size–1}
        self.action_space = spaces.Tuple((
            spaces.Discrete(grid_size),
            spaces.Discrete(grid_size)
        ))
        # dummy observation
        self.observation_space = spaces.Discrete(1)

    def step(self, actions):
        a_i, a_j = actions
        p_i = self.prices[a_i]
        p_j = self.prices[a_j]

        # determine demand
        if p_i < p_j and p_i <= self.prices[-1]:
            d_i, d_j = 1.0, 0.0
        elif p_j < p_i and p_j <= self.prices[-1]:
            d_i, d_j = 0.0, 1.0
        elif p_i == p_j and p_i <= self.prices[-1]:
            d_i, d_j = 0.5, 0.5
        else:
            d_i, d_j = 0.0, 0.0

        # profits
        r_i = (p_i - self.cost) * d_i
        r_j = (p_j - self.cost) * d_j

        return 0, (r_i, r_j), False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return 0, {}