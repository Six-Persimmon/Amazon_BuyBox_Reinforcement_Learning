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
    A two-firm repeated Bertrand pricing environment.
    Actions: tuple (a_i, a_j) with each in {0,...,M-1}, indexing price grid.
    Observation: dummy constant (state is singleton).
    Rewards: profits for each firm given marginal cost and demand rule.
    """
    metadata = {"render_modes": []}

    def __init__(self,
                 price_min: float = 0.1,
                 price_max: float = 10.0,
                 grid_size: int = 100,
                 marginal_cost: float = 2.0):
        super().__init__()
        # discretized price grid
        self.prices = np.linspace(price_min, price_max, grid_size)
        self.cost = marginal_cost

        # multi‐agent action space
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

        # demand rule
        if p_i < p_j and p_i <= 10:
            d_i, d_j = 1.0, 0.0
        elif p_j < p_i and p_j <= 10:
            d_i, d_j = 0.0, 1.0
        elif p_i == p_j and p_i <= 10:
            d_i, d_j = 0.5, 0.5
        else:
            d_i, d_j = 0.0, 0.0

        # profits
        r_i = (p_i - self.cost) * d_i
        r_j = (p_j - self.cost) * d_j

        # no terminal state
        obs = 0
        done = False
        info = {}
        return obs, (r_i, r_j), done, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # always return the same dummy observation
        return 0, {}