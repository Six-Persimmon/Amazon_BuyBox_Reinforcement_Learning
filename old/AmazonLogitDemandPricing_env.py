#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   AmazonLogitDemandPricing_env.py
@Time    :   2025/04/26 14:33:16
@Author  :   Shijian Liu
@Version :   1.0
@Contact :   lshijian405@gmail.com
@Desc    :   A two-firm repeated pricing environment.
    Actions: tuple (a_i, a_j) with each in {0,...,M-1}, indexing price grid.
    Observation: tuple (p_i_lag, p_j_lag) with each in {0,...,M-1}, indexing price grid. Price pair played in last step.
    Rewards: profits for each firm given marginal cost and demand rule.
    Demand: Logit demand with outside option. The Amazon Featured Offer (or Buy Box) winner has an additional utility term in the demand function.
    The Buy Box is assigned to one of the sellers base on its features using a prediction model trained on real-world Amazon Buy Box assignment data.
    More about the prediction model:
        1. We obtained the historical pricingd data for the top 5000 best sellers in the Books category on Amazon. For each ASIN, we collecte the pricing history data for all the sellers, their offer/seller feature, and the Buy Box winner at each time step.
        2. We trained a prediction model to predict the Buy Box winner based on the seller features and the prices of all the sellers.
        3. We used the prediction model to generate the additional utility term for the Buy Box winner in the demand function. The prediction result is also stored in a tabular to avoid repeated computation.
'''

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class AmazonLogitDemandPricingEnv(gym.Env):
    """
    Two‑firm discrete Bertrand pricing game with Logit demand.
    - Prices on a grid from price_min to price_max (inclusive).
    - Demand: Logit demand with outside option.
    - Profit = (price – cost) * demand.
    State is a tuple of last prices played by each firm.
    """
    metadata = {"render_modes": []}

    def __init__(self,
                 price_min: float = 0.01,
                 price_max: float = 10.0,
                 grid_size: int = 100,
                 marginal_cost: float = 2.0,
                 beta: float = 0.95,
                 a_0=0, # parameter for logit demand. OUtside option
                 a_12=10, # parameter for logit demand. Inside option
                 mu=0.25 # parameter for logit demand. Vertical differentiation
                 ):
        super().__init__()
        self.prices = np.linspace(price_min, price_max, grid_size)
        self.price_min = price_min
        self.price_max = price_max
        self.cost = marginal_cost
        self.beta = beta
        self.a_0 = a_0
        self.a_12 = a_12
        self.mu = mu

        # each firm’s action is picking an index in {0,…,grid_size–1}
        self.action_space = spaces.Tuple((
            spaces.Discrete(grid_size),
            spaces.Discrete(grid_size)
        ))
        # observation is the last prices played by each firm
        self.observation_state = spaces.Tuple((
            spaces.Discrete(grid_size),
            spaces.Discrete(grid_size)
        ))

    def step(self, actions, buy_box):
        '''
        Logit demand system with outside option.
        The Buy Box winner receives additional utility to capture ~80% of inside-good demand.
        '''
        a_i, a_j = actions
        bb1, bb2 = buy_box  # 1 means winner, 0 means loser
        p_i = self.prices[a_i]
        p_j = self.prices[a_j]

        # Buy Box utility boost to shift ~80% of inside-good demand to winner
        bb_utility = 1.5  # Tunable parameter to achieve ~80% share, depending on mu

        # Utilities
        u_i = (self.a_12 - p_i) / self.mu + bb1 * bb_utility
        u_j = (self.a_12 - p_j) / self.mu + bb2 * bb_utility
        u_0 = self.a_0 / self.mu  # outside option utility

        # Softmax denominator
        deno = np.exp(u_i) + np.exp(u_j) + np.exp(u_0)

        # Choice probabilities
        d_i = np.exp(u_i) / deno
        d_j = np.exp(u_j) / deno

        # Profits  
        r_i = (p_i - self.cost) * d_i
        r_j = (p_j - self.cost) * d_j

        # Update state
        self.state = (a_i, a_j)

        return (a_i, a_j), (r_i, r_j), False, False, {}
    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # reset state to a random price pair
        a_i = np.random.randint(self.action_space[0].n)
        a_j = np.random.randint(self.action_space[1].n)
        self.state = (a_i, a_j)
        return (a_i, a_j), {}