#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   LogitDemandPricingEnv.py
@Time    :   2025/04/24 23:06:10
@Author  :   Shijian Liu
@Version :   1.0
@Contact :   lshijian405@gmail.com
@Desc    :   A two-firm repeated pricing environment.
    Actions: tuple (a_i, a_j) with each in {0,...,M-1}, indexing price grid.
    Observation: tuple (p_i_lag, p_j_lag) with each in {0,...,M-1}, indexing price grid. Price pair played in last step.
    Rewards: profits for each firm given marginal cost and demand rule.
    Demand: Logit demand with outside option.

'''


import numpy as np
import gymnasium as gym
from gymnasium import spaces

class LogitDemandPricingEnv(gym.Env):
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

    def step(self, actions):
        '''
        Logit demand system with outside option'''
        a_i, a_j = actions
        p_i = self.prices[a_i]
        p_j = self.prices[a_j]
        # print("a_i:", a_i, "a_j:", a_j)
        # print("p_i:", p_i, "p_j:", p_j)

        deno = np.exp((self.a_12 - p_i) / self.mu) + np.exp((self.a_12 - p_j) / self.mu) + np.exp((self.a_0) / self.mu)

        d_i = np.exp((self.a_12 - p_i) / self.mu) / deno
        d_j = np.exp((self.a_12 - p_j) / self.mu) / deno
        # print("d_i:", d_i, "d_j:", d_j)


        # profits
        r_i = (p_i - self.cost) * d_i
        r_j = (p_j - self.cost) * d_j
        # print("r_i:", r_i, "r_j:", r_j)
        # update state
        self.state = (a_i, a_j)

        return (a_i, a_j), (r_i, r_j), False, False, {}
    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # reset state to a random price pair
        a_i = np.random.randint(self.action_space[0].n)
        a_j = np.random.randint(self.action_space[1].n)
        self.state = (a_i, a_j)
        return (a_i, a_j), {}

class LogitDemandPricingAmazonEnv(gym.Env):
    """
    Two‑firm discrete Bertrand pricing game with Logit demand and the Amazon Featured Offer (or Buy Box) mechanism.
    - Prices on a grid from price_min to price_max (inclusive).
    - Demand: Logit demand with outside option. The Buy Box winner gets an additional term in the utility to give it a higher probability of being purchased.
    - Profit = (price – cost) * demand.
    State is a tuple of last prices played by each firm.
    """
    metadata = {"render_modes": []}




if __name__ == "__main__":
    env = LogitDemandPricingEnv()
    obs, info = env.reset()
    print("Initial observation:", obs)
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print("Action:", action, "Observation:", obs, "Reward:", reward)
        if done or truncated:
            break
    env.close()