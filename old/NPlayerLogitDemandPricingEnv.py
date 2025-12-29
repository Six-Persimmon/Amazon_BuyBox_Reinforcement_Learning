#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   NPlayerLogitDemandPricingEnv.py
@Time    :   2025/07/11
@Author  :   Shijian Liu  
@Version :   1.0
@Contact :   lshijian405@gmail.com
@Desc    :   N-player repeated pricing environment with Logit demand.
    Actions: tuple (a_1, a_2, ..., a_N) with each in {0,...,M-1}, indexing price grid.
    Observation: tuple (p_1_lag, p_2_lag, ..., p_N_lag) indexing price grid from last step.
    Rewards: profits for each firm given marginal cost and logit demand rule.
    Demand: Logit demand with outside option, properly allocated among N players.
'''

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class NPlayerLogitDemandPricingEnv(gym.Env):
    """
    N-player discrete Bertrand pricing game with Logit demand.
    - Prices on a grid from price_min to price_max (inclusive).
    - Demand: Logit demand with outside option.
    - Profit = (price - cost) * demand.
    State is a tuple of last prices played by each firm.
    """
    metadata = {"render_modes": []}

    def __init__(self,
                 n_players: int = 2,
                 price_min: float = 0.01,
                 price_max: float = 10.0,
                 grid_size: int = 25,
                 marginal_cost: float = 2.0,
                 beta: float = 0.95,
                 a_0: float = 0,    # parameter for logit demand. Outside option
                 a_12: float = 10,  # parameter for logit demand. Inside option
                 mu: float = 0.25   # parameter for logit demand. Vertical differentiation
                 ):
        super().__init__()
        self.n_players = n_players
        self.prices = np.linspace(price_min, price_max, grid_size)
        self.price_min = price_min
        self.price_max = price_max
        self.cost = marginal_cost
        self.beta = beta
        self.a_0 = a_0
        self.a_12 = a_12
        self.mu = mu

        # each firm's action is picking an index in {0,…,grid_size–1}
        self.action_space = spaces.Tuple([spaces.Discrete(grid_size) for _ in range(n_players)])
        
        # observation is the last prices played by each firm
        self.observation_space = spaces.Tuple([spaces.Discrete(grid_size) for _ in range(n_players)])

    def step(self, actions):
        '''
        N-player Logit demand system with outside option
        
        Args:
            actions: tuple of price indices (a_1, a_2, ..., a_N)
            
        Returns:
            observations: tuple of price indices from this step
            rewards: tuple of profits for each player
            terminated: always False
            truncated: always False  
            info: dict with additional information
        '''
        # Convert actions to actual prices
        prices = [self.prices[a] for a in actions]
        
        # Calculate logit demand for each player
        # Denominator includes all players plus outside option
        deno = sum(np.exp((self.a_12 - p) / self.mu) for p in prices) + np.exp(self.a_0 / self.mu)
        
        # Demand for each player
        demands = []
        for i, p_i in enumerate(prices):
            d_i = np.exp((self.a_12 - p_i) / self.mu) / deno
            demands.append(d_i)
        
        # Calculate profits for each player
        rewards = []
        for i, (p_i, d_i) in enumerate(zip(prices, demands)):
            r_i = (p_i - self.cost) * d_i
            rewards.append(r_i)
        
        # Update state
        self.state = actions
        
        # Additional info
        info = {
            "profits": rewards,
            "prices": prices,
            "demands": demands
        }
        
        return actions, tuple(rewards), False, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset state to random price indices for all players
        actions = tuple(np.random.randint(len(self.prices)) for _ in range(self.n_players))
        self.state = actions
        return actions, {}

    def get_monopoly_price(self):
        """Calculate monopoly price for single firm"""
        from scipy.optimize import minimize_scalar
        
        def monopoly_profit(p):
            d = np.exp((self.a_12 - p) / self.mu) / (np.exp((self.a_12 - p) / self.mu) + np.exp(self.a_0 / self.mu))
            return (p - self.cost) * d
        
        result = minimize_scalar(lambda p: -monopoly_profit(p), 
                               bounds=(self.price_min, self.price_max), 
                               method='bounded')
        return result.x
    
    def get_nash_equilibrium_price(self):
        """Calculate symmetric Nash equilibrium price for N players"""
        from scipy.optimize import fsolve
        
        def nash_condition(p):
            # For symmetric equilibrium, all players charge same price p
            # First-order condition: derivative of profit w.r.t. own price = 0
            
            # Demand when all charge price p
            deno = self.n_players * np.exp((self.a_12 - p) / self.mu) + np.exp(self.a_0 / self.mu)
            d = np.exp((self.a_12 - p) / self.mu) / deno
        
            # First-order condition:
            foc = d - (p - self.cost) * np.exp(self.a_0 / self.mu)
            return foc
        
        # Solve for Nash equilibrium price
        p_nash = fsolve(nash_condition, self.cost + 0.2)[0]
        return max(self.price_min, min(self.price_max, p_nash))


if __name__ == "__main__":
    # Test the N-player environment
    print("Testing N-player environment...")
    
    # Test with N=2 (should match original behavior)
    env_2 = NPlayerLogitDemandPricingEnv(n_players=2)
    obs, info = env_2.reset()
    print(f"N=2 Initial observation: {obs}")
    
    for _ in range(3):
        action = tuple(np.random.randint(len(env_2.prices)) for _ in range(2))
        obs, reward, done, truncated, info = env_2.step(action)
        print(f"N=2 Action: {action}, Prices: {[round(p, 3) for p in info['prices']]}, Rewards: {[round(r, 4) for r in reward]}")
    
    print()
    
    # Test with N=3
    env_3 = NPlayerLogitDemandPricingEnv(n_players=3)
    obs, info = env_3.reset()
    print(f"N=3 Initial observation: {obs}")
    
    for _ in range(3):
        action = tuple(np.random.randint(len(env_3.prices)) for _ in range(3))
        obs, reward, done, truncated, info = env_3.step(action)
        print(f"N=3 Action: {action}, Prices: {[round(p, 3) for p in info['prices']]}, Rewards: {[round(r, 4) for r in reward]}")
    
    # Test equilibrium calculations
    print(f"\nN=2 Monopoly price: {env_2.get_monopoly_price():.3f}")
    print(f"N=2 Nash price: {env_2.get_nash_equilibrium_price():.3f}")
    
    print(f"N=3 Monopoly price: {env_3.get_monopoly_price():.3f}")
    print(f"N=3 Nash price: {env_3.get_nash_equilibrium_price():.3f}")

    print(f"N=6 Monopoly price: {NPlayerLogitDemandPricingEnv(n_players=6).get_monopoly_price():.3f}")
    print(f"N=6 Nash price: {NPlayerLogitDemandPricingEnv(n_players=6).get_nash_equilibrium_price():.3f}")

    print(f"N=20 Monopoly price: {NPlayerLogitDemandPricingEnv(n_players=20).get_monopoly_price():.3f}")
    print(f"N=20 Nash price: {NPlayerLogitDemandPricingEnv(n_players=20).get_nash_equilibrium_price():.3f}")