import gymnasium as gym
import numpy as np
from gymnasium import spaces

class SequentialPricingEnv(gym.Env):
    """
    Sequential Pricing Environment based on Klein (2021)
    
    This environment simulates a sequential pricing duopoly with homogeneous goods.
    Each firm takes turns to set prices, and demand follows a simple linear model.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, n_firms=2, n_prices=6, discount_factor=0.95, render_mode=None):
        """
        Initialize the environment with default parameters from Klein (2021)
        
        Args:
            n_firms: Number of firms in the market (default: 2)
            n_prices: Number of discrete price levels (default: 6)
            discount_factor: Discount factor for future rewards (default: 0.95)
            render_mode: Mode for rendering the environment
        """
        self.n_firms = n_firms
        self.n_prices = n_prices
        self.discount_factor = discount_factor
        self.render_mode = render_mode
        
        # Price space from 0 to 1 with n_prices intervals
        self.prices = np.linspace(0, 1, n_prices)
        
        # Calculate Nash equilibrium and monopoly prices
        self.p_nash = 0  # Price index corresponding to Nash equilibrium
        self.p_monopoly = n_prices - 1  # Price index corresponding to monopoly price
        
        # Define action and observation spaces
        self.action_space = spaces.MultiDiscrete([n_prices] * n_firms)
        # Each action is a tuple (p1, p2), where one of the entries will be 0 for the inactive firm
        
        # Observation is the prices chosen by both firms in the previous period
        self.observation_space = spaces.MultiDiscrete([n_prices] * n_firms)
        
        # Current state
        self.previous_prices = np.zeros(n_firms, dtype=int)  # Previous prices set by firms. Can be set to include all the hisotry

        self.previous_profit = np.zeros(n_firms, dtype=int)  # Previous profit for each firm
        
        # Reset the environment
        self.reset()
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state
        
        Returns:
            observation: Initial observation
        """
        super().reset(seed=seed)
        
        # Reset state variables
        self.previous_prices = np.zeros(self.n_firms, dtype=int)
        self.previous_profit = np.zeros(self.n_firms, dtype=int)
        
        return self.previous_prices, {}
    
    def _calculate_demand(self, price_idx_list):
        """
        Calculate demand for each firm based on linear demand function from Klein (2021)
        
        Args:
            price_idx_list: List of price indices chosen by all firms
            
        Returns:
            demand: A vector of demand quantities for each firm
        """
        # print(f"Price indices: {price_idx_list}")
        # print(f"Prices: {self.prices}")
        prices = [self.prices[i] for i in price_idx_list]  # Convert indices to actual prices
        demand = np.zeros(self.n_firms)
        
        for i in range(self.n_firms):
            own_price = prices[i]
            competitor_prices = np.delete(prices, i)
            
            if own_price < np.min(competitor_prices):
                # Firm captures entire market
                demand[i] = 1 - own_price
            elif own_price == np.min(competitor_prices):
                # Firms with the lowest price split the market
                num_firms_with_min_price = np.sum(prices == own_price)
                demand[i] = (1 - own_price) / num_firms_with_min_price
            else:
                # Firm captures no market
                demand[i] = 0
        
        return demand
    
    def _calculate_profit(self, action):
        """
        Calculate profit for both firms based on their prices
        
        Args:
            action: Price index chosen by all the firms in the current period
            
        Returns:
            profit: Profit for each firm in the current period
        """
        demand = self._calculate_demand(action)
        prices = self.prices[action]  # Get the actual prices based on indices
        
        return prices * demand # shape: (n_firms,)
    
    def step(self, action_idx): 
        """
        Take a step in the environment by setting a price
        
        Args:
            action: Price index chosen by all the firms in the current period. Shape: (n_firms,)
            
        Returns:
            observation: Price set by opponent
            reward: Profit for current firm
            terminated: Whether episode is terminated
            truncated: Whether episode is truncated
            info: Additional information
        """
        
        # Calculate profit for current firm
        profit = self._calculate_profit(action_idx)
        
        # Update state. Note: In the future we may want to include more history.
        self.previous_prices_idx = action_idx
        self.previous_profit = profit
        
        # Next observation is the price just set by the current firm
        observation = action_idx
        
        # In this environment, episodes don't terminate
        terminated = False
        truncated = False
        
        # Additional info
        info = {
            "profit": profit,
            "price": [self.prices[i] for i in action_idx],
        }
        
        return observation, profit, terminated, truncated, info
    
    def render(self):
        """
        Render the environment
        
        Returns:
            None
        """
        if self.render_mode == "human":
            # Simple text rendering of current state
            print(f"Previous prices Action: {self.previous_prices}")
            print(f"Actual prices: {self.prices[self.previous_prices]}")
        
    def close(self):
        """
        Close the environment
        
        Returns:
            None
        """
        pass