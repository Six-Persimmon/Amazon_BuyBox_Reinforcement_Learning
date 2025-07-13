#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   n_player_rule_agent.py
@Time    :   2025/07/11
@Author  :   Shijian Liu
@Version :   1.0
@Contact :   lshijian405@gmail.com
@Desc    :   N-player Q-learning rule-based agent for pricing games.
    Agent learns to pick one of 4 pricing rules that target the lowest competitor price.
    Rules:
    - Rule 0: Match the lowest competitor price
    - Rule 1: Price one step above the lowest competitor price
    - Rule 2: Price one step below the lowest competitor price  
    - Rule 3: Keep current price (do nothing)
'''

import numpy as np

class NPlayerQLearningRuleAgent:
    """
    Q-learning agent for N-player pricing game using rule-based actions.
    Instead of learning specific prices, agent learns which pricing rule to use.
    All rules target the lowest price among competitors.
    
    State representation: (own_price_idx, min_competitor_price_idx)
    This reduces Q-table size from grid_size^N to grid_size^2.
    """
    
    def __init__(self, player_id, n_players, n_price_actions, alpha=0.1, gamma=0.9, 
                 cost=2.0, prices=None, rule_timer_thr=4):
        self.player_id = player_id
        self.n_players = n_players
        self.n_price_actions = n_price_actions
        self.n_rules = 4
        self.alpha = alpha
        self.gamma = gamma
        self.omega = 1.5e-5
        self.t = 0
        
        # Q-table: rows = reduced state index (own_price, min_competitor), cols = rule index (0-3)
        # State space is now grid_size^2 instead of grid_size^n_players
        self.Q = np.random.uniform(10, 20, size=(n_price_actions * n_price_actions, self.n_rules))
        
        # Rule management
        self.current_rule = 0
        self.rule_timer_thr = rule_timer_thr
        self.rule_timer = self.rule_timer_thr  # start fresh
        
        # Track prices
        self.prices = prices
        self.cost = cost
        self.last_price_idx = None

    def take_action(self, own_price_idx, min_competitor_price_idx):
        """
        Take action based on reduced state representation.
        
        Args:
            own_price_idx: This agent's previous price index
            min_competitor_price_idx: Minimum price index among competitors
            
        Returns:
            price_idx: This agent's chosen price index
        """
        # Convert to state index for Q-table lookup
        state_index = reduced_state_to_index(own_price_idx, min_competitor_price_idx, self.n_price_actions)
        
        # Check if we need to pick a new rule
        if self.rule_timer >= self.rule_timer_thr:
            epsilon = np.exp(-self.t * self.omega)
            if np.random.rand() < epsilon:
                new_rule = np.random.randint(self.n_rules)
            else:
                new_rule = int(np.argmax(self.Q[state_index]))
            self.current_rule = new_rule
            self.rule_timer = 0
        
        # Apply current rule to get price index
        price_idx = self._apply_rule(self.current_rule, own_price_idx, min_competitor_price_idx)
        self.rule_timer += 1
        self.last_price_idx = price_idx
        
        return price_idx

    def _apply_rule(self, rule, own_price_idx, min_competitor_price_idx):
        """
        Apply pricing rule based on own price and minimum competitor price.
        
        Args:
            rule: Rule index (0-3)
            own_price_idx: Own previous price index
            min_competitor_price_idx: Minimum competitor price index
            
        Returns:
            price_idx: Chosen price index
        """
        # Apply rule targeting lowest competitor price
        if rule == 0:
            # Match lowest competitor price
            return min_competitor_price_idx
        elif rule == 1:
            # Price one step above lowest competitor
            return min(min_competitor_price_idx + 1, self.n_price_actions - 1)
        elif rule == 2:
            # Price one step below lowest competitor
            return max(min_competitor_price_idx - 1, 0)
        elif rule == 3:
            # Keep own previous price
            return own_price_idx
        else:
            raise ValueError(f"Invalid rule: {rule}")

    def update(self, state_index, rule, reward, next_state_index):
        """
        Update Q-table using Q-learning.
        
        Args:
            state_index: Current reduced state index
            rule: Rule that was applied
            reward: Reward received
            next_state_index: Next reduced state index
        """
        best_next = np.max(self.Q[next_state_index])
        td_target = reward + self.gamma * best_next
        self.Q[state_index, rule] += self.alpha * (td_target - self.Q[state_index, rule])
        self.t += 1


def reduced_state_to_index(own_price_idx, min_competitor_price_idx, grid_size):
    """
    Convert reduced state (own_price, min_competitor_price) to single index.
    
    Args:
        own_price_idx: Own price index
        min_competitor_price_idx: Minimum competitor price index
        grid_size: Number of price levels
        
    Returns:
        state_index: Single integer index for Q-table lookup
    """
    return own_price_idx * grid_size + min_competitor_price_idx


def index_to_reduced_state(state_index, grid_size):
    """
    Convert single state index back to reduced state representation.
    
    Args:
        state_index: Single integer index
        grid_size: Number of price levels
        
    Returns:
        (own_price_idx, min_competitor_price_idx): Tuple of price indices
    """
    own_price_idx = state_index // grid_size
    min_competitor_price_idx = state_index % grid_size
    return own_price_idx, min_competitor_price_idx


def extract_reduced_state(all_price_indices, player_id):
    """
    Extract reduced state for a specific player from all players' prices.
    
    Args:
        all_price_indices: tuple of all players' price indices
        player_id: ID of the player (0-indexed)
        
    Returns:
        (own_price_idx, min_competitor_price_idx): Reduced state representation
    """
    own_price_idx = all_price_indices[player_id]
    
    # Get competitor prices (exclude own price)
    competitor_prices = [all_price_indices[i] for i in range(len(all_price_indices)) 
                        if i != player_id]
    
    # Find minimum competitor price
    if competitor_prices:
        min_competitor_price_idx = min(competitor_prices)
    else:
        # Edge case: only one player (shouldn't happen in normal usage)
        min_competitor_price_idx = 0
    
    return own_price_idx, min_competitor_price_idx


if __name__ == "__main__":
    # Test the N-player rule agent with reduced state space
    print("Testing N-player rule agent with reduced state space...")
    
    # Test reduced state indexing
    print("\nTesting reduced state indexing:")
    grid_size = 5
    for _ in range(5):
        # Random state
        own_price = np.random.randint(grid_size)
        min_comp_price = np.random.randint(grid_size)
        
        state_idx = reduced_state_to_index(own_price, min_comp_price, grid_size)
        recovered_own, recovered_min = index_to_reduced_state(state_idx, grid_size)
        
        print(f"({own_price}, {min_comp_price}) -> {state_idx} -> ({recovered_own}, {recovered_min}), "
              f"Match: {(own_price, min_comp_price) == (recovered_own, recovered_min)}")
    
    # Test extract_reduced_state function
    print("\nTesting extract_reduced_state:")
    all_prices = (2, 1, 4, 0, 3)  # 5 players with different prices
    for player_id in range(len(all_prices)):
        own_price, min_comp = extract_reduced_state(all_prices, player_id)
        competitor_prices = [all_prices[i] for i in range(len(all_prices)) if i != player_id]
        expected_min = min(competitor_prices)
        print(f"Player {player_id}: own={own_price}, min_comp={min_comp}, "
              f"expected_min={expected_min}, match={min_comp == expected_min}")
    
    # Test agent behavior with reduced state
    print("\nTesting agent behavior with reduced state:")
    n_players = 3
    grid_size = 10
    prices = np.linspace(1.0, 5.0, grid_size)
    
    agents = [NPlayerQLearningRuleAgent(i, n_players, grid_size, prices=prices) 
              for i in range(n_players)]
    
    # Simulate a few steps
    current_state = (3, 5, 7)  # Some initial prices
    
    for step in range(5):
        print(f"\nStep {step + 1}:")
        print(f"Current prices: {current_state}")
        
        actions = []
        state_indices = []
        
        for agent in agents:
            # Extract reduced state for this agent
            own_price, min_comp = extract_reduced_state(current_state, agent.player_id)
            state_idx = reduced_state_to_index(own_price, min_comp, grid_size)
            state_indices.append(state_idx)
            
            # Take action
            action = agent.take_action(own_price, min_comp)
            actions.append(action)
            print(f"Agent {agent.player_id}: state=({own_price}, {min_comp}), "
                  f"Rule {agent.current_rule} -> Price idx {action}")
        
        # Simulate rewards (dummy values)
        rewards = np.random.uniform(0, 1, n_players)
        
        # Update agents
        next_state = tuple(actions)
        
        for i, agent in enumerate(agents):
            # Calculate next state for this agent
            next_own, next_min_comp = extract_reduced_state(next_state, agent.player_id)
            next_state_idx = reduced_state_to_index(next_own, next_min_comp, grid_size)
            
            # Update Q-table
            agent.update(state_indices[i], agent.current_rule, rewards[i], next_state_idx)
        
        # Move to next state
        current_state = next_state
    
    # Test Q-table size reduction
    print(f"\nQ-table size comparison:")
    print(f"Original size (grid_size^n_players): {grid_size**n_players} = {grid_size}^{n_players}")
    print(f"Reduced size (grid_size^2): {grid_size**2} = {grid_size}^2")
    print(f"Reduction factor: {(grid_size**n_players) / (grid_size**2):.1f}x smaller")