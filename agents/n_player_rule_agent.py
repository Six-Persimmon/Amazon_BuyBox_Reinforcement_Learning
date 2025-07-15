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
    
    Rules:
    - Rule 0: Match lowest competitor price
    - Rule 1: Price one step above lowest competitor
    - Rule 2: Price one step below lowest competitor (but not below marginal cost)
    - Rule 3: Keep current price
    - Rule 4: Raise price by one step (only available when agent is lowest price)
    """
    
    def __init__(self, player_id, n_players, n_price_actions, alpha=0.1, gamma=0.9, 
                 cost=2.0, prices=None, rule_timer_thr=4):
        self.player_id = player_id
        self.n_players = n_players
        self.n_price_actions = n_price_actions
        self.n_rules = 5  # Now 5 rules (0-4)
        self.alpha = alpha
        self.gamma = gamma
        self.omega = 1.5e-5
        self.t = 0
        self.cost = cost
        self.prices = prices
        
        # Find marginal cost index in price grid
        if prices is not None:
            self.marginal_cost_idx = np.argmin(np.abs(prices - cost))
        else:
            self.marginal_cost_idx = 0  # Fallback
        
        # Q-table: rows = reduced state index (own_price, min_competitor), cols = rule index (0-4)
        # State space is now grid_size^2 instead of grid_size^n_players
        self.Q = np.random.uniform(10, 20, size=(n_price_actions * n_price_actions, self.n_rules))
        
        # Rule management
        self.current_rule = 0
        self.rule_timer_thr = rule_timer_thr
        self.rule_timer = self.rule_timer_thr  # start fresh
        
        self.last_price_idx = None

    def take_action(self, own_price_idx, min_competitor_price_idx, all_price_indices):
        """
        Take action based on reduced state representation.
        
        Args:
            own_price_idx: This agent's previous price index
            min_competitor_price_idx: Minimum price index among competitors
            all_price_indices: All players' current price indices (for Rule 4 eligibility)
            
        Returns:
            price_idx: This agent's chosen price index
        """
        # Convert to state index for Q-table lookup
        state_index = reduced_state_to_index(own_price_idx, min_competitor_price_idx, self.n_price_actions)
        
        # Check if we need to pick a new rule
        if self.rule_timer >= self.rule_timer_thr:
            # Get available rules (Rule 4 may be masked)
            available_rules = self._get_available_rules(own_price_idx, all_price_indices)
            
            epsilon = np.exp(-self.t * self.omega)
            if np.random.rand() < epsilon:
                # Random selection from available rules only
                new_rule = np.random.choice(available_rules)
            else:
                # Greedy selection from available rules only
                q_values_available = self.Q[state_index, available_rules]
                best_available_idx = np.argmax(q_values_available)
                new_rule = available_rules[best_available_idx]
            
            self.current_rule = new_rule
            self.rule_timer = 0
        
        # Apply current rule to get price index
        price_idx = self._apply_rule(self.current_rule, own_price_idx, min_competitor_price_idx, all_price_indices)
        self.rule_timer += 1
        self.last_price_idx = price_idx
        
        return price_idx

    def _get_available_rules(self, own_price_idx, all_price_indices):
        """
        Get list of available rules based on current state.
        Rule 4 is only available when agent is the lowest price.
        
        Args:
            own_price_idx: Agent's current price index
            all_price_indices: All players' current price indices
            
        Returns:
            available_rules: List of available rule indices
        """
        # Rules 0-3 are always available
        available_rules = [0, 1, 2, 3]
        
        # Rule 4 only available if agent is currently the lowest price
        # (or tied for lowest)
        min_price_idx = min(all_price_indices)
        if own_price_idx == min_price_idx:
            available_rules.append(4)
        
        return np.array(available_rules)

    def _apply_rule(self, rule, own_price_idx, min_competitor_price_idx, all_price_indices):
        """
        Apply pricing rule based on own price and minimum competitor price.
        
        Args:
            rule: Rule index (0-4)
            own_price_idx: Own previous price index
            min_competitor_price_idx: Minimum competitor price index
            all_price_indices: All players' current price indices
            
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
            # Price one step below lowest competitor, but not below marginal cost
            below_competitor = max(min_competitor_price_idx - 1, 0)
            return max(below_competitor, self.marginal_cost_idx)
        elif rule == 3:
            # Keep own previous price
            return own_price_idx
        elif rule == 4:
            # Raise price by one step (only available when agent is lowest price)
            return min(own_price_idx + 1, self.n_price_actions - 1)
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
    # Test the improved N-player rule agent
    print("Testing improved N-player rule agent with new rules...")
    
    # Test with marginal cost constraint
    grid_size = 10
    marginal_cost = 2.0
    prices = np.linspace(0.5, 5.0, grid_size)
    
    print(f"Price grid: {prices}")
    print(f"Marginal cost: {marginal_cost}")
    
    # Create agent
    agent = NPlayerQLearningRuleAgent(0, 3, grid_size, cost=marginal_cost, prices=prices)
    print(f"Marginal cost index: {agent.marginal_cost_idx} (price = {prices[agent.marginal_cost_idx]:.3f})")
    
    # Test Rule 2 with marginal cost constraint
    print(f"\nTesting Rule 2 (below competitor but not below marginal cost):")
    test_cases = [
        (agent.marginal_cost_idx + 2, agent.marginal_cost_idx + 1),  # Normal case
        (agent.marginal_cost_idx + 1, agent.marginal_cost_idx),      # At marginal cost
        (agent.marginal_cost_idx, 1),                               # Would go below marginal cost
    ]
    
    for own_idx, min_comp_idx in test_cases:
        result = agent._apply_rule(2, own_idx, min_comp_idx, (own_idx, min_comp_idx, 5))
        own_price = prices[own_idx]
        min_comp_price = prices[min_comp_idx]
        result_price = prices[result]
        print(f"  Own: {own_price:.3f}, Min comp: {min_comp_price:.3f} -> Result: {result_price:.3f}")
    
    # Test Rule 4 availability and behavior
    print(f"\nTesting Rule 4 (raise price when lowest):")
    
    # Case 1: Agent is lowest price (Rule 4 should be available)
    all_prices_1 = (2, 5, 7)  # Agent 0 is lowest
    available_rules_1 = agent._get_available_rules(2, all_prices_1)
    print(f"Agent price=2, others=[5,7]: Available rules = {available_rules_1}")
    print(f"Rule 4 available: {4 in available_rules_1}")
    
    if 4 in available_rules_1:
        rule4_result = agent._apply_rule(4, 2, 5, all_prices_1)
        print(f"  Rule 4 result: index {rule4_result} (price {prices[rule4_result]:.3f})")
    
    # Case 2: Agent is not lowest price (Rule 4 should not be available)
    all_prices_2 = (5, 2, 7)  # Agent 0 is not lowest
    available_rules_2 = agent._get_available_rules(5, all_prices_2)
    print(f"Agent price=5, others=[2,7]: Available rules = {available_rules_2}")
    print(f"Rule 4 available: {4 in available_rules_2}")
    
    # Test complete action selection with masking
    print(f"\nTesting complete action selection with masking:")
    
    # Force rule timer to trigger new rule selection
    agent.rule_timer = agent.rule_timer_thr
    agent.t = 10000  # Low exploration
    
    # Initialize Q-values to prefer Rule 4 when available
    state_idx = reduced_state_to_index(2, 5, grid_size)
    agent.Q[state_idx, 4] = 50  # Make Rule 4 highly preferred
    agent.Q[state_idx, 0:4] = 10  # Lower values for other rules
    
    # Test when Rule 4 is available (should be selected)
    action_1 = agent.take_action(2, 5, all_prices_1)
    print(f"When Rule 4 available: selected rule {agent.current_rule}, action {action_1}")
    
    # Reset for next test
    agent.rule_timer = agent.rule_timer_thr
    
    # Test when Rule 4 is not available (should select best among 0-3)
    action_2 = agent.take_action(5, 2, all_prices_2)
    print(f"When Rule 4 not available: selected rule {agent.current_rule}, action {action_2}")
    
    print(f"\nAll tests completed successfully!")