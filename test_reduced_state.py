#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Test script for reduced state space functionality
'''

import numpy as np
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.n_player_rule_agent import (
    NPlayerQLearningRuleAgent, 
    reduced_state_to_index, 
    extract_reduced_state,
    index_to_reduced_state
)

def test_n2_compatibility():
    """Test that N=2 case works correctly with reduced state representation."""
    print("Testing N=2 compatibility with reduced state representation...")
    
    # Create 2 agents
    n_players = 2
    grid_size = 10
    
    agent0 = NPlayerQLearningRuleAgent(0, n_players, grid_size)
    agent1 = NPlayerQLearningRuleAgent(1, n_players, grid_size)
    
    # Test various 2-player scenarios
    test_cases = [
        (3, 5),  # agent0=3, agent1=5
        (7, 2),  # agent0=7, agent1=2
        (1, 1),  # both same price
        (9, 0),  # extreme prices
    ]
    
    print(f"Q-table size for N=2: {grid_size**2} (vs original {grid_size**n_players})")
    
    for case_num, (price0, price1) in enumerate(test_cases):
        current_state = (price0, price1)
        print(f"\nTest case {case_num + 1}: {current_state}")
        
        # Extract reduced states for both agents
        own0, min_comp0 = extract_reduced_state(current_state, 0)
        own1, min_comp1 = extract_reduced_state(current_state, 1)
        
        print(f"  Agent 0: own={own0}, min_comp={min_comp0}")
        print(f"  Agent 1: own={own1}, min_comp={min_comp1}")
        
        # Verify correctness
        assert own0 == price0, f"Agent 0 own price mismatch: {own0} != {price0}"
        assert own1 == price1, f"Agent 1 own price mismatch: {own1} != {price1}"
        assert min_comp0 == price1, f"Agent 0 min competitor mismatch: {min_comp0} != {price1}"
        assert min_comp1 == price0, f"Agent 1 min competitor mismatch: {min_comp1} != {price0}"
        
        # Test state indexing
        state_idx0 = reduced_state_to_index(own0, min_comp0, grid_size)
        state_idx1 = reduced_state_to_index(own1, min_comp1, grid_size)
        
        # Verify we can recover state
        recovered0 = index_to_reduced_state(state_idx0, grid_size)
        recovered1 = index_to_reduced_state(state_idx1, grid_size)
        
        assert recovered0 == (own0, min_comp0), f"State recovery failed for agent 0"
        assert recovered1 == (own1, min_comp1), f"State recovery failed for agent 1"
        
        # Test agent actions
        action0 = agent0.take_action(own0, min_comp0)
        action1 = agent1.take_action(own1, min_comp1)
        
        print(f"  Agent 0: rule={agent0.current_rule}, action={action0}")
        print(f"  Agent 1: rule={agent1.current_rule}, action={action1}")
        
        # Verify actions are valid
        assert 0 <= action0 < grid_size, f"Invalid action for agent 0: {action0}"
        assert 0 <= action1 < grid_size, f"Invalid action for agent 1: {action1}"
    
    print("\nN=2 compatibility test PASSED!")

def test_n_player_scaling():
    """Test that reduced state works for larger N."""
    print("\nTesting N-player scaling...")
    
    grid_size = 5
    
    for n_players in [3, 5, 10]:
        print(f"\nN={n_players} players:")
        
        # Create agents
        agents = [NPlayerQLearningRuleAgent(i, n_players, grid_size) 
                 for i in range(n_players)]
        
        # Random price state
        current_state = tuple(np.random.randint(grid_size) for _ in range(n_players))
        print(f"  Current state: {current_state}")
        
        # Test each agent
        for agent in agents:
            own_price, min_comp = extract_reduced_state(current_state, agent.player_id)
            
            # Verify min_comp is actually minimum among competitors
            competitor_prices = [current_state[i] for i in range(n_players) 
                               if i != agent.player_id]
            expected_min = min(competitor_prices)
            
            assert min_comp == expected_min, f"Min competitor calculation error for agent {agent.player_id}"
            assert own_price == current_state[agent.player_id], f"Own price error for agent {agent.player_id}"
        
        # Calculate Q-table size savings
        original_size = grid_size ** n_players
        reduced_size = grid_size ** 2
        reduction_factor = original_size / reduced_size
        
        print(f"  Q-table size: {reduced_size} (vs {original_size}) = {reduction_factor:.1f}x smaller")
    
    print("N-player scaling test PASSED!")

def test_rule_behavior():
    """Test that all 4 rules work correctly with reduced state."""
    print("\nTesting rule behavior with reduced state...")
    
    grid_size = 10
    n_players = 3
    
    # Test each rule
    for rule in range(4):
        agent = NPlayerQLearningRuleAgent(0, n_players, grid_size)
        agent.current_rule = rule  # Force specific rule
        
        # Test case: own_price=5, min_competitor=3
        own_price = 5
        min_comp = 3
        
        action = agent._apply_rule(rule, own_price, min_comp)
        
        if rule == 0:  # Match lowest competitor
            expected = min_comp
        elif rule == 1:  # One step above lowest competitor
            expected = min(min_comp + 1, grid_size - 1)
        elif rule == 2:  # One step below lowest competitor
            expected = max(min_comp - 1, 0)
        elif rule == 3:  # Keep own price
            expected = own_price
        
        assert action == expected, f"Rule {rule} failed: got {action}, expected {expected}"
        print(f"  Rule {rule}: own={own_price}, min_comp={min_comp} -> action={action} ✓")
    
    # Test boundary conditions
    print("  Testing boundary conditions...")
    
    agent = NPlayerQLearningRuleAgent(0, n_players, grid_size)
    
    # Test rule 1 at max price
    agent.current_rule = 1
    action = agent._apply_rule(1, 5, grid_size - 1)  # min_comp at max
    assert action == grid_size - 1, f"Rule 1 boundary test failed: {action}"
    
    # Test rule 2 at min price
    agent.current_rule = 2
    action = agent._apply_rule(2, 5, 0)  # min_comp at min
    assert action == 0, f"Rule 2 boundary test failed: {action}"
    
    print("Rule behavior test PASSED!")

if __name__ == "__main__":
    print("Testing reduced state space implementation...")
    print("=" * 50)
    
    # Run all tests
    test_n2_compatibility()
    test_n_player_scaling()
    test_rule_behavior()
    
    print("\n" + "=" * 50)
    print("All tests PASSED! ✅")
    print("Reduced state space implementation is working correctly.")
    
    # Show efficiency gains
    print("\nEfficiency gains:")
    grid_size = 25  # Typical grid size
    for n in [2, 3, 4, 5, 10]:
        original = grid_size ** n
        reduced = grid_size ** 2
        factor = original / reduced
        print(f"  N={n}: {original:,} -> {reduced:,} ({factor:,.0f}x smaller)")