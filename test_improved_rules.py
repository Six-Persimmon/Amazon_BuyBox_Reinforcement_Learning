#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Test the improved rules to verify they prevent below-cost pricing
'''

import numpy as np
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.n_player_rule_agent import NPlayerQLearningRuleAgent, extract_reduced_state, reduced_state_to_index

def simple_logit_demand(prices, a_0=0, a_12=10, mu=0.25):
    """Calculate logit demand for given prices."""
    n_players = len(prices)
    deno = sum(np.exp((a_12 - p) / mu) for p in prices) + np.exp(a_0 / mu)
    demands = []
    for p in prices:
        d = np.exp((a_12 - p) / mu) / deno
        demands.append(d)
    return np.array(demands)

def simulate_improved_pricing():
    """Simulate pricing with improved rules."""
    print("Simulating improved pricing rules...")
    print("=" * 50)
    
    # Setup
    n_players = 5
    grid_size = 25
    marginal_cost = 2.0
    price_min = 0.01  # Still allow low prices to test the constraint
    price_max = 10.0
    prices = np.linspace(price_min, price_max, grid_size)
    
    print(f"N players: {n_players}")
    print(f"Marginal cost: {marginal_cost}")
    print(f"Price grid: [{price_min:.2f}, {price_max:.2f}] with {grid_size} points")
    
    # Create agents with improved rules
    agents = [NPlayerQLearningRuleAgent(
        i, n_players, grid_size, alpha=0.2, gamma=0.9,
        cost=marginal_cost, prices=prices, rule_timer_thr=4
    ) for i in range(n_players)]
    
    # Adjust exploration decay manually
    for agent in agents:
        agent.omega = 1e-4  # Faster exploration decay
    
    cost_idx = agents[0].marginal_cost_idx
    print(f"Marginal cost index: {cost_idx} (price = {prices[cost_idx]:.3f})")
    
    # Start with random prices including some below cost
    current_state = tuple(np.random.randint(grid_size) for _ in range(n_players))
    print(f"Initial prices: {[prices[idx] for idx in current_state]}")
    
    # Track statistics
    periods = 5000
    price_history = np.zeros((n_players, periods))
    profit_history = np.zeros((n_players, periods))
    rule_history = np.zeros((n_players, periods))
    below_cost_count = np.zeros(n_players)
    
    for t in range(periods):
        actions = []
        state_indices = []
        
        for agent in agents:
            # Extract reduced state
            own_price_idx, min_comp_idx = extract_reduced_state(current_state, agent.player_id)
            state_idx = reduced_state_to_index(own_price_idx, min_comp_idx, grid_size)
            state_indices.append(state_idx)
            
            # Take action
            action = agent.take_action(own_price_idx, min_comp_idx, current_state)
            actions.append(action)
        
        # Calculate rewards
        current_prices = [prices[idx] for idx in current_state]
        demands = simple_logit_demand(current_prices)
        rewards = [(prices[current_state[i]] - marginal_cost) * demands[i] for i in range(n_players)]
        
        # Update agents
        next_state = tuple(actions)
        for i, agent in enumerate(agents):
            next_own_idx, next_min_comp_idx = extract_reduced_state(next_state, agent.player_id)
            next_state_idx = reduced_state_to_index(next_own_idx, next_min_comp_idx, grid_size)
            agent.update(state_indices[i], agent.current_rule, rewards[i], next_state_idx)
        
        # Record history
        for i in range(n_players):
            price_history[i, t] = prices[current_state[i]]
            profit_history[i, t] = rewards[i]
            rule_history[i, t] = agents[i].current_rule
            
            # Count below-cost pricing
            if prices[current_state[i]] < marginal_cost:
                below_cost_count[i] += 1
        
        # Move to next state
        current_state = next_state
        
        # Print progress occasionally
        if (t + 1) % 1000 == 0:
            current_prices = [prices[idx] for idx in current_state]
            mean_price = np.mean(current_prices)
            min_price = np.min(current_prices)
            print(f"Period {t+1:4d}: Mean price = {mean_price:.3f}, Min price = {min_price:.3f}")
    
    # Analyze results
    print(f"\nFinal Analysis (last 500 periods):")
    print("-" * 30)
    
    final_prices = price_history[:, -500:].mean(axis=1)
    final_profits = profit_history[:, -500:].mean(axis=1)
    
    for i in range(n_players):
        below_cost_rate = below_cost_count[i] / periods
        print(f"Player {i}: Price = {final_prices[i]:.3f}, Profit = {final_profits[i]:.4f}, "
              f"Below-cost rate = {below_cost_rate:.3f}")
    
    overall_below_cost_rate = np.sum(below_cost_count) / (periods * n_players)
    print(f"\nOverall below-cost pricing rate: {overall_below_cost_rate:.3f}")
    
    # Check if any final prices are below marginal cost
    prices_below_cost = np.sum(final_prices < marginal_cost)
    print(f"Players with final price below marginal cost: {prices_below_cost}")
    
    # Rule usage analysis
    print(f"\nRule usage in final 500 periods:")
    rule_names = ["Match", "Above", "Below*", "Hold", "Raise"]
    for rule in range(5):
        usage_rate = np.mean(rule_history[:, -500:] == rule)
        print(f"Rule {rule} ({rule_names[rule]}): {usage_rate:.3f}")
    
    return {
        'final_prices': final_prices,
        'below_cost_rate': overall_below_cost_rate,
        'prices_below_cost': prices_below_cost,
        'price_history': price_history,
        'profit_history': profit_history
    }

def compare_old_vs_new_rules():
    """Compare behavior with old vs new rules."""
    print("\n" + "=" * 60)
    print("COMPARISON: Impact of Rule Improvements")
    print("=" * 60)
    
    # The key improvements:
    print("Rule improvements:")
    print("1. Rule 2: Below competitor BUT not below marginal cost")
    print("2. Rule 4: Raise price (only when lowest)")
    print("3. Q-table now 5 rules instead of 4")
    
    # Run simulation
    results = simulate_improved_pricing()
    
    print(f"\nKey Results:")
    print(f"- Below-cost pricing rate: {results['below_cost_rate']:.1%}")
    print(f"- Players below marginal cost: {results['prices_below_cost']}/{len(results['final_prices'])}")
    print(f"- Mean final price: {np.mean(results['final_prices']):.3f}")
    
    if results['below_cost_rate'] < 0.01 and results['prices_below_cost'] == 0:
        print("✅ SUCCESS: Rules successfully prevent below-cost pricing!")
    else:
        print("⚠️  WARNING: Some below-cost pricing still occurs")
    
    return results

if __name__ == "__main__":
    print("Testing improved pricing rules...")
    results = compare_old_vs_new_rules()
    print("\nTest completed!")