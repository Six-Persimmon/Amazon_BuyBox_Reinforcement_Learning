#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Debug script to analyze why agents set prices below marginal cost
'''

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.n_player_rule_agent import (
    NPlayerQLearningRuleAgent, 
    reduced_state_to_index, 
    extract_reduced_state
)

def simple_logit_demand(prices, a_0=0, a_12=10, mu=0.25):
    """Calculate logit demand for given prices."""
    n_players = len(prices)
    deno = sum(np.exp((a_12 - p) / mu) for p in prices) + np.exp(a_0 / mu)
    demands = []
    for p in prices:
        d = np.exp((a_12 - p) / mu) / deno
        demands.append(d)
    return np.array(demands)

def calculate_profits(prices, marginal_cost=2.0):
    """Calculate profits for given prices."""
    demands = simple_logit_demand(prices)
    profits = (np.array(prices) - marginal_cost) * demands
    return profits

def analyze_nash_equilibrium():
    """Analyze Nash equilibrium for different N."""
    print("Analyzing Nash Equilibrium for different N...")
    print("=" * 50)
    
    marginal_cost = 2.0
    
    for n_players in [2, 3, 5, 10]:
        # Calculate symmetric Nash equilibrium
        def nash_foc(p):
            # All players charge price p
            deno = n_players * np.exp((10 - p) / 0.25) + np.exp(0 / 0.25)
            d = np.exp((10 - p) / 0.25) / deno
            # FOC: d + (p-c)*d' = 0
            d_prime = (-1/0.25) * d * (1 - d * n_players)
            return d + (p - marginal_cost) * d_prime
        
        # Solve numerically
        from scipy.optimize import fsolve
        try:
            p_nash = fsolve(nash_foc, marginal_cost + 0.5)[0]
            
            # Calculate demand and profit at Nash
            demands = simple_logit_demand([p_nash] * n_players)
            profit = (p_nash - marginal_cost) * demands[0]
            
            print(f"N={n_players:2d}: Nash price = {p_nash:.3f}, demand = {demands[0]:.3f}, profit = {profit:.3f}")
            
            # Check if Nash price is above marginal cost
            if p_nash < marginal_cost:
                print(f"      WARNING: Nash price below marginal cost!")
                
        except Exception as e:
            print(f"N={n_players:2d}: Nash calculation failed: {e}")
    
    print()

def analyze_rule_incentives():
    """Analyze what happens when agents follow different rules."""
    print("Analyzing rule incentives in high-N scenarios...")
    print("=" * 50)
    
    marginal_cost = 2.0
    n_players = 10
    
    # Scenario: Most agents price at marginal cost, one agent considers different rules
    base_prices = [marginal_cost] * (n_players - 1)  # 9 agents at marginal cost
    
    test_prices = np.linspace(0.5, 4.0, 20)  # Test different prices for the 10th agent
    
    print(f"Scenario: {n_players-1} agents price at marginal cost ({marginal_cost})")
    print(f"Analyzing 10th agent's best response...\n")
    
    best_profit = -float('inf')
    best_price = None
    
    for test_price in test_prices:
        all_prices = base_prices + [test_price]
        profits = calculate_profits(all_prices, marginal_cost)
        test_agent_profit = profits[-1]
        
        if test_agent_profit > best_profit:
            best_profit = test_agent_profit
            best_price = test_price
        
        print(f"Price = {test_price:.2f}: profit = {test_agent_profit:.4f}")
    
    print(f"\nBest response price: {best_price:.3f}")
    print(f"Best profit: {best_profit:.4f}")
    print(f"Is best price below marginal cost? {best_price < marginal_cost}")

def analyze_exploration_vs_exploitation():
    """Analyze exploration behavior that might lead to below-cost pricing."""
    print("\nAnalyzing exploration vs exploitation...")
    print("=" * 50)
    
    # Create agent and check exploration probability over time
    agent = NPlayerQLearningRuleAgent(0, 10, 25)  # N=10, 25 price levels
    
    time_steps = [0, 1000, 10000, 50000, 100000, 200000]
    
    print("Exploration probability (epsilon) over time:")
    for t in time_steps:
        agent.t = t
        epsilon = np.exp(-t * agent.omega)
        print(f"t = {t:6d}: epsilon = {epsilon:.6f}")
    
    print(f"\nWith omega = {agent.omega}, exploration continues for a very long time!")
    print("This means agents keep trying 'bad' rules even late in training.")

def analyze_price_grid_effects():
    """Analyze how price grid affects behavior."""
    print("\nAnalyzing price grid effects...")
    print("=" * 50)
    
    price_min, price_max = 0.01, 10.0
    grid_size = 25
    marginal_cost = 2.0
    
    prices = np.linspace(price_min, price_max, grid_size)
    
    print(f"Price grid: {price_min} to {price_max} with {grid_size} points")
    print(f"Marginal cost: {marginal_cost}")
    
    # Find price indices around marginal cost
    cost_idx = np.argmin(np.abs(prices - marginal_cost))
    print(f"Closest price to marginal cost: index {cost_idx}, price {prices[cost_idx]:.3f}")
    
    # Show prices near marginal cost
    print("\nPrices near marginal cost:")
    for i in range(max(0, cost_idx-3), min(grid_size, cost_idx+4)):
        mark = " <-- marginal cost" if i == cost_idx else ""
        print(f"  Index {i:2d}: {prices[i]:.3f}{mark}")
    
    # Show what happens with rules when min_competitor is very low
    print(f"\nRule behavior when min_competitor = 0 (price = {prices[0]:.3f}):")
    print(f"  Rule 0 (match): price = {prices[0]:.3f}")
    print(f"  Rule 1 (above): price = {prices[1]:.3f}")  
    print(f"  Rule 2 (below): price = {prices[0]:.3f} (can't go lower)")
    print(f"  Rule 3 (hold):  depends on own previous price")

def suggest_fixes():
    """Suggest potential fixes for the below-cost pricing issue."""
    print("\nSuggested fixes for below-cost pricing:")
    print("=" * 50)
    
    print("1. CONSTRAIN PRICE GRID:")
    print("   - Set price_min = marginal_cost (2.0) instead of 0.01")
    print("   - This prevents any price below marginal cost")
    
    print("\n2. MODIFY REWARD FUNCTION:")
    print("   - Add penalty for negative profits")
    print("   - reward = max(profit, -penalty) where penalty > 0")
    
    print("\n3. ADJUST EXPLORATION:")
    print("   - Reduce omega (faster epsilon decay)")
    print("   - Or use epsilon-greedy with manual decay schedule")
    
    print("\n4. RULE MODIFICATION:")
    print("   - Add 'Rule 4: Price at marginal cost' as safe fallback")
    print("   - Modify Rule 2 to never go below marginal cost")
    
    print("\n5. INITIALIZE Q-VALUES:")
    print("   - Initialize Q-values for profitable rules higher")
    print("   - Initialize Q-values for loss-making rules lower")

if __name__ == "__main__":
    print("Debugging negative profit issue in N-player pricing...")
    print("=" * 60)
    
    analyze_nash_equilibrium()
    analyze_rule_incentives() 
    analyze_exploration_vs_exploitation()
    analyze_price_grid_effects()
    suggest_fixes()
    
    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("The below-cost pricing happens due to:")
    print("1. Price grid allows prices below marginal cost")
    print("2. Slow exploration decay keeps agents trying bad rules")
    print("3. In high-N scenarios, Nash equilibrium approaches marginal cost")
    print("4. Rules can drive prices below competitors even if unprofitable")
    print("\nRecommendation: Constrain price grid to start at marginal cost")