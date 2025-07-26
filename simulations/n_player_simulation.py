#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   n_player_simulation.py
@Time    :   2025/07/11
@Author  :   Shijian Liu
@Version :   1.0
@Contact :   lshijian405@gmail.com
@Desc    :   N-player simulation framework for rule-based pricing competition.
'''

import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.NPlayerLogitDemandPricingEnv import NPlayerLogitDemandPricingEnv
from agents.n_player_rule_agent import NPlayerQLearningRuleAgent, reduced_state_to_index, extract_reduced_state


def simulate_n_player_competition(env, n_players, periods, alpha=0.1, gamma=0.9, rule_timer_thr=4):
    """
    Run N-player Q-learning rule vs rule competition.
    
    Args:
        env: NPlayerLogitDemandPricingEnv instance
        n_players: Number of players
        periods: Number of simulation periods
        alpha: Learning rate
        gamma: Discount factor
        rule_timer_thr: How long agents stick with same rule
        
    Returns:
        history_prices: np.ndarray of shape (n_players, periods) - price histories
        history_actions: np.ndarray of shape (n_players, periods) - rule histories  
        history_profits: np.ndarray of shape (n_players, periods) - profit histories
    """
    n_price_actions = len(env.prices)
    
    # Create agents
    agents = [NPlayerQLearningRuleAgent(
        player_id=i,
        n_players=n_players,
        n_price_actions=n_price_actions,
        alpha=alpha,
        gamma=gamma,
        cost=env.cost,
        prices=env.prices,
        rule_timer_thr=rule_timer_thr
    ) for i in range(n_players)]
    
    # Initialize histories
    history_prices = np.zeros((n_players, periods))
    history_actions = np.zeros((n_players, periods))  # Rule indices
    history_profits = np.zeros((n_players, periods))
    
    # Reset environment
    obs, info = env.reset()
    
    for t in range(periods):
        # Each agent picks a price based on reduced state
        actions = []
        state_indices = []
        
        for agent in agents:
            # Extract reduced state for this agent
            own_price_idx, min_comp_price_idx = extract_reduced_state(obs, agent.player_id)
            state_idx = reduced_state_to_index(own_price_idx, min_comp_price_idx, n_price_actions)
            state_indices.append(state_idx)
            
            # Agent takes action based on reduced state (now needs all prices for Rule 4)
            action = agent.take_action(own_price_idx, min_comp_price_idx, obs)
            actions.append(action)
        
        actions = tuple(actions)
        
        # Step environment
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        
        # Update all agents using reduced state representation
        for i, agent in enumerate(agents):
            # Calculate next reduced state for this agent
            next_own_idx, next_min_comp_idx = extract_reduced_state(next_obs, agent.player_id)
            next_state_idx = reduced_state_to_index(next_own_idx, next_min_comp_idx, n_price_actions)
            
            # Update Q-table
            agent.update(state_indices[i], agent.current_rule, rewards[i], next_state_idx)
        
        # Record histories
        for i in range(n_players):
            history_prices[i, t] = env.prices[actions[i]]
            history_actions[i, t] = agents[i].current_rule
            history_profits[i, t] = rewards[i]
        
        # Move to next state
        obs = next_obs
        
        if terminated or truncated:
            break
    
    return history_prices, history_actions, history_profits


def get_equilibrium_prices(n_players, env_params):
    """Get monopoly and Nash equilibrium prices from the environment."""
    # Create environment to calculate equilibrium prices
    env = NPlayerLogitDemandPricingEnv(n_players=n_players, **env_params)
    
    monopoly_price = env.get_monopoly_price()
    nash_price = env.get_nash_equilibrium_price()
    
    return monopoly_price, nash_price


def simulate_batch_n_player(n_players, periods, runs, alpha, gamma, env_params, rule_timer_thr=4):
    """
    Run multiple N-player simulations in batch.
    
    Args:
        n_players: Number of players
        periods: Number of periods per simulation
        runs: Number of simulation runs
        alpha: Learning rate
        gamma: Discount factor
        env_params: Dictionary of environment parameters
        rule_timer_thr: Rule persistence parameter
        
    Returns:
        batch_prices: np.ndarray of shape (runs, n_players, periods)
        batch_actions: np.ndarray of shape (runs, n_players, periods)
        batch_profits: np.ndarray of shape (runs, n_players, periods)
    """
    batch_prices = []
    batch_actions = []
    batch_profits = []
    
    for run in range(runs):
        if run % 10 == 0:
            print(f"Running simulation {run + 1}/{runs}")
        
        # Create fresh environment for each run
        env = NPlayerLogitDemandPricingEnv(n_players=n_players, **env_params)
        
        # Run simulation
        prices, actions, profits = simulate_n_player_competition(
            env, n_players, periods, alpha, gamma, rule_timer_thr
        )
        
        batch_prices.append(prices)
        batch_actions.append(actions)
        batch_profits.append(profits)
    
    return np.array(batch_prices), np.array(batch_actions), np.array(batch_profits)


if __name__ == "__main__":
    # Test the simulation framework
    print("Testing N-player simulation framework...")
    
    # Environment parameters
    env_params = {
        'price_min': 0.01,
        'price_max': 10.0,
        'grid_size': 25,
        'marginal_cost': 2.0,
        'beta': 0.95,
        'a_0': 0,
        'a_12': 10,
        'mu': 0.25
    }
    
    # Simulation parameters
    n_players = 5
    periods = 200_000
    runs = 50
    alpha = 0.15
    gamma = 0.9
    
    print(f"Running {runs} simulations with {n_players} players for {periods} periods each...")
    
    # Run batch simulation
    batch_prices, batch_actions, batch_profits = simulate_batch_n_player(
        n_players, periods, runs, alpha, gamma, env_params
    )
    
    print(f"Batch simulation completed!")
    print(f"Batch prices shape: {batch_prices.shape}")
    print(f"Batch actions shape: {batch_actions.shape}")
    print(f"Batch profits shape: {batch_profits.shape}")
    
    # Show some results
    print(f"\nFinal prices in last run:")
    for i in range(n_players):
        final_price = batch_prices[-1, i, -1]
        print(f"Player {i}: {final_price:.3f}")

    print(f"\nN=5 Mean prices over last 100 periods:")
    for i in range(n_players):
        mean_price = np.mean(batch_prices[:, i, -100:])
        std_price = np.std(batch_prices[:, i, -100:])
        print(f"Player {i}: {mean_price:.3f} ± {std_price:.3f}")
    
    # Test N=2 case for comparison
    print(f"\n\nTesting N=2 case for comparison with original:")
    batch_prices_2, batch_actions_2, batch_profits_2 = simulate_batch_n_player(
        2, periods, runs, alpha, gamma, env_params
    )
    
    print(f"N=2 Mean prices over last 100 periods:")
    for i in range(2):
        mean_price = np.mean(batch_prices_2[:, i, -100:])
        std_price = np.std(batch_prices_2[:, i, -100:])
        print(f"Player {i}: {mean_price:.3f} ± {std_price:.3f}")