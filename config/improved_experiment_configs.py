#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Improved experiment configurations that prevent below-cost pricing
'''

from dataclasses import dataclass
from typing import Dict, Any, Optional
import os

@dataclass
class ImprovedEnvironmentConfig:
    """Improved environment configuration that prevents below-cost pricing."""
    price_min: float = 2.0    # Set to marginal cost to prevent losses
    price_max: float = 10.0
    grid_size: int = 25
    marginal_cost: float = 2.0
    beta: float = 0.95
    a_0: float = 0
    a_12: float = 10
    mu: float = 0.25

@dataclass 
class ImprovedAgentConfig:
    """Improved agent configuration with faster exploration decay."""
    alpha: float = 0.15
    gamma: float = 0.9
    rule_timer_thr: int = 4
    omega: float = 5e-5      # Faster exploration decay (was 1.5e-5)
    
@dataclass
class RewardShapingConfig:
    """Configuration for reward shaping to discourage negative profits."""
    use_reward_shaping: bool = True
    negative_profit_penalty: float = 0.1  # Penalty for negative profits
    min_reward: float = -0.1  # Floor for minimum reward

def get_improved_n_player_config(n_players: int, 
                                periods: int = 200_000, 
                                runs: int = 50) -> dict:
    """Get improved N-player configuration that prevents below-cost pricing."""
    
    # Adjust exploration decay based on number of players
    # More players = need faster convergence to avoid chaos
    omega_adjustment = min(2.0, 1.0 + (n_players - 2) * 0.2)
    
    config = {
        'name': f"improved_n_player_N{n_players}",
        'environment': ImprovedEnvironmentConfig(),
        'agent': ImprovedAgentConfig(omega=5e-5 * omega_adjustment),
        'reward_shaping': RewardShapingConfig(),
        'simulation': {
            'n_players': n_players,
            'periods': periods,
            'runs': runs,
            'random_seed': 42
        },
        'data': {
            'data_dir': f"./data/improved_N{n_players}",
            'figure_dir': f"./figure/improved_N{n_players}",
            'save_data': True,
            'save_last_periods': 1000,
            'save_time_series': True,
            'time_series_interval': 5000,
            'time_series_window': 1000
        }
    }
    
    return config

def get_conservative_pricing_config(n_players: int) -> dict:
    """Configuration with additional safeguards against below-cost pricing."""
    
    config = get_improved_n_player_config(n_players)
    
    # Even more conservative settings
    config['environment'].price_min = 2.1  # 5% above marginal cost
    config['agent'].omega = 1e-4  # Even faster exploration decay
    config['reward_shaping'].negative_profit_penalty = 0.5  # Higher penalty
    config['name'] = f"conservative_n_player_N{n_players}"
    config['data']['data_dir'] = f"./data/conservative_N{n_players}"
    config['data']['figure_dir'] = f"./figure/conservative_N{n_players}"
    
    return config

if __name__ == "__main__":
    # Test the improved configurations
    print("Testing improved configurations...")
    
    for n in [2, 5, 10]:
        config = get_improved_n_player_config(n)
        env_config = config['environment']
        agent_config = config['agent']
        
        print(f"\nN={n} players:")
        print(f"  Price range: [{env_config.price_min}, {env_config.price_max}]")
        print(f"  Marginal cost: {env_config.marginal_cost}")
        print(f"  Exploration decay (omega): {agent_config.omega}")
        print(f"  Can price below cost: {env_config.price_min < env_config.marginal_cost}")
        
        # Conservative version
        conservative = get_conservative_pricing_config(n)
        conservative_env = conservative['environment']
        print(f"  Conservative price_min: {conservative_env.price_min}")