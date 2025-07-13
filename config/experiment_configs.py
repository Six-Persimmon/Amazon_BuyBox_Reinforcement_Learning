#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   experiment_configs.py
@Time    :   2025/07/11
@Author  :   Shijian Liu
@Version :   1.0
@Contact :   lshijian405@gmail.com
@Desc    :   Configuration management for N-player pricing experiments.
'''

from dataclasses import dataclass
from typing import Dict, Any, Optional
import os

@dataclass
class EnvironmentConfig:
    """Configuration for the pricing environment."""
    price_min: float = 0.01
    price_max: float = 10.0
    grid_size: int = 25
    marginal_cost: float = 2.0
    beta: float = 0.95
    a_0: float = 0        # Outside option parameter
    a_12: float = 10      # Inside option parameter
    mu: float = 0.25      # Vertical differentiation parameter

@dataclass
class AgentConfig:
    """Configuration for Q-learning agents."""
    alpha: float = 0.15   # Learning rate
    gamma: float = 0.9    # Discount factor
    rule_timer_thr: int = 4  # How long to stick with same rule

@dataclass
class SimulationConfig:
    """Configuration for simulation parameters."""
    n_players: int = 2
    periods: int = 200_000
    runs: int = 50
    random_seed: Optional[int] = 42

@dataclass
class DataConfig:
    """Configuration for data saving."""
    save_data: bool = True
    data_dir: str = "./data"
    figure_dir: str = "./figure"
    save_last_periods: int = 1000
    save_time_series: bool = True
    time_series_interval: int = 5000
    time_series_window: int = 1000

@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str
    environment: EnvironmentConfig
    agent: AgentConfig
    simulation: SimulationConfig
    data: DataConfig
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'name': self.name,
            'environment': self.environment.__dict__,
            'agent': self.agent.__dict__,
            'simulation': self.simulation.__dict__,
            'data': self.data.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        return cls(
            name=config_dict['name'],
            environment=EnvironmentConfig(**config_dict['environment']),
            agent=AgentConfig(**config_dict['agent']),
            simulation=SimulationConfig(**config_dict['simulation']),
            data=DataConfig(**config_dict['data'])
        )

# Predefined experiment configurations

def get_baseline_2player_config() -> ExperimentConfig:
    """Get baseline 2-player configuration matching original CIST experiment."""
    return ExperimentConfig(
        name="baseline_2player",
        environment=EnvironmentConfig(
            price_min=0.01,
            price_max=10.0,
            grid_size=25,
            marginal_cost=2.0,
            beta=0.95,
            a_0=0,
            a_12=10,
            mu=0.25
        ),
        agent=AgentConfig(
            alpha=0.15,
            gamma=0.9,
            rule_timer_thr=2  # Original used 2
        ),
        simulation=SimulationConfig(
            n_players=2,
            periods=200_000,
            runs=50,
            random_seed=42
        ),
        data=DataConfig(
            save_data=True,
            data_dir="./data/N2",
            figure_dir="./figure/N2",
            save_last_periods=1000,
            save_time_series=True,
            time_series_interval=5000,
            time_series_window=1000
        )
    )

def get_n_player_config(n_players: int, 
                       periods: int = 200_000, 
                       runs: int = 50) -> ExperimentConfig:
    """Get N-player configuration."""
    return ExperimentConfig(
        name=f"n_player_N{n_players}",
        environment=EnvironmentConfig(),  # Use defaults
        agent=AgentConfig(),
        simulation=SimulationConfig(
            n_players=n_players,
            periods=periods,
            runs=runs,
            random_seed=42
        ),
        data=DataConfig(
            data_dir=f"./data/N{n_players}",
            figure_dir=f"./figure/N{n_players}"
        )
    )

def get_quick_test_config(n_players: int = 3) -> ExperimentConfig:
    """Get configuration for quick testing."""
    return ExperimentConfig(
        name=f"quick_test_N{n_players}",
        environment=EnvironmentConfig(grid_size=10),  # Smaller grid for speed
        agent=AgentConfig(rule_timer_thr=2),
        simulation=SimulationConfig(
            n_players=n_players,
            periods=1000,
            runs=3,
            random_seed=42
        ),
        data=DataConfig(
            data_dir="./test_data",
            figure_dir="./test_figures",
            save_last_periods=500,
            time_series_interval=200,
            time_series_window=100
        )
    )

def get_parameter_sweep_configs() -> Dict[str, ExperimentConfig]:
    """Get configurations for parameter sensitivity analysis."""
    base_config = get_n_player_config(3)
    configs = {}
    
    # Vary learning rate
    for alpha in [0.05, 0.10, 0.15, 0.20]:
        config = ExperimentConfig(
            name=f"alpha_sweep_a{alpha:.2f}",
            environment=base_config.environment,
            agent=AgentConfig(alpha=alpha, gamma=0.9, rule_timer_thr=4),
            simulation=base_config.simulation,
            data=DataConfig(
                data_dir=f"./data/alpha_sweep/a{alpha:.2f}",
                figure_dir=f"./figure/alpha_sweep/a{alpha:.2f}"
            )
        )
        configs[f"alpha_{alpha:.2f}"] = config
    
    # Vary rule persistence
    for timer in [2, 4, 6, 8]:
        config = ExperimentConfig(
            name=f"timer_sweep_t{timer}",
            environment=base_config.environment,
            agent=AgentConfig(alpha=0.15, gamma=0.9, rule_timer_thr=timer),
            simulation=base_config.simulation,
            data=DataConfig(
                data_dir=f"./data/timer_sweep/t{timer}",
                figure_dir=f"./figure/timer_sweep/t{timer}"
            )
        )
        configs[f"timer_{timer}"] = config
    
    # Vary number of players
    for n in [2, 3, 4, 5]:
        config = get_n_player_config(n)
        config.name = f"n_player_sweep_N{n}"
        config.data.data_dir = f"./data/n_sweep/N{n}"
        config.data.figure_dir = f"./figure/n_sweep/N{n}"
        configs[f"n_players_{n}"] = config
    
    return configs

def create_experiment_dirs(config: ExperimentConfig):
    """Create necessary directories for an experiment."""
    os.makedirs(config.data.data_dir, exist_ok=True)
    os.makedirs(config.data.figure_dir, exist_ok=True)
    print(f"Created directories for experiment '{config.name}'")

def save_config(config: ExperimentConfig, filepath: str):
    """Save experiment configuration to file."""
    import json
    with open(filepath, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Configuration saved to {filepath}")

def load_config(filepath: str) -> ExperimentConfig:
    """Load experiment configuration from file."""
    import json
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    return ExperimentConfig.from_dict(config_dict)


if __name__ == "__main__":
    # Test configuration system
    print("Testing configuration system...")
    
    # Test baseline config
    baseline = get_baseline_2player_config()
    print(f"Baseline config: {baseline.name}")
    print(f"N players: {baseline.simulation.n_players}")
    print(f"Periods: {baseline.simulation.periods}")
    print(f"Grid size: {baseline.environment.grid_size}")
    
    # Test N-player configs
    for n in [3, 4, 5]:
        config = get_n_player_config(n)
        print(f"N={n} config: {config.name}, data_dir: {config.data.data_dir}")
    
    # Test quick test config
    quick = get_quick_test_config()
    print(f"Quick test: {quick.simulation.periods} periods, {quick.simulation.runs} runs")
    
    # Test parameter sweep
    sweep_configs = get_parameter_sweep_configs()
    print(f"Parameter sweep has {len(sweep_configs)} configurations")
    
    # Test serialization
    test_file = "test_config.json"
    save_config(baseline, test_file)
    loaded = load_config(test_file)
    print(f"Serialization test: {baseline.name == loaded.name}")
    
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)
    
    print("Configuration system test completed!")