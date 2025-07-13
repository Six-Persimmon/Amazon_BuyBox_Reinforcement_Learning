#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   n_player_experiments.py
@Time    :   2025/07/11
@Author  :   Shijian Liu
@Version :   1.0
@Contact :   lshijian405@gmail.com
@Desc    :   N-player pricing experiments and validation against 2-player case.
'''

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.experiment_configs import *
from simulations.n_player_simulation import simulate_batch_n_player
from analysis.plotting import *
from utils.data_utils import *
from env.NPlayerLogitDemandPricingEnv import NPlayerLogitDemandPricingEnv

def run_n_player_experiment(config: ExperimentConfig, verbose: bool = True):
    """
    Run a complete N-player experiment based on configuration.
    
    Args:
        config: Experiment configuration
        verbose: Whether to print progress updates
        
    Returns:
        results: Dictionary containing experiment results
    """
    if verbose:
        print(f"Running experiment: {config.name}")
        print(f"N players: {config.simulation.n_players}")
        print(f"Periods: {config.simulation.periods}, Runs: {config.simulation.runs}")
    
    # Create directories
    create_experiment_dirs(config)
    
    # Set random seed
    if config.simulation.random_seed is not None:
        np.random.seed(config.simulation.random_seed)
    
    # Environment parameters
    env_params = config.environment.__dict__
    
    # Run batch simulation
    batch_prices, batch_actions, batch_profits = simulate_batch_n_player(
        n_players=config.simulation.n_players,
        periods=config.simulation.periods,
        runs=config.simulation.runs,
        alpha=config.agent.alpha,
        gamma=config.agent.gamma,
        env_params=env_params,
        rule_timer_thr=config.agent.rule_timer_thr
    )
    
    if verbose:
        print(f"Simulation completed. Shape: {batch_prices.shape}")
    
    # Calculate equilibrium prices for reference
    env = NPlayerLogitDemandPricingEnv(
        n_players=config.simulation.n_players, 
        **env_params
    )
    monopoly_price = env.get_monopoly_price()
    nash_price = env.get_nash_equilibrium_price()
    
    if verbose:
        print(f"Monopoly price: {monopoly_price:.3f}")
        print(f"Nash equilibrium price: {nash_price:.3f}")
    
    # Save data if requested
    if config.data.save_data:
        # Save main results
        save_n_player_simulation_results(
            batch_prices, batch_actions, batch_profits,
            config.name,
            config.data.data_dir,
            config.data.save_last_periods,
            metadata={
                'config': config.to_dict(),
                'monopoly_price': monopoly_price,
                'nash_price': nash_price
            }
        )
        
        # Save time series if requested
        if config.data.save_time_series:
            save_time_series_snapshots(
                batch_prices, batch_actions, batch_profits,
                config.name,
                config.data.data_dir,
                config.data.time_series_interval,
                config.data.time_series_window
            )
    
    # Generate plots
    player_names = [f"Player {i+1}" for i in range(config.simulation.n_players)]
    
    # Percentile plot
    plot_n_player_percentiles(
        batch_prices, config.simulation.periods, config.simulation.n_players,
        f"{config.name} - Price Competition",
        save_path=os.path.join(config.data.figure_dir, f"{config.name}-percentiles.png"),
        ne_price=nash_price,
        mono_price=monopoly_price,
        player_names=player_names
    )
    
    # Mean with SE plot
    plot_n_player_mean_with_se(
        batch_prices, config.simulation.periods, config.simulation.n_players,
        f"{config.name} - Mean Prices with Standard Error",
        save_path=os.path.join(config.data.figure_dir, f"{config.name}-mean-se.png"),
        ne_price=nash_price,
        mono_price=monopoly_price,
        player_names=player_names
    )
    
    # Rule distribution heatmap
    plot_rule_distribution_heatmap(
        batch_actions, config.simulation.n_players, config.simulation.periods,
        f"{config.name} - Rule Distribution",
        save_path=os.path.join(config.data.figure_dir, f"{config.name}-rule-dist.png")
    )
    
    # Convergence analysis
    variance_over_time = plot_convergence_analysis(
        batch_prices, config.simulation.n_players,
        f"{config.name} - Convergence",
        save_path=os.path.join(config.data.figure_dir, f"{config.name}-convergence.png")
    )
    
    # Calculate summary statistics
    stats = calculate_summary_statistics(batch_prices)
    
    if verbose:
        print(f"Summary statistics:")
        for i, (mean_p, std_p) in enumerate(zip(stats['mean_prices'], stats['std_prices'])):
            print(f"  Player {i+1}: {mean_p:.3f} ± {std_p:.3f}")
        print(f"Cross-player variance: {stats['cross_player_variance']:.6f}")
    
    # Return results
    results = {
        'batch_prices': batch_prices,
        'batch_actions': batch_actions, 
        'batch_profits': batch_profits,
        'monopoly_price': monopoly_price,
        'nash_price': nash_price,
        'statistics': stats,
        'variance_over_time': variance_over_time,
        'config': config
    }
    
    return results

def validate_n2_against_original():
    """
    Validate that N=2 case produces similar results to original implementation.
    
    Returns:
        comparison_results: Dictionary with comparison metrics
    """
    print("Validating N=2 case against original implementation...")
    
    # Run N=2 case with exact same parameters as original
    config = get_baseline_2player_config()
    config.simulation.runs = 10  # Reduce for quick test
    config.simulation.periods = 10000
    
    results = run_n_player_experiment(config, verbose=False)
    
    # Compare with theoretical expectations
    monopoly_price = results['monopoly_price']
    nash_price = results['nash_price'] 
    
    # Check if prices converge to reasonable range
    final_prices = results['batch_prices'][:, :, -1000:].mean(axis=2)  # Average over last 1000 periods
    mean_final_prices = np.mean(final_prices, axis=0)
    
    print(f"Monopoly price: {monopoly_price:.3f}")
    print(f"Nash price: {nash_price:.3f}")
    print(f"Final mean prices: {mean_final_prices}")
    
    # Check if prices are above Nash but below monopoly (expected for collusion)
    above_nash = all(p > nash_price for p in mean_final_prices)
    below_monopoly = all(p < monopoly_price for p in mean_final_prices)
    
    comparison_results = {
        'monopoly_price': monopoly_price,
        'nash_price': nash_price,
        'final_mean_prices': mean_final_prices,
        'above_nash': above_nash,
        'below_monopoly': below_monopoly,
        'price_convergence': results['statistics']['cross_player_variance'] < 0.01,
        'validation_passed': above_nash and below_monopoly
    }
    
    print(f"Validation results:")
    print(f"  Prices above Nash: {above_nash}")
    print(f"  Prices below monopoly: {below_monopoly}")
    print(f"  Price convergence: {comparison_results['price_convergence']}")
    print(f"  Overall validation: {'PASSED' if comparison_results['validation_passed'] else 'FAILED'}")
    
    return comparison_results

def run_n_player_comparison_study():
    """
    Run comparison study across different numbers of players.
    
    Returns:
        comparison_data: Dictionary with results for each N
    """
    print("Running N-player comparison study...")
    
    comparison_data = {}
    
    for n_players in [2, 3, 4, 5]:
        print(f"\nRunning N={n_players} experiment...")
        
        config = get_n_player_config(n_players)
        config.simulation.runs = 20  # Moderate number for comparison
        config.simulation.periods = 50000
        
        results = run_n_player_experiment(config, verbose=False)
        
        comparison_data[n_players] = {
            'final_prices': results['batch_prices'][:, :, -1000:].mean(axis=2),
            'price_volatility': results['statistics']['price_volatility'],
            'cross_player_variance': results['statistics']['cross_player_variance'],
            'monopoly_price': results['monopoly_price'],
            'nash_price': results['nash_price']
        }
        
        print(f"  Final prices: {comparison_data[n_players]['final_prices'].mean(axis=0)}")
        print(f"  Cross-player variance: {comparison_data[n_players]['cross_player_variance']:.6f}")
    
    # Analyze trends
    print(f"\nComparison across N:")
    print(f"{'N':<3} {'Mean Price':<12} {'Nash Price':<12} {'Variance':<12}")
    print("-" * 45)
    
    for n in [2, 3, 4, 5]:
        data = comparison_data[n]
        mean_price = data['final_prices'].mean()
        nash_price = data['nash_price']
        variance = data['cross_player_variance']
        print(f"{n:<3} {mean_price:<12.3f} {nash_price:<12.3f} {variance:<12.6f}")
    
    return comparison_data


if __name__ == "__main__":
    print("Running N-player experiments...")
    
    # Test 1: Validate N=2 case
    print("=" * 60)
    print("TEST 1: Validating N=2 case")
    print("=" * 60)
    validation_results = validate_n2_against_original()
    
    # Test 2: Quick N=3 experiment
    print("\n" + "=" * 60)
    print("TEST 2: Quick N=3 experiment")
    print("=" * 60)
    quick_config = get_quick_test_config(3)
    quick_results = run_n_player_experiment(quick_config)
    
    # Test 3: N-player comparison (if time permits)
    print("\n" + "=" * 60)
    print("TEST 3: N-player comparison study")
    print("=" * 60)
    
    # This is computationally intensive, so make it optional
    run_comparison = input("Run full N-player comparison? (y/n): ").lower().startswith('y')
    
    if run_comparison:
        comparison_results = run_n_player_comparison_study()
    else:
        print("Skipping comparison study.")
    
    print("\nAll experiments completed!")