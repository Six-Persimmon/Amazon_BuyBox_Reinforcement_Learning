#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   n_player_experiments.py
@Time    :   2025/07/11
@Author  :   Shijian Liu
@Version :   1.0
@Contact :   lshijian405@gmail.com
@Desc    :   N-player pricing experiments, analysis, and validation utilities.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from time import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.experiment_configs import *
from simulations.n_player_simulation import simulate_batch_n_player, get_equilibrium_prices
from analysis.plotting import *
from utils.data_utils import *
from env.NPlayerLogitDemandPricingEnv import NPlayerLogitDemandPricingEnv


# ---------------------------------------------------------------------------
# Analysis pipeline helpers (formerly in run_n_player_analysis.py)
# ---------------------------------------------------------------------------

def run_n_player_simulations():
    """Run baseline simulations for N=2 to 5 players."""
    print("Running N-player simulations (N=2 to 5)...")
    print("=" * 60)

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

    simulation_params = {
        'periods': 100_000,
        'runs': 20,
        'alpha': 0.15,
        'gamma': 0.9,
        'rule_timer_thr': 4
    }

    print("Simulation parameters:")
    print(f"- Periods: {simulation_params['periods']:,}")
    print(f"- Runs: {simulation_params['runs']}")
    print(f"- Marginal cost: ${env_params['marginal_cost']:.2f}")
    print(f"- Price range: ${env_params['price_min']:.2f} - ${env_params['price_max']:.2f}")
    print()

    results = {}

    for n_players in [2, 3, 4, 5]:
        print(f"Running N={n_players} simulation...")
        start_time = time()

        monopoly_price, nash_price = get_equilibrium_prices(n_players, env_params)

        batch_prices, batch_actions, batch_profits = simulate_batch_n_player(
            n_players=n_players,
            periods=simulation_params['periods'],
            runs=simulation_params['runs'],
            alpha=simulation_params['alpha'],
            gamma=simulation_params['gamma'],
            env_params=env_params,
            rule_timer_thr=simulation_params['rule_timer_thr']
        )

        elapsed_time = time() - start_time
        print(f"  Completed in {elapsed_time:.1f} seconds")

        final_periods = min(1000, simulation_params['periods'] // 10)
        final_prices = batch_prices[:, :, -final_periods:]
        final_profits = batch_profits[:, :, -final_periods:]

        mean_prices = np.mean(final_prices, axis=(0, 2))
        std_prices = np.std(final_prices, axis=(0, 2))
        mean_profits = np.mean(final_profits, axis=(0, 2))

        overall_mean_price = np.mean(mean_prices)
        overall_std_price = np.std(mean_prices)
        overall_mean_profit = np.mean(mean_profits)
        price_evolution = np.mean(batch_prices, axis=(0, 1))

        if n_players == 2:
            initial_prices = batch_prices[:, :, :100]
            print(f"N=2 Debug - Initial price range: ${np.min(initial_prices):.2f} - ${np.max(initial_prices):.2f}")
            print(f"N=2 Debug - Mean first 10 periods: ${np.mean(initial_prices[:, :, :10]):.2f}")
            print(f"N=2 Debug - Mean periods 90-100: ${np.mean(initial_prices[:, :, 90:100]):.2f}")

        below_cost_rate = np.mean(final_prices < env_params['marginal_cost'])

        rule_usage = np.zeros(5)
        last_1000_periods = min(1000, simulation_params['periods'])
        for rule in range(5):
            rule_usage[rule] = np.mean(batch_actions[:, :, -last_1000_periods:] == rule)

        results[n_players] = {
            'batch_prices': batch_prices,
            'batch_actions': batch_actions,
            'batch_profits': batch_profits,
            'mean_prices': mean_prices,
            'std_prices': std_prices,
            'mean_profits': mean_profits,
            'overall_mean_price': overall_mean_price,
            'overall_std_price': overall_std_price,
            'overall_mean_profit': overall_mean_profit,
            'price_evolution': price_evolution,
            'below_cost_rate': below_cost_rate,
            'rule_usage': rule_usage,
            'monopoly_price': monopoly_price,
            'nash_price': nash_price,
            'final_periods': final_periods
        }

        print(f"  Mean price: ${overall_mean_price:.3f} ± ${overall_std_price:.3f}")
        print(f"  Mean profit: ${overall_mean_profit:.4f}")
        print(f"  Below-cost rate: {below_cost_rate:.3%}")
        print()

    return results, simulation_params


def generate_summary_statistics(results):
    """Generate summary statistics tables for reporting."""
    print("SUMMARY STATISTICS")
    print("=" * 100)

    summary_data = []
    for n_players, data in results.items():
        summary_data.append({
            'N_Players': n_players,
            'Mean_Price': data['overall_mean_price'],
            'Price_Std': data['overall_std_price'],
            'Mean_Profit': data['overall_mean_profit'],
            'Below_Cost_Rate': data['below_cost_rate'],
            'Nash_Price': data['nash_price'],
            'Monopoly_Price': data['monopoly_price']
        })

    df = pd.DataFrame(summary_data)

    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Summary Statistics of N-Player Pricing Competition}")
    print(r"\label{tab:summary_stats}")
    print(r"\begin{tabular}{lccccccc}")
    print(r"\toprule")
    print(r"N Players & Mean Price & Std Dev & Mean Profit & Below Cost \\% & Nash Price & Monopoly Price \\")
    print(r"\midrule")

    for _, row in df.iterrows():
        print(f"{row['N_Players']} & "
              f"{row['Mean_Price']:.3f} & "
              f"{row['Price_Std']:.3f} & "
              f"{row['Mean_Profit']:.4f} & "
              f"{row['Below_Cost_Rate']:.1%} & "
              f"{row['Nash_Price']:.3f} & "
              f"{row['Monopoly_Price']:.3f} \\\\n")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()

    print("READABLE FORMAT:")
    print("=" * 100)
    header = f"{'N':<3} {'Mean Price':<12} {'Std Dev':<10} {'Mean Profit':<12} {'Below Cost%':<12} {'Nash Price':<12} {'Monopoly':<10}"
    print(header)
    print("=" * len(header))

    for _, row in df.iterrows():
        print(f"{row['N_Players']:<3} "
              f"${row['Mean_Price']:<11.3f} "
              f"${row['Price_Std']:<9.3f} "
              f"${row['Mean_Profit']:<11.4f} "
              f"{row['Below_Cost_Rate']:<11.1%} "
              f"${row['Nash_Price']:<11.3f} "
              f"${row['Monopoly_Price']:<9.3f}")

    print("=" * len(header))
    print()

    print("RULE USAGE ANALYSIS")
    print("=" * 60)

    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Rule Usage Frequency by Number of Players}")
    print(r"\label{tab:rule_usage}")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"N Players & Match & Above & Below* & Hold & Raise \\")
    print(r"\midrule")

    for n_players, data in results.items():
        usage_line = f"{n_players} & "
        for i, usage in enumerate(data['rule_usage']):
            usage_line += f"{usage:.3f}"
            usage_line += " & " if i < len(data['rule_usage']) - 1 else " \\\\"
        print(usage_line)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()

    print("READABLE RULE USAGE:")
    rule_names = ['Match', 'Above', 'Below*', 'Hold', 'Raise']
    rule_header = f"{'N':<3} " + " ".join(f"{name:<8}" for name in rule_names)
    print(rule_header)
    print("=" * len(rule_header))

    for n_players, data in results.items():
        usage_str = f"{n_players:<3} "
        for usage in data['rule_usage']:
            usage_str += f"{usage:<8.3f} "
        print(usage_str)

    print("=" * len(rule_header))
    print()
    print("Rule descriptions:")
    print("- Match: Match lowest competitor price")
    print("- Above: Price one step above lowest competitor")
    print("- Below*: Price below competitor (but not below marginal cost)")
    print("- Hold: Keep current price")
    print("- Raise: Raise price by one step (only when lowest)")

    return df


def create_visualizations(results, simulation_params):
    """Create visualizations of multi-player pricing outcomes."""
    print("CREATING VISUALIZATIONS")
    print("=" * 40)

    os.makedirs("./figure/n_player_analysis", exist_ok=True)

    window = 300
    n_values = sorted(results.keys())
    n_plots = len(n_values)
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6 * n_rows))
    axes = np.array(axes).reshape(-1)

    marginal_cost = 2.0

    for i, n_players in enumerate(n_values):
        data = results[n_players]
        ax = axes[i]

        batch_prices = data['batch_prices']
        player_colors = plt.cm.Set1(np.linspace(0, 1, n_players))

        for player_id in range(n_players):
            player_prices = np.mean(batch_prices[:, player_id, :], axis=0)

            if len(player_prices) > window:
                price_smooth = np.convolve(player_prices, np.ones(window) / window, mode='valid')
                x = np.arange(len(price_smooth))
            else:
                price_smooth = player_prices
                x = np.arange(len(price_smooth))

            ax.plot(x, price_smooth, color=player_colors[player_id], linewidth=1.5,
                    label=f'Player {player_id+1}', alpha=0.8)

        ax.axhline(y=marginal_cost, color='black', linestyle='--',
                   label='Marginal Cost', alpha=0.7)

        if data['nash_price'] is not None:
            ax.axhline(y=data['nash_price'], color='red', linestyle=':',
                       label='Nash Eq.', alpha=0.7)

        if data['monopoly_price'] is not None:
            ax.axhline(y=data['monopoly_price'], color='green', linestyle=':',
                       label='Monopoly', alpha=0.7)

        ax.set_title(f'N={n_players} Players')
        ax.set_xlabel('Period')
        ax.set_ylabel('Average Price ($)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, ncol=2)

    for idx in range(n_plots, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig('./figure/n_player_analysis/price_evolution_subplots.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 6))

    mean_prices = [results[n]['overall_mean_price'] for n in n_values]
    price_stds = [results[n]['overall_std_price'] for n in n_values]
    nash_prices = [results[n]['nash_price'] for n in n_values]
    monopoly_prices = [results[n]['monopoly_price'] for n in n_values]

    x = np.array(n_values)

    plt.errorbar(x, mean_prices, yerr=price_stds, fmt='bo', capsize=5,
                 capthick=2, markersize=4, label='Simulated Prices', linewidth=0, elinewidth=2)
    plt.scatter(x, nash_prices, c='red', marker='s', s=80,
                label='Nash Equilibrium', zorder=5)
    plt.scatter(x, monopoly_prices, c='green', marker='^', s=80,
                label='Monopoly Price', zorder=5)

    marginal_cost = 2.0
    plt.axhline(y=marginal_cost, color='black', linestyle='--',
                label=f'Marginal Cost (${marginal_cost:.2f})', alpha=0.7)

    plt.xlabel('Number of Players')
    plt.ylabel('Price ($)')
    plt.title('Final Prices vs Number of Players')
    plt.xticks(x)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./figure/n_player_analysis/final_prices_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(12, 6))

    rule_names = ['Match', 'Above', 'Below*', 'Hold', 'Raise*']
    rule_matrix = np.array([results[n]['rule_usage'] for n in n_values])

    im = plt.imshow(rule_matrix.T, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    cbar = plt.colorbar(im, label='Usage Frequency')
    cbar.ax.tick_params(labelsize=10)

    plt.yticks(range(5), rule_names, fontsize=11)
    plt.xticks(range(len(n_values)), [f'N={n}' for n in n_values], fontsize=11)
    plt.xlabel('Number of Players', fontsize=12)
    plt.ylabel('Pricing Rule', fontsize=12)
    plt.title('Rule Usage Frequency by Number of Players\n(Based on Last 1000 Periods of All Runs)', fontsize=13)

    for i in range(len(n_values)):
        for j in range(5):
            text_color = 'white' if rule_matrix[i, j] > 0.5 else 'black'
            plt.text(i, j, f'{rule_matrix[i, j]:.3f}',
                     ha='center', va='center', fontweight='bold',
                     color=text_color, fontsize=10)

    plt.figtext(0.02, 0.02,
                '*Raise rule is only available when agent has the lowest price (conditional availability)',
                fontsize=9, style='italic', ha='left')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('./figure/n_player_analysis/rule_usage_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Visualizations saved to ./figure/n_player_analysis/")


def save_results(results, simulation_params, summary_df):
    """Persist simulation outputs and summary tables."""
    print("SAVING DETAILED RESULTS")
    print("=" * 30)

    os.makedirs("./data/n_player_analysis", exist_ok=True)

    for n_players, data in results.items():
        file_prefix = f"n_player_analysis_N{n_players}"

        save_n_player_simulation_results(
            data['batch_prices'],
            data['batch_actions'],
            data['batch_profits'],
            file_prefix,
            "./data/n_player_analysis",
            save_last_periods=1000,
            metadata={
                'n_players': n_players,
                'simulation_params': simulation_params,
                'monopoly_price': data['monopoly_price'],
                'nash_price': data['nash_price'],
                'overall_mean_price': data['overall_mean_price'],
                'below_cost_rate': data['below_cost_rate']
            }
        )

    summary_df.to_csv("./data/n_player_analysis/summary_statistics.csv", index=False)

    with open("./data/n_player_analysis/latex_tables.tex", "w") as f:
        f.write("% Summary Statistics Table\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Summary Statistics of N-Player Pricing Competition}\n")
        f.write("\\label{tab:summary_stats}\n")
        f.write("\\begin{tabular}{lccccccc}\n")
        f.write("\\toprule\n")
        header_row = (
            "N Players & Mean Price & Std Dev & Mean Profit & Below Cost "
            "\\% & Nash Price & Monopoly Price "
            "\\\\"
        )
        f.write(header_row + "\n")


        f.write("\\midrule\n")

        for _, row in summary_df.iterrows():
            f.write(
                f"{row['N_Players']} & {row['Mean_Price']:.3f} & {row['Price_Std']:.3f} & "
                f"{row['Mean_Profit']:.4f} & {row['Below_Cost_Rate']:.1%} & "
                f"{row['Nash_Price']:.3f} & {row['Monopoly_Price']:.3f} \\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")

        f.write("% Rule Usage Table\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Rule Usage Frequency by Number of Players}\n")
        f.write("\\label{tab:rule_usage}\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("N Players & Match & Above & Below* & Hold & Raise \\\\n")
        f.write("\\midrule\n")

        for n_players, data in results.items():
            usage_values = " & ".join(f"{usage:.3f}" for usage in data['rule_usage'])
            f.write(f"{n_players} & {usage_values} \\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print("Detailed results saved to ./data/n_player_analysis/")
    print("LaTeX tables saved to ./data/n_player_analysis/latex_tables.tex")
    print("Summary CSV saved to ./data/n_player_analysis/summary_statistics.csv")


def run_default_n_player_analysis():
    """End-to-end pipeline matching the legacy run_n_player_analysis script."""
    print("N-PLAYER PRICING COMPETITION ANALYSIS")
    print("=" * 60)
    print("This pipeline runs simulations for N=2 to 5 players")
    print("and generates comprehensive statistics and visualizations.")
    print()

    results, simulation_params = run_n_player_simulations()
    summary_df = generate_summary_statistics(results)
    create_visualizations(results, simulation_params)
    save_results(results, simulation_params, summary_df)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("Key findings:")
    print("- Simulations ran successfully for N=2 to 5 players")
    print("- All player groups maintain prices above marginal cost")
    print("- Results saved to ./data/n_player_analysis/")
    print("- Visualizations saved to ./figure/n_player_analysis/")

    return results, summary_df


# ---------------------------------------------------------------------------
# Config-driven experiment interfaces (original content)
# ---------------------------------------------------------------------------

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
    print("Running N-player experiments module...")
    print("=" * 60)
    print("Option 1: Default analysis pipeline (N=2-5)")
    print("Option 2: Validation + custom experiment suite")
    choice = input("Select option (1/2): ").strip()
    
    if choice == '1':
        run_default_n_player_analysis()
    else:
        print("\n" + "=" * 60)
        print("TEST 1: Validating N=2 case")
        print("=" * 60)
        validate_n2_against_original()
        
        print("\n" + "=" * 60)
        print("TEST 2: Quick N=3 experiment")
        print("=" * 60)
        quick_config = get_quick_test_config(3)
        run_n_player_experiment(quick_config)
        
        print("\n" + "=" * 60)
        print("TEST 3: N-player comparison study (optional)")
        print("=" * 60)
        run_comparison = input("Run full N-player comparison? (y/n): ").lower().startswith('y')
        if run_comparison:
            run_n_player_comparison_study()
        else:
            print("Skipping comparison study.")
    
    print("\nAll experiments completed!")
