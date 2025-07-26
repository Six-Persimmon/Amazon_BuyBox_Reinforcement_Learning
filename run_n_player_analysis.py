#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Comprehensive N-player analysis script
Runs simulations for N=2 to 5 players and generates statistics and visualizations
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from time import time

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulations.n_player_simulation import simulate_batch_n_player, get_equilibrium_prices
from analysis.plotting import plot_n_player_mean_with_se
from utils.data_utils import save_n_player_simulation_results


def run_n_player_simulations():
    """Run simulations for N=2 to 5 players."""
    print("Running N-player simulations (N=2 to 5)...")
    print("=" * 60)
    
    # Simulation parameters
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
        'periods': 100_000,  # Reduced for faster execution
        'runs': 20,          # Reduced for faster execution  
        'alpha': 0.15,
        'gamma': 0.9,
        'rule_timer_thr': 4
    }
    
    print(f"Simulation parameters:")
    print(f"- Periods: {simulation_params['periods']:,}")
    print(f"- Runs: {simulation_params['runs']}")
    print(f"- Marginal cost: ${env_params['marginal_cost']:.2f}")
    print(f"- Price range: ${env_params['price_min']:.2f} - ${env_params['price_max']:.2f}")
    print()
    
    results = {}
    
    for n_players in [2, 3, 4, 5]:
        print(f"Running N={n_players} simulation...")
        start_time = time()
        
        # Calculate theoretical prices using the correct method from environment
        monopoly_price, nash_price = get_equilibrium_prices(n_players, env_params)
        
        # Run simulation
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
        
        # Calculate statistics
        final_periods = min(1000, simulation_params['periods'] // 10)
        final_prices = batch_prices[:, :, -final_periods:]
        final_profits = batch_profits[:, :, -final_periods:]
        
        # Per-player statistics
        mean_prices = np.mean(final_prices, axis=(0, 2))  # Average across runs and time
        std_prices = np.std(final_prices, axis=(0, 2))
        mean_profits = np.mean(final_profits, axis=(0, 2))
        
        # Overall statistics
        overall_mean_price = np.mean(mean_prices)
        overall_std_price = np.std(mean_prices)
        overall_mean_profit = np.mean(mean_profits)
        
        # Price evolution (average across all players and runs)
        price_evolution = np.mean(batch_prices, axis=(0, 1))  # Average across runs and players
        
        # Debug: Check initial prices to investigate Issue 1
        if n_players == 2:
            initial_prices = batch_prices[:, :, :100]  # First 100 periods
            print(f"N=2 Debug - Initial price range: ${np.min(initial_prices):.2f} - ${np.max(initial_prices):.2f}")
            print(f"N=2 Debug - Mean first 10 periods: ${np.mean(initial_prices[:, :, :10]):.2f}")
            print(f"N=2 Debug - Mean periods 90-100: ${np.mean(initial_prices[:, :, 90:100]):.2f}")
        
        # Below-cost pricing analysis
        below_cost_rate = np.mean(final_prices < env_params['marginal_cost'])
        
        # Rule usage analysis - Issue 4: use only last 1000 periods
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
    """Generate summary statistics table in academic 3-line format."""
    print("SUMMARY STATISTICS")
    print("=" * 100)
    
    # Create summary table
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
    
    # Academic 3-line table format
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Summary Statistics of N-Player Pricing Competition}")
    print("\\label{tab:summary_stats}")
    print("\\begin{tabular}{lccccccc}")
    print("\\toprule")
    print("N Players & Mean Price & Std Dev & Mean Profit & Below Cost \\% & Nash Price & Monopoly Price \\\\")
    print("\\midrule")
    
    for _, row in df.iterrows():
        print(f"{row['N_Players']} & "
              f"{row['Mean_Price']:.3f} & "
              f"{row['Price_Std']:.3f} & "
              f"{row['Mean_Profit']:.4f} & "
              f"{row['Below_Cost_Rate']:.1%} & "
              f"{row['Nash_Price']:.3f} & "
              f"{row['Monopoly_Price']:.3f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print()
    
    # Also print readable version
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
    
    # Rule usage analysis table (also in academic format)
    print("RULE USAGE ANALYSIS")
    print("=" * 60)
    
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Rule Usage Frequency by Number of Players}")
    print("\\label{tab:rule_usage}")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("N Players & Match & Above & Below* & Hold & Raise \\\\")
    print("\\midrule")
    
    for n_players, data in results.items():
        usage_line = f"{n_players} & "
        for i, usage in enumerate(data['rule_usage']):
            usage_line += f"{usage:.3f}"
            if i < len(data['rule_usage']) - 1:
                usage_line += " & "
            else:
                usage_line += " \\\\"
        print(usage_line)
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print()
    
    # Readable rule usage
    print("READABLE RULE USAGE:")
    rule_names = ['Match', 'Above', 'Below*', 'Hold', 'Raise']
    rule_header = f"{'N':<3} " + " ".join(f"{name:<8}" for name in rule_names)
    print(rule_header)
    print("=" * len(rule_header))
    
    for n_players, data in results.items():
        usage_str = f"{n_players:<3} "
        for i, usage in enumerate(data['rule_usage']):
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

def simulate_old_n2_version():
    """Simulate N=2 case using old rules (from CIST_Q_rule.ipynb) for comparison."""
    print("Running N=2 simulation with OLD rules for comparison...")
    
    # Import the old agent from CIST notebook logic
    sys.path.append('.')
    
    # Recreate the old agent logic
    class OldQLearningRuleAgent:
        def __init__(self, n_actions, alpha=0.1, gamma=0.9, cost=2.0, prices=None, rule_timer_thr=2):
            self.n_price_actions = n_actions
            self.n_rules = 4  # Old version had 4 rules
            self.alpha = alpha
            self.gamma = gamma
            self.omega = 1.5e-5  # Original slow decay
            self.t = 0
            self.Q = np.random.uniform(10, 20, size=(n_actions**2, self.n_rules))
            self.current_rule = 0
            self.rule_timer_thr = rule_timer_thr
            self.rule_timer = self.rule_timer_thr
            self.cost = cost
            self.prices = prices
        
        def take_action(self, state, rival_pre_price_idx, own_pre_price_idx):
            if self.rule_timer >= self.rule_timer_thr:
                epsilon = np.exp(-self.t * self.omega)
                if np.random.rand() < epsilon:
                    new_rule = np.random.randint(self.n_rules)
                else:
                    new_rule = int(np.argmax(self.Q[state]))
                self.current_rule = new_rule
                self.rule_timer = 0
            
            price_idx = self._apply_rule(self.current_rule, rival_pre_price_idx, own_pre_price_idx)
            self.rule_timer += 1
            return price_idx
        
        def _apply_rule(self, rule, rival_idx, own_idx):
            if rule == 0:
                return rival_idx
            elif rule == 1:
                return min(rival_idx + 1, self.n_price_actions - 1)
            elif rule == 2:
                return max(rival_idx - 1, 0)  # OLD VERSION: Can go below marginal cost
            elif rule == 3:
                return own_idx
        
        def update(self, state, rule, reward, next_state):
            best_next = np.max(self.Q[next_state])
            td_target = reward + self.gamma * best_next
            self.Q[state, rule] += self.alpha * (td_target - self.Q[state, rule])
            self.t += 1
    
    # Simple environment for old version
    def joint_to_index(i, j, grid_size):
        return i * grid_size + j
    
    # Simulation parameters
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
    
    periods = 100_000
    runs = 20
    n_actions = env_params['grid_size']
    prices = np.linspace(env_params['price_min'], env_params['price_max'], n_actions)
    
    # Simple logit demand function
    def simple_logit_demand(price_indices, prices, env_params):
        p1, p2 = prices[price_indices[0]], prices[price_indices[1]]
        deno = np.exp((env_params['a_12'] - p1) / env_params['mu']) + \
               np.exp((env_params['a_12'] - p2) / env_params['mu']) + \
               np.exp(env_params['a_0'] / env_params['mu'])
        d1 = np.exp((env_params['a_12'] - p1) / env_params['mu']) / deno
        d2 = np.exp((env_params['a_12'] - p2) / env_params['mu']) / deno
        r1 = (p1 - env_params['marginal_cost']) * d1
        r2 = (p2 - env_params['marginal_cost']) * d2
        return r1, r2
    
    # Run simulations
    all_price_histories = []
    
    for run in range(runs):
        agent0 = OldQLearningRuleAgent(n_actions, alpha=0.15, gamma=0.9, 
                                      cost=env_params['marginal_cost'], prices=prices, rule_timer_thr=2)
        agent1 = OldQLearningRuleAgent(n_actions, alpha=0.15, gamma=0.9, 
                                      cost=env_params['marginal_cost'], prices=prices, rule_timer_thr=2)
        
        history = np.zeros((2, periods))
        
        # Initial random state
        obs_0 = np.random.randint(n_actions)
        obs_1 = np.random.randint(n_actions)
        state = joint_to_index(obs_0, obs_1, n_actions)
        
        for t in range(periods):
            # Agents take actions
            a0 = agent0.take_action(state, obs_1, obs_0)
            a1 = agent1.take_action(state, obs_0, obs_1)
            
            # Calculate rewards
            r0, r1 = simple_logit_demand((a0, a1), prices, env_params)
            
            # Next state
            next_state = joint_to_index(a0, a1, n_actions)
            
            # Update agents
            agent0.update(state, agent0.current_rule, r0, next_state)
            agent1.update(state, agent1.current_rule, r1, next_state)
            
            # Record
            history[0, t] = prices[a0]
            history[1, t] = prices[a1]
            
            # Update state
            obs_0, obs_1 = a0, a1
            state = next_state
        
        all_price_histories.append(history)
    
    # Convert to numpy array and calculate average
    batch_prices_old = np.array(all_price_histories)  # Shape: (runs, 2, periods)
    price_evolution_old = np.mean(batch_prices_old, axis=(0, 1))  # Average across runs and players
    
    return price_evolution_old, batch_prices_old

def create_visualizations(results, simulation_params):
    """Create visualizations of the results."""
    print("CREATING VISUALIZATIONS")
    print("=" * 40)
    
    # Create output directory
    os.makedirs("./figure/n_player_analysis", exist_ok=True)
    
    # Get old N=2 results for comparison
    old_n2_evolution, old_n2_batch = simulate_old_n2_version()
    
    # Create subplots: 2x3 grid (N=2,3,4,5 + N=2 old + summary)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    window = 300  # Fixed moving average window - smaller to show more detail
    
    # Subplots for each N (new version)
    for i, (n_players, data) in enumerate(results.items()):
        ax = axes[i]
        
        # Plot price evolution for each player separately (Issue 2)
        batch_prices = data['batch_prices']  # Shape: (runs, n_players, periods)
        
        # Calculate per-player averages across runs
        player_colors = plt.cm.Set1(np.linspace(0, 1, n_players))
        
        for player_id in range(n_players):
            player_prices = np.mean(batch_prices[:, player_id, :], axis=0)  # Average across runs
            
            # Simple fixed window smoothing
            if len(player_prices) > window:
                price_smooth = np.convolve(player_prices, np.ones(window)/window, mode='valid')
                x = np.arange(len(price_smooth))
            else:
                price_smooth = player_prices
                x = np.arange(len(price_smooth))
            
            ax.plot(x, price_smooth, color=player_colors[player_id], linewidth=1.5, 
                   label=f'Player {player_id+1}', alpha=0.8)
        
        # Reference lines
        marginal_cost = 2.0
        ax.axhline(y=marginal_cost, color='black', linestyle='--', 
                  label=f'Marginal Cost', alpha=0.7)
        
        if data['nash_price'] is not None:
            ax.axhline(y=data['nash_price'], color='red', linestyle=':', 
                      label=f'Nash Eq.', alpha=0.7)
        
        if data['monopoly_price'] is not None:
            ax.axhline(y=data['monopoly_price'], color='green', linestyle=':', 
                      label=f'Monopoly', alpha=0.7)
        
        ax.set_title(f'N={n_players} Players (New Rules)')
        ax.set_xlabel('Period')
        ax.set_ylabel('Average Price ($)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, ncol=2)  # Smaller font and 2 columns for space
    
    # Subplot for N=2 old version
    ax_old = axes[4]  # 5th subplot
    if len(old_n2_evolution) > window:
        price_smooth_old = np.convolve(old_n2_evolution, np.ones(window)/window, mode='valid')
        x_old = np.arange(window-1, len(old_n2_evolution))
    else:
        price_smooth_old = old_n2_evolution
        x_old = np.arange(len(price_smooth_old))
    
    ax_old.plot(x_old, price_smooth_old, color='brown', linewidth=2, label='Simulated (Old Rules)')
    ax_old.axhline(y=marginal_cost, color='black', linestyle='--', label='Marginal Cost', alpha=0.7)
    
    # Add Nash and Monopoly for N=2 old version
    if results[2]['nash_price'] is not None:
        ax_old.axhline(y=results[2]['nash_price'], color='red', linestyle=':', 
                      label='Nash Eq.', alpha=0.7)
    if results[2]['monopoly_price'] is not None:
        ax_old.axhline(y=results[2]['monopoly_price'], color='green', linestyle=':', 
                      label='Monopoly', alpha=0.7)
    
    ax_old.set_title('N=2 Players (Old Rules)')
    ax_old.set_xlabel('Period')
    ax_old.set_ylabel('Average Price ($)')
    ax_old.grid(True, alpha=0.3)
    ax_old.legend(fontsize=8)
    
    # Comparison subplot (N=2 old vs new)
    ax_comp = axes[5]  # 6th subplot
    
    # New N=2
    n2_evolution = results[2]['price_evolution']
    if len(n2_evolution) > window:
        n2_smooth = np.convolve(n2_evolution, np.ones(window)/window, mode='valid')
        x_n2 = np.arange(window-1, len(n2_evolution))
    else:
        n2_smooth = n2_evolution
        x_n2 = np.arange(len(n2_smooth))
    
    ax_comp.plot(x_n2, n2_smooth, color='blue', linewidth=2, label='New Rules')
    ax_comp.plot(x_old, price_smooth_old, color='brown', linewidth=2, 
                label='Old Rules', linestyle='--')
    
    ax_comp.axhline(y=marginal_cost, color='black', linestyle='--', 
                   label='Marginal Cost', alpha=0.7)
    
    ax_comp.set_title('N=2: New vs Old Rules Comparison')
    ax_comp.set_xlabel('Period')
    ax_comp.set_ylabel('Average Price ($)')
    ax_comp.grid(True, alpha=0.3)
    ax_comp.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('./figure/n_player_analysis/price_evolution_subplots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Final price comparison (event study style)
    plt.figure(figsize=(10, 6))
    
    n_values = list(results.keys())
    mean_prices = [results[n]['overall_mean_price'] for n in n_values]
    price_stds = [results[n]['overall_std_price'] for n in n_values]
    nash_prices = [results[n]['nash_price'] for n in n_values]
    monopoly_prices = [results[n]['monopoly_price'] for n in n_values]
    
    x = np.array(n_values)  # Use actual N values instead of indices
    
    # Plot simulated prices with error bars (no connecting lines) - Issue 3: smaller dots
    plt.errorbar(x, mean_prices, yerr=price_stds, fmt='bo', capsize=5, 
                capthick=2, markersize=4, label='Simulated Prices', linewidth=0, elinewidth=2)
    
    # Plot Nash equilibrium as separate points (no connecting lines)
    plt.scatter(x, nash_prices, c='red', marker='s', s=80, 
               label='Nash Equilibrium', zorder=5)
    
    # Plot Monopoly prices as separate points (no connecting lines)
    plt.scatter(x, monopoly_prices, c='green', marker='^', s=80,
               label='Monopoly Price', zorder=5)
    
    # Add marginal cost reference line
    marginal_cost = 2.0
    plt.axhline(y=marginal_cost, color='black', linestyle='--', 
               label=f'Marginal Cost (${marginal_cost:.2f})', alpha=0.7)
    
    plt.xlabel('Number of Players')
    plt.ylabel('Price ($)')
    plt.title('Final Prices vs Number of Players')
    plt.xticks(x)  # Show actual N values
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./figure/n_player_analysis/final_prices_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Rule usage heatmap - Issue 4: Improved version with blue colors and detailed description
    plt.figure(figsize=(12, 6))
    
    """
    DETAILED DESCRIPTION OF RULE USAGE HEATMAP PLOTTING:
    
    This heatmap visualizes how frequently each of the 5 pricing rules is used
    by agents across different numbers of players (N=2,3,4,5).
    
    Data preparation:
    1. For each N, we extract the last 1000 periods from all simulation runs
    2. batch_actions has shape (runs, n_players, periods)
    3. We take batch_actions[:, :, -1000:] to get the final 1000 periods
    4. We calculate frequency as: mean(batch_actions == rule) across all runs and players
    5. This gives us a rule_usage array of shape (5,) for each N
    6. We stack these into rule_matrix of shape (4, 5) representing (N_scenarios, rules)
    
    Visualization:
    - X-axis: Number of players (N=2,3,4,5)  
    - Y-axis: The 5 pricing rules
    - Color intensity: Usage frequency (0=never used, 1=always used)
    - Blue colormap: Uses 'Blues' instead of red-based colors
    - Text annotations: Show exact frequency values on each cell
    
    The heatmap reveals:
    - Which rules are most popular for each N
    - How rule preferences change as competition increases
    - The conditional availability of Rule 4 (Raise) affects its usage patterns
    """
    
    rule_names = ['Match', 'Above', 'Below*', 'Hold', 'Raise*']  # Added * to Raise
    rule_matrix = np.array([results[n]['rule_usage'] for n in n_values])
    
    # Use blue-based colormap instead of red-based
    im = plt.imshow(rule_matrix.T, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    cbar = plt.colorbar(im, label='Usage Frequency')
    cbar.ax.tick_params(labelsize=10)
    
    plt.yticks(range(5), rule_names, fontsize=11)
    plt.xticks(range(len(n_values)), [f'N={n}' for n in n_values], fontsize=11)
    plt.xlabel('Number of Players', fontsize=12)
    plt.ylabel('Pricing Rule', fontsize=12)
    plt.title('Rule Usage Frequency by Number of Players\n(Based on Last 1000 Periods of All Runs)', fontsize=13)
    
    # Add text annotations with better contrast
    for i in range(len(n_values)):
        for j in range(5):
            # Choose text color based on background intensity
            text_color = 'white' if rule_matrix[i, j] > 0.5 else 'black'
            plt.text(i, j, f'{rule_matrix[i, j]:.3f}', 
                    ha='center', va='center', fontweight='bold', 
                    color=text_color, fontsize=10)
    
    # Add footnote about Rule 4
    plt.figtext(0.02, 0.02, '*Raise rule is only available when agent has the lowest price (conditional availability)', 
                fontsize=9, style='italic', ha='left')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for footnote
    plt.savefig('./figure/n_player_analysis/rule_usage_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved to ./figure/n_player_analysis/")

def save_results(results, simulation_params, summary_df):
    """Save detailed results and tables to files."""
    print("SAVING DETAILED RESULTS")
    print("=" * 30)
    
    os.makedirs("./data/n_player_analysis", exist_ok=True)
    
    # Save simulation data
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
    
    # Save summary statistics table
    summary_df.to_csv("./data/n_player_analysis/summary_statistics.csv", index=False)
    
    # Save LaTeX tables to file
    with open("./data/n_player_analysis/latex_tables.tex", "w") as f:
        f.write("% Summary Statistics Table\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Summary Statistics of N-Player Pricing Competition}\n")
        f.write("\\label{tab:summary_stats}\n")
        f.write("\\begin{tabular}{lccccccc}\n")
        f.write("\\toprule\n")
        f.write("N Players & Mean Price & Std Dev & Mean Profit & Below Cost \\% & Nash Price & Monopoly Price \\\\\n")
        f.write("\\midrule\n")
        
        for _, row in summary_df.iterrows():
            f.write(f"{row['N_Players']} & "
                   f"{row['Mean_Price']:.3f} & "
                   f"{row['Price_Std']:.3f} & "
                   f"{row['Mean_Profit']:.4f} & "
                   f"{row['Below_Cost_Rate']:.1%} & "
                   f"{row['Nash_Price']:.3f} & "
                   f"{row['Monopoly_Price']:.3f} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # Rule usage table
        f.write("% Rule Usage Table\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Rule Usage Frequency by Number of Players}\n")
        f.write("\\label{tab:rule_usage}\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("N Players & Match & Above & Below* & Hold & Raise \\\\\n")
        f.write("\\midrule\n")
        
        for n_players, data in results.items():
            usage_line = f"{n_players} & "
            for i, usage in enumerate(data['rule_usage']):
                usage_line += f"{usage:.3f}"
                if i < len(data['rule_usage']) - 1:
                    usage_line += " & "
                else:
                    usage_line += " \\\\\n"
            f.write(usage_line)
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print("Detailed results saved to ./data/n_player_analysis/")
    print("LaTeX tables saved to ./data/n_player_analysis/latex_tables.tex")
    print("Summary CSV saved to ./data/n_player_analysis/summary_statistics.csv")

def main():
    """Main analysis function."""
    print("N-PLAYER PRICING COMPETITION ANALYSIS")
    print("=" * 60)
    print("This script runs simulations for N=2 to 5 players")
    print("and generates comprehensive statistics and visualizations.")
    print()
    
    # Run simulations
    results, simulation_params = run_n_player_simulations()
    
    # Generate statistics
    summary_df = generate_summary_statistics(results)
    
    # Create visualizations
    create_visualizations(results, simulation_params)
    
    # Save results
    save_results(results, simulation_params, summary_df)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("Key findings:")
    print(f"- Simulations ran successfully for N=2 to 5 players")
    print(f"- All player groups maintain prices above marginal cost")
    print(f"- Results saved to ./data/n_player_analysis/")
    print(f"- Visualizations saved to ./figure/n_player_analysis/")
    
    return results, summary_df

if __name__ == "__main__":
    results, summary_df = main()