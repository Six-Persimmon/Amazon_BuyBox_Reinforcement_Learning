#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   plotting.py
@Time    :   2025/07/11
@Author  :   Shijian Liu
@Version :   1.0
@Contact :   lshijian405@gmail.com
@Desc    :   Plotting utilities for N-player pricing competition analysis.
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def move_average(x, window):
    """Compute moving average of x with window size."""
    return np.convolve(x, np.ones(window), 'valid') / window

def plot_n_player_percentiles(price_histories, periods, n_players, title, 
                             save_path=None, ne_price=None, mono_price=None, 
                             window=300, player_names=None):
    """
    Plot percentiles for N-player price competition.
    
    Args:
        price_histories: np.ndarray of shape (runs, n_players, periods)
        periods: Number of periods
        n_players: Number of players
        title: Plot title
        save_path: Path to save figure
        ne_price: Nash equilibrium price (optional)
        mono_price: Monopoly price (optional)
        window: Moving average window size
        player_names: List of player names (optional)
    """
    if player_names is None:
        player_names = [f"Player {i+1}" for i in range(n_players)]
    
    # Create subplots
    fig, axes = plt.subplots(2, (n_players + 1) // 2, figsize=(15, 8))
    if n_players == 1:
        axes = [axes]
    elif n_players <= 2:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    colors = plt.cm.Set1(np.linspace(0, 1, 5))
    
    for player in range(n_players):
        ax = axes[player]
        
        # Calculate percentiles for this player
        player_data = price_histories[:, player, :]  # shape: (runs, periods)
        percentiles = np.percentile(player_data, [0, 25, 50, 75, 100], axis=0)
        
        # Apply moving average
        x = np.arange(periods - window + 1)
        labels_linestyle = [
            ('Min', '-', colors[0]),
            ('25th percentile', '--', colors[1]),
            ('Median', '-', colors[2]),
            ('75th percentile', '--', colors[3]),
            ('Max', '-', colors[4]),
        ]
        
        for idx, (label, ls, color) in enumerate(labels_linestyle):
            lw = 2 if label == 'Median' else 1
            move_ave_pct = move_average(percentiles[idx], window=window)
            ax.plot(x, move_ave_pct, linestyle=ls, linewidth=lw, 
                   label=label, color=color)
        
        # Add reference lines
        if ne_price is not None:
            ax.axhline(y=ne_price, color='grey', linestyle='--', 
                      label='Nash Equilibrium', alpha=0.7)
        if mono_price is not None:
            ax.axhline(y=mono_price, color='black', linestyle='--', 
                      label='Monopoly Price', alpha=0.7)
        
        ax.set_title(player_names[player])
        ax.set_xlabel("Period")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.3)
        if player == 0:  # Only show legend for first subplot
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Remove empty subplots
    for i in range(n_players, len(axes)):
        fig.delaxes(axes[i])
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_n_player_mean_with_se(price_histories, periods, n_players, title,
                              save_path=None, ne_price=None, mono_price=None,
                              window=300, player_names=None):
    """
    Plot mean prices with standard error bands for N players.
    
    Args:
        price_histories: np.ndarray of shape (runs, n_players, periods)
        periods: Number of periods
        n_players: Number of players
        title: Plot title
        save_path: Path to save figure
        ne_price: Nash equilibrium price (optional)
        mono_price: Monopoly price (optional)
        window: Moving average window size
        player_names: List of player names (optional)
    """
    if player_names is None:
        player_names = [f"Player {i+1}" for i in range(n_players)]
    
    runs = price_histories.shape[0]
    colors = plt.cm.Set1(np.linspace(0, 1, n_players))
    
    plt.figure(figsize=(12, 8))
    
    x = np.arange(periods - window + 1)
    
    for player in range(n_players):
        player_data = price_histories[:, player, :]  # shape: (runs, periods)
        
        # Calculate mean and standard error
        mean_series = np.mean(player_data, axis=0)
        se_series = np.std(player_data, axis=0, ddof=1) / np.sqrt(runs)
        
        # Apply moving average
        mean_smooth = move_average(mean_series, window)
        se_smooth = move_average(se_series, window)
        
        plt.plot(x, mean_smooth, label=player_names[player], 
                linewidth=2, color=colors[player])
        plt.fill_between(x, mean_smooth - 2*se_smooth, mean_smooth + 2*se_smooth, 
                        alpha=0.2, color=colors[player])
    
    # Add reference lines
    if ne_price is not None:
        plt.axhline(y=ne_price, color='grey', linestyle='--', 
                   label='Nash Equilibrium', alpha=0.7)
    if mono_price is not None:
        plt.axhline(y=mono_price, color='black', linestyle='--', 
                   label='Monopoly Price', alpha=0.7)
    
    plt.title(title)
    plt.xlabel("Period")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_rule_distribution_heatmap(action_histories, n_players, periods, title,
                                  save_path=None, window=1000):
    """
    Plot heatmap showing distribution of rules used by each player over time.
    
    Args:
        action_histories: np.ndarray of shape (runs, n_players, periods)
        n_players: Number of players
        periods: Number of periods
        title: Plot title
        save_path: Path to save figure
        window: Window size for aggregation
    """
    rule_names = ['Match Lowest', 'Above Lowest', 'Below Lowest', 'Hold Price']
    
    # Aggregate data into windows
    n_windows = periods // window
    rule_dist = np.zeros((n_players, 4, n_windows))
    
    for player in range(n_players):
        for w in range(n_windows):
            start_idx = w * window
            end_idx = (w + 1) * window
            window_actions = action_histories[:, player, start_idx:end_idx]
            
            for rule in range(4):
                rule_dist[player, rule, w] = np.mean(window_actions == rule)
    
    # Create subplots
    fig, axes = plt.subplots(1, n_players, figsize=(4*n_players, 6))
    if n_players == 1:
        axes = [axes]
    
    for player in range(n_players):
        ax = axes[player]
        
        im = ax.imshow(rule_dist[player], aspect='auto', cmap='YlOrRd',
                      vmin=0, vmax=1)
        
        ax.set_title(f'Player {player + 1}')
        ax.set_xlabel('Time Window')
        ax.set_ylabel('Pricing Rule')
        ax.set_yticks(range(4))
        ax.set_yticklabels(rule_names)
        
        # Add colorbar
        if player == n_players - 1:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Frequency')
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_convergence_analysis(price_histories, n_players, title, save_path=None):
    """
    Plot convergence analysis showing price variance over time.
    
    Args:
        price_histories: np.ndarray of shape (runs, n_players, periods)
        n_players: Number of players
        title: Plot title
        save_path: Path to save figure
    """
    runs, _, periods = price_histories.shape
    
    # Calculate cross-player price variance for each time period
    variance_over_time = []
    for t in range(periods):
        # Get prices at time t across all runs and players
        prices_t = price_histories[:, :, t]  # shape: (runs, n_players)
        # Calculate variance across players for each run, then average
        cross_player_variance = np.var(prices_t, axis=1)  # variance across players
        mean_variance = np.mean(cross_player_variance)  # average across runs
        variance_over_time.append(mean_variance)
    
    # Apply smoothing
    window = min(1000, periods // 10)
    smooth_variance = move_average(variance_over_time, window)
    x = np.arange(len(smooth_variance))
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, smooth_variance, linewidth=2)
    plt.title(f'{title} - Price Convergence Analysis')
    plt.xlabel('Period')
    plt.ylabel('Cross-Player Price Variance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return variance_over_time


if __name__ == "__main__":
    # Test plotting functions with dummy data
    print("Testing plotting functions with dummy data...")
    
    # Create dummy data
    runs = 5
    n_players = 3
    periods = 1000
    
    # Generate some realistic price patterns
    np.random.seed(42)
    price_histories = np.random.uniform(2, 8, (runs, n_players, periods))
    action_histories = np.random.randint(0, 4, (runs, n_players, periods))
    
    # Add some trends to make it more interesting
    for run in range(runs):
        for player in range(n_players):
            trend = np.linspace(0, np.random.uniform(-1, 1), periods)
            noise = np.random.normal(0, 0.5, periods)
            price_histories[run, player] += trend + noise
    
    # Test plotting functions
    print("Creating plots...")
    
    plot_n_player_percentiles(
        price_histories, periods, n_players,
        "Test N-Player Price Competition"
    )
    
    plot_n_player_mean_with_se(
        price_histories, periods, n_players,
        "Test N-Player Mean Prices"
    )
    
    plot_rule_distribution_heatmap(
        action_histories, n_players, periods,
        "Test Rule Distribution"
    )
    
    plot_convergence_analysis(
        price_histories, n_players,
        "Test Convergence"
    )
    
    print("All plots created successfully!")