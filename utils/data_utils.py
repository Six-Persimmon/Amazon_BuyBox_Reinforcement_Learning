#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   data_utils.py
@Time    :   2025/07/11
@Author  :   Shijian Liu
@Version :   1.0
@Contact :   lshijian405@gmail.com
@Desc    :   Data utilities for saving and loading simulation results.
'''

import numpy as np
import pandas as pd
import os
import pickle
from typing import Tuple, Optional, Dict, Any

def save_n_player_simulation_results(price_histories: np.ndarray, 
                                   action_histories: np.ndarray,
                                   profit_histories: np.ndarray,
                                   file_prefix: str,
                                   data_dir: str = "./data",
                                   save_last_periods: int = 1000,
                                   metadata: Optional[Dict[str, Any]] = None):
    """
    Save N-player simulation results to CSV files and metadata to pickle.
    
    Args:
        price_histories: Array of shape (runs, n_players, periods)
        action_histories: Array of shape (runs, n_players, periods)  
        profit_histories: Array of shape (runs, n_players, periods)
        file_prefix: Prefix for output filenames
        data_dir: Directory to save files
        save_last_periods: Number of final periods to save
        metadata: Additional metadata to save
    """
    runs, n_players, periods = price_histories.shape
    
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Save last periods for each data type and player
    for player in range(n_players):
        # Price data
        df_price = pd.DataFrame(
            price_histories[:, player, -save_last_periods:].T,
            columns=[f"Run {i+1}" for i in range(runs)]
        )
        price_file = os.path.join(data_dir, f"{file_prefix}-player{player+1}-price.csv")
        df_price.to_csv(price_file, index=False)
        
        # Action data
        df_action = pd.DataFrame(
            action_histories[:, player, -save_last_periods:].T,
            columns=[f"Run {i+1}" for i in range(runs)]
        )
        action_file = os.path.join(data_dir, f"{file_prefix}-player{player+1}-action.csv")
        df_action.to_csv(action_file, index=False)
        
        # Profit data
        df_profit = pd.DataFrame(
            profit_histories[:, player, -save_last_periods:].T,
            columns=[f"Run {i+1}" for i in range(runs)]
        )
        profit_file = os.path.join(data_dir, f"{file_prefix}-player{player+1}-profit.csv")
        df_profit.to_csv(profit_file, index=False)
    
    # Save metadata
    if metadata is not None:
        metadata_file = os.path.join(data_dir, f"{file_prefix}-metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
    
    print(f"Simulation results saved to {data_dir} with prefix '{file_prefix}'")

def save_time_series_snapshots(price_histories: np.ndarray,
                              action_histories: np.ndarray, 
                              profit_histories: np.ndarray,
                              file_prefix: str,
                              data_dir: str = "./data",
                              interval: int = 5000,
                              window: int = 1000):
    """
    Save simulation results as time series snapshots similar to CIST approach.
    
    Args:
        price_histories: Array of shape (runs, n_players, periods)
        action_histories: Array of shape (runs, n_players, periods)
        profit_histories: Array of shape (runs, n_players, periods)
        file_prefix: Prefix for output filenames
        data_dir: Directory to save files
        interval: Interval between snapshots
        window: Size of each snapshot window
    """
    runs, n_players, total_periods = price_histories.shape
    
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Save snapshots at regular intervals
    for t in range(0, total_periods, interval):
        if t + window <= total_periods:
            for player in range(n_players):
                # Extract window for this player
                price_window = price_histories[:, player, t:t+window]
                action_window = action_histories[:, player, t:t+window]
                profit_window = profit_histories[:, player, t:t+window]
                
                # Save as CSV files
                df_price = pd.DataFrame(
                    price_window.T,
                    columns=[f"Run {i+1}" for i in range(runs)]
                )
                df_action = pd.DataFrame(
                    action_window.T,
                    columns=[f"Run {i+1}" for i in range(runs)]
                )
                df_profit = pd.DataFrame(
                    profit_window.T,
                    columns=[f"Run {i+1}" for i in range(runs)]
                )
                
                # File names with time index
                time_prefix = f"{file_prefix}-p{t:06d}-player{player+1}"
                df_price.to_csv(os.path.join(data_dir, f"{time_prefix}-price.csv"), index=False)
                df_action.to_csv(os.path.join(data_dir, f"{time_prefix}-action.csv"), index=False)
                df_profit.to_csv(os.path.join(data_dir, f"{time_prefix}-profit.csv"), index=False)
    
    # Save final slice if it doesn't fit the interval
    if total_periods % interval != 0:
        start = (total_periods // interval) * interval
        end = total_periods
        for player in range(n_players):
            price_window = price_histories[:, player, start:end]
            action_window = action_histories[:, player, start:end]
            profit_window = profit_histories[:, player, start:end]
            
            df_price = pd.DataFrame(
                price_window.T,
                columns=[f"Run {i+1}" for i in range(runs)]
            )
            df_action = pd.DataFrame(
                action_window.T,
                columns=[f"Run {i+1}" for i in range(runs)]
            )
            df_profit = pd.DataFrame(
                profit_window.T,
                columns=[f"Run {i+1}" for i in range(runs)]
            )
            
            time_prefix = f"{file_prefix}-p{start:06d}-player{player+1}"
            df_price.to_csv(os.path.join(data_dir, f"{time_prefix}-price.csv"), index=False)
            df_action.to_csv(os.path.join(data_dir, f"{time_prefix}-action.csv"), index=False)
            df_profit.to_csv(os.path.join(data_dir, f"{time_prefix}-profit.csv"), index=False)
    
    print(f"Time series snapshots saved to {data_dir} with prefix '{file_prefix}'")

def load_simulation_results(file_prefix: str, 
                          data_dir: str = "./data",
                          n_players: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load simulation results from CSV files.
    
    Args:
        file_prefix: Prefix of the files to load
        data_dir: Directory containing the files
        n_players: Number of players (if None, will auto-detect)
        
    Returns:
        price_histories: Array of shape (runs, n_players, periods)
        action_histories: Array of shape (runs, n_players, periods)
        profit_histories: Array of shape (runs, n_players, periods)
    """
    # Auto-detect number of players if not specified
    if n_players is None:
        player_files = [f for f in os.listdir(data_dir) 
                       if f.startswith(f"{file_prefix}-player") and f.endswith("-price.csv")]
        n_players = len(player_files)
        if n_players == 0:
            raise ValueError(f"No player files found with prefix '{file_prefix}'")
    
    price_data = []
    action_data = []
    profit_data = []
    
    for player in range(n_players):
        # Load price data
        price_file = os.path.join(data_dir, f"{file_prefix}-player{player+1}-price.csv")
        df_price = pd.read_csv(price_file)
        price_data.append(df_price.values.T)  # Transpose to get (runs, periods)
        
        # Load action data
        action_file = os.path.join(data_dir, f"{file_prefix}-player{player+1}-action.csv")
        df_action = pd.read_csv(action_file)
        action_data.append(df_action.values.T)
        
        # Load profit data
        profit_file = os.path.join(data_dir, f"{file_prefix}-player{player+1}-profit.csv")
        df_profit = pd.read_csv(profit_file)
        profit_data.append(df_profit.values.T)
    
    # Stack into arrays of shape (runs, n_players, periods)
    price_histories = np.stack(price_data, axis=1)
    action_histories = np.stack(action_data, axis=1)
    profit_histories = np.stack(profit_data, axis=1)
    
    return price_histories, action_histories, profit_histories

def load_metadata(file_prefix: str, data_dir: str = "./data") -> Optional[Dict[str, Any]]:
    """
    Load metadata from pickle file.
    
    Args:
        file_prefix: Prefix of the metadata file
        data_dir: Directory containing the file
        
    Returns:
        metadata: Dictionary of metadata or None if file doesn't exist
    """
    metadata_file = os.path.join(data_dir, f"{file_prefix}-metadata.pkl")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'rb') as f:
            return pickle.load(f)
    return None

def calculate_summary_statistics(price_histories: np.ndarray,
                               periods_to_analyze: int = 1000) -> Dict[str, np.ndarray]:
    """
    Calculate summary statistics for price histories.
    
    Args:
        price_histories: Array of shape (runs, n_players, periods)
        periods_to_analyze: Number of final periods to analyze
        
    Returns:
        stats: Dictionary containing various statistics
    """
    runs, n_players, periods = price_histories.shape
    
    # Use last periods for analysis
    analysis_data = price_histories[:, :, -periods_to_analyze:]
    
    stats = {
        'mean_prices': np.mean(analysis_data, axis=(0, 2)),  # Average across runs and time
        'std_prices': np.std(analysis_data, axis=(0, 2)),
        'min_prices': np.min(analysis_data, axis=(0, 2)),
        'max_prices': np.max(analysis_data, axis=(0, 2)),
        'median_prices': np.median(analysis_data, axis=(0, 2)),
        'price_volatility': np.std(analysis_data, axis=2).mean(axis=0),  # Time volatility per player
        'cross_player_variance': np.var(analysis_data, axis=1).mean(axis=(0, 1))  # Variance across players
    }
    
    return stats


if __name__ == "__main__":
    # Test data utilities with dummy data
    print("Testing data utilities...")
    
    # Create dummy data
    runs = 3
    n_players = 2
    periods = 5000
    
    np.random.seed(42)
    price_histories = np.random.uniform(2, 8, (runs, n_players, periods))
    action_histories = np.random.randint(0, 4, (runs, n_players, periods))
    profit_histories = np.random.uniform(0, 5, (runs, n_players, periods))
    
    # Test metadata
    metadata = {
        'n_players': n_players,
        'runs': runs,
        'periods': periods,
        'alpha': 0.1,
        'gamma': 0.9,
        'environment': 'NPlayerLogitDemandPricingEnv'
    }
    
    # Test saving
    test_dir = "./test_data"
    print("Saving simulation results...")
    save_n_player_simulation_results(
        price_histories, action_histories, profit_histories,
        "test-n-player-simulation", test_dir, metadata=metadata
    )
    
    # Test saving time series snapshots
    print("Saving time series snapshots...")
    save_time_series_snapshots(
        price_histories, action_histories, profit_histories,
        "test-n-player-timeseries", test_dir
    )
    
    # Test loading
    print("Loading simulation results...")
    loaded_prices, loaded_actions, loaded_profits = load_simulation_results(
        "test-n-player-simulation", test_dir, n_players
    )
    
    # Test metadata loading
    loaded_metadata = load_metadata("test-n-player-simulation", test_dir)
    print(f"Loaded metadata: {loaded_metadata}")
    
    # Test summary statistics
    print("Calculating summary statistics...")
    stats = calculate_summary_statistics(price_histories)
    print(f"Mean prices: {stats['mean_prices']}")
    print(f"Price volatility: {stats['price_volatility']}")
    
    # Verify data integrity
    print(f"Original shape: {price_histories.shape}")
    print(f"Loaded shape: {loaded_prices.shape}")
    print(f"Data matches: {np.allclose(price_histories[:, :, -1000:], loaded_prices)}")
    
    # Clean up test files
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    print("Test completed and cleaned up!")