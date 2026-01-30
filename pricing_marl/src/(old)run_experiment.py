import pandas as pd
import numpy as np
import time
import os
from pathlib import Path
from joblib import Parallel, delayed
from src.config import Config
from src.environment import PricingEnvironment
from src.simulation import run_simulation
from src.strategies import (
    ACT_UNDERCUT, ACT_MATCH, ACT_ABOVE, 
    ACT_UNDER_RESET, ACT_MATCH_RESET
)

# ==========================================
# 0. Worker Function (Run & Save Locally)
# ==========================================

def single_run_and_save(run_id, config, output_folder):
    """
    运行单个 Simulation 并直接保存为 Parquet 文件。
    """
    file_name = f"run_{run_id}.parquet"
    save_path = output_folder / file_name
    
    if save_path.exists():
        return True

    try:
        np.random.seed(run_id) 
        
        # [MODIFIED] disable_tqdm=True to prevent stdout flooding
        df = run_simulation(config, run_id=run_id, disable_tqdm=True)
        
        # Float optimization
        float_cols = df.select_dtypes(include=['float64']).columns
        for c in float_cols:
            df[c] = df[c].astype('float32')
            
        int_cols = df.select_dtypes(include=['int64']).columns
        for c in int_cols:
            df[c] = df[c].astype('int32')

        df.to_parquet(save_path, index=False, compression='snappy')
        
        return True
    except Exception as e:
        print(f"!!! Error in Run {run_id}: {e}")
        return False

# ==========================================
# 1. Batch Controller
# ==========================================

def run_experiment_batch(experiment_name, mu, active_strategies, n_list, n_rounds):
    """
    执行一组实验逻辑
    """
    print(f"\n{'='*60}")
    print(f"Starting Experiment: {experiment_name}")
    print(f"Params: mu={mu}, Rounds={n_rounds}")
    print(f"Strategies: {active_strategies}")
    print(f"{'='*60}")

    for n_sellers in n_list:
        print(f"\n--- Processing N = {n_sellers} ---")
        
        # 1. Config
        # We assume run_experiment.py is in the project root.
        # But we rely on src.config paths to be safe.
        cfg = Config(
            num_sellers=n_sellers,
            num_grids=10,
            mu=mu,
            active_strategies=active_strategies,
            T=1_000_000,
            converge_period=100_000
        )
        
        # [MODIFIED] Use the path defined in Config to ensure consistency
        # data/results/experiment_name/N_x
        current_output_dir = cfg.results_dir / experiment_name / f"N_{n_sellers}"
        current_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. 预热 Lookup Table (单线程)
        print(f"Step 1: Checking/Computing Lookup Table for N={n_sellers}...")
        t0_table = time.time()
        _ = PricingEnvironment(cfg) 
        print(f"        Lookup Table ready ({time.time() - t0_table:.1f}s).")

        # 3. 并行运行
        print(f"Step 2: Running {n_rounds} simulations in parallel...")
        t0_sim = time.time()
        
        # joblib produces its own clean progress output
        Parallel(n_jobs=-1, verbose=5)(
            delayed(single_run_and_save)(seed, cfg, current_output_dir) 
            for seed in range(n_rounds)
        )
        
        elapsed = time.time() - t0_sim
        print(f"        Finished N={n_sellers} in {elapsed:.2f} seconds.")
        print(f"        Data saved to: {current_output_dir}")

# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":
    
    # 全局设置
    N_VALUES = [2, 3, 4, 5] 
    ROUNDS = 50 
    
    # --- Strategy Sets ---
    
    # Set 1: Classic (3 actions)
    # [Undercut, Match, Above]
    STRATS_CLASSIC = [ACT_UNDERCUT, ACT_MATCH, ACT_ABOVE]
    
    # Set 2: Reset Only (3 actions) - [MODIFIED as requested]
    # [Undercut+Reset, Match+Reset, Above]
    STRATS_RESET_3 = [ ACT_ABOVE, ACT_UNDER_RESET, ACT_MATCH_RESET]

    # --- 实验 1.1: Classic, mu=0.25 ---
    run_experiment_batch(
        experiment_name="exp_1_classic_mu025",
        mu=0.25,
        active_strategies=STRATS_CLASSIC,
        n_list=N_VALUES,
        n_rounds=ROUNDS
    )

    # --- 实验 1.2: Classic, mu=0.02 ---
    run_experiment_batch(
        experiment_name="exp_2_classic_mu002",
        mu=0.02,
        active_strategies=STRATS_CLASSIC,
        n_list=N_VALUES,
        n_rounds=ROUNDS
    )

    # --- 实验 1.3: Reset (3 Actions), mu=0.25 ---
    run_experiment_batch(
        experiment_name="exp_3_reset_mu025",
        mu=0.25,
        active_strategies=STRATS_RESET_3, # 使用新的3个动作
        n_list=N_VALUES,
        n_rounds=ROUNDS
    )

    # --- 实验 1.4: Reset (3 Actions), mu=0.02 ---
    run_experiment_batch(
        experiment_name="exp_4_reset_mu002",
        mu=0.02,
        active_strategies=STRATS_RESET_3, # 使用新的3个动作
        n_list=N_VALUES,
        n_rounds=ROUNDS
    )

    print("\nAll experiments completed!")


    # How to run? 
    # cd /Users/liushijian/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning/pricing_marl
    # PYTHONPATH=$(pwd) python src/run_experiment.py