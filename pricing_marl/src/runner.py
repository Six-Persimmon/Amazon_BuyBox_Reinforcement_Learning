import pandas as pd
import numpy as np
import time
import json
import dataclasses
from pathlib import Path
from joblib import Parallel, delayed

from src.config import Config
from src.environment import PricingEnvironment
from src.simulation import run_simulation

# ==========================================
# 0. Worker Function (Run & Save Locally)
# ==========================================

def single_run_and_save(run_id, config, output_folder):
    """
    运行单个 Simulation 并直接保存为 Parquet 文件。
    """
    file_name = f"run_{run_id}.parquet"
    save_path = output_folder / file_name
    
    # check if file exists
    if save_path.exists():
        # print(f"Run {run_id} exists. Skipping.") 
        return True

    try:
        np.random.seed(run_id) 
        
        # disable_tqdm=True
        df_eval = run_simulation(config, run_id=run_id, disable_tqdm=True)
        df_eval.to_parquet(save_path, index=False, compression='snappy')
        return True
    except Exception as e:
        print(f"!!! Error in Run {run_id}: {e}")
        return False

# ==========================================
# 1. Batch Controller
# ==========================================

def run_experiment_batch(experiment_name, mu, active_strategies, n_list, n_rounds, **kwargs):
    """
    Args:
        experiment_name: 实验名称，用于生成文件夹
        mu: 产品差异化参数
        active_strategies: 策略 ID 列表
        n_list: 需要遍历的卖家数量列表 [2, 3, 5, 10]
        n_rounds: 每个设定跑多少个随机种子
        **kwargs: 用于覆盖 Config 默认值的其他参数
    """
    print(f"\n{'='*60}")
    print(f"Starting Experiment: {experiment_name}")
    print(f"Params: mu={mu}, Rounds={n_rounds}")
    print(f"Strategies: {active_strategies}")
    print(f"{'='*60}")

    base_output_dir = Path(__file__).resolve().parent.parent / "data" / "results"/experiment_name
    base_output_dir.mkdir(parents=True, exist_ok=True)

    for n_sellers in n_list:
        # 使用 Config 定义的路径，确保和之前的实验路径完全一致
        # Path: pricing_marl/data/results/experiment_name/N_x
        current_output_dir = base_output_dir / f"N_{n_sellers}"
        current_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Processing N = {n_sellers} ---")

        # 1. 构建 Config 对象
        cfg = Config(
            num_sellers=n_sellers,
            mu=mu,
            active_strategies=active_strategies,
        )

        for key, value in kwargs.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                print(f"Warning: Config has no attribute '{key}', ignoring.")
        

        
        # ==================================================
        # [NEW] Save Config as JSON
        # Path: pricing_marl/data/results/experiment_name/Config_N_x.json
        # ==================================================
        config_file_path = current_output_dir.parent / f"Config_N_{n_sellers}.json"
        
        # 将 Config 对象转换为字典
        cfg_dict = dataclasses.asdict(cfg)
        
        # JSON 无法直接序列化 Path 对象，需要转为字符串
        for k, v in cfg_dict.items():
            if isinstance(v, Path):
                cfg_dict[k] = str(v)
        
        # 写入文件
        try:
            with open(config_file_path, "w") as f:
                json.dump(cfg_dict, f, indent=4)
            print(f"        [Config] Saved to {config_file_path.name}")
        except Exception as e:
            print(f"        [Warning] Failed to save config: {e}")
        # ==================================================

        # 2. 预热 Lookup Table
        print(f"Step 1: Checking/Computing Lookup Table for N={n_sellers}...")
        t0_table = time.time()
        _ = PricingEnvironment(cfg) 
        print(f"        Lookup Table ready ({time.time() - t0_table:.1f}s).")

        # 3. 并行运行
        print(f"Step 2: Running {n_rounds} simulations in parallel...")
        t0_sim = time.time()
        
        Parallel(n_jobs=-1, verbose=5)(
            delayed(single_run_and_save)(seed, cfg, current_output_dir) 
            for seed in range(n_rounds)
        )
        
        elapsed = time.time() - t0_sim
        print(f"        Finished N={n_sellers} in {elapsed:.2f} seconds.")
        print(f"        Data saved to: {current_output_dir}")