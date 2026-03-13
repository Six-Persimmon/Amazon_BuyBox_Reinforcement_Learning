import sys
import os
from pathlib import Path
import itertools
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.config import Config
from src.runner import run_experiment_batch
from src.strategies import (
    ACT_UNDERCUT, ACT_MATCH, ACT_ABOVE, 
    ACT_UNDER_RESET, ACT_MATCH_RESET
)

def _fmt_duration(seconds: float) -> str:
    """Format seconds into H:MM:SS (with 1 decimal if < 60s)."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _parse_csv_env(env_name: str, cast_fn):
    raw = os.getenv(env_name)
    if not raw:
        return None
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(cast_fn(token))
    if not values:
        raise ValueError(f"{env_name} is set but empty.")
    return values


def _filter_values(full_values, requested_values, label: str, float_mode: bool = False):
    if requested_values is None:
        return full_values

    if float_mode:
        matched = [
            v for v in full_values
            if any(abs(v - req) <= 1e-12 for req in requested_values)
        ]
        missing = [
            req for req in requested_values
            if not any(abs(v - req) <= 1e-12 for v in full_values)
        ]
    else:
        requested_set = set(requested_values)
        matched = [v for v in full_values if v in requested_set]
        missing = [req for req in requested_values if req not in full_values]

    if missing:
        raise ValueError(f"{label} filter contains unsupported values: {missing}")
    if not matched:
        raise ValueError(f"{label} filter removed all values.")
    return matched

if __name__ == "__main__":
    
    # --- Experiment Parameter Sweep ---
    # Total combinations = 4 * 7 * 7 = 196 (per strategy set)
    
    # # ---before Jan 4 2026----
    # N_VALUES = [2, 3, 5, 10]
    # MU_VALUES = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
    # K_VALUES = [1, 2, 4, 8, 10, 20, 50]
    # ROUNDS_PER_CONFIG = 10

    # ---Jan 4 2026----
    N_VALUES = [2, 3, 5, 7, 10]
    MU_VALUES = [0.04, 0.07, 0.1, 0.13, 0.16, 0.19, 0.22, 0.25, 0.28, 0.31]
    # K_VALUES = [5, 10, 15, 20, 25, 30, 35, 40]
    # K_VALUES = [5, 10, 15, 20, 25, 27, 30, 35, 37, 39, 40, 50, 60, 70]
    K_VALUES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    ROUNDS_PER_CONFIG = 100 # used to be 30

    # Optional grid filters for targeted reruns (comma-separated lists).
    # Examples:
    #   PRICING_MARL_FILTER_N="7,10"
    #   PRICING_MARL_FILTER_MU="0.01,0.04"
    #   PRICING_MARL_FILTER_K="10,20,30"
    n_filter = _parse_csv_env("PRICING_MARL_FILTER_N", int)
    mu_filter = _parse_csv_env("PRICING_MARL_FILTER_MU", float)
    k_filter = _parse_csv_env("PRICING_MARL_FILTER_K", int)

    N_VALUES = _filter_values(N_VALUES, n_filter, "N")
    MU_VALUES = _filter_values(MU_VALUES, mu_filter, "Mu", float_mode=True)
    K_VALUES = _filter_values(K_VALUES, k_filter, "K")

    rounds_override = os.getenv("PRICING_MARL_ROUNDS_PER_CONFIG")
    if rounds_override:
        ROUNDS_PER_CONFIG = int(rounds_override)

    # Parallelism control (HPC-friendly)
    # Priority: PRICING_MARL_N_JOBS > SLURM_CPUS_PER_TASK > default (all cores)
    env_n_jobs = os.getenv("PRICING_MARL_N_JOBS")
    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    if env_n_jobs is not None:
        N_JOBS = int(env_n_jobs)
    elif slurm_cpus is not None:
        N_JOBS = int(slurm_cpus)
    else:
        N_JOBS = None

    # Optional results directory override (e.g., scratch on HPC)
    results_dir = os.getenv("PRICING_MARL_RESULTS_DIR")
    
    # --- 定义两组实验的策略集 ---
    
    # 实验 A: 基础策略 (3 Strategies)
    STRATS_BASIC = [ACT_UNDERCUT, ACT_MATCH, ACT_ABOVE]
    
    # 实验 B: 含 Reset 的合谋策略 (4 Strategies)
    STRATS_COLLUSIVE = [ACT_UNDERCUT, ACT_MATCH, ACT_ABOVE, ACT_UNDER_RESET]
    
    # 将它们放入一个字典，方便遍历并生成有区分度的文件夹名
    EXPERIMENT_SETS = {
        "3strats": STRATS_BASIC,
        "4strats": STRATS_COLLUSIVE
    }

    # Optional: run a single experiment set without editing code
    only_set = os.getenv("PRICING_MARL_EXPERIMENT_SET")
    if only_set:
        if only_set not in EXPERIMENT_SETS:
            raise ValueError(
                f"Unknown PRICING_MARL_EXPERIMENT_SET='{only_set}'. "
                f"Valid: {list(EXPERIMENT_SETS.keys())}"
            )
        EXPERIMENT_SETS = {only_set: EXPERIMENT_SETS[only_set]}
    
    print(f"Starting Heatmap Scan Experiment...")
    print(f"N: {N_VALUES}")
    print(f"Mu: {MU_VALUES}")
    print(f"K: {K_VALUES}")
    print(f"Rounds per config: {ROUNDS_PER_CONFIG}")
    print("-" * 50)

    overall_start = time.perf_counter()
    # 预先算总 batch 数，便于显示进度
    combinations = list(itertools.product(MU_VALUES, K_VALUES))
    total_batches = len(EXPERIMENT_SETS) * len(combinations)
    batch_counter = 0

    # 遍历每一组策略设定 (3strats vs 4strats)
    for label, current_strats in EXPERIMENT_SETS.items():
        print(f"\n[[[ STARTING EXPERIMENT SET: {label} ({len(current_strats)} strategies) ]]]")
        
        # Loop through Mu and K
        combinations = list(itertools.product(MU_VALUES, K_VALUES))
        
        for mu_val, k_val in combinations:
            
            # [关键修改] 在文件夹名中加入 label (3strats 或 4strats)
            # 结果路径: data/results/scan_3strats_mu0.05_k1/N_2/...
            exp_batch_name = f"scan_{label}_mu{mu_val}_k{k_val}"
            
            print(f"\n>>> Running Batch: {exp_batch_name}")

            batch_start = time.perf_counter()

            run_experiment_batch(
                experiment_name=exp_batch_name,
                mu=mu_val,
                active_strategies=current_strats, # 使用当前循环的策略集
                n_list=N_VALUES,
                n_rounds=ROUNDS_PER_CONFIG,
                n_jobs=N_JOBS,
                output_root=results_dir,
                
                # Additional Config Overrides passed as kwargs
                K=k_val,
                
                # 固定参数
                max_episodes=2_000_000,
                converge_period=10_000,
                eval_H=2_000,
                beta = 1e-5,
                save_training_data=False 
            )
            batch_elapsed = time.perf_counter() - batch_start
            total_elapsed = time.perf_counter() - overall_start
            print(f">>> Batch finished: {exp_batch_name} | elapsed {_fmt_duration(batch_elapsed)}")
            print(f">>> Total elapsed so far: {_fmt_duration(total_elapsed)}")


    print("\n" + "="*50)
    print("All experiments completed.")
    overall_elapsed = time.perf_counter() - overall_start
    print("\n" + "=" * 50)
    print(f"End time: {datetime.now().isoformat(timespec='seconds')}")
    print(f"TOTAL TIME: {_fmt_duration(overall_elapsed)}")
