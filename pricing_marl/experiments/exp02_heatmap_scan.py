import sys
from pathlib import Path
import itertools

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.config import Config
from src.runner import run_experiment_batch
from src.strategies import (
    ACT_UNDERCUT, ACT_MATCH, ACT_ABOVE, 
    ACT_UNDER_RESET, ACT_MATCH_RESET
)

if __name__ == "__main__":
    
    # --- Experiment Parameter Sweep ---
    # Total combinations = 4 * 7 * 7 = 196 (per strategy set)
    
    N_VALUES = [2, 3, 5, 10]
    MU_VALUES = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
    K_VALUES = [1, 2, 4, 8, 10, 20, 50]
    
    ROUNDS_PER_CONFIG = 10
    
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
    
    print(f"Starting Heatmap Scan Experiment...")
    print(f"N: {N_VALUES}")
    print(f"Mu: {MU_VALUES}")
    print(f"K: {K_VALUES}")
    print("-" * 50)

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
            
            run_experiment_batch(
                experiment_name=exp_batch_name,
                mu=mu_val,
                active_strategies=current_strats, # 使用当前循环的策略集
                n_list=N_VALUES,
                n_rounds=ROUNDS_PER_CONFIG,
                
                # Additional Config Overrides passed as kwargs
                K=k_val,
                
                # 固定参数
                max_episodes=2_000_000,
                converge_period=10_000,
                eval_H=10_000,
                beta = 1e-5,
                save_training_data=False 
            )

    print("\n" + "="*50)
    print("All experiments completed.")