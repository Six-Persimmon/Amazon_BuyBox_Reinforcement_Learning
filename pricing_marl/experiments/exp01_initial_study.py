# experiments/exp01_initial_study.py
import sys
from pathlib import Path

# current file is at pricing_marl/experiments/
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.strategies import (
    ACT_UNDERCUT, ACT_MATCH, ACT_ABOVE, 
    ACT_UNDER_RESET, ACT_MATCH_RESET
)
from src.runner import run_experiment_batch  # 调用刚才写的 runner

if __name__ == "__main__":
    
    # 全局设置
    N_VALUES = [2, 3, 4, 5] 
    ROUNDS = 50 
    
    # --- Strategy Sets ---
    
    # Set 1: Classic (3 actions): 0, 1, 2
    STRATS_CLASSIC = [ACT_UNDERCUT, ACT_MATCH, ACT_ABOVE]
    
    # Set 2: Reset Only (3 actions)
    # 按全局 ID 从小到大排序：2, 3, 4
    STRATS_RESET_3 = [ACT_ABOVE, ACT_UNDER_RESET, ACT_MATCH_RESET]
    
    # Set 3: Classic + Under+Reset (4 actions)，按全局 ID 升序: 0, 1, 2, 3
    STRATS_RESET_4 = [ACT_UNDERCUT, ACT_MATCH, ACT_ABOVE, ACT_UNDER_RESET]

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
        active_strategies=STRATS_RESET_3,
        n_list=N_VALUES,
        n_rounds=ROUNDS
    )

    # --- 实验 1.4: Reset (3 Actions), mu=0.02 ---
    run_experiment_batch(
        experiment_name="exp_4_reset_mu002",
        mu=0.02,
        active_strategies=STRATS_RESET_3,
        n_list=N_VALUES,
        n_rounds=ROUNDS
    )

    # --- 实验 1.5: Reset (4 Actions), mu=0.25 ---
    run_experiment_batch(
        experiment_name="exp_5_reset4_mu025",
        mu=0.25,
        active_strategies=STRATS_RESET_4,
        n_list=N_VALUES,
        n_rounds=ROUNDS
    )

    # --- 实验 1.6: Reset (4 Actions), mu=0.02 ---
    run_experiment_batch(
        experiment_name="exp_6_reset4_mu002",
        mu=0.02,
        active_strategies=STRATS_RESET_4,
        n_list=N_VALUES,
        n_rounds=ROUNDS
    )

    print("\nAll experiments completed!")


    # run the experiments by:
    # cd /Users/liushijian/Documents/GitHub/Amazon_BuyBox_Reinforcement_Learning/pricing_marl
    # python experiments/exp01_initial_study.py
