import sys
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import time
from datetime import datetime

# Project Root
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.strategies import ID_TO_NAME, ACT_UNDERCUT, ACT_MATCH, ACT_ABOVE, ACT_UNDER_RESET, ACT_MATCH_RESET

# --- 全局颜色映射 ---
ACTION_COLOR_MAP = {
    ACT_UNDERCUT:    "#e74c3c",  # Red
    ACT_MATCH:       "#3498db",  # Blue
    ACT_ABOVE:       "#2ecc71",  # Green
    ACT_UNDER_RESET: "#9b59b6",  # Purple
    ACT_MATCH_RESET: "#f1c40f"   # Yellow
}
MAX_ACTION_ID = 4

def load_all_heatmap_data():
    results_dir = project_root / "data" / "results"
    if not results_dir.exists():
        print(f"ERROR: Directory {results_dir} does not exist!")
        return pd.DataFrame()

    summary_records = []
    exp_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    print(f"Found {len(exp_dirs)} experiment folders. Scanning...")

    for i, exp_dir in enumerate(exp_dirs):
        n_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("N_")])
        if not n_dirs: continue
            
        for n_dir in n_dirs:
            # Config Search
            config_files = list(n_dir.glob("Config_*.json"))
            if not config_files:
                candidate = exp_dir / f"Config_{n_dir.name}.json"
                if candidate.exists(): config_files = [candidate]
            if not config_files: continue
            
            # Parquet Search
            parquet_files = list(n_dir.glob("*.parquet"))
            if not parquet_files: continue

            try:
                import json
                with open(config_files[0], 'r') as f:
                    cfg = json.load(f)
                
                N = cfg.get('num_sellers')
                mu = cfg.get('mu')
                K = cfg.get('K')
                
                # --- 关键修改：统计 Action 的具体分布 ---
                # 我们需要记录每个 Action ID 出现了多少次
                action_counts_global = {aid: 0 for aid in range(MAX_ACTION_ID + 1)}
                total_steps = 0
                
                run_metrics = []
                for pf in parquet_files:
                    try:
                        df = pd.read_parquet(pf)
                        m, run_counts, run_len = compute_metrics_for_run(df, N)
                        run_metrics.append(m)
                        
                        total_steps += run_len
                        for aid, count in run_counts.items():
                            action_counts_global[aid] += count
                    except: pass 

                if run_metrics:
                    df_runs = pd.DataFrame(run_metrics)
                    avg_metrics = df_runs.mean()
                    
                    # 计算每个 Action 的全局 Share (0.0 - 1.0)
                    action_shares = {}
                    if total_steps > 0:
                        for aid, count in action_counts_global.items():
                            action_shares[f"share_{aid}"] = count / total_steps
                    else:
                        for aid in range(MAX_ACTION_ID + 1):
                            action_shares[f"share_{aid}"] = 0.0

                    record = {
                        "Experiment": exp_dir.name,
                        "N": N, "mu": mu, "K": K,
                        "avg_delta": avg_metrics['delta'],
                        "price_std": avg_metrics['price_std'],
                        "avg_price": avg_metrics['avg_price'],
                        **action_shares # 展开 share_0, share_1 ...
                    }
                    summary_records.append(record)
            
            except Exception as e:
                print(f"  [ERROR] processing {n_dir.name}: {e}")

    return pd.DataFrame(summary_records)

def compute_metrics_for_run(df, N):
    if df.empty: return {}, {}, 0

    avg_delta = df['delta'].mean()
    price_seq = df['price_min'].values
    avg_price = np.mean(price_seq)
    price_std = np.std(price_seq)
    
    # Action Counting
    act_cols = [c for c in df.columns if c.startswith('a_')]
    if not act_cols: return {}, {}, 0
        
    all_actions = df[act_cols].values.flatten()
    all_actions = all_actions[~np.isnan(all_actions)]
    total_count = len(all_actions)
    
    unique, counts = np.unique(all_actions, return_counts=True)
    action_counts = dict(zip(unique.astype(int), counts))
    
    return {
        "delta": avg_delta,
        "price_std": price_std,
        "avg_price": avg_price
    }, action_counts, total_count

def draw_split_action_heatmap(ax, df_n, mu_values, k_values, valid_actions):
    """
    自定义绘图函数：在 Grid 中绘制分割矩形
    """
    # 设置坐标轴
    ax.set_xlim(0, len(mu_values))
    ax.set_ylim(0, len(k_values))
    
    # 标签
    ax.set_xticks(np.arange(len(mu_values)) + 0.5)
    ax.set_xticklabels(mu_values)
    ax.set_yticks(np.arange(len(k_values)) + 0.5)
    ax.set_yticklabels(k_values)
    ax.set_xlabel("mu")
    ax.set_ylabel("K")
    
    # 反转 Y 轴让 K 从大到小排列 (和 Heatmap 保持一致)
    # 注意：我们绘图是 row 从上到下，所以数据处理要注意
    # 最简单的方法：Y轴 0 在底部，但 Heatmap 通常 0 在顶部。
    # 我们这里手动控制：row 0 是 K_max。
    
    sorted_k = sorted(k_values, reverse=True) # Top row is max K
    
    # 遍历每个格子
    for row_idx, k_val in enumerate(sorted_k):
        for col_idx, mu_val in enumerate(mu_values):
            
            # 获取该格子的数据
            mask = (df_n['K'] == k_val) & (df_n['mu'] == mu_val)
            if not mask.any(): continue
            
            row_data = df_n[mask].iloc[0]
            
            # 提取 Shares
            shares = []
            for aid in range(MAX_ACTION_ID + 1):
                s = row_data.get(f"share_{aid}", 0.0)
                shares.append((aid, s))
            
            # 排序找出 Top 2
            shares.sort(key=lambda x: x[1], reverse=True)
            top1_id, top1_share = shares[0]
            top2_id, top2_share = shares[1]
            
            # 坐标转换：Heatmap 的 row 0 在最上面
            # 在 Matplotlib 坐标系中，y=0 是底部。
            # 所以 row_idx 0 (K=50) 应该画在 y = len(k) - 1 - row_idx
            y_pos = len(k_values) - 1 - row_idx
            x_pos = col_idx
            
            # 1. 绘制 Top 1 矩形 (左侧)
            if top1_share > 0:
                color1 = ACTION_COLOR_MAP.get(top1_id, 'gray')
                rect1 = mpatches.Rectangle(
                    (x_pos, y_pos), width=top1_share, height=1, 
                    facecolor=color1, edgecolor='none'
                )
                ax.add_patch(rect1)
            
            # 2. 绘制 Top 2 矩形 (紧接在 Top 1 之后)
            if top2_share > 0.01: # 只有当份额 > 1% 才画，避免太细
                color2 = ACTION_COLOR_MAP.get(top2_id, 'gray')
                rect2 = mpatches.Rectangle(
                    (x_pos + top1_share, y_pos), width=top2_share, height=1, 
                    facecolor=color2, edgecolor='none'
                )
                ax.add_patch(rect2)
                
            # 3. 画个白框把格子隔开
            border = mpatches.Rectangle(
                (x_pos, y_pos), 1, 1, 
                fill=False, edgecolor='white', linewidth=1
            )
            ax.add_patch(border)

    # 去除默认边框线，让图看起来像 Heatmap
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    # 构建 Legend (根据 valid_actions 过滤)
    patches = []
    for aid in valid_actions:
        color = ACTION_COLOR_MAP.get(aid, 'black')
        label = f"{ID_TO_NAME.get(aid, str(aid))}"
        patches.append(mpatches.Patch(color=color, label=label))
    
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10, title="Action Share")

def plot_heatmaps(df_summary):
    sns.set_context("talk")
    plt.rcParams.update({'font.size': 12})
    
    df_summary['Strategy_Set'] = df_summary['Experiment'].apply(
        lambda x: "4 Strategies" if "4strats" in x else "3 Strategies" # maybe need to be finer.
    )
    
    unique_Ns = sorted(df_summary['N'].unique())
    unique_Strats = df_summary['Strategy_Set'].unique()

    fig_dir = project_root / "analysis" / "figures" / "heatmaps_k_mu_by_n"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to {fig_dir} ...")

    count = 0
    for strat_label in unique_Strats:
        df_strat = df_summary[df_summary['Strategy_Set'] == strat_label]
        
        # --- 需求 1: 确定 Legend 的过滤列表 ---
        if "3 Strategies" in strat_label:
            valid_actions = [ACT_UNDERCUT, ACT_MATCH, ACT_ABOVE]
        else:
            valid_actions = [ACT_UNDERCUT, ACT_MATCH, ACT_ABOVE, ACT_UNDER_RESET] # 4 actions

        for n in unique_Ns:
            df_n = df_strat[df_strat['N'] == n]
            if df_n.empty: continue
            
            try:
                pivot_delta = df_n.pivot(index="K", columns="mu", values="avg_delta").sort_index(ascending=False)
                pivot_price = df_n.pivot(index="K", columns="mu", values="avg_price").sort_index(ascending=False)
                pivot_std   = df_n.pivot(index="K", columns="mu", values="price_std").sort_index(ascending=False)
                
                # 获取坐标轴刻度 (Sorted K for Y, Sorted Mu for X)
                mu_values = sorted(df_n['mu'].unique())
                k_values = sorted(df_n['K'].unique()) # Draw function handles reverse sort
                
                fig, axes = plt.subplots(2, 2, figsize=(22, 16))
                fig.suptitle(f"N={n} Sellers | {strat_label}", fontsize=24, y=0.98)
                
                # 1. Delta
                sns.heatmap(pivot_delta, annot=True, fmt=".2f", cmap="RdYlBu_r", ax=axes[0,0], vmin=0, vmax=1)
                axes[0,0].set_title("1. Collusion Index (Delta)")
                
                # 2. Avg Price
                sns.heatmap(pivot_price, annot=True, fmt=".2f", cmap="viridis", ax=axes[0,1])
                axes[0,1].set_title("2. Average Lowest Price")

                # 3. Action Share (Split-Tile Plot) --- [NEW]
                axes[1,0].set_title("3. Action Share (Top 2 Actions)")
                draw_split_action_heatmap(axes[1,0], df_n, mu_values, k_values, valid_actions)

                # 4. Std
                sns.heatmap(pivot_std, annot=True, fmt=".3f", cmap="magma", ax=axes[1,1])
                axes[1,1].set_title("4. Price Instability (Std Dev)")
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                safe_label = "4strats" if "4 Strategies" in strat_label else "3strats"
                save_path = fig_dir / f"heatmap_shares_{safe_label}_N{n}.png"
                plt.savefig(save_path, dpi=150)
                plt.close()
                count += 1
                
            except Exception as e:
                print(f"Error plotting N={n}: {e}")
                import traceback
                traceback.print_exc()

    print(f"Done! Generated {count} heatmaps with Split-Tile Actions.")

if __name__ == "__main__":
    start_time = time.time()
    print(f"Loading heatmap data...")
    print(f"Start time: {datetime.now().isoformat(timespec='seconds')}")
    df = load_all_heatmap_data()
    if not df.empty:
        plot_heatmaps(df)
        print("All heatmaps generated.")
    else:
        print("No data found.")
    
    elapsed = time.time() - start_time
    print(f"Total elapsed time: {elapsed/60:.2f} minutes.")