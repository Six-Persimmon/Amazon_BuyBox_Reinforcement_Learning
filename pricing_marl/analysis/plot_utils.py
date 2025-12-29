import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.strategies import ID_TO_NAME

def load_experiment_data(exp_name, downsample_rate=1000):
    """
    load and merge all Parquet files from a given experiment.
    
    Args:
        exp_name (str): name of the exp, e.g. "exp_1_classic_mu025"
        downsample_rate (int): downsampling rate for long-term trend plots. Should always be > 1 otherwise memory issues.
    Returns:
        pd.DataFrame: merged large table
    """
    # 定位结果目录: pricing_marl/data/results/{exp_name}
    base_path = project_root / "data" / "results" / exp_name
    
    if not base_path.exists():
        print(f"Error: Path not found: {base_path}")
        return None
    
    all_dfs = []
    
    # 遍历该实验下的所有 N_x 子文件夹
    # 使用 rglob 递归查找所有 .parquet 文件
    parquet_files = list(base_path.rglob("*.parquet"))
    
    print(f"Loading {len(parquet_files)} files from {exp_name}...")
    
    for f in parquet_files:
        try:
            # 读取 Parquet
            df = pd.read_parquet(f)
            
            # 解析 N (从父文件夹名字 N_2, N_3...)
            # f.parent.name 应该是 "N_2" 这种格式
            n_str = f.parent.name.split("_")[-1]
            try:
                n_sellers = int(n_str)
            except:
                n_sellers = -1 # Fallback
            
            df["n_sellers"] = n_sellers
            
            # --- 降采样 (Downsampling) ---
            # 对于长期趋势图，我们不需要每一步的数据
            if downsample_rate > 1:
                df = df.iloc[::downsample_rate, :].copy()
            
            all_dfs.append(df)
            
        except Exception as e:
            print(f"Skipping {f.name}: {e}")
            
    if not all_dfs:
        print("No data loaded.")
        return pd.DataFrame()
        
    final_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Loaded successfully. Total rows: {len(final_df):,}")
    return final_df

def load_experiment_configs(exp_name):
    """
    加载指定实验下所有的 Config JSON 文件。
    Returns: Dict { n_sellers (int) : config_dict (dict) }
    """
    base_path = project_root / "data" / "results" / exp_name
    if not base_path.exists():
        print(f"Path not found: {base_path}")
        return {}
    
    configs = {}
    # 查找所有 Config_N_*.json
    json_files = list(base_path.glob("Config_N_*.json"))
    
    for jf in json_files:
        try:
            # 文件名格式 Config_N_2.json
            n_str = jf.stem.split("_")[-1]
            n = int(n_str)
            
            with open(jf, "r") as f:
                cfg_data = json.load(f)
                configs[n] = cfg_data
        except Exception as e:
            print(f"Error loading config {jf.name}: {e}")
            
    print(f"Loaded {len(configs)} configs for N={sorted(configs.keys())}")
    return configs

def _apply_smoothing(df, col, window, group_cols=["n_sellers", "run_id"]):
    """
    辅助函数：对指定列进行滑动平均平滑处理。
    必须按 Run 分组平滑，否则不同 Run 的数据会混在一起.
    """
    if window <= 1:
        return df
    
    # 也就是对每个 run_id 内部的时间序列做平滑
    # transform 会保持索引不变，方便赋值回去
    smoothed = df.groupby(group_cols)[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
    return smoothed


# ==========================================
# 3. 核心绘图函数
# ==========================================

def plot_metric_for_specific_n(df, n_sellers, y_col, title, ylabel, 
                               smoothing=100, ci=90, 
                               nash_price=None, monopoly_price=None):
    """
    专门画某个具体的 N 下的某个具体的变量。
    Input:
        - df: 总的大表 (函数内部会筛选 N)
        - n_sellers: 指定的 seller 数量
        - y_col: 要画的列名 (e.g. 'average_price', 'delta')
        - title: 图片标题
        - ylabel: Y轴标签
        - smoothing: 平滑窗口
        - ci: Error bar 分位数 (e.g. 90)
        - nash_price: (Optional) 纳什均衡价格，画虚线用
        - monopoly_price: (Optional) 垄断价格，画虚线用
    """
    # 1. 筛选数据
    sub_df = df[df["n_sellers"] == n_sellers].copy()
    
    if sub_df.empty:
        print(f"No data found for N={n_sellers}")
        return

    # 2. 平滑处理
    if smoothing > 1:
        sub_df[y_col] = _apply_smoothing(sub_df, y_col, smoothing)
        title += f" (Smoothed: {smoothing})"

    # 3. 绘图
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # 主曲线
    sns.lineplot(
        data=sub_df,
        x="t",
        y=y_col,
        errorbar=("ci", ci), # Confidence Interval
        linewidth=2,
        label=ylabel
    )
    
    # 4. 画基准线 (如果提供了)
    if monopoly_price is not None:
        plt.axhline(monopoly_price, color='red', linestyle='--', alpha=0.6, 
                    label=f"Monopoly ({monopoly_price:.2f})")
        
    if nash_price is not None:
        plt.axhline(nash_price, color='gray', linestyle=':', alpha=0.8, 
                    label=f"Nash ({nash_price:.2f})")

    plt.title(title, fontsize=14)
    plt.xlabel("Time Period (t)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

def plot_time_series(df, y_col, title, ylabel, hue="n_sellers", ci=90, smoothing=100):
    """
    将所有 N 画在一张图里 (适合对比 Delta)。
    """
    # Copy to avoid modifying original
    plot_df = df.copy()
    
    if smoothing > 1:
        plot_df[y_col] = _apply_smoothing(plot_df, y_col, smoothing)
        title += f" (Smoothed: {smoothing})"

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    sns.lineplot(
        data=plot_df,
        x="t",
        y=y_col,
        hue=hue,
        palette="viridis",
        errorbar=("ci", ci), # Confidence Interval
        linewidth=2
    )
    
    plt.title(title, fontsize=14)
    plt.xlabel("Time Period (t)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(title="Number of Sellers", loc="best")
    plt.tight_layout()
    plt.show()

    

def plot_action_distribution(df, last_percent=0.1):
    """
    绘制收敛后的策略分布图。
    只取最后 last_percent (例如 10%) 的时间段的数据。
    """
    # 1. 筛选后期数据
    max_t = df["t"].max()
    cutoff = max_t * (1 - last_percent)
    late_df = df[df["t"] > cutoff].copy()
    
    # 2. 融合 (Melt) 所有 Agent 的 Action 列
    # 找到所有以 'a_' 开头的列
    act_cols = [c for c in df.columns if c.startswith("a_")]
    
    melted = late_df.melt(
        id_vars=["n_sellers"], 
        value_vars=act_cols, 
        value_name="action_id"
    )
    
    # 3. 映射 ID -> Name
    # 只有存在于数据中的 ID 才会被映射，防止报错
    present_ids = melted["action_id"].unique()
    # 过滤 ID_TO_NAME 只保留存在的 ID，用于排序
    order_ids = sorted([i for i in present_ids if i in ID_TO_NAME])
    order_names = [ID_TO_NAME[i] for i in order_ids]
    
    melted["Strategy"] = melted["action_id"].map(ID_TO_NAME)
    
    # 4. 绘图
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    sns.histplot(
        data=melted,
        x="Strategy",
        hue="n_sellers",
        multiple="dodge", # 分组柱状图
        stat="probability", # 显示百分比
        shrink=0.8,
        palette="viridis",
        common_norm=False # 每个 N 单独归一化 (即每个 N 的总和都是 100%)
    )
    
    plt.title(f"Strategy Distribution (Last {int(last_percent*100)}% periods)", fontsize=14)
    plt.ylabel("Frequency (Probability)", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_single_run_cycles(exp_name, n_sellers, run_id=0, periods=100):
    """
    深入查看某一次 Run 的最后 periods 期的价格循环 (Edgeworth Cycles)。
    这里不使用降采样，需要加载原始文件。
    """
    # 构造具体的文件路径
    file_path = project_root / "data" / "results" / exp_name / f"N_{n_sellers}" / f"run_{run_id}.parquet"
    
    if not file_path.exists():
        print(f"Run file not found: {file_path}")
        return
    
    df = pd.read_parquet(file_path)
    
    # 只取最后 periods 期
    tail_df = df.tail(periods).copy()
    
    plt.figure(figsize=(12, 5))
    
    # 画出每个 Seller 的价格
    # 找到 p_0, p_1... 列
    price_cols = [c for c in df.columns if c.startswith("p_")]
    
    for p_col in price_cols:
        plt.plot(tail_df["t"], tail_df[p_col], label=p_col, alpha=0.8, marker='o', markersize=3)
        
    plt.title(f"Price Cycles (Last {periods} periods) - Exp: {exp_name}, N={n_sellers}, Run={run_id}")
    plt.xlabel("Time Period")
    plt.ylabel("Price")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_action_share_for_specific_n(df, n_sellers, smoothing=100, ci=95):
    """
    [Fixed v3] 针对特定的 N，画出不同 Action 的占比随时间的变化。
    修复了 KeyError: 'n_sellers' 问题。
    """
    # 1. 筛选数据
    sub_df = df[df["n_sellers"] == n_sellers].copy()
    
    if sub_df.empty:
        print(f"No data found for N={n_sellers}")
        return

    # 2. 剔除全是 NaN 的列 (防止 N=2 时读取到 N=4 的空列)
    sub_df = sub_df.dropna(axis=1, how='all')

    # 3. 强制转换 Action 列为整数
    act_cols = [c for c in sub_df.columns if c.startswith("a_")]
    for c in act_cols:
        sub_df[c] = sub_df[c].astype(int)

    # 4. 数据转换：计算 Share
    # Melt
    melted = sub_df.melt(id_vars=["t", "run_id"], value_vars=act_cols, value_name="act_id")
    
    # One-Hot Encoding
    dummies = pd.get_dummies(melted["act_id"], prefix="act")
    
    # 合并并按 (t, run_id) 聚合求平均
    combined = pd.concat([melted[["t", "run_id"]], dummies], axis=1)
    share_df = combined.groupby(["t", "run_id"]).mean().reset_index()
    
    # ==================================================
    # [FIX] 把 n_sellers 列加回去！
    # 否则 _apply_smoothing 试图按 n_sellers 分组时会报错
    # ==================================================
    share_df["n_sellers"] = n_sellers
    
    # 排序以保证平滑正确
    share_df.sort_values(["run_id", "t"], inplace=True)

    # 5. 绘图
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    all_ids = sorted(ID_TO_NAME.keys())
    lines_drawn = 0
    
    for aid in all_ids:
        col_name = f"act_{aid}"
        
        if col_name not in share_df.columns:
            continue
            
        if smoothing > 1:
            share_df[col_name] = _apply_smoothing(share_df, col_name, smoothing)
        
        sns.lineplot(
            data=share_df,
            x="t",
            y=col_name,
            label=ID_TO_NAME[aid], 
            errorbar=("ci", ci),   
            linewidth=2
        )
        lines_drawn += 1

    if lines_drawn == 0:
        print(f"Warning: No action columns matched for N={n_sellers}.")
    else:
        plt.title(f"Action Share Evolution (N={n_sellers}, Smoothed={smoothing})", fontsize=14)
        plt.xlabel("Time Period (t)", fontsize=12)
        plt.ylabel("Proportion of Sellers", fontsize=12)
        plt.ylim(-0.05, 1.05) 
        plt.legend(title="Strategy", loc="center right")
        plt.tight_layout()
        plt.show()