import sys
from pathlib import Path
import time
from datetime import datetime
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Project Root
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import environment utilities for grid computation
from src.environment import compute_nash_and_monopoly_static


def get_grid_index(price, price_grid):
    """Find the closest grid index for a given price."""
    return int(np.argmin(np.abs(price_grid - price)))


def get_region(grid_idx):
    """
    Map grid index to region.
    - 'H' (High): grids 8, 9
    - 'L' (Low): grids 0, 1
    - 'M' (Mid): grids 2-7
    """
    if grid_idx in [7, 8, 9]:
        return 'H'
    elif grid_idx in [0, 1, 2]:
        return 'L'
    else:
        return 'M'


def has_internal_cycle_in_K_period(prices_in_K, price_grid):
    """
    Check if a single K-period has an internal cycle (C-K).
    
    A C-K exists if H region or L region is visited at least twice,
    with a "departure" (visiting a different region) in between.
    
    Parameters:
    -----------
    prices_in_K : list of float
        Minimum prices at each step within a K-period
    price_grid : np.ndarray
        Array of 10 grid prices
    
    Returns:
    --------
    bool
        True if internal cycle detected
    """
    # Get region sequence for this K-period
    regions = [get_region(get_grid_index(p, price_grid)) for p in prices_in_K]
    
    # Check H region for "fold-back": two H visits with non-H in between
    h_indices = [i for i, r in enumerate(regions) if r == 'H']
    for i in range(len(h_indices) - 1):
        t1, t2 = h_indices[i], h_indices[i + 1]
        # Check if there's any non-H between t1 and t2
        if any(regions[t] != 'H' for t in range(t1 + 1, t2)):
            return True  # H region fold-back detected
    
    # Check L region for "fold-back": two L visits with non-L in between
    l_indices = [i for i, r in enumerate(regions) if r == 'L']
    for i in range(len(l_indices) - 1):
        t1, t2 = l_indices[i], l_indices[i + 1]
        # Check if there's any non-L between t1 and t2
        if any(regions[t] != 'L' for t in range(t1 + 1, t2)):
            return True  # L region fold-back detected
    
    return False


def classify_equilibrium(df, price_grid, K, return_detailed=False):
    """
    Classify equilibrium based on states visited using two-layer classification.
    
    Layer 1 (C-K detection): Check if any K-period has internal cycle (fold-back)
    Layer 2 (K-1 endpoint analysis): Analyze states at end of each K-period
    
    Classification rules:
    - 'H' (High): All K-1 endpoints in grids 8, 9; no internal cycle
    - 'L' (Low): All K-1 endpoints in grids 0, 1; no internal cycle
    - 'C' (Cycle): Either has internal cycle (C-K) or K-1 endpoints span both H and L (C-T)
    - 'O' (Other): Everything else (no internal cycle, endpoints don't fit H/L/C)
    
    Parameters:
    -----------
    df : pd.DataFrame
        The run data from parquet file
    price_grid : np.ndarray
        Array of 10 grid prices
    K : int
        K parameter from config
    return_detailed : bool
        If True, return tuple (type, subtype) where subtype is 'C-K', 'C-T', or None
    
    Returns:
    --------
    str or tuple
        'H', 'L', 'C', or 'O' (or tuple with subtype if return_detailed=True)
    """
    # Get price columns
    p_cols = sorted([c for c in df.columns if c.startswith('p_')],
                    key=lambda x: int(x.split('_')[1]))
    
    # Use only the last 20% of data
    total_rows = len(df)
    start_idx = int(total_rows * 0.8)
    df_tail = df.iloc[start_idx:].reset_index(drop=True)
    
    # ===== Layer 1: Check for internal cycle (C-K) =====
    has_CK = False
    
    # Find indices where step_in_k == 0 (start of each K-period)
    period_starts = df_tail.index[df_tail['step_in_k'] == 0].tolist()
    
    for start_idx in period_starts:
        # Extract one complete K-period
        end_idx = start_idx + K
        if end_idx > len(df_tail):
            continue  # Skip incomplete K-period
        
        period_data = df_tail.iloc[start_idx:end_idx]
        
        # Get minimum price at each step within this K-period
        min_prices_in_K = []
        for _, row in period_data.iterrows():
            prices = [row[col] for col in p_cols]
            min_prices_in_K.append(min(prices))
        
        if has_internal_cycle_in_K_period(min_prices_in_K, price_grid):
            has_CK = True
            break  # Only need one K-period with internal cycle
    
    # ===== Layer 2: Analyze K-1 endpoint states =====
    end_rows = df_tail[df_tail['step_in_k'] == K - 1]
    
    # Collect all state grid indices at K-1 endpoints
    endpoint_grids = []
    for _, row in end_rows.iterrows():
        prices = [row[col] for col in p_cols]
        min_price = min(prices)
        grid_idx = get_grid_index(min_price, price_grid)
        endpoint_grids.append(grid_idx)
    
    # Categorize endpoints (using new H/L definition)
    has_high = any(g in [7, 8, 9] for g in endpoint_grids)
    has_mid = any(g in [3, 4, 5, 6] for g in endpoint_grids)
    has_low = any(g in [0, 1, 2] for g in endpoint_grids)
    
    # ===== Final Classification =====
    if has_CK:
        eq_type = 'C'
        subtype = 'C-K'
    elif has_high and has_low:
        # Cross-T cycle: K-1 endpoints span both H and L
        eq_type = 'C'
        subtype = 'C-T'
    elif has_high and not has_low and not has_mid:
        eq_type = 'H'
        subtype = None
    elif has_low and not has_high and not has_mid:
        eq_type = 'L'
        subtype = None
    else:
        eq_type = 'O'
        subtype = None
    
    if return_detailed:
        return eq_type, subtype
    return eq_type


def compute_price_grid(cfg):
    """
    Compute price grid from config parameters.
    
    Parameters:
    -----------
    cfg : dict
        Config dictionary
        
    Returns:
    --------
    np.ndarray
        Array of 10 grid prices
    """
    num_sellers = cfg['num_sellers']
    a_val = cfg['a_val']
    c_val = cfg['c_val']
    mu_val = cfg['mu']
    a0 = cfg['a0']
    
    # Compute Nash and Monopoly prices
    p_nash, p_monopoly = compute_nash_and_monopoly_static(
        num_sellers=num_sellers,
        a_val=a_val,
        mu=mu_val,
        a0=a0,
        c_val=c_val
    )
    
    # Build price grid (10 grids, Nash at index 1, Monopoly at index 8)
    step = (p_monopoly - p_nash) / 7
    price_grid = np.linspace(p_nash - step, p_monopoly + step, 10)
    
    return price_grid


def _process_one_config(args):
    """
    Process a single (exp_dir, n_dir) configuration. 
    Designed for parallel execution.
    """
    exp_dir, n_dir, config_path = args
    from collections import Counter
    
    # Load config
    with open(config_path, "r") as f:
        cfg = json.load(f)

    N = cfg.get("num_sellers")
    mu = cfg.get("mu")
    K = cfg.get("K")
    
    # Compute price grid
    price_grid = compute_price_grid(cfg)
    
    # Load and classify all runs
    parquet_files = list(n_dir.glob("*.parquet"))
    eq_types = []
    eq_subtypes = []
    
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            eq_type, subtype = classify_equilibrium(df, price_grid, K, return_detailed=True)
            eq_types.append(eq_type)
            eq_subtypes.append(subtype)
        except Exception:
            continue
    
    # Count equilibrium types
    type_counts = Counter(eq_types)
    subtype_counts = Counter(eq_subtypes)
    diversity = len(set(eq_types))
    
    return {
        "Experiment": exp_dir.name,
        "N": N,
        "mu": mu,
        "K": K,
        "eq_diversity": diversity,
        "count_H": type_counts.get('H', 0),
        "count_L": type_counts.get('L', 0),
        "count_C": type_counts.get('C', 0),
        "count_O": type_counts.get('O', 0),
        "count_CK": subtype_counts.get('C-K', 0),
        "count_CT": subtype_counts.get('C-T', 0),
        "run_count": len(eq_types),
    }


def load_all_diversity_data(max_workers=8):
    """
    Load and classify all equilibria across all experiments (parallelized).
    
    Parameters:
    -----------
    max_workers : int
        Number of parallel workers
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns:
        - Experiment, N, mu, K
        - eq_diversity: number of distinct equilibrium types
        - count_H, count_L, count_C, count_O: counts of each type
        - count_CK, count_CT: detailed cycle counts
        - run_count: total number of runs
    """
    results_dir = project_root / "data" / "results"
    
    # Collect all tasks
    tasks = []
    exp_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    print(f"Found {len(exp_dirs)} experiment folders. Scanning...")

    for exp_dir in exp_dirs:
        n_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("N_")])

        for n_dir in n_dirs:
            # Find config file
            config_candidates = list(n_dir.glob("Config_*.json"))
            if config_candidates:
                config_path = config_candidates[0]
            else:
                config_path = exp_dir / f"Config_{n_dir.name}.json"
                if not config_path.exists():
                    continue
            tasks.append((exp_dir, n_dir, config_path))

    print(f"Processing {len(tasks)} configurations with {max_workers} workers...")
    
    # Parallel execution
    summary_records = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_one_config, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                record = future.result()
                summary_records.append(record)
                if i % 20 == 0:
                    print(f"  Processed {i}/{len(tasks)} configurations...")
            except Exception as e:
                print(f"  Warning: Error processing a config: {e}")

    return pd.DataFrame(summary_records)


def draw_split_eq_type_heatmap(ax, df_n, mu_values, k_values, split_cycle=False):
    """
    Draw equilibrium type distribution as split rectangles in each grid cell.
    Similar to action share visualization.
    
    Parameters:
    -----------
    ax : matplotlib axis
        The axis to draw on
    df_n : pd.DataFrame
        Data for specific N
    mu_values : list
        Sorted mu values
    k_values : list
        Sorted K values
    split_cycle : bool
        If True, show C-K and C-T separately instead of unified C
    """
    # Color map for equilibrium types
    if split_cycle:
        EQ_COLOR_MAP = {
            'H': '#2ecc71',   # Green - High/Collusive
            'L': '#e74c3c',   # Red - Low/Competitive
            'C-K': '#e67e22', # Dark Orange - Cycle within K-period
            'C-T': '#f1c40f', # Yellow - Cycle across T-periods
            'O': '#95a5a6'    # Gray - Other
        }
        eq_types_to_plot = ['H', 'L', 'C-K', 'C-T', 'O']
    else:
        EQ_COLOR_MAP = {
            'H': '#2ecc71',  # Green - High/Collusive
            'L': '#e74c3c',  # Red - Low/Competitive
            'C': '#f39c12',  # Orange - Cycle
            'O': '#95a5a6'   # Gray - Other
        }
        eq_types_to_plot = ['H', 'L', 'C', 'O']
    
    # Set up axis
    ax.set_xlim(0, len(mu_values))
    ax.set_ylim(0, len(k_values))
    
    ax.set_xticks(np.arange(len(mu_values)) + 0.5)
    ax.set_xticklabels(mu_values)
    ax.set_yticks(np.arange(len(k_values)) + 0.5)
    ax.set_yticklabels(k_values)
    ax.set_xlabel("mu")
    ax.set_ylabel("K")
    
    sorted_k = sorted(k_values, reverse=True)
    
    # Draw each grid cell
    for row_idx, k_val in enumerate(sorted_k):
        for col_idx, mu_val in enumerate(mu_values):
            
            # Get data for this cell
            mask = (df_n['K'] == k_val) & (df_n['mu'] == mu_val)
            if not mask.any():
                continue
            
            row_data = df_n[mask].iloc[0]
            total_runs = row_data['run_count']
            
            if total_runs == 0:
                continue
            
            # Get counts for each type
            type_shares = []
            if split_cycle:
                # Use detailed C-K and C-T counts
                for eq_type in ['H', 'L', 'C-K', 'C-T', 'O']:
                    if eq_type == 'C-K':
                        count = row_data.get('count_CK', 0)
                    elif eq_type == 'C-T':
                        count = row_data.get('count_CT', 0)
                    else:
                        count = row_data.get(f'count_{eq_type}', 0)
                    share = count / total_runs
                    if share > 0:
                        type_shares.append((eq_type, share, count))
            else:
                # Use unified C count
                for eq_type in ['H', 'L', 'C', 'O']:
                    count = row_data.get(f'count_{eq_type}', 0)
                    share = count / total_runs
                    if share > 0:
                        type_shares.append((eq_type, share, count))
            
            # Sort by share (descending)
            type_shares.sort(key=lambda x: x[1], reverse=True)
            
            # Coordinate transformation
            y_pos = len(k_values) - 1 - row_idx
            x_pos = col_idx
            
            # Draw rectangles for each type
            current_x = x_pos
            for eq_type, share, count in type_shares:
                if share > 0.01:  # Only draw if > 1%
                    color = EQ_COLOR_MAP.get(eq_type, 'gray')
                    rect = mpatches.Rectangle(
                        (current_x, y_pos), width=share, height=1,
                        facecolor=color, edgecolor='none'
                    )
                    ax.add_patch(rect)
                    current_x += share
            
            # Draw white border
            border = mpatches.Rectangle(
                (x_pos, y_pos), 1, 1,
                fill=False, edgecolor='white', linewidth=1
            )
            ax.add_patch(border)
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Create legend
    if split_cycle:
        patches = [
            mpatches.Patch(color='#2ecc71', label='H (High/Collusive)'),
            mpatches.Patch(color='#e74c3c', label='L (Low/Competitive)'),
            mpatches.Patch(color='#e67e22', label='C-K (Cycle within K)'),
            mpatches.Patch(color='#f1c40f', label='C-T (Cycle across T)'),
            mpatches.Patch(color='#95a5a6', label='O (Other)')
        ]
    else:
        patches = [
            mpatches.Patch(color='#2ecc71', label='H (High/Collusive)'),
            mpatches.Patch(color='#e74c3c', label='L (Low/Competitive)'),
            mpatches.Patch(color='#f39c12', label='C (Cycle)'),
            mpatches.Patch(color='#95a5a6', label='O (Other)')
        ]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', 
             borderaxespad=0., fontsize=10, title="EQ Type")


def plot_heatmaps(df_summary, split_cycle=False):
    """
    Plot heatmaps showing equilibrium diversity and type distribution.
    
    Parameters:
    -----------
    df_summary : pd.DataFrame
        Summary data with equilibrium counts
    split_cycle : bool
        If True, show C-K and C-T separately in the type distribution plot
    
    For each N and strategy set:
    - Left subfigure: Heatmap of equilibrium diversity (1-4)
    - Right subfigure: Split-tile visualization of equilibrium type distribution
    """
    sns.set_context("talk")
    plt.rcParams.update({"font.size": 12})

    df_summary["Strategy_Set"] = df_summary["Experiment"].apply(
        lambda x: "4 Strategies" if "4strats" in x else "3 Strategies"
    )

    unique_Ns = sorted(df_summary["N"].unique())
    unique_Strats = df_summary["Strategy_Set"].unique()
    max_diversity = 4  # Maximum possible diversity with our classification

    fig_dir = project_root / "analysis" / "figures" / "heatmaps_eq_diversity_k_mu_by_n"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to {fig_dir} ...")

    count = 0
    for strat_label in unique_Strats:
        df_strat = df_summary[df_summary["Strategy_Set"] == strat_label]

        for n in unique_Ns:
            df_n = df_strat[df_strat["N"] == n]
            
            if df_n.empty:
                continue
            
            try:
                # Get mu and K values for axis labels
                mu_values = sorted(df_n['mu'].unique())
                k_values = sorted(df_n['K'].unique())
                
                # Create pivot table for diversity
                pivot_div = df_n.pivot(index="K", columns="mu", values="eq_diversity").sort_index(ascending=False)
                
                # Create figure with 2 subplots
                fig, axes = plt.subplots(1, 2, figsize=(20, 8))
                fig.suptitle(f"N={n} Sellers | {strat_label}\nEquilibrium Classification", 
                           fontsize=16, fontweight='bold')
                
                # Left: Diversity heatmap
                sns.heatmap(
                    pivot_div,
                    annot=True,
                    fmt=".0f",
                    cmap="YlOrRd",
                    vmin=1,
                    vmax=max_diversity,
                    ax=axes[0],
                    cbar_kws={'label': 'Number of Distinct Equilibrium Types'}
                )
                axes[0].set_title("Equilibrium Diversity (1-4 types)")
                axes[0].set_xlabel("mu")
                axes[0].set_ylabel("K")
                
                # Right: Type distribution split-tile
                title_suffix = " (C-K/C-T split)" if split_cycle else ""
                axes[1].set_title(f"Equilibrium Type Distribution{title_suffix}")
                draw_split_eq_type_heatmap(axes[1], df_n, mu_values, k_values, split_cycle=split_cycle)
                
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                
                safe_label = "4strats" if "4 Strategies" in strat_label else "3strats"
                cycle_suffix = "_split" if split_cycle else ""
                save_path = fig_dir / f"heatmap_eq_diversity_{safe_label}_N{n}{cycle_suffix}.png"
                plt.savefig(save_path, dpi=150)
                plt.close()
                count += 1
                
            except Exception as e:
                print(f"  Error plotting N={n}: {e}")
                import traceback
                traceback.print_exc()

    print(f"Done! Generated {count} equilibrium diversity heatmaps with type distribution.")


if __name__ == "__main__":
    start_time = time.time()
    print("Loading equilibrium diversity data...")
    print(f"Start time: {datetime.now().isoformat(timespec='seconds')}")

    df = load_all_diversity_data()
    plot_heatmaps(df)
    print("All heatmaps generated.")

    elapsed = time.time() - start_time
    print(f"Total elapsed time: {elapsed/60:.2f} minutes.")
