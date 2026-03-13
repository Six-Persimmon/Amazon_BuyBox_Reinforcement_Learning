import sys
from pathlib import Path
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Project Root
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

EVAL_RUN_FILE_RE = re.compile(r"^run_\d+\.parquet$")


def _list_eval_parquet_files(n_dir: Path):
    return sorted(
        p for p in n_dir.glob("run_*.parquet")
        if EVAL_RUN_FILE_RE.match(p.name)
    )


def load_all_heatmap_data():
    results_dir = project_root / "data" / "results"
    if not results_dir.exists():
        print(f"ERROR: Directory {results_dir} does not exist!")
        return pd.DataFrame()

    summary_records = []
    exp_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    print(f"Found {len(exp_dirs)} experiment folders. Scanning for Delta Diff analysis...")

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
            parquet_files = _list_eval_parquet_files(n_dir)
            if not parquet_files: continue

            try:
                import json
                with open(config_files[0], 'r') as f:
                    cfg = json.load(f)
                
                N = cfg.get('num_sellers')
                mu = cfg.get('mu')
                K = cfg.get('K')
                
                deltas = []
                for pf in parquet_files:
                    try:
                        # Only reading delta column for speed
                        df = pd.read_parquet(pf, columns=['delta']) 
                        if not df.empty:
                            deltas.append(df['delta'].mean())
                    except: pass 

                if deltas:
                    avg_delta = np.mean(deltas)
                    record = {
                        "Experiment": exp_dir.name,
                        "N": N, "mu": mu, "K": K,
                        "avg_delta": avg_delta
                    }
                    summary_records.append(record)
            
            except Exception as e:
                print(f"  [ERROR] processing {n_dir.name}: {e}")

    return pd.DataFrame(summary_records)

def plot_diff_heatmaps(df_summary):
    sns.set_context("talk")
    plt.rcParams.update({'font.size': 12})
    df_summary = df_summary[~np.isclose(df_summary['mu'], 0.01)].copy()
    
    # Define Strategy Sets based on Experiment name
    # "4 Strategies" if "4strats" in x
    df_summary['Strategy_Set'] = df_summary['Experiment'].apply(
        lambda x: "4strats" if "4strats" in x else "3strats"
    )
    
    unique_Ns = sorted(df_summary['N'].unique())

    # Create output directory
    fig_dir = project_root / "analysis" / "figures" / "heatmaps_strat_diff"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to {fig_dir} ...")

    count = 0
    for n in unique_Ns:
        df_n = df_summary[df_summary['N'] == n]
        if df_n.empty: continue
        
        # Split into 3 vs 4 strategies
        df_3 = df_n[df_n['Strategy_Set'] == '3strats']
        df_4 = df_n[df_n['Strategy_Set'] == '4strats']
        
        if df_3.empty or df_4.empty:
            print(f"Skipping N={n}: Missing data for comparison (Has 3strats: {not df_3.empty}, Has 4strats: {not df_4.empty})")
            continue

        try:
            # Pivot table: K (rows) x mu (columns)
            # Duplicate entries for (K, mu) are not expected per strategy set per N, 
            # but if they exist, pivot will raise error. We can use pivot_table with aggregation.
            pivot_3 = df_3.pivot_table(index="K", columns="mu", values="avg_delta", aggfunc='mean')
            pivot_4 = df_4.pivot_table(index="K", columns="mu", values="avg_delta", aggfunc='mean')
            
            # Align indices and columns to ensure subtraction works correctly
            all_K = sorted(list(set(pivot_3.index) | set(pivot_4.index)), reverse=True)
            all_mu = sorted(list(set(pivot_3.columns) | set(pivot_4.columns)))
            
            pivot_3 = pivot_3.reindex(index=all_K, columns=all_mu)
            pivot_4 = pivot_4.reindex(index=all_K, columns=all_mu)
            
            # Calculate Diff: 4strats - 3strats
            pivot_diff = pivot_4 - pivot_3
            
            # Plot setup
            plt.figure(figsize=(12, 10))
            
            # Create title
            plt.title(
                f"Delta Difference (4strats - 3strats) | N={n}\n"
                f"Values > 0: 4strats is more collusive. Values < 0: 3strats is more collusive.", 
                fontsize=16, pad=20
            )
            
            # Heatmap with Diverging Colormap centered at 0
            # vmin/vmax can be set to be symmetric if desired, generally let seaborn infer or fix range -0.5 to 0.5
            max_abs = max(abs(pivot_diff.min().min()), abs(pivot_diff.max().max()))
            limit = max(0.1, max_abs) # ensure at least some range
            
            sns.heatmap(
                pivot_diff, 
                annot=True, 
                fmt=".2f", 
                cmap="coolwarm", 
                center=0, 
                vmin=-limit,
                vmax=limit,
                square=True, 
                linewidths=.5, 
                cbar_kws={'label': 'Delta (4strats) - Delta (3strats)'}
            )
            
            plt.tight_layout()
            
            save_path = fig_dir / f"heatmap_delta_diff_N{n}.png"
            plt.savefig(save_path, dpi=150)
            plt.close()
            count += 1
            print(f"Generated diff heatmap for N={n}")
            
        except Exception as e:
            print(f"Error plotting N={n}: {e}")
            import traceback
            traceback.print_exc()

    print(f"Done! Generated {count} difference heatmaps.")

if __name__ == "__main__":
    start_time = time.time()
    print(f"Loading heatmap data for Diff Analysis...")
    df = load_all_heatmap_data()
    if not df.empty:
        plot_diff_heatmaps(df)
        print("All difference heatmaps generated.")
    else:
        print("No data found.")
    
    elapsed = time.time() - start_time
    print(f"Total elapsed time: {elapsed/60:.2f} minutes.")
