"""
Generate all figures for a given experiment (headless, saved to disk).

Edit EXP_NAME below, then run:
    PYTHONPATH=$(pwd)/pricing_marl python pricing_marl/analysis/export_figures.py
Images will be saved under pricing_marl/analysis/figures/<EXP_NAME>/.
"""
import matplotlib

# Use non-interactive backend to avoid GUI requirements
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Ensure project root on path for relative imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from analysis.plot_utils import (
    load_experiment_data,
    load_experiment_configs,
    plot_time_series,
    plot_action_distribution,
    plot_metric_for_specific_n,
    plot_action_share_for_specific_n,
)
from src.environment import compute_nash_and_monopoly_static

# Suppress interactive windows inside imported plotting helpers
plt.show = lambda *args, **kwargs: None

# === Configure target experiment ===
EXP_NAME = "exp_1_classic_mu025"  # <--- change this to target different experiments


def save_current(fig_path):
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()

def main():
    output_dir = project_root / "analysis" / "figures" / EXP_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data (downsample to keep plotting fast)
    df = load_experiment_data(EXP_NAME, downsample_rate=1000)
    if df is None or df.empty:
        return

    configs = load_experiment_configs(EXP_NAME)

    # Benchmarks by N
    benchmarks = {}
    for n, cfg in configs.items():
        p_nash, p_mon = compute_nash_and_monopoly_static(
            num_sellers=n,
            a_val=cfg["a_val"],
            mu=cfg["mu"],
            a0=cfg["a0"],
            c_val=cfg["c_val"],
        )
        benchmarks[n] = (p_nash, p_mon)

    all_ns = sorted(df["n_sellers"].unique())

    # # Average price per N (original) - kept for reference
    # for n in all_ns:
    #     p_nash, p_mon = benchmarks.get(n, (None, None))
    #     plot_metric_for_specific_n(
    #         df=df,
    #         n_sellers=n,
    #         y_col="average_price",
    #         title=f"Average Price Evolution (N={n})",
    #         ylabel="Average Price",
    #         smoothing=1,
    #         ci=95,
    #         nash_price=p_nash,
    #         monopoly_price=p_mon,
    #     )
    #     save_current(output_dir / f"avg_price_N{n}.png")

    # Average K-step price per N
    for n in all_ns:
        p_nash, p_mon = benchmarks.get(n, (None, None))
        plot_metric_for_specific_n(
            df=df,
            n_sellers=n,
            y_col="average_K_price",
            title=f"Average K-Step Price (N={n})",
            ylabel="Average K-Step Price",
            smoothing=1,
            ci=95,
            nash_price=p_nash,
            monopoly_price=p_mon,
        )
        save_current(output_dir / f"avgK_price_N{n}.png")

    # # Lowest price per N (original) - kept for reference
    # if "lowest_price" in df.columns:
    #     for n in all_ns:
    #         p_nash, p_mon = benchmarks.get(n, (None, None))
    #         plot_metric_for_specific_n(
    #             df=df,
    #             n_sellers=n,
    #             y_col="lowest_price",
    #             title=f"Lowest Market Price Evolution (N={n})",
    #             ylabel="Lowest Price",
    #             smoothing=1,
    #             ci=95,
    #             nash_price=p_nash,
    #             monopoly_price=p_mon,
    #         )
    #         save_current(output_dir / f"lowest_price_N{n}.png")

    # Average K-step lowest price per N
    if "average_lowest_K_price" in df.columns:
        for n in all_ns:
            p_nash, p_mon = benchmarks.get(n, (None, None))
            plot_metric_for_specific_n(
                df=df,
                n_sellers=n,
                y_col="average_lowest_K_price",
                title=f"Average K-Step Lowest Price (N={n})",
                ylabel="Average K-Step Lowest Price",
                smoothing=1,
                ci=95,
                nash_price=p_nash,
                monopoly_price=p_mon,
            )
            save_current(output_dir / f"avgK_lowest_price_N{n}.png")

    # Delta across Ns
    plot_time_series(
        df,
        y_col="delta",
        title=f"Collusion Index ($\\Delta$) Evolution ({EXP_NAME})",
        ylabel="$\\Delta$ (0=Nash, 1=Monopoly)",
        hue="n_sellers",
        smoothing=1,
        ci=95,
    )
    save_current(output_dir / "delta_timeseries.png")

    # Action share per N
    for n in all_ns:
        plot_action_share_for_specific_n(
            df=df,
            n_sellers=n,
            smoothing=1,
            ci=95,
        )
        save_current(output_dir / f"action_share_N{n}.png")

    # Action distribution (last 10%)
    plot_action_distribution(df, last_percent=0.1)
    save_current(output_dir / "action_distribution_last10pct.png")


if __name__ == "__main__":
    main()
