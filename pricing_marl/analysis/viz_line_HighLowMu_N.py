import sys
from pathlib import Path
import json
import time
from datetime import datetime
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project Root
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

EVAL_RUN_FILE_RE = re.compile(r"^run_\d+\.parquet$")


def _list_eval_parquet_files(n_dir: Path):
    return sorted(
        p for p in n_dir.glob("run_*.parquet")
        if EVAL_RUN_FILE_RE.match(p.name)
    )


TARGET_MUS = [0.04, 0.13, 0.25]
TARGET_K = 30


def compute_metrics_for_run(df):
    if df.empty:
        return None

    avg_delta = df["delta"].mean()
    price_seq = df["price_min"].values
    avg_price = float(np.mean(price_seq))

    return {
        "avg_delta": avg_delta,
        "avg_price": avg_price,
    }


def load_line_data(target_k=TARGET_K, target_mus=None):
    results_dir = project_root / "data" / "results"
    if not results_dir.exists():
        print(f"ERROR: Directory {results_dir} does not exist!")
        return pd.DataFrame()

    if target_mus is None:
        target_mus = TARGET_MUS

    summary_records = []
    exp_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    print(f"Found {len(exp_dirs)} experiment folders. Scanning...")

    for exp_dir in exp_dirs:
        n_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("N_")])
        if not n_dirs:
            continue

        for n_dir in n_dirs:
            config_files = list(n_dir.glob("Config_*.json"))
            if not config_files:
                candidate = exp_dir / f"Config_{n_dir.name}.json"
                if candidate.exists():
                    config_files = [candidate]
            if not config_files:
                continue

            parquet_files = _list_eval_parquet_files(n_dir)
            if not parquet_files:
                continue

            try:
                with open(config_files[0], "r") as f:
                    cfg = json.load(f)

                n_sellers = cfg.get("num_sellers")
                mu = cfg.get("mu")
                k_val = cfg.get("K")

                if k_val != target_k:
                    continue
                if mu not in target_mus:
                    continue

                run_metrics = []
                for pf in parquet_files:
                    try:
                        df = pd.read_parquet(pf)
                        metrics = compute_metrics_for_run(df)
                        if metrics is not None:
                            run_metrics.append(metrics)
                    except Exception:
                        continue

                if run_metrics:
                    df_runs = pd.DataFrame(run_metrics)
                    avg_metrics = df_runs.mean()

                    record = {
                        "Experiment": exp_dir.name,
                        "N": n_sellers,
                        "mu": mu,
                        "K": k_val,
                        "avg_delta": avg_metrics["avg_delta"],
                        "avg_price": avg_metrics["avg_price"],
                    }
                    summary_records.append(record)
            except Exception as e:
                print(f"  [ERROR] processing {n_dir.name}: {e}")

    return pd.DataFrame(summary_records)


def plot_lines(df_summary, target_k=TARGET_K, target_mus=None):
    if target_mus is None:
        target_mus = TARGET_MUS

    if df_summary.empty:
        print("No data found.")
        return

    plt.style.use("ggplot")
    plt.rcParams.update({"font.size": 12})

    df_summary["Strategy_Set"] = df_summary["Experiment"].apply(
        lambda x: "4 Rules Set" if "4strats" in x else "3 Rules Set"
    )

    fig_dir = project_root / "analysis" / "figures" / "lines_high_low_mu_by_n"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to {fig_dir} ...")

    metrics = [
        ("avg_delta", r"$\Delta$"),
        ("avg_price", "Price"),
    ]

    line_styles = {
        "3 Rules Set": "-",
        "4 Rules Set": "--",
    }

    y_limits = {}
    for metric_key, _ in metrics:
        series = df_summary[metric_key].dropna()
        if series.empty:
            y_limits[metric_key] = None
            continue
        y_min = float(series.min())
        y_max = float(series.max())
        if y_min == y_max:
            pad = 0.1 if y_min == 0 else abs(y_min) * 0.1
        else:
            pad = (y_max - y_min) * 0.05
        y_min -= pad
        y_max += pad
        y_limits[metric_key] = (y_min, y_max)

    for mu in target_mus:
        df_mu_all = df_summary[df_summary["mu"] == mu]
        if df_mu_all.empty:
            continue

        for metric_key, metric_label in metrics:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.set_title(f"K={target_k} | $\mu$={mu} | {metric_label}")
            ax.set_xlabel("N")
            ax.set_ylabel(metric_label)

            for strat_label in sorted(df_summary["Strategy_Set"].unique()):
                df_strat = df_mu_all[df_mu_all["Strategy_Set"] == strat_label].sort_values("N")
                if df_strat.empty:
                    continue

                ax.plot(
                    df_strat["N"],
                    df_strat[metric_key],
                    marker="o",
                    linewidth=2,
                    linestyle=line_styles.get(strat_label, "-"),
                    color="black",
                    label=strat_label,
                )

            ax.legend(title="Rules Set")
            ax.grid(True, alpha=0.4)

            if y_limits.get(metric_key) is not None:
                ax.set_ylim(y_limits[metric_key])

            metric_slug = "delta" if metric_key == "avg_delta" else "price"
            mu_slug = str(mu).replace(".", "p")
            save_path = fig_dir / f"line_{metric_slug}_mu{mu_slug}_K{target_k}.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()

    print("Done! Generated line plots for Delta and Price.")


if __name__ == "__main__":
    start_time = time.time()
    print("Loading line data...")
    print(f"Start time: {datetime.now().isoformat(timespec='seconds')}")

    df = load_line_data(target_k=TARGET_K, target_mus=TARGET_MUS)
    plot_lines(df, target_k=TARGET_K, target_mus=TARGET_MUS)

    elapsed = time.time() - start_time
    print(f"Total elapsed time: {elapsed/60:.2f} minutes.")
