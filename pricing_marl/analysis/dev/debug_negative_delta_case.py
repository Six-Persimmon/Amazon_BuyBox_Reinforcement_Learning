import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.config import Config
from src.environment import compute_nash_and_monopoly_static, get_demand_and_profit_static


def build_config_from_json(cfg_json: dict) -> Config:
    """Only pass fields that Config.__init__ accepts and that affect simulation logic."""
    return Config(
        num_sellers=cfg_json["num_sellers"],
        num_grids=cfg_json["num_grids"],
        active_strategies=cfg_json["active_strategies"],
        a_val=cfg_json["a_val"],
        c_val=cfg_json["c_val"],
        a0=cfg_json["a0"],
        mu=cfg_json["mu"],
        xi=cfg_json["xi"],
        alpha=cfg_json["alpha"],
        gamma=cfg_json["gamma"],
        max_episodes=cfg_json["max_episodes"],
        beta=cfg_json["beta"],
        converge_period=cfg_json["converge_period"],
        K=cfg_json["K"],
        eval_H=cfg_json["eval_H"],
        save_training_data=cfg_json["save_training_data"],
    )


def symmetric_market_metrics(p: float, n: int, a: float, mu: float, a0: float, c: float):
    """
    Symmetric-profile closed form under logit demand:
      D_i(p) = exp((a-p)/mu) / (n*exp((a-p)/mu) + exp(a0/mu))
      pi_i(p)= (p-c) * D_i(p)
    """
    inside_term = np.exp((a - p) / mu)
    outside_term = np.exp(a0 / mu)
    denom = n * inside_term + outside_term
    demand_i = inside_term / denom
    demand_outside = outside_term / denom
    profit_i = (p - c) * demand_i
    return demand_i, demand_outside, profit_i


def compute_benchmarks(cfg: Config):
    p_nash, p_mono = compute_nash_and_monopoly_static(
        cfg.num_sellers, cfg.a_val, cfg.mu, cfg.a0, cfg.c_val
    )
    step = (p_mono - p_nash) / (cfg.num_grids - 3)
    price_grid = np.linspace(p_nash - step, p_mono + step, cfg.num_grids)
    pi_nash = float(
        get_demand_and_profit_static(
            [p_nash] * cfg.num_sellers, cfg.a_val, cfg.mu, cfg.a0, cfg.c_val
        )[0]
    )
    pi_mono = float(
        get_demand_and_profit_static(
            [p_mono] * cfg.num_sellers, cfg.a_val, cfg.mu, cfg.a0, cfg.c_val
        )[0]
    )
    return {
        "p_nash": float(p_nash),
        "p_mono": float(p_mono),
        "price_grid": price_grid,
        "pi_nash": pi_nash,
        "pi_mono": pi_mono,
    }


def print_benchmark_explanation(bench: dict, cfg: Config):
    p_nash = bench["p_nash"]
    p_mono = bench["p_mono"]

    d_nash, d0_nash, pi_nash = symmetric_market_metrics(
        p_nash, cfg.num_sellers, cfg.a_val, cfg.mu, cfg.a0, cfg.c_val
    )
    d_mono, d0_mono, pi_mono = symmetric_market_metrics(
        p_mono, cfg.num_sellers, cfg.a_val, cfg.mu, cfg.a0, cfg.c_val
    )

    # Use the same static demand/profit function as environment logic.
    pi_nash_env = bench["pi_nash"]
    pi_mono_env = bench["pi_mono"]

    print("\n=== 1) Benchmark Prices and Profits ===")
    print("Symmetric demand/profit formulas:")
    print("  D_i(p) = exp((a-p)/mu) / (N*exp((a-p)/mu) + exp(a0/mu))")
    print("  pi_i(p)= (p-c) * D_i(p)")
    print("Nash price p_nash and monopoly price p_mono are solved numerically in src/environment.py")
    print("  - p_nash: fixed point of best response under symmetric rivals")
    print("  - p_mono: maximizer of joint profit N * pi_i(p) under symmetric pricing")

    print(f"\nParameters: N={cfg.num_sellers}, a={cfg.a_val}, c={cfg.c_val}, a0={cfg.a0}, mu={cfg.mu}")
    print(f"p_nash    = {p_nash:.12f}")
    print(f"p_mono    = {p_mono:.12f}")

    print("\nAt p_nash:")
    print(f"  demand_i       = {d_nash:.12e}")
    print(f"  demand_outside = {d0_nash:.12e}")
    print(f"  pi_nash (closed-form) = {pi_nash:.12f}")
    print(f"  pi_nash (env func)    = {pi_nash_env:.12f}")

    print("\nAt p_mono:")
    print(f"  demand_i       = {d_mono:.12e}")
    print(f"  demand_outside = {d0_mono:.12e}")
    print(f"  pi_mono (closed-form) = {pi_mono:.12f}")
    print(f"  pi_mono (env func)    = {pi_mono_env:.12f}")

    denom = pi_mono_env - pi_nash_env
    print(f"\nNormalization denominator (pi_mono - pi_nash) = {denom:.12f}")

    print("\nPrice grid (idx -> price):")
    for i, p in enumerate(bench["price_grid"]):
        print(f"  {i:2d} -> {float(p):.12f}")

    print("\nSymmetric grid check (all sellers at same grid price):")
    print("  idx   price           pi_i            delta")
    for i, p in enumerate(bench["price_grid"]):
        pi_i = float(
            get_demand_and_profit_static(
                [float(p)] * cfg.num_sellers,
                cfg.a_val,
                cfg.mu,
                cfg.a0,
                cfg.c_val,
            )[0]
        )
        delta = (pi_i - pi_nash_env) / denom
        print(f"  {i:2d}  {float(p):.12f}  {pi_i:.12f}  {delta:.9f}")


def analyze_run(bench: dict, cfg: Config, run_path: Path):
    df = pd.read_parquet(run_path)

    pi_cols = [f"pi_{i}" for i in range(cfg.num_sellers)]
    p_cols = [f"p_{i}" for i in range(cfg.num_sellers)]

    pi_nash = bench["pi_nash"]
    pi_mono = bench["pi_mono"]
    denom = pi_mono - pi_nash

    mean_pi = df[pi_cols].mean(axis=1)
    recomputed_delta = (mean_pi - pi_nash) / denom
    max_delta_diff = float(np.max(np.abs(df["delta"].to_numpy() - recomputed_delta.to_numpy())))

    neg = df[df["delta"] < 0]

    print("\n=== 2) Run-Level Diagnostics ===")
    print(f"Run file: {run_path}")
    print(f"Rows: {len(df)}")
    print(f"Converged flags: {df['converged'].value_counts().to_dict()}")
    print(f"Unique training episode tag(s): {sorted(df['episode'].unique().tolist())[:5]} ...")

    print("\nDelta consistency check:")
    print(f"  max |delta_file - delta_recomputed| = {max_delta_diff:.3e}")

    print("\nOverall stats:")
    print(f"  price_mean min/max/mean = {df['price_mean'].min():.6f} / {df['price_mean'].max():.6f} / {df['price_mean'].mean():.6f}")
    print(f"  delta      min/max/mean = {df['delta'].min():.6f} / {df['delta'].max():.6f} / {df['delta'].mean():.6f}")
    print(f"  mean_pi    min/max/mean = {mean_pi.min():.12f} / {mean_pi.max():.12f} / {mean_pi.mean():.12f}")

    print("\nNegative-delta region:")
    print(f"  count = {len(neg)} / {len(df)} ({len(neg) / len(df):.2%})")
    if len(neg) > 0:
        print(f"  price_mean min/max/mean = {neg['price_mean'].min():.6f} / {neg['price_mean'].max():.6f} / {neg['price_mean'].mean():.6f}")

    print("\nPer-seller averages (all rows):")
    for i in range(cfg.num_sellers):
        print(f"  seller {i:2d}: avg p = {df[f'p_{i}'].mean():.6f}, avg pi = {df[f'pi_{i}'].mean():.12f}")

    if len(neg) > 0:
        print("\nPer-seller averages (delta < 0 rows):")
        for i in range(cfg.num_sellers):
            print(f"  seller {i:2d}: avg p = {neg[f'p_{i}'].mean():.6f}, avg pi = {neg[f'pi_{i}'].mean():.12f}")

    print("\nTop 8 price vectors by frequency:")
    vc = df[p_cols].round(12).value_counts().head(8)
    for idx, cnt in vc.items():
        as_tuple = tuple(float(x) for x in idx)
        print(f"  {cnt:5d} -> {as_tuple}")

    print("\nFirst 20 rows (t_global, step_in_k, price_mean, delta):")
    print(df[["t_global", "step_in_k", "price_mean", "delta"]].head(20).to_string(index=False))

    return df


def maybe_plot(df: pd.DataFrame, bench: dict, out_file: Path):
    import matplotlib.pyplot as plt

    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(df["t_global"], df["price_mean"], linewidth=1.0, color="#1f77b4")
    axes[0].axhline(float(bench["p_nash"]), linestyle="--", linewidth=1.0, color="#ff7f0e", label="p_nash")
    axes[0].axhline(float(bench["p_mono"]), linestyle="--", linewidth=1.0, color="#2ca02c", label="p_mono")
    axes[0].axhline(float(bench["price_grid"][-1]), linestyle=":", linewidth=1.0, color="#9467bd", label="grid max")
    axes[0].set_ylabel("price_mean")
    axes[0].legend(loc="best")

    axes[1].plot(df["t_global"], df["delta"], linewidth=1.0, color="#d62728")
    axes[1].axhline(0.0, linestyle="--", linewidth=1.0, color="black")
    axes[1].axhline(1.0, linestyle="--", linewidth=1.0, color="gray")
    axes[1].set_xlabel("t_global")
    axes[1].set_ylabel("delta")

    plt.tight_layout()
    fig.savefig(out_file, dpi=160)
    plt.close(fig)
    print(f"\nFigure saved: {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Debug a specific negative-delta run with detailed benchmark decomposition."
    )
    parser.add_argument("--experiment", type=str, default="scan_3strats_mu0.01_k90")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--results-root", type=str, default=None)
    parser.add_argument("--no-fig", action="store_true")
    args = parser.parse_args()

    results_root = Path(args.results_root) if args.results_root else project_root / "data" / "results"
    cfg_path = results_root / args.experiment / f"Config_N_{args.n}.json"
    run_path = results_root / args.experiment / f"N_{args.n}" / f"run_{args.run}.parquet"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    if not run_path.exists():
        raise FileNotFoundError(f"Run file not found: {run_path}")

    with open(cfg_path, "r") as f:
        cfg_json = json.load(f)

    cfg = build_config_from_json(cfg_json)
    bench = compute_benchmarks(cfg)

    print_benchmark_explanation(bench, cfg)
    df = analyze_run(bench, cfg, run_path)

    if not args.no_fig:
        fig_path = (
            project_root
            / "analysis"
            / "figures"
            / f"debug_negative_delta_{args.experiment}_N{args.n}_run{args.run}.png"
        )
        maybe_plot(df, bench, fig_path)


if __name__ == "__main__":
    main()
