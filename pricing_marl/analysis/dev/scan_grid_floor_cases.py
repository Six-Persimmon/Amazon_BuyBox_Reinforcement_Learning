import argparse
import ast
import csv
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.environment import compute_nash_and_monopoly_static, get_demand_and_profit_static


def load_scan_axes(exp_file: Path):
    wanted = {"N_VALUES", "MU_VALUES", "K_VALUES", "ROUNDS_PER_CONFIG"}
    found = {}
    tree = ast.parse(exp_file.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                try:
                    found[target.id] = ast.literal_eval(node.value)
                except Exception:
                    pass
    missing = wanted - set(found.keys())
    if missing:
        raise ValueError(f"Could not parse {sorted(missing)} from {exp_file}")
    return (
        [int(x) for x in found["N_VALUES"]],
        [float(x) for x in found["MU_VALUES"]],
        [int(x) for x in found["K_VALUES"]],
        int(found["ROUNDS_PER_CONFIG"]),
    )


def mu_tag(mu: float) -> str:
    scaled = mu * 100
    if abs(scaled - round(scaled)) <= 1e-12:
        return f"m{int(round(scaled)):02d}"
    txt = f"{mu:.6f}".rstrip("0").rstrip(".")
    return f"m{txt.replace('.', 'p')}"


def build_old_grid(p_nash: float, p_mono: float, num_grids: int) -> np.ndarray:
    step = (p_mono - p_nash) / (num_grids - 3)
    return np.linspace(p_nash - step, p_mono + step, num_grids)


def _fmt_table(headers, rows):
    str_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    out = []
    out.append("  ".join(headers[i].rjust(widths[i]) for i in range(len(headers))))
    for row in str_rows:
        out.append("  ".join(row[i].rjust(widths[i]) for i in range(len(row))))
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(
        description="Scan N/mu grid-floor anomalies where old lowest price grid falls below marginal cost."
    )
    parser.add_argument(
        "--exp-file",
        type=Path,
        default=project_root / "experiments" / "exp02_heatmap_scan.py",
        help="Experiment script to parse N/MU/K/ROUNDS from.",
    )
    parser.add_argument("--num-grids", type=int, default=10)
    parser.add_argument("--a", type=float, default=2.0)
    parser.add_argument("--a0", type=float, default=0.0)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument(
        "--strategy-sets",
        type=str,
        default="3strats,4strats",
        help="Comma-separated strategy set labels for rerun manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "analysis" / "tables",
    )
    args = parser.parse_args()

    n_values, mu_values, k_values, rounds_per_config = load_scan_axes(args.exp_file)
    strategy_sets = [s.strip() for s in args.strategy_sets.split(",") if s.strip()]
    if not strategy_sets:
        raise ValueError("No strategy sets specified.")

    records = []
    for n in n_values:
        for mu in mu_values:
            p_nash, p_mono = compute_nash_and_monopoly_static(
                num_sellers=n,
                a_val=args.a,
                mu=mu,
                a0=args.a0,
                c_val=args.c,
            )
            old_grid = build_old_grid(float(p_nash), float(p_mono), args.num_grids)
            old_min = float(old_grid[0])
            new_min = max(old_min, args.c)
            floor_applied = old_min < args.c - 1e-12

            old_min_profit = float(
                get_demand_and_profit_static(
                    [old_min] * n, args.a, mu, args.a0, args.c
                )[0]
            )
            new_min_profit = float(
                get_demand_and_profit_static(
                    [new_min] * n, args.a, mu, args.a0, args.c
                )[0]
            )

            records.append(
                {
                    "N": n,
                    "mu": mu,
                    "mu_tag": mu_tag(mu),
                    "p_nash": float(p_nash),
                    "p_monopoly": float(p_mono),
                    "old_grid_min": old_min,
                    "new_grid_min": new_min,
                    "floor_applied": floor_applied,
                    "old_min_profit": old_min_profit,
                    "new_min_profit": new_min_profit,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / "grid_floor_scan_by_n_mu.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "N",
                "mu",
                "mu_tag",
                "p_nash",
                "p_monopoly",
                "old_grid_min",
                "new_grid_min",
                "floor_applied",
                "old_min_profit",
                "new_min_profit",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    # Build rerun manifest (affected N/mu across all K and strategy sets)
    affected = [r for r in records if r["floor_applied"]]
    rerun_manifest = []
    for rec in affected:
        for k in k_values:
            for label in strategy_sets:
                rerun_manifest.append(
                    {
                        "strategy_set": label,
                        "N": rec["N"],
                        "mu": rec["mu"],
                        "K": k,
                        "experiment_name": f"scan_{label}_mu{rec['mu']}_k{k}",
                        "runs_to_rerun": rounds_per_config,
                    }
                )

    manifest_csv = args.output_dir / "grid_floor_rerun_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "strategy_set",
                "N",
                "mu",
                "K",
                "experiment_name",
                "runs_to_rerun",
            ],
        )
        writer.writeheader()
        writer.writerows(rerun_manifest)

    mu_headers = [mu_tag(mu) for mu in mu_values]
    mu_index = {mu: idx for idx, mu in enumerate(mu_values)}

    print("=== Grid Floor Scan Summary ===")
    print(f"Experiment file: {args.exp_file}")
    print(f"N values: {n_values}")
    print(f"mu values: {mu_values}")
    print(f"K values: {k_values}")
    print(f"rounds per config: {rounds_per_config}")
    print(f"marginal cost c: {args.c}")
    print(f"affected (N,mu) cells: {len(affected)} / {len(records)}")
    print(f"summary csv: {summary_csv}")
    print(f"rerun manifest csv: {manifest_csv}")

    print("\nAffected (N, mu) combinations:")
    if not affected:
        print("  none")
    else:
        for rec in affected:
            print(
                "  "
                f"N={rec['N']}, mu={rec['mu']:.3f}, "
                f"old_min={rec['old_grid_min']:.6f}, "
                f"new_min={rec['new_grid_min']:.6f}, "
                f"old_min_profit={rec['old_min_profit']:.6f}"
            )

    # N x mu boolean table.
    bool_headers = ["N"] + mu_headers + ["RowSum"]
    bool_rows = []
    col_sums = [0] * len(mu_values)
    for n in n_values:
        row_vals = [0] * len(mu_values)
        for rec in records:
            if rec["N"] == n and rec["floor_applied"]:
                idx = mu_index[rec["mu"]]
                row_vals[idx] = 1
        for i, v in enumerate(row_vals):
            col_sums[i] += v
        bool_rows.append([n] + row_vals + [sum(row_vals)])
    bool_rows.append(["ColSum"] + col_sums + [sum(col_sums)])

    print("\nTable A: floor_applied by N x mu (1 means old grid min < c)")
    print("```text")
    print(_fmt_table(bool_headers, bool_rows))
    print("```")

    # K x mu table per N, in the style of investigation table.
    print("\nTable B: per-N K x mu rerun counts (per strategy set)")
    for n in n_values:
        table_headers = ["K"] + mu_headers + ["RowSum"]
        rows = []
        col_sums = [0] * len(mu_values)
        for k in k_values:
            row_vals = [0] * len(mu_values)
            for rec in records:
                if rec["N"] == n and rec["floor_applied"]:
                    idx = mu_index[rec["mu"]]
                    row_vals[idx] = rounds_per_config
            for i, v in enumerate(row_vals):
                col_sums[i] += v
            rows.append([f"K{k}"] + row_vals + [sum(row_vals)])
        rows.append(["ColSum"] + col_sums + [sum(col_sums)])

        non_zero_cells = sum(1 for row in rows[:-1] for v in row[1:-1] if int(v) > 0)
        print(
            f"\nN={n}: non-zero cells={non_zero_cells}/{len(k_values) * len(mu_values)}, "
            f"total runs per strategy set={rows[-1][-1]}"
        )
        print("```text")
        print(_fmt_table(table_headers, rows))
        print("```")


if __name__ == "__main__":
    main()
