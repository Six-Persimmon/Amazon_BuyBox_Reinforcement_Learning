from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def find_pricing_root(start: Path) -> Path:
    for candidate in [start] + list(start.parents):
        candidate = candidate.resolve()
        if (candidate / "src").exists() and (candidate / "data").exists():
            return candidate
        pricing_root = candidate / "pricing_marl"
        if (pricing_root / "src").exists() and (pricing_root / "data").exists():
            return pricing_root.resolve()
    raise FileNotFoundError("Could not find pricing_marl root containing src/ and data/")


PRICING_ROOT = find_pricing_root(Path.cwd())

if str(PRICING_ROOT) not in sys.path:
    sys.path.insert(0, str(PRICING_ROOT))

from src.config import Config
from src.environment import compute_nash_and_monopoly_static, get_demand_and_profit_static
from src.strategies import get_strategy_function


EXPERIMENT_DIR_PATTERN = re.compile(
    r"^scan_(?P<strategy_label>.+?)_mu(?P<mu>[-0-9.]+)_k(?P<K>[-0-9.]+)$"
)
RUN_PATH_PATTERN = re.compile(r"^run_(?P<run_id>\d+)\.parquet$")


@dataclass(frozen=True)
class RunTask:
    strategy_label: str
    mu: float
    K: int
    N: int
    run_id: int
    run_path: str
    qtable_path: str
    config_path: str


def get_results_root(results_dir_name: str) -> Path:
    results_root = PRICING_ROOT / "data" / results_dir_name
    if not results_root.exists():
        raise FileNotFoundError(results_root)
    return results_root


def load_config(config_path: Path) -> Config:
    config_raw = json.loads(config_path.read_text())
    config_fields = set(Config.__dataclass_fields__.keys())
    path_fields = {"base_dir", "data_dir", "lookup_dir", "results_dir"}
    config_kwargs = {
        key: value
        for key, value in config_raw.items()
        if key in config_fields and key not in path_fields
    }
    return Config(**config_kwargs)


def build_price_grid(config: Config) -> np.ndarray:
    p_nash, p_monopoly = compute_nash_and_monopoly_static(
        config.num_sellers,
        config.a_val,
        config.mu,
        config.a0,
        config.c_val,
    )
    step = (p_monopoly - p_nash) / (config.num_grids - 3)
    price_grid = np.linspace(p_nash - step, p_monopoly + step, config.num_grids)
    if price_grid[0] < config.c_val:
        price_grid[0] = config.c_val
    return price_grid


def extract_reference_window(run_df: pd.DataFrame, K: int, window_blocks: int) -> pd.DataFrame:
    window_steps = window_blocks * K
    if len(run_df) < window_steps:
        raise ValueError(
            f"Run only has {len(run_df)} rows, smaller than requested {window_steps}-step reference window."
        )

    ref_df = run_df.iloc[-window_steps:].reset_index(drop=True).copy()
    expected = np.tile(np.arange(K), window_blocks)
    actual = ref_df["step_in_k"].to_numpy()
    if not np.array_equal(actual, expected):
        raise ValueError("The last reference window is not aligned on full K-block boundaries.")
    return ref_df


def nearest_state_idx(price_grid: np.ndarray, price_value: float) -> int:
    return int(np.argmin(np.abs(price_grid - float(price_value))))


def build_q_lookup(qtable_df: pd.DataFrame) -> dict[tuple[int, int], pd.DataFrame]:
    q_final = qtable_df[qtable_df["qtable_type"] == "final"].copy()
    lookup: dict[tuple[int, int], pd.DataFrame] = {}
    for (seller_id, state_idx), group in q_final.groupby(["seller_id", "state_idx"], sort=True):
        lookup[(int(seller_id), int(state_idx))] = group.sort_values("action_idx").reset_index(drop=True)
    return lookup


def best_action_idx(q_rows: pd.DataFrame) -> int:
    max_q = q_rows["q_value"].max()
    best_rows = q_rows[np.isclose(q_rows["q_value"], max_q)].sort_values("action_idx")
    return int(best_rows.iloc[0]["action_idx"])


def action_profile_for_state(
    config: Config,
    q_lookup: dict[tuple[int, int], pd.DataFrame],
    state_idx: int,
    deviator_id: int | None = None,
    override_action_idx: int | None = None,
) -> list[int]:
    action_indices = []
    for seller_id in range(config.num_sellers):
        q_rows = q_lookup[(seller_id, state_idx)]
        action_idx = best_action_idx(q_rows)
        if deviator_id is not None and seller_id == deviator_id and override_action_idx is not None:
            action_idx = override_action_idx
        action_indices.append(int(action_idx))
    return action_indices


def simulate_k_block(
    config: Config,
    price_grid: np.ndarray,
    start_state_idx: int,
    action_indices: list[int],
) -> tuple[pd.DataFrame, int]:
    current_indices = [int(start_state_idx)] * config.num_sellers
    strategy_funcs = [get_strategy_function(config.active_strategies[action_idx]) for action_idx in action_indices]
    rows = []

    for step_in_k in range(config.K):
        prices = np.array([price_grid[idx] for idx in current_indices], dtype=float)
        profits = get_demand_and_profit_static(
            prices,
            config.a_val,
            config.mu,
            config.a0,
            config.c_val,
        )

        row = {
            "step_in_k": step_in_k,
            "price_min": float(prices.min()),
            "price_mean": float(prices.mean()),
        }
        for seller_id in range(config.num_sellers):
            row[f"a_{seller_id}"] = int(config.active_strategies[action_indices[seller_id]])
            row[f"p_{seller_id}"] = float(prices[seller_id])
            row[f"pi_{seller_id}"] = float(profits[seller_id])
        rows.append(row)

        next_indices = []
        for seller_id, strategy_func in enumerate(strategy_funcs):
            others = current_indices[:seller_id] + current_indices[seller_id + 1 :]
            next_idx = strategy_func(current_indices[seller_id], min(others), config.num_grids)
            next_indices.append(int(next_idx))
        current_indices = next_indices

    next_state_idx = int(min(current_indices))
    return pd.DataFrame(rows), next_state_idx


def compare_profit(dev_profit: float, baseline_profit: float, tol: float) -> str:
    diff = dev_profit - baseline_profit
    if abs(diff) < tol:
        return "same"
    return "higher" if diff > 0 else "lower"


def evaluate_deviation_run(
    config: Config,
    price_grid: np.ndarray,
    reference_df: pd.DataFrame,
    q_lookup: dict[tuple[int, int], pd.DataFrame],
    seller_id_to_deviate: int,
    profit_tolerance: float,
    post_deviation_blocks: int,
) -> dict[str, object]:
    start_of_second_block = reference_df.loc[config.K, "price_min"]
    pre_deviation_state_idx = nearest_state_idx(price_grid, start_of_second_block)
    pre_deviation_price = float(price_grid[pre_deviation_state_idx])

    seller_q_rows = q_lookup[(seller_id_to_deviate, pre_deviation_state_idx)]
    best_idx = best_action_idx(seller_q_rows)
    deviation_action_indices = [action_idx for action_idx in range(config.num_actions) if action_idx != best_idx]

    if not deviation_action_indices:
        return {
            "is_nash_equilibrium": True,
            "all_scenarios_returned": True,
            "all_scenarios_profit_safe": True,
            "num_scenarios": 0,
            "num_scenarios_passed": 0,
        }

    scenario_results = []
    seller_profit_col = f"pi_{seller_id_to_deviate}"

    for deviation_action_idx in deviation_action_indices:
        current_state_idx = pre_deviation_state_idx
        deviated_blocks = []
        block_diagnostics = []

        for block_offset in range(post_deviation_blocks):
            chosen_actions = action_profile_for_state(
                config,
                q_lookup,
                current_state_idx,
                deviator_id=seller_id_to_deviate if block_offset == 0 else None,
                override_action_idx=deviation_action_idx if block_offset == 0 else None,
            )

            block_df, next_state_idx = simulate_k_block(
                config=config,
                price_grid=price_grid,
                start_state_idx=current_state_idx,
                action_indices=chosen_actions,
            )
            block_df["block_offset"] = block_offset + 1
            deviated_blocks.append(block_df)

            block_diagnostics.append(
                {
                    "block_offset": block_offset + 1,
                    "start_state_idx": int(current_state_idx),
                    "next_state_idx": int(next_state_idx),
                }
            )
            current_state_idx = next_state_idx

        deviation_df = pd.concat(deviated_blocks, ignore_index=True)
        state_returned_anytime = bool(np.isclose(deviation_df["price_min"].to_numpy(), pre_deviation_price).any())
        state_never_left = bool(np.isclose(deviation_df["price_min"].to_numpy(), pre_deviation_price).all())

        if state_never_left:
            profit_horizon_steps = 0
        else:
            return_block_offset = None
            for block in block_diagnostics[1:]:
                if block["start_state_idx"] == pre_deviation_state_idx:
                    return_block_offset = int(block["block_offset"])
                    break
            if return_block_offset is None:
                profit_horizon_steps = len(deviation_df)
            else:
                profit_horizon_steps = (return_block_offset - 1) * config.K

        deviation_profit = (
            float(deviation_df.iloc[:profit_horizon_steps][seller_profit_col].sum())
            if profit_horizon_steps > 0
            else 0.0
        )
        baseline_profit = (
            float(reference_df.iloc[config.K : config.K + profit_horizon_steps][seller_profit_col].sum())
            if profit_horizon_steps > 0
            else 0.0
        )
        profit_result = compare_profit(deviation_profit, baseline_profit, profit_tolerance)
        scenario_passed = state_returned_anytime and profit_result in {"same", "lower"}

        scenario_results.append(
            {
                "deviation_action_idx": int(deviation_action_idx),
                "state_returned_anytime": state_returned_anytime,
                "profit_result": profit_result,
                "scenario_passed": scenario_passed,
            }
        )

    num_scenarios_passed = int(sum(int(row["scenario_passed"]) for row in scenario_results))
    return {
        "is_nash_equilibrium": num_scenarios_passed == len(scenario_results),
        "all_scenarios_returned": all(bool(row["state_returned_anytime"]) for row in scenario_results),
        "all_scenarios_profit_safe": all(row["profit_result"] in {"same", "lower"} for row in scenario_results),
        "num_scenarios": len(scenario_results),
        "num_scenarios_passed": num_scenarios_passed,
    }


def evaluate_run_task(
    task: RunTask,
    seller_id_to_deviate: int,
    reference_window_blocks: int,
    post_deviation_blocks: int,
    profit_tolerance: float,
) -> dict[str, object]:
    config = load_config(Path(task.config_path))
    run_df = pd.read_parquet(Path(task.run_path))
    qtable_df = pd.read_parquet(Path(task.qtable_path))

    reference_df = extract_reference_window(
        run_df=run_df,
        K=config.K,
        window_blocks=reference_window_blocks,
    )
    price_grid = build_price_grid(config)
    q_lookup = build_q_lookup(qtable_df)
    result = evaluate_deviation_run(
        config=config,
        price_grid=price_grid,
        reference_df=reference_df,
        q_lookup=q_lookup,
        seller_id_to_deviate=seller_id_to_deviate,
        profit_tolerance=profit_tolerance,
        post_deviation_blocks=post_deviation_blocks,
    )

    return {
        "strategy_label": task.strategy_label,
        "N": task.N,
        "mu": task.mu,
        "K": task.K,
        "run_id": task.run_id,
        **result,
    }


def discover_run_tasks(results_root: Path) -> list[RunTask]:
    tasks = []
    for experiment_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        match = EXPERIMENT_DIR_PATTERN.match(experiment_dir.name)
        if not match:
            continue

        strategy_label = match.group("strategy_label")
        mu = float(match.group("mu"))
        K = int(float(match.group("K")))

        for n_dir in sorted(p for p in experiment_dir.iterdir() if p.is_dir() and p.name.startswith("N_")):
            N = int(n_dir.name.split("_", maxsplit=1)[1])
            config_path = experiment_dir / f"Config_N_{N}.json"
            if not config_path.exists():
                raise FileNotFoundError(config_path)

            for run_path in sorted(n_dir.glob("run_*.parquet")):
                if run_path.name.endswith("_qtable.parquet"):
                    continue

                run_match = RUN_PATH_PATTERN.match(run_path.name)
                if not run_match:
                    continue

                qtable_path = run_path.with_name(f"run_{run_match.group('run_id')}_qtable.parquet")
                if not qtable_path.exists():
                    raise FileNotFoundError(qtable_path)

                tasks.append(
                    RunTask(
                        strategy_label=strategy_label,
                        mu=mu,
                        K=K,
                        N=N,
                        run_id=int(run_match.group("run_id")),
                        run_path=str(run_path),
                        qtable_path=str(qtable_path),
                        config_path=str(config_path),
                    )
                )
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count how many runs satisfy the one-shot deviation Nash-equilibrium check."
    )
    parser.add_argument(
        "--results-dir-name",
        default="result_K30_qtable",
        help="Folder name under pricing_marl/data/ that contains the experiment results.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV path. Defaults to analysis/tables/nash_equilibrium_summary_<results_dir_name>.csv",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--seller-id",
        type=int,
        default=0,
        help="Seller index used for the one-shot deviation check.",
    )
    parser.add_argument(
        "--reference-window-blocks",
        type=int,
        default=5,
        help="How many trailing K-blocks to use from the evaluation trajectory.",
    )
    parser.add_argument(
        "--post-deviation-blocks",
        type=int,
        default=4,
        help="How many K-blocks to simulate after the injected deviation.",
    )
    parser.add_argument(
        "--profit-tolerance",
        type=float,
        default=1e-3,
        help="Tolerance used in the same/lower profit comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = get_results_root(args.results_dir_name)
    output_path = (
        Path(args.output)
        if args.output is not None
        else PRICING_ROOT
        / "analysis"
        / "tables"
        / f"nash_equilibrium_summary_{args.results_dir_name}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = discover_run_tasks(results_root)
    if not tasks:
        raise RuntimeError(f"No run tasks found under {results_root}")

    print(f"Results root: {results_root}")
    print(f"Output CSV:   {output_path}")
    print(f"Worker jobs:  {args.jobs}")
    print(f"Total runs:   {len(tasks)}")

    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = [
            executor.submit(
                evaluate_run_task,
                task,
                args.seller_id,
                args.reference_window_blocks,
                args.post_deviation_blocks,
                args.profit_tolerance,
            )
            for task in tasks
        ]

        for future in tqdm(as_completed(futures), total=len(futures), unit="run", desc="Checking runs"):
            results.append(future.result())

    detail_df = pd.DataFrame(results).sort_values(
        ["strategy_label", "mu", "K", "N", "run_id"],
        ascending=[True, True, True, True, True],
    )
    summary_df = (
        detail_df.groupby(["strategy_label", "N", "mu", "K"], as_index=False)
        .agg(
            num_runs=("run_id", "count"),
            num_NashEqu=("is_nash_equilibrium", "sum"),
        )
        .sort_values(["strategy_label", "mu", "K", "N"], ascending=[True, True, True, True])
        .reset_index(drop=True)
    )

    summary_df.to_csv(output_path, index=False)
    print(f"Wrote {len(summary_df)} summary rows to {output_path}")


if __name__ == "__main__":
    main()
