#!/usr/bin/env python3
"""
Repair incorrect 'init' Q-table rows in run_*_qtable.parquet files.

Design goals:
- Recompute the true initialization Q-table from config + lookup table.
- Replace ONLY rows where qtable_type == "init".
- Keep non-init rows (e.g., "final") unchanged.
- Do not touch evaluation parquet files.

Usage:
  cd pricing_marl
  python fix_initial_qtables.py --results-root data/results
  python fix_initial_qtables.py --results-root data/results --dry-run
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.agent import calculate_heuristic_init_values
from src.config import Config
from src.environment import PricingEnvironment
from src.strategies import ACT_ABOVE, ACT_MATCH, ACT_UNDERCUT, ACT_UNDER_RESET


QTABLE_RE = re.compile(r"^run_(\d+)_qtable\.parquet$")
SCAN_RE = re.compile(r"^scan_(3strats|4strats)_mu(.+)_k(.+)$")


def _discover_qtables(results_root: Path) -> List[Path]:
    return sorted(results_root.glob("scan_*/N_*/run_*_qtable.parquet"))


def _group_by_config(qtable_paths: Iterable[Path]) -> Dict[Tuple[Path, int], List[Path]]:
    groups: Dict[Tuple[Path, int], List[Path]] = defaultdict(list)
    for p in qtable_paths:
        n_dir = p.parent
        scan_dir = n_dir.parent
        try:
            n_val = int(n_dir.name.split("_", 1)[1])
        except Exception:
            continue
        groups[(scan_dir, n_val)].append(p)
    return groups


def _load_config_from_json(config_json: Path) -> Config:
    with open(config_json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    init_fields = {f.name for f in dataclasses.fields(Config) if f.init}
    kwargs = {k: v for k, v in raw.items() if k in init_fields}
    return Config(**kwargs)


def _build_config_from_scan_dir(scan_dir: Path, n_val: int) -> Config:
    m = SCAN_RE.match(scan_dir.name)
    if not m:
        raise ValueError(f"Cannot parse scan directory name: {scan_dir.name}")

    label = m.group(1)
    mu_val = float(m.group(2))
    k_val = int(float(m.group(3)))

    if label == "3strats":
        active_strategies = [ACT_UNDERCUT, ACT_MATCH, ACT_ABOVE]
    elif label == "4strats":
        active_strategies = [ACT_UNDERCUT, ACT_MATCH, ACT_ABOVE, ACT_UNDER_RESET]
    else:
        raise ValueError(f"Unsupported label in scan directory: {label}")

    # exp02 defaults (only fields relevant for init-q reconstruction are required).
    return Config(
        num_sellers=n_val,
        mu=mu_val,
        active_strategies=active_strategies,
        K=k_val,
        max_episodes=2_000_000,
        converge_period=100_000,
        eval_H=2_000,
        beta=1e-5,
        save_training_data=False,
    )


def _resolve_config(scan_dir: Path, n_val: int) -> Config:
    """
    Prefer Config_N_*.json. If any failure occurs, fallback to scan-dir parsing.
    """
    config_json = scan_dir / f"Config_N_{n_val}.json"
    if config_json.exists():
        try:
            cfg = _load_config_from_json(config_json)
            print(f"[CFG] Using config file: {config_json.name}")
            return cfg
        except Exception as e:
            print(f"[WARN] Failed to load {config_json.name}: {e}")
            print("[WARN] Falling back to scan-dir parsing.")
    else:
        print(f"[WARN] Missing {config_json.name}; fallback to scan-dir parsing.")

    return _build_config_from_scan_dir(scan_dir, n_val)


def _build_init_template(cfg: Config, init_q: np.ndarray) -> pd.DataFrame:
    rows = []
    for seller_id in range(cfg.num_sellers):
        for state_idx in range(cfg.num_grids):
            for action_idx in range(cfg.num_actions):
                rows.append(
                    {
                        "run_id": 0,  # filled per file later
                        "seller_id": int(seller_id),
                        "state_idx": int(state_idx),
                        "action_idx": int(action_idx),
                        "action_id": int(cfg.active_strategies[action_idx]),
                        "q_value": float(init_q[state_idx, action_idx]),
                        "training_stop_episode": -1,
                        "qtable_type": "init",
                    }
                )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["run_id"] = df["run_id"].astype("int32")
        df["seller_id"] = df["seller_id"].astype("int16")
        df["state_idx"] = df["state_idx"].astype("int16")
        df["action_idx"] = df["action_idx"].astype("int16")
        df["action_id"] = df["action_id"].astype("int16")
        df["training_stop_episode"] = df["training_stop_episode"].astype("int32")
        df["q_value"] = df["q_value"].astype("float32")
    return df


def _parse_run_id(path: Path) -> int:
    m = QTABLE_RE.match(path.name)
    if not m:
        raise ValueError(f"Unexpected qtable filename: {path.name}")
    return int(m.group(1))


def _has_correct_init(df: pd.DataFrame, expected_init: pd.DataFrame, atol: float = 1e-8) -> bool:
    if "qtable_type" not in df.columns:
        return False

    cur = df[df["qtable_type"] == "init"]
    if len(cur) != len(expected_init):
        return False

    key_cols = ["run_id", "seller_id", "state_idx", "action_idx", "action_id"]
    cur_cmp = cur[key_cols + ["q_value", "training_stop_episode"]].sort_values(key_cols).reset_index(drop=True)
    exp_cmp = expected_init[key_cols + ["q_value", "training_stop_episode"]].sort_values(key_cols).reset_index(drop=True)

    if not cur_cmp[key_cols].equals(exp_cmp[key_cols]):
        return False

    if not np.array_equal(cur_cmp["training_stop_episode"].to_numpy(), exp_cmp["training_stop_episode"].to_numpy()):
        return False

    return np.allclose(
        cur_cmp["q_value"].to_numpy(dtype=np.float64),
        exp_cmp["q_value"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=atol,
    )


def _replace_init_rows(df: pd.DataFrame, expected_init: pd.DataFrame) -> pd.DataFrame:
    if "qtable_type" not in df.columns:
        raise ValueError("Missing qtable_type column; cannot safely replace init rows.")

    non_init = df[df["qtable_type"] != "init"].copy()
    repaired = pd.concat([expected_init, non_init], ignore_index=True)

    # Keep original column order and dtypes where possible.
    for col in df.columns:
        if col not in repaired.columns:
            repaired[col] = pd.NA
    repaired = repaired[df.columns]

    for col in df.columns:
        try:
            repaired[col] = repaired[col].astype(df[col].dtype)
        except Exception:
            pass

    # Non-init rows must stay byte-equivalent on values/order.
    repaired_non_init = repaired[repaired["qtable_type"] != "init"].reset_index(drop=True)
    old_non_init = non_init.reset_index(drop=True)
    if not repaired_non_init.equals(old_non_init):
        raise RuntimeError("Non-init rows changed unexpectedly.")

    return repaired


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair init qtable rows in parquet files.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("data/results"),
        help="Root directory containing scan_*/N_*/run_*_qtable.parquet",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report only; do not modify files.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Optional max number of qtable files to process (for testing).",
    )
    args = parser.parse_args()

    results_root = args.results_root.expanduser().resolve()
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    qtable_files = _discover_qtables(results_root)
    if args.limit_files is not None:
        qtable_files = qtable_files[: args.limit_files]

    print(f"Results root: {results_root}")
    print(f"Q-table files discovered: {len(qtable_files)}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'WRITE'}")

    groups = _group_by_config(qtable_files)
    print(f"Config groups: {len(groups)}")

    fixed = 0
    skipped_already_ok = 0
    skipped_missing_col = 0
    errors = 0

    for idx, ((scan_dir, n_val), files) in enumerate(sorted(groups.items()), start=1):
        try:
            cfg = _resolve_config(scan_dir, n_val)
            env = PricingEnvironment(cfg)
            init_q = calculate_heuristic_init_values(env, cfg)
            init_template = _build_init_template(cfg, init_q)
        except Exception as e:
            print(f"[ERROR] Failed to prepare init template for {scan_dir.name}/N_{n_val}: {e}")
            errors += len(files)
            continue

        print(f"[{idx}/{len(groups)}] {scan_dir.name}/N_{n_val} files={len(files)}")

        for p in files:
            try:
                df = pd.read_parquet(p)
                if "qtable_type" not in df.columns:
                    print(f"  [SKIP missing qtable_type] {p}")
                    skipped_missing_col += 1
                    continue

                run_id = _parse_run_id(p)
                expected_init = init_template.copy()
                expected_init["run_id"] = np.int32(run_id)

                if _has_correct_init(df, expected_init):
                    skipped_already_ok += 1
                    continue

                repaired = _replace_init_rows(df, expected_init)
                if not args.dry_run:
                    tmp = p.with_suffix(".parquet.tmp")
                    repaired.to_parquet(tmp, index=False, compression="zstd")
                    tmp.replace(p)
                fixed += 1
            except Exception as e:
                print(f"  [ERROR] {p}: {e}")
                errors += 1

    print("-" * 60)
    print(f"Fixed files: {fixed}")
    print(f"Skipped (already correct): {skipped_already_ok}")
    print(f"Skipped (missing qtable_type): {skipped_missing_col}")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
