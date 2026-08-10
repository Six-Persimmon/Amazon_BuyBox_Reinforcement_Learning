"""
Batch runner for atomic-time experiments (exp07 / exp08).

Mirrors src.runner: parallel seeds via joblib, Parquet output with zstd,
skip-existing resume logic, Config JSON per N.
"""

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from src_atomic.config import AtomicConfig
from src_atomic.agent import heuristic_init_from_baseline_lookup
from src_atomic.simulation import run_simulation


def single_run_and_save(run_id, config, output_folder):
    file_name = f"run_{run_id}.parquet"
    save_path = output_folder / file_name
    q_table_file_name = f"run_{run_id}_qtable.parquet"
    q_table_save_path = output_folder / q_table_file_name

    if save_path.exists() and q_table_save_path.exists():
        return True

    try:
        np.random.seed(run_id)

        df_eval, q_table_snapshot = run_simulation(
            config,
            run_id=run_id,
            disable_tqdm=True,
            return_q_snapshot=True,
        )
        df_eval.to_parquet(save_path, index=False, compression="zstd")
        q_table_snapshot.to_parquet(q_table_save_path, index=False, compression="zstd")
        return True
    except Exception as e:
        print(f"!!! Error in Run {run_id}: {e}")
        return False


def run_experiment_batch(
    experiment_name,
    mu,
    active_strategies,
    n_list,
    n_rounds,
    n_jobs=None,
    output_root=None,
    **kwargs,
):
    print(f"\n{'='*60}")
    print(f"Starting Atomic Experiment: {experiment_name}")
    print(f"Params: mu={mu}, Rounds={n_rounds}")
    print(f"Strategies: {active_strategies}")
    print(f"{'='*60}")

    if output_root is None:
        base_output_dir = Path(__file__).resolve().parent.parent / "data" / "results" / experiment_name
    else:
        base_output_dir = Path(output_root).expanduser().resolve() / experiment_name
    base_output_dir.mkdir(parents=True, exist_ok=True)

    for n_sellers in n_list:
        current_output_dir = base_output_dir / f"N_{n_sellers}"
        current_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Processing N = {n_sellers} ---")

        cfg = AtomicConfig(
            num_sellers=n_sellers,
            mu=mu,
            active_strategies=active_strategies,
        )
        for key, value in kwargs.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                print(f"Warning: AtomicConfig has no attribute '{key}', ignoring.")

        config_file_path = current_output_dir.parent / f"Config_N_{n_sellers}.json"
        cfg_dict = dataclasses.asdict(cfg)
        for k, v in cfg_dict.items():
            if isinstance(v, Path):
                cfg_dict[k] = str(v)
        try:
            with open(config_file_path, "w") as f:
                json.dump(cfg_dict, f, indent=4)
            print(f"        [Config] Saved to {config_file_path.name}")
        except Exception as e:
            print(f"        [Warning] Failed to save config: {e}")

        # Warm the heuristic-init lookup table once, before the workers fan
        # out (they would otherwise serialize on the lookup file lock).
        print(f"Step 1: Checking/Computing init lookup table for N={n_sellers}...")
        t0_table = time.time()
        _ = heuristic_init_from_baseline_lookup(cfg)
        print(f"        Init lookup ready ({time.time() - t0_table:.1f}s).")

        print(f"Step 2: Running {n_rounds} simulations in parallel...")
        t0_sim = time.time()

        Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(single_run_and_save)(seed, cfg, current_output_dir)
            for seed in range(n_rounds)
        )

        elapsed = time.time() - t0_sim
        print(f"        Finished N={n_sellers} in {elapsed:.2f} seconds.")
        print(f"        Data saved to: {current_output_dir}")
