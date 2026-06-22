import itertools
import os
import sys
import time
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.runner import run_experiment_batch
from src.strategies import ACT_ABOVE, ACT_MATCH, ACT_UNDER_RESET, ACT_UNDERCUT


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _parse_csv_env(env_name: str, cast_fn):
    raw = os.getenv(env_name)
    if not raw:
        return None
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(cast_fn(token))
    if not values:
        raise ValueError(f"{env_name} is set but empty.")
    return values


def _filter_values(full_values, requested_values, label: str, float_mode: bool = False):
    if requested_values is None:
        return full_values

    if float_mode:
        matched = [
            v for v in full_values
            if any(abs(v - req) <= 1e-12 for req in requested_values)
        ]
        missing = [
            req for req in requested_values
            if not any(abs(v - req) <= 1e-12 for v in full_values)
        ]
    else:
        requested_set = set(requested_values)
        matched = [v for v in full_values if v in requested_set]
        missing = [req for req in requested_values if req not in full_values]

    if missing:
        raise ValueError(f"{label} filter contains unsupported values: {missing}")
    if not matched:
        raise ValueError(f"{label} filter removed all values.")
    return matched


if __name__ == "__main__":
    N_VALUES = [2, 3, 5, 7, 10]
    MU_VALUES = [0.04, 0.07, 0.1, 0.13, 0.16, 0.19, 0.22, 0.25, 0.28, 0.31]
    K_VALUES = [1]

    ROUNDS_PER_CONFIG = 30
    MAX_EPISODES = int(os.getenv("PRICING_MARL_EXP05_MAX_EPISODES", "2000000"))
    CONVERGE_PERIOD = int(os.getenv("PRICING_MARL_EXP05_CONVERGE_PERIOD", "10000"))
    EVAL_H = int(os.getenv("PRICING_MARL_EXP05_EVAL_H", "2000"))

    n_filter = _parse_csv_env("PRICING_MARL_EXP05_FILTER_N", int)
    mu_filter = _parse_csv_env("PRICING_MARL_EXP05_FILTER_MU", float)

    N_VALUES = _filter_values(N_VALUES, n_filter, "N")
    MU_VALUES = _filter_values(MU_VALUES, mu_filter, "Mu", float_mode=True)

    rounds_override = os.getenv("PRICING_MARL_EXP05_ROUNDS_PER_CONFIG")
    if rounds_override:
        ROUNDS_PER_CONFIG = int(rounds_override)

    env_n_jobs = os.getenv("PRICING_MARL_EXP05_N_JOBS") or os.getenv("PRICING_MARL_N_JOBS")
    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    if env_n_jobs is not None:
        N_JOBS = int(env_n_jobs)
    elif slurm_cpus is not None:
        N_JOBS = int(slurm_cpus)
    else:
        N_JOBS = None

    results_dir = os.getenv("PRICING_MARL_EXP05_RESULTS_DIR") or (
        project_root / "data" / "results_exp05"
    )

    STRATS_BASIC = [ACT_UNDERCUT, ACT_MATCH, ACT_ABOVE]
    STRATS_COLLUSIVE = [ACT_UNDERCUT, ACT_MATCH, ACT_ABOVE, ACT_UNDER_RESET]

    EXPERIMENT_SETS = {
        "3strats": STRATS_BASIC,
        "4strats": STRATS_COLLUSIVE,
    }

    only_set = os.getenv("PRICING_MARL_EXP05_EXPERIMENT_SET")
    if only_set:
        if only_set not in EXPERIMENT_SETS:
            raise ValueError(
                f"Unknown PRICING_MARL_EXP05_EXPERIMENT_SET='{only_set}'. "
                f"Valid: {list(EXPERIMENT_SETS.keys())}"
            )
        EXPERIMENT_SETS = {only_set: EXPERIMENT_SETS[only_set]}

    print("Starting exp05 K=1 scan...")
    print(f"N: {N_VALUES}")
    print(f"Mu: {MU_VALUES}")
    print(f"K: {K_VALUES}")
    print(f"Rounds per config: {ROUNDS_PER_CONFIG}")
    print(f"Max episodes: {MAX_EPISODES}")
    print(f"Converge period: {CONVERGE_PERIOD}")
    print(f"Eval horizon: {EVAL_H}")
    print(f"Results dir: {results_dir}")
    print("-" * 50)

    overall_start = time.perf_counter()
    combinations = list(itertools.product(MU_VALUES, K_VALUES))
    total_batches = len(EXPERIMENT_SETS) * len(combinations)
    batch_counter = 0

    for label, current_strats in EXPERIMENT_SETS.items():
        print(f"\n[[[ STARTING EXP05 SET: {label} ({len(current_strats)} strategies) ]]]")

        for mu_val, k_val in combinations:
            batch_counter += 1
            exp_batch_name = f"scan_k1_{label}_mu{mu_val}_k{k_val}"

            print(f"\n>>> Running exp05 batch {batch_counter}/{total_batches}: {exp_batch_name}")
            batch_start = time.perf_counter()

            run_experiment_batch(
                experiment_name=exp_batch_name,
                mu=mu_val,
                active_strategies=current_strats,
                n_list=N_VALUES,
                n_rounds=ROUNDS_PER_CONFIG,
                n_jobs=N_JOBS,
                output_root=results_dir,
                K=k_val,
                max_episodes=MAX_EPISODES,
                converge_period=CONVERGE_PERIOD,
                eval_H=EVAL_H,
                beta=1e-5,
                save_training_data=False,
            )

            batch_elapsed = time.perf_counter() - batch_start
            total_elapsed = time.perf_counter() - overall_start
            print(f">>> Batch finished: {exp_batch_name} | elapsed {_fmt_duration(batch_elapsed)}")
            print(f">>> Total elapsed so far: {_fmt_duration(total_elapsed)}")

    print("\n" + "=" * 50)
    print("All exp05 experiments completed.")
    overall_elapsed = time.perf_counter() - overall_start
    print("\n" + "=" * 50)
    print(f"End time: {datetime.now().isoformat(timespec='seconds')}")
    print(f"TOTAL TIME: {_fmt_duration(overall_elapsed)}")
