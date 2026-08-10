"""
exp09: Calvano ladder -- where does the collusion gain come from?

A 2x2 factorial (state in {full price vector, market minimum} x actions in
{all 10 prices, the 3 repricer rules}) plus a fifth commitment rung at K = 30,
so the gain can be attributed additively:

    D_state       = Delta(C2) - Delta(C1)
    D_action      = Delta(C3) - Delta(C1)
    D_interaction = Delta(C4) - Delta(C2) - Delta(C3) + Delta(C1)
    D_commitment  = Delta(C5) - Delta(C4)
    total         = Delta(C5) - Delta(C1)

Scope: N = 3, all 10 baseline mu values, 30 seeds, beta = 1e-5,
converge_period = 100,000 (Calvano's own criterion), eval_H = 10,000.
All five rungs are produced by this script -- exp09 does not reuse exp07 data.

See Extention_Plan.md section 4 for the design, including the remark (4.5) on
why C1 is a Calvano-STYLE baseline rather than a strict replication.

Env-var overrides:
    PRICING_MARL_EXP09_FILTER_MU        e.g. "0.04,0.25"
    PRICING_MARL_EXP09_FILTER_CELL      e.g. "cal_full,cal_both"
    PRICING_MARL_EXP09_FILTER_N         e.g. "3"
    PRICING_MARL_EXP09_ROUNDS_PER_CONFIG
    PRICING_MARL_EXP09_MAX_EPISODES
    PRICING_MARL_EXP09_CONVERGE_PERIOD
    PRICING_MARL_EXP09_EVAL_H
    PRICING_MARL_EXP09_BETA             (the optional Arm B of 4.5 is just
                                         BETA=1e-6 + a different RESULTS_DIR)
    PRICING_MARL_EXP09_TIE_BREAK        "random" (default) | "lowest"
    PRICING_MARL_EXP09_RESULTS_DIR
    PRICING_MARL_EXP09_N_JOBS
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.strategies import ACT_ABOVE, ACT_MATCH, ACT_UNDERCUT
from src_calvano.runner import run_experiment_batch


# The five rungs. Order matters only for the log; each is independent.
CELLS = {
    "cal_full":     {"state_mode": "full_vector", "action_mode": "price", "K": 1},
    "cal_smin":     {"state_mode": "market_min",  "action_mode": "price", "K": 1},
    "cal_arule":    {"state_mode": "full_vector", "action_mode": "rule",  "K": 1},
    "cal_both":     {"state_mode": "market_min",  "action_mode": "rule",  "K": 1},
    "cal_both_k30": {"state_mode": "market_min",  "action_mode": "rule",  "K": 30},
}

CELL_DESCRIPTION = {
    "cal_full":     "C1 Calvano-style baseline (full state, all prices)",
    "cal_smin":     "C2 state reduction only (market min, all prices)",
    "cal_arule":    "C3 action reduction only (full state, 3 rules)",
    "cal_both":     "C4 both reductions (market min, 3 rules)",
    "cal_both_k30": "C5 both reductions + commitment (K=30)",
}


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
    values = [cast_fn(tok.strip()) for tok in raw.split(",") if tok.strip()]
    if not values:
        raise ValueError(f"{env_name} is set but empty.")
    return values


def _filter_values(full_values, requested_values, label: str, float_mode: bool = False):
    if requested_values is None:
        return full_values

    if float_mode:
        matched = [v for v in full_values if any(abs(v - r) <= 1e-12 for r in requested_values)]
        missing = [r for r in requested_values if not any(abs(v - r) <= 1e-12 for v in full_values)]
    else:
        requested_set = set(requested_values)
        matched = [v for v in full_values if v in requested_set]
        missing = [r for r in requested_values if r not in full_values]

    if missing:
        raise ValueError(f"{label} filter contains unsupported values: {missing}")
    if not matched:
        raise ValueError(f"{label} filter removed all values.")
    return matched


if __name__ == "__main__":
    N_VALUES = [3]
    MU_VALUES = [0.04, 0.07, 0.1, 0.13, 0.16, 0.19, 0.22, 0.25, 0.28, 0.31]
    K_VALUES = [1, 30]  # declared for the progress reporter; set per cell

    ROUNDS_PER_CONFIG = 30
    MAX_EPISODES = int(os.getenv("PRICING_MARL_EXP09_MAX_EPISODES", "5000000"))
    CONVERGE_PERIOD = int(os.getenv("PRICING_MARL_EXP09_CONVERGE_PERIOD", "100000"))
    EVAL_H = int(os.getenv("PRICING_MARL_EXP09_EVAL_H", "10000"))
    BETA = float(os.getenv("PRICING_MARL_EXP09_BETA", "1e-5"))
    TIE_BREAK = os.getenv("PRICING_MARL_EXP09_TIE_BREAK", "random")

    n_filter = _parse_csv_env("PRICING_MARL_EXP09_FILTER_N", int)
    mu_filter = _parse_csv_env("PRICING_MARL_EXP09_FILTER_MU", float)
    cell_filter = _parse_csv_env("PRICING_MARL_EXP09_FILTER_CELL", str)

    N_VALUES = _filter_values(N_VALUES, n_filter, "N")
    MU_VALUES = _filter_values(MU_VALUES, mu_filter, "Mu", float_mode=True)
    cell_names = _filter_values(list(CELLS.keys()), cell_filter, "Cell")

    rounds_override = os.getenv("PRICING_MARL_EXP09_ROUNDS_PER_CONFIG")
    if rounds_override:
        ROUNDS_PER_CONFIG = int(rounds_override)

    env_n_jobs = os.getenv("PRICING_MARL_EXP09_N_JOBS") or os.getenv("PRICING_MARL_N_JOBS")
    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    if env_n_jobs is not None:
        N_JOBS = int(env_n_jobs)
    elif slurm_cpus is not None:
        N_JOBS = int(slurm_cpus)
    else:
        N_JOBS = None

    results_dir = os.getenv("PRICING_MARL_EXP09_RESULTS_DIR") or (
        project_root / "data" / "results_exp09"
    )

    # The 3-rule set; only consulted by the rule-action cells.
    STRATS_BASIC = [ACT_UNDERCUT, ACT_MATCH, ACT_ABOVE]

    print("Starting exp09 Calvano-ladder scan...")
    print(f"N: {N_VALUES}")
    print(f"Mu: {MU_VALUES}")
    print(f"Cells: {cell_names}")
    print(f"Rounds per config: {ROUNDS_PER_CONFIG}")
    print(f"Max episodes: {MAX_EPISODES}")
    print(f"Converge period: {CONVERGE_PERIOD}")
    print(f"Eval horizon (atomic): {EVAL_H}")
    print(f"Beta: {BETA}")
    print(f"Tie-break: {TIE_BREAK}")
    print(f"Results dir: {results_dir}")
    print("-" * 50)

    overall_start = time.perf_counter()
    total_batches = len(cell_names) * len(MU_VALUES)
    batch_counter = 0

    for cell_name in cell_names:
        spec = CELLS[cell_name]
        print(f"\n[[[ STARTING EXP09 CELL: {cell_name} -- {CELL_DESCRIPTION[cell_name]} ]]]")

        for mu_val in MU_VALUES:
            batch_counter += 1
            k_val = spec["K"]
            exp_batch_name = f"scan_{cell_name}_mu{mu_val}_k{k_val}"

            print(f"\n>>> Running exp09 batch {batch_counter}/{total_batches}: {exp_batch_name}")
            batch_start = time.perf_counter()

            run_experiment_batch(
                experiment_name=exp_batch_name,
                mu=mu_val,
                active_strategies=STRATS_BASIC,
                n_list=N_VALUES,
                n_rounds=ROUNDS_PER_CONFIG,
                n_jobs=N_JOBS,
                output_root=results_dir,
                state_mode=spec["state_mode"],
                action_mode=spec["action_mode"],
                K=k_val,
                max_episodes=MAX_EPISODES,
                converge_period=CONVERGE_PERIOD,
                eval_H=EVAL_H,
                beta=BETA,
                tie_break=TIE_BREAK,
                save_training_data=False,
            )

            batch_elapsed = time.perf_counter() - batch_start
            total_elapsed = time.perf_counter() - overall_start
            print(f">>> Batch finished: {exp_batch_name} | elapsed {_fmt_duration(batch_elapsed)}")
            print(f">>> Total elapsed so far: {_fmt_duration(total_elapsed)}")

    print("\n" + "=" * 50)
    print("All exp09 experiments completed.")
    overall_elapsed = time.perf_counter() - overall_start
    print(f"End time: {datetime.now().isoformat(timespec='seconds')}")
    print(f"TOTAL TIME: {_fmt_duration(overall_elapsed)}")
