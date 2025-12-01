#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Simple experiment harness for the tabular Q-learning simulation."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd

from simulations.n_player_simulation_simple import EnvironmentConfig, SimulationConfig, run_simulation  # noqa: E402
from agents.repricer_meta_actions import MetaActionLibrary  # noqa: E402

RESULTS_DIR = ROOT / "analysis" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# These are the per-episode fields we record and later save to a file.
_ACTION_SHARE_FIELDS = [f"action_share_{action.name}" for action in MetaActionLibrary().list_actions()]
SUMMARY_FIELDS = [
    "episode",
    "mean_profit",
    "avg_price",
    "avg_lowest_price",
    "repricer_share",
    "network_density",
    "weighted_density",
] + _ACTION_SHARE_FIELDS


def _timestamp():
    """Helper to create a timestamp string for filenames."""
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _run_single(config):
    """Run one simulation with the given config and return its results."""
    result = run_simulation(config)
    return {"config": asdict(config), "result": result}


def _run_batch(configs, max_workers=None):
    """Run many simulations in parallel."""
    if not configs:
        return []
    records = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_single, cfg): cfg for cfg in configs}
        for future in as_completed(futures):
            records.append(future.result())
    return records


def _clean_summary(summary):
    """Keep only numeric fields we want to save."""
    cleaned = {}
    for key in SUMMARY_FIELDS:
        if key in summary:
            value = summary[key]
            if isinstance(value, (int, float)):
                cleaned[key] = float(value)
    return cleaned


def _records_to_frame(records):
    """Turn a list of simulation records into a pandas DataFrame for easy saving."""
    rows = []
    run_info = []

    for record in records:
        config = record.get("config", {})
        result = record.get("result", {})
        share_flag = bool(config.get("share_parameters", False))
        scenario_label = config.get("scenario_label") or ("shared_parameters" if share_flag else "independent_agents")
        seed = config.get("seed")

        summaries = result.get("summaries", [])
        for summary in summaries:
            row = {
                "scenario": scenario_label,
                "share_parameters": share_flag,
                "seed": seed,
            }
            for field in SUMMARY_FIELDS:
                if field in summary:
                    row[field] = summary[field]
            rows.append(row)

        final_summary = _clean_summary(summaries[-1]) if summaries else {}
        run_info.append(
            {
                "scenario": scenario_label,
                "share_parameters": share_flag,
                "seed": seed,
                "config": config,
                "final_summary": final_summary,
                "loss_updates": len(result.get("losses", [])),
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.sort_values(["scenario", "seed", "episode"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
    return frame, run_info


def _aggregate_final_metrics(frame):
    """Compute the average of the final-episode metrics across runs."""
    if frame.empty:
        return {}
    final_df = frame.sort_values("episode").groupby(["scenario", "seed"]).tail(1)
    aggregates = {}
    for field in SUMMARY_FIELDS:
        if field == "episode" or field not in final_df.columns:
            continue
        series = final_df[field].dropna()
        if not series.empty:
            aggregates[f"mean_{field}"] = float(series.mean())
    return aggregates


def _save_table(frame, file_stem):
    """Save the per-episode data to disk (Parquet preferred, CSV as fallback)."""
    parquet_path = RESULTS_DIR / f"{file_stem}.parquet"
    try:
        frame.to_parquet(parquet_path, index=False)
        return parquet_path.name, "parquet"
    except Exception:
        csv_path = RESULTS_DIR / f"{file_stem}.csv.gz"
        frame.to_csv(csv_path, index=False, compression="gzip")
        return csv_path.name, "csv.gz"


def _save_payload(experiment_name, payload, frame, params_dict=None):
    """Save a summary JSON that points to the per-episode data file.

    - ``payload`` holds the high-level results (e.g., per-scenario summaries).
    - The episode-level table is saved separately and referenced by filename.
    """
    params_dict = params_dict or {}
    timestamp = _timestamp()

    def _value_text(value):
        if isinstance(value, float):
            text = f"{value:.3f}".rstrip("0").rstrip(".")
            return text if text else "0"
        if isinstance(value, (list, tuple, set)):
            return "-".join(_value_text(item) for item in value)
        return str(value)

    name_bits = [experiment_name]
    for key, value in sorted(params_dict.items()):
        if key == "scenarios":
            continue
        name_bits.append(f"{key}{_value_text(value)}")
    base_id = "_".join(bit for bit in name_bits if bit)
    if not base_id:
        base_id = experiment_name or "experiment"
    artifact_name = f"{base_id}_{timestamp}"

    # Save the per-episode table (usually Parquet).
    data_stem = f"{artifact_name}_episodes"
    data_name, data_format = _save_table(frame, data_stem)

    payload = dict(payload)
    payload["data_file"] = data_name
    payload["data_format"] = data_format
    path = RESULTS_DIR / f"{artifact_name}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def experiment_n_player_market_composition(
    experiment_name="n_player_market_composition",
    scenarios=(),
    default_n_players=5,
    default_lambda=1.0,
    default_rho=0.5,
    default_marginal_costs=None,
    default_repricer_cost=0.0,
    share_parameters=False,
    outer_episodes=100_000,
    inner_periods=50,
    seeds=(0, 1, 2),
    max_workers=None,
    verbose=True,
    log_interval=None,
    carry_over_prices=True,
    allowed_action_ids=None,
):
    """Run a suite of market composition experiments using the tabular agent."""

    scenario_list = [dict(item) for item in scenarios]
    log_every = log_interval if log_interval is not None else max(1, outer_episodes // 10)

    configs = []
    scenario_details = []

    for idx, spec in enumerate(scenario_list):
        n_players = spec.get("n_players", default_n_players)
        lam_value = spec.get("lam", spec.get("lambda", default_lambda))
        rho_value = spec.get("rho", default_rho)
        repricer_cost = spec.get("repricer_cost", default_repricer_cost)
        share_flag = bool(spec.get("share_parameters", share_parameters))
        label = spec.get("label") or f"rho_{rho_value}"

        env_kwargs = {
            **spec.get("environment", {}),
            "lam": lam_value,
            "rho": rho_value,
            "repricer_cost": repricer_cost,
        }
        environment = EnvironmentConfig(**env_kwargs)

        marginal_costs = spec.get("marginal_costs")
        if marginal_costs is None:
            if default_marginal_costs is not None:
                marginal_costs = list(default_marginal_costs)
            else:
                marginal_costs = [2.0] * n_players
        else:
            marginal_costs = list(marginal_costs)

        if len(marginal_costs) != n_players:
            raise ValueError(f"Scenario '{label}' expected {n_players} marginal costs, got {len(marginal_costs)}")

        base_config = SimulationConfig(
            n_players=n_players,
            outer_episodes=outer_episodes,
            inner_periods=inner_periods,
            share_parameters=share_flag,
            seed=None,
            environment=environment,
            marginal_costs=marginal_costs,
            verbose=verbose,
            log_interval=log_every,
            scenario_label=label,
            carry_over_prices=carry_over_prices,
            allowed_action_ids=allowed_action_ids,
        )

        for seed in seeds:
            configs.append(replace(base_config, seed=seed))

        scenario_details.append(
            {
                "label": label,
                "n_players": n_players,
                "rho": rho_value,
                "lam": lam_value,
                "repricer_cost": repricer_cost,
                "marginal_costs": marginal_costs,
                "share_parameters": share_flag,
            }
        )

    # Run all configs (possibly in parallel).
    records = _run_batch(configs, max_workers=max_workers)
    episode_frame, run_info = _records_to_frame(records)

    # Summaries by scenario.
    scenario_payloads = []
    for detail in scenario_details:
        label = detail["label"]
        scenario_rows = episode_frame[episode_frame["scenario"] == label]
        runs = [info for info in run_info if info["scenario"] == label]
        scenario_payloads.append({**detail, "runs": runs, "summary": _aggregate_final_metrics(scenario_rows)})

    payload = {
        "experiment": experiment_name,
        "generated_at": _timestamp(),
        "outer_episodes": outer_episodes,
        "inner_periods": inner_periods,
        "seeds": list(seeds),
        "share_parameters_default": share_parameters,
        "scenarios": scenario_payloads,
    }

    distinct_players = sorted({detail["n_players"] for detail in scenario_details})
    distinct_lambdas = sorted({detail["lam"] for detail in scenario_details})
    rho_values = [detail["rho"] for detail in scenario_details]
    distinct_costs = sorted({detail["repricer_cost"] for detail in scenario_details})

    params_for_name = {}
    if distinct_players:
        params_for_name["N"] = distinct_players if len(distinct_players) > 1 else distinct_players[0]
    if distinct_lambdas:
        params_for_name["lambda"] = distinct_lambdas if len(distinct_lambdas) > 1 else distinct_lambdas[0]
    if rho_values:
        params_for_name["rho"] = rho_values if len(set(rho_values)) > 1 else rho_values[0]
    if distinct_costs and any(cost != 0 for cost in distinct_costs):
        params_for_name["c"] = distinct_costs if len(distinct_costs) > 1 else distinct_costs[0]

    path = _save_payload(experiment_name, payload, episode_frame, params_for_name)
    print(f"Saved market composition results to {path}")
    return path


if __name__ == "__main__":
    import time

    start_time = time.time()
    print(f"Starting Simulation at {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}")

    rho_sweep_scenarios = (
        {"label": "rho_1", "rho": 1.0},
        {"label": "rho_0.9", "rho": 0.9},
        {"label": "rho_0.5", "rho": 0.5},
        {"label": "rho_0.1", "rho": 0.1},
        {"label": "rho_0", "rho": 0.0},
    )

    experiment_n_player_market_composition(
        scenarios=rho_sweep_scenarios,
        default_n_players=5,
        default_lambda=1.0,
        default_marginal_costs=[2.0, 2.0, 2.0, 2.0, 2.0],
        seeds=(0, 1, 2, 3, 4, 5),
        outer_episodes=120_000,
        inner_periods=50,
        carry_over_prices=True,
        experiment_name="n_player_carryover_simplerule_tab_q_learning_3_state",
        allowed_action_ids=(0, 1, 5, 9),
    )

    experiment_n_player_market_composition(
        scenarios=rho_sweep_scenarios,
        default_n_players=5,
        default_lambda=1.0,
        default_marginal_costs=[2.0, 2.0, 2.0, 2.0, 2.0],
        seeds=(0, 1, 2, 3, 4, 5),
        outer_episodes=120_000,
        inner_periods=50,
        carry_over_prices=True,
        experiment_name="n_player_carryover_simplerule_reset_tab_q_learning_3_state",
        allowed_action_ids=(0, 1, 3, 5, 7, 9, 11),
    )

    end_time = time.time()
    print(
        f"Total experiment time (hours and minutes): {(end_time - start_time)//3600}h "
        f"{((end_time - start_time)%3600)//60}m"
    )

'''
	•	0: no_repricer
	•	1: undercut
	•	2: undercut_raise
	•	3: undercut_reset
	•	4: undercut_reset_raise
	•	5: match
	•	6: match_raise
	•	7: match_reset
	•	8: match_reset_raise
	•	9: above
	•	10: above_raise
	•	11: above_reset
	•	12: above_reset_raise
'''