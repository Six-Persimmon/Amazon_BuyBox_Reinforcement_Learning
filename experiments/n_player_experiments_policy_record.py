#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Experiment harness variant that records policy probe outputs."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from simulations.n_player_simulation_policy_record import (  # noqa: E402
    EnvironmentConfig,
    PolicyRecordSimulationConfig,
    run_simulation_policy_record,
)
from agents.repricer_meta_actions import MetaActionLibrary  # noqa: E402

RESULTS_DIR = ROOT / "analysis" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_ACTION_SHARE_FIELDS = [f"action_share_{action.name}" for action in MetaActionLibrary().list_actions()]

SUMMARY_FIELDS = [
    "episode",
    "mean_profit",
    "avg_price",
    "avg_lowest_price",
    "repricer_share",
] + _ACTION_SHARE_FIELDS


def _timestamp() -> str:
    from datetime import datetime

    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _run_single(config: PolicyRecordSimulationConfig) -> Dict[str, object]:
    result = run_simulation_policy_record(config)
    return {"config": asdict(config), "result": result}


def _run_batch(configs: Sequence[PolicyRecordSimulationConfig], max_workers: int | None = None) -> List[Dict[str, object]]:
    if not configs:
        return []
    records: List[Dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_single, cfg): cfg for cfg in configs}
        for future in as_completed(futures):
            records.append(future.result())
    return records


def _clean_summary(summary: Dict[str, object]) -> Dict[str, float]:
    cleaned: Dict[str, float] = {}
    for key in SUMMARY_FIELDS:
        if key in summary:
            value = summary[key]
            if isinstance(value, (int, float)):
                cleaned[key] = float(value)
    return cleaned


def _records_to_frame(
    records: Sequence[Dict[str, object]]
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    # Convert simulation records into a DataFrame of episode summaries and a list of run info dicts.
    rows: List[Dict[str, object]] = []
    run_info: List[Dict[str, object]] = []

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
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    row[key] = float(value)
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


def _aggregate_final_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    # Aggregate final metrics from the last episode of each (scenario, seed) combination.
    if frame.empty:
        return {}
    final_df = frame.sort_values("episode").groupby(["scenario", "seed"]).tail(1)
    aggregates: Dict[str, float] = {}
    for field in SUMMARY_FIELDS:
        if field == "episode" or field not in final_df.columns:
            continue
        series = final_df[field].dropna()
        if not series.empty:
            aggregates[f"mean_{field}"] = float(series.mean())
    return aggregates


def _save_table(frame: pd.DataFrame, base_path: Path) -> Tuple[str, str]:
    csv_path = base_path.with_suffix(".csv.gz")
    frame.to_csv(csv_path, index=False, compression="gzip")
    return csv_path.name, "csv.gz"


def _value_text(value: object) -> str:
    # Convert a parameter value to a string suitable for filenames.
    if isinstance(value, float):
        text = f"{value:.3f}".rstrip("0").rstrip(".")
        return text if text else "0"
    if isinstance(value, (list, tuple, set)):
        return "-".join(_value_text(item) for item in value)
    return str(value)


def _build_base_name(experiment_name: str, params_dict: Dict[str, object]) -> str:
    # Build a base filename for the experiment results based on parameters.
    timestamp = _timestamp()
    detail_bits: List[str] = []
    for key, value in sorted(params_dict.items()):
        if key == "scenarios":
            continue
        detail_bits.append(f"{key}{_value_text(value)}")
    detail_text = "_".join(bit for bit in detail_bits if bit)
    if detail_text:
        return f"{experiment_name}_policy_record_{detail_text}_{timestamp}"
    return f"{experiment_name}_policy_record_{timestamp}"


def _write_payload(base_name: str, payload: Dict[str, object], frame: pd.DataFrame) -> Path:
    # save the experiment data, then write the payload JSON referencing the data file. Payload is like a high-level summary of the experiment.
    data_name, data_format = _save_table(frame, RESULTS_DIR / f"{base_name}_episodes")
    payload = dict(payload)
    payload["data_file"] = data_name
    payload["data_format"] = data_format
    path = RESULTS_DIR / f"{base_name}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def experiment_n_player_market_composition_policy_record(
    *,
    experiment_name: str = "n_player_market_composition",
    scenarios: Sequence[Dict[str, object]] = (),
    default_n_players: int = 5,
    default_lambda: float = 1.0,
    default_rho: float = 0.5,
    default_marginal_costs: Sequence[float] | None = None,
    default_repricer_cost: float = 0.0,
    share_parameters: bool = False,
    outer_episodes: int = 100_000,
    inner_periods: int = 50,
    seeds: Sequence[int] = (0, 1, 2),
    max_workers: int | None = None,
    verbose: bool = True,
    log_interval: int | None = None,
    carry_over_prices: bool = False,
    policy_probe_states: Sequence[Sequence[float]] | None = None,
    policy_probe_labels: Sequence[str] | None = None,
    policy_probe_states_normalised: bool = False,
    policy_probe_use_greedy: bool = True,
    policy_qnet_snapshot_episodes: Sequence[int] = (),
    policy_save_final_qnets: bool = False,
    policy_output_dir: str | None = None,
    policy_file_prefix: str | None = None,
) -> Path:
    """Run market composition experiments with policy probing enabled."""

    scenario_list = [dict(item) for item in scenarios]
    log_every = log_interval if log_interval is not None else max(1, outer_episodes // 10)

    configs: List[PolicyRecordSimulationConfig] = []
    scenario_details: List[Dict[str, object]] = []

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

        scenario_probe_states = spec.get("policy_probe_states", policy_probe_states)
        if scenario_probe_states is not None:
            scenario_probe_states = [list(state) for state in scenario_probe_states]
        scenario_probe_labels = spec.get("policy_probe_labels", policy_probe_labels)
        if scenario_probe_labels is not None:
            scenario_probe_labels = list(scenario_probe_labels)
        scenario_probe_normalised = spec.get("policy_probe_states_normalised", policy_probe_states_normalised)
        if scenario_probe_normalised is None:
            scenario_probe_normalised = policy_probe_states_normalised
        scenario_probe_greedy = spec.get("policy_probe_use_greedy", policy_probe_use_greedy)
        if scenario_probe_greedy is None:
            scenario_probe_greedy = policy_probe_use_greedy
        scenario_snapshot_episodes = spec.get("policy_qnet_snapshot_episodes", policy_qnet_snapshot_episodes)
        if scenario_snapshot_episodes is None:
            scenario_snapshot_episodes = ()
        scenario_save_final_qnets = spec.get("policy_save_final_qnets", policy_save_final_qnets)
        if scenario_save_final_qnets is None:
            scenario_save_final_qnets = policy_save_final_qnets
        scenario_output_dir = spec.get("policy_output_dir", policy_output_dir)
        scenario_file_prefix = spec.get("policy_file_prefix", policy_file_prefix)

        base_config = PolicyRecordSimulationConfig(
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
            policy_probe_states=scenario_probe_states,
            policy_probe_labels=scenario_probe_labels,
            policy_probe_states_normalised=scenario_probe_normalised,
            policy_probe_use_greedy=scenario_probe_greedy,
            policy_qnet_snapshot_episodes=tuple(int(ep) for ep in scenario_snapshot_episodes),
            policy_save_final_qnets=bool(scenario_save_final_qnets),
            policy_output_dir=str(scenario_output_dir) if scenario_output_dir is not None else None,
            policy_file_prefix=scenario_file_prefix,
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

    records = _run_batch(configs, max_workers=max_workers)
    episode_frame, run_info = _records_to_frame(records)

    scenario_payloads: List[Dict[str, object]] = []
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

    params_for_name: Dict[str, object] = {}
    if distinct_players:
        params_for_name["N"] = distinct_players if len(distinct_players) > 1 else distinct_players[0]
    if distinct_lambdas:
        params_for_name["lambda"] = distinct_lambdas if len(distinct_lambdas) > 1 else distinct_lambdas[0]
    if rho_values:
        params_for_name["rho"] = rho_values if len(set(rho_values)) > 1 else rho_values[0]
    if distinct_costs and any(cost != 0 for cost in distinct_costs):
        params_for_name["c"] = distinct_costs if len(distinct_costs) > 1 else distinct_costs[0]

    base_name = _build_base_name(experiment_name, params_for_name)
    path = _write_payload(base_name, payload, episode_frame)
    print(f"Saved policy-record market composition results to {path}")
    return path


__all__ = [
    "experiment_n_player_market_composition_policy_record",
]


if __name__ == "__main__":
    import time

    start_time = time.time()
    print(f"[PolicyRecordExperiments] Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(start_time))}")

    rho_sweep_scenarios: Sequence[Dict[str, object]] = (
        {"label": "rho_1", "rho": 1.0},
        {"label": "rho_0.9", "rho": 0.9},
        {"label": "rho_0.5", "rho": 0.5},
        {"label": "rho_0.1", "rho": 0.1},
        {"label": "rho_0", "rho": 0.0},
    )
    probe_states = [(2.09, 2.09, 2.09)]
    probe_labels = ["near_marginal_cost"]

    # experiment_n_player_market_composition_policy_record(
    #     scenarios=rho_sweep_scenarios,
    #     default_n_players=5,
    #     default_lambda=1.0,
    #     default_marginal_costs=[2.0] * 5,
    #     seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    #     outer_episodes=150_000,
    #     inner_periods=50,
    #     carry_over_prices=True,
    #     experiment_name="n_player_market_composition_carryover",
    #     policy_probe_states=probe_states,
    #     policy_probe_labels=probe_labels,
    #     policy_probe_states_normalised=False,
    #     policy_probe_use_greedy=True,
    # )

    experiment_n_player_market_composition_policy_record(
        scenarios=rho_sweep_scenarios,
        default_n_players=10,
        default_lambda=1.0,
        default_marginal_costs=[2.0] * 10,
        seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        outer_episodes=150_000,
        inner_periods=50,
        carry_over_prices=True,
        experiment_name="n_player_market_composition_carryover",
        policy_probe_states=probe_states,
        policy_probe_labels=probe_labels,
        policy_probe_states_normalised=False,
        policy_probe_use_greedy=True,
    )

    end_time = time.time()
    total = end_time - start_time
    print(
        "[PolicyRecordExperiments] Total runtime: "
        f"{int(total // 3600)}h {int((total % 3600) // 60)}m"
    )
