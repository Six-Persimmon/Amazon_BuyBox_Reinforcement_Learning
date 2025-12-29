#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Experiment harness for MARL repricer simulations."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd

from simulations.n_player_simulation import EnvironmentConfig, SimulationConfig, run_simulation  # noqa: E402
from agents.repricer_meta_actions import MetaActionLibrary  # noqa: E402

RESULTS_DIR = ROOT / "analysis" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_ACTION_SHARE_FIELDS = [f"action_share_{action.name}" for action in MetaActionLibrary().list_actions()]

SUMMARY_FIELDS = [
    "episode",
    "mean_profit",
    "norm_profit",
    "avg_price",
    "avg_lowest_price",
    "repricer_share",
    "network_density",
    "weighted_density",
    "in_degree_centralization",
] + _ACTION_SHARE_FIELDS


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _run_single(config: SimulationConfig) -> Dict[str, object]:
    result = run_simulation(config)
    return {"config": asdict(config), "result": result}


def _run_batch(configs: Sequence[SimulationConfig], max_workers: int | None = None) -> List[Dict[str, object]]:
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


def _records_to_frame(records: Sequence[Dict[str, object]]) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
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


def _aggregate_final_metrics(frame: pd.DataFrame) -> Dict[str, float]:
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


def _save_table(frame: pd.DataFrame, file_stem: str) -> Tuple[str, str]:
    """Persist the per-episode frame, preferring Parquet but falling back to CSV.

    The ``file_stem`` argument should already embed any identifying metadata
    (e.g. experiment name, parameter sweep details, timestamp).  This helper
    appends the appropriate extension and returns the resulting filename and
    format tag so callers can reference the saved artifact.
    """

    parquet_path = RESULTS_DIR / f"{file_stem}.parquet"
    try:
        frame.to_parquet(parquet_path, index=False)
        return parquet_path.name, "parquet"
    except Exception:
        csv_path = RESULTS_DIR / f"{file_stem}.csv.gz"
        frame.to_csv(csv_path, index=False, compression="gzip")
        return csv_path.name, "csv.gz"


def _save_payload(
    experiment_name: str,
    payload: Dict[str, object],
    frame: pd.DataFrame,
    params_dict: Dict[str, object] | None = None,
) -> Path:
    params_dict = params_dict or {}
    timestamp = _timestamp()

    def _value_text(value: object) -> str:
        if isinstance(value, float):
            text = f"{value:.3f}".rstrip("0").rstrip(".")
            return text if text else "0"
        if isinstance(value, (list, tuple, set)):
            return "-".join(_value_text(item) for item in value)
        return str(value)

    name_bits: List[str] = [experiment_name]
    for key, value in sorted(params_dict.items()):
        if key == "scenarios":
            continue
        name_bits.append(f"{key}{_value_text(value)}")
    base_id = "_".join(bit for bit in name_bits if bit)
    if not base_id:
        base_id = experiment_name or "experiment"
    artifact_name = f"{base_id}_{timestamp}"

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
    carry_over_prices: bool = True,
    allowed_action_ids: Sequence[int] | None = None,
) -> Path:
    """Run a simple suite of market composition experiments."""

    scenario_list = [dict(item) for item in scenarios]
    log_every = log_interval if log_interval is not None else max(1, outer_episodes // 10)

    configs: List[SimulationConfig] = []
    scenario_details: List[Dict[str, object]] = []

    for idx, spec in enumerate(scenario_list):
        n_players = spec.get("n_players", default_n_players)
        lam_value = spec.get("lam", spec.get("lambda", default_lambda))
        rho_value = spec.get("rho", default_rho)
        repricer_cost = spec.get("repricer_cost", default_repricer_cost)
        share_flag = bool(spec.get("share_parameters", share_parameters))
        label = spec.get("label") or f"rho_{rho_value}"

        # EnvironmentConfig no longer has lam or rho; they are kept only as scenario metadata.
        env_kwargs = {
            **spec.get("environment", {}),
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

    path = _save_payload(experiment_name, payload, episode_frame, params_for_name)
    print(f"Saved market composition results to {path}")
    return path


def experiment_n_player_logit(
    *,
    experiment_name: str = "n_player_logit",
    simulation_config: SimulationConfig | None = None,
    environment_config: EnvironmentConfig | None = None,
    seeds: Sequence[int] = (0, 1, 2),
    max_workers: int | None = None,
) -> Path:
    """Run repeated logit-demand simulations for a fixed config over multiple seeds.

    This is a simplified experiment harness for the canonical logit environment.
    It takes a single SimulationConfig and EnvironmentConfig, runs multiple
    seeds in parallel, and aggregates the per-episode summaries into the same
    payload format used by ``experiment_n_player_market_composition``.
    """

    # Default configs: 5-player logit market, no parameter sharing.
    base_env = environment_config or EnvironmentConfig()
    base_sim = simulation_config or SimulationConfig(
        n_players=5,
        share_parameters=False,
    )

    # Build per-seed SimulationConfig instances.
    configs: List[SimulationConfig] = []
    for seed in seeds:
        cfg = replace(
            base_sim,
            seed=seed,
            environment=base_env,
        )
        configs.append(cfg)

    # Run all seeds (in parallel if max_workers > 0).
    records = _run_batch(configs, max_workers=max_workers)
    episode_frame, run_info = _records_to_frame(records)

    # For this experiment, there is a single logical scenario.
    label = base_sim.scenario_label or "logit"
    scenario_rows = episode_frame[episode_frame["scenario"] == label] if not episode_frame.empty else episode_frame
    runs = [info for info in run_info if info["scenario"] == label] if run_info else run_info

    # Infer marginal costs used in simulation for metadata.
    if base_sim.marginal_costs is not None:
        mc_list = list(base_sim.marginal_costs)
    else:
        mc_list = [float(base_env.base_mc)] * base_sim.n_players

    scenario_detail: Dict[str, object] = {
        "label": label,
        "n_players": base_sim.n_players,
        # lam and rho are not used in the logit environment; they are omitted here.
        "repricer_cost": base_env.repricer_cost,
        "marginal_costs": mc_list,
        "share_parameters": base_sim.share_parameters,
        "environment": asdict(base_env),
    }

    scenario_payload = {
        **scenario_detail,
        "runs": runs,
        "summary": _aggregate_final_metrics(scenario_rows),
    }

    payload = {
        "experiment": experiment_name,
        "generated_at": _timestamp(),
        "outer_episodes": base_sim.outer_episodes,
        "inner_periods": base_sim.inner_periods,
        "seeds": list(seeds),
        "share_parameters_default": base_sim.share_parameters,
        "scenarios": [scenario_payload],
    }

    # Build a concise identifier for the saved artifact.
    params_for_name: Dict[str, object] = {
        "N": base_sim.n_players,
    }
    if base_sim.allowed_action_ids is not None:
        params_for_name["actions"] = list(base_sim.allowed_action_ids)

    path = _save_payload(experiment_name, payload, episode_frame, params_for_name)
    print(f"Saved logit-demand results to {path}")
    return path


if __name__ == "__main__":
    import time

    start_time = time.time()
    print(f"Starting simulation at {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}")

    # Derive action ID sets for:
    # (1) basic actions only (no reset/raise); and
    # (2) basic actions plus reset (still no raise_if_below_min).
    library = MetaActionLibrary()
    actions = library.list_actions()

    basic_action_ids = [
        a.action_id
        for a in actions
        if (
            a.is_static
            or (
                a.base_rule in {"match", "undercut", "above"}
                and not a.reset_when_below_cost
                and not a.raise_if_below_min
            )
        )
    ]

    basic_plus_reset_ids = [
        a.action_id
        for a in actions
        if (
            a.is_static
            or (
                a.base_rule in {"match", "undercut", "above"}
                and a.reset_when_below_cost
                and not a.raise_if_below_min
            )
        )
    ]

    # Common environment configuration: canonical logit demand with Calvano/Wang parameters.
    env_cfg = EnvironmentConfig()

    # New Env: hold others same but let a12 = 10
    env_maxprice_4_cfg = replace(env_cfg,
                             max_price=4.0)

    # (A) Experiment with basic actions only.
    sim_cfg_basic = SimulationConfig(
        n_players=5,
        outer_episodes=120_000,
        inner_periods=50,
        share_parameters=False,
        allowed_action_ids=basic_action_ids,
        verbose=True,
        log_interval=1000
    )
    path_basic = experiment_n_player_logit(
        # experiment_name="5_player_logit_dqn_baseactions_3_state_avgprice_avglowest_lastlowest",
        experiment_name="5_player_logit_maxprice_4_dqn_baseactions_3_state_avgprice_avglowest_lastlowest",
        simulation_config=sim_cfg_basic,
        # environment_config=env_cfg,
        environment_config=env_maxprice_4_cfg,
        seeds=(0, 1, 2, 3, 4),
        max_workers=None,
    )
    print(f"Basic-actions experiment saved to: {path_basic}")

    # (B) Experiment with basic actions plus reset variants.
    sim_cfg_reset = replace(
        sim_cfg_basic,
        allowed_action_ids=basic_plus_reset_ids,
    )
    path_reset = experiment_n_player_logit(
        experiment_name="5_player_logit_maxprice_4_dqn_baseactions_reset_3_state_avgprice_avglowest_lastlowest",
        simulation_config=sim_cfg_reset,
        # environment_config=env_cfg,
        environment_config=env_maxprice_4_cfg,
        seeds=(0, 1, 2, 3, 4),
        max_workers=None,
    )
    print(f"Basic+reset experiment saved to: {path_reset}")

    end_time = time.time()
    print(
        f"Total experiment time (hours and minutes): "
        f"{(end_time - start_time) // 3600}h {((end_time - start_time) % 3600) // 60}m"
    )
