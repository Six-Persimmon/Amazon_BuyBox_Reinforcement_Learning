#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Plotting utilities for MARL repricer experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")


def load_experiment(path: Path | str) -> dict:
    """Load a saved experiment JSON file."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    payload["_source_path"] = path
    return payload


def runs_to_frame(data: dict) -> pd.DataFrame:
    """Flatten scenario runs into a tidy DataFrame."""

    data_file = data.get("data_file")
    source_path = Path(data.get("_source_path", ".")) if "_source_path" in data else None

    if data_file and source_path:
        data_path = source_path.parent / data_file
        if not data_path.exists():
            raise FileNotFoundError(f"Episode data file '{data_path}' not found.")
        if data.get("data_format") == "csv.gz" or data_path.suffix == ".gz":
            frame = pd.read_csv(data_path)
        else:
            frame = pd.read_parquet(data_path)
        frame.sort_values(["scenario", "seed", "episode"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
        return frame

    records: List[dict] = []
    scenarios = data.get("scenarios", [])
    for scenario in scenarios:
        label = scenario.get("label", "scenario")
        for run in scenario.get("runs", []):
            seed = run.get("config", {}).get("seed")
            summaries = run.get("result", {}).get("summaries", [])
            for summary in summaries:
                row = {
                    "scenario": label,
                    "seed": seed,
                }
                for key, value in summary.items():
                    if isinstance(value, (int, float, bool)):
                        row[key] = value
                records.append(row)
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    frame.sort_values(["scenario", "seed", "episode"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def plot_metric_over_episodes(
    frame: pd.DataFrame,
    metric: str,
    *,
    smoothing: int = 1,
    figsize: tuple[int, int] = (14, 8),
    title: str | None = None,
    layout: tuple[int, int] = (2, 3),
    include_overall: bool = True,
    use_log_x: bool = False,
    save_path: Path | str | None = None,
) -> None:
    """Plot the requested metric averaged over seeds for each scenario."""

    if metric not in frame.columns:
        raise ValueError(f"Unknown metric '{metric}'")

    df = frame.copy()
    if smoothing > 1:
        df[metric] = (
            df.groupby(["scenario", "seed"])[metric]
            .transform(lambda x: x.rolling(window=smoothing, min_periods=1).mean())
        )

    scenarios = sorted(df["scenario"].dropna().unique())
    rows, cols = layout
    total_axes = rows * cols
    required_axes = len(scenarios) + (1 if include_overall else 0)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes_list = list(axes.flatten()) if total_axes > 1 else [axes]

    plot_col = "episode_plot" if use_log_x else "episode"
    if use_log_x:
        df[plot_col] = df["episode"] + 1.0
    else:
        df[plot_col] = df["episode"]

    for idx, scenario in enumerate(scenarios):
        if idx >= total_axes:
            break
        ax = axes_list[idx]
        subset = df[df["scenario"] == scenario]
        sns.lineplot(
            data=subset,
            x=plot_col,
            y=metric,
            hue="scenario",
            estimator="mean",
            errorbar="sd",
            legend=False,
            ax=ax,
        )
        ax.set_title(scenario)
        ax.set_xlabel("Episode (log10 scale)" if use_log_x else "Episode")
        if use_log_x:
            ax.set_xscale("log")

    if include_overall and len(scenarios) < total_axes:
        ax = axes_list[len(scenarios)]
        overall = (
            df.groupby("episode", as_index=False)[metric]
            .mean(numeric_only=True)
            .rename(columns={metric: "value"})
        )
        x_values = overall["episode"] + 1.0 if use_log_x else overall["episode"]
        ax.plot(x_values, overall["value"], color="black")
        ax.set_title("Average (all scenarios)")
        ax.set_xlabel("Episode (log10 scale)" if use_log_x else "Episode")
        ax.set_ylabel(metric)
        if use_log_x:
            ax.set_xscale("log")

    for idx in range(required_axes, total_axes):
        axes_list[idx].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.97] if title else None)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_final_metric(
    frame: pd.DataFrame,
    metric: str,
    *,
    figsize: tuple[int, int] = (6, 4),
    tail_length: int = 1,
    save_path: Path | str | None = None,
) -> None:
    """Compare final-episode metrics across scenarios."""

    if metric not in frame.columns:
        raise ValueError(f"Unknown metric '{metric}'")

    final_df = (
        frame.sort_values("episode")
        .groupby(["scenario", "seed"])
        .tail(tail_length)
        .groupby(["scenario", "seed"], as_index=False)
        .mean(numeric_only=True)
    )
    plt.figure(figsize=figsize)
    sns.barplot(data=final_df, x="scenario", y=metric, hue="scenario", dodge=False)
    plt.title(f"Final episode {metric} (avg last {tail_length})")
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


DEFAULT_ACTION_FAMILIES: Dict[str, Callable[[str], bool]] = {
    "Match Style": lambda name: name.startswith("match"),
    "Undercut Style": lambda name: name.startswith("undercut"),
    "Above Style": lambda name: name.startswith("above"),
    "No Repricer": lambda name: name == "no_repricer",
    "Raise Style": lambda name: "raise" in name,
    "Reset Style": lambda name: "reset" in name,
}


def _aggregate_action_families(
    frame: pd.DataFrame,
    families: Sequence[str] | None = None, # which families to include and their order. e.g. ["Match Style", "Undercut Style"]
    family_rules: Mapping[str, Callable[[str], bool]] | None = None, # custom family definitions
) -> pd.DataFrame:
    """Expand action-share columns into family-level shares."""

    action_cols = [col for col in frame.columns if col.startswith("action_share_")]
    if not action_cols:
        raise ValueError("No action_share_* columns found in the frame.")

    rules = family_rules or DEFAULT_ACTION_FAMILIES
    family_order = list(families) if families is not None else list(rules.keys())
    action_names = {col: col.replace("action_share_", "") for col in action_cols}

    base_cols = frame[["scenario", "seed", "episode"]].copy()
    pieces: List[pd.DataFrame] = []
    for family in family_order:
        predicate = rules.get(family)
        if predicate is None:
            continue
        family_columns = [col for col, name in action_names.items() if predicate(name)]
        if not family_columns:
            continue
        part = base_cols.copy()
        part["family"] = family
        part["share"] = frame[family_columns].sum(axis=1)
        pieces.append(part)

    if not pieces:
        raise ValueError("No action families matched the provided data.")

    return pd.concat(pieces, ignore_index=True)


def plot_action_family_shares(
    frame: pd.DataFrame,
    *,
    families: Sequence[str] | None = None,
    family_rules: Mapping[str, Callable[[str], bool]] | None = None,
    figsize: tuple[int, int] = (14, 8),
    layout: tuple[int, int] = (2, 3),
    smoothing: int = 1,
    use_log_x: bool = False,
    save_path: Path | str | None = None,
) -> None:
    """Plot aggregated action-family shares (mean across seeds) over episodes."""

    agg = _aggregate_action_families(frame, families=families, family_rules=family_rules)
    families_present = list(dict.fromkeys(agg["family"]))
    scenarios = sorted(agg["scenario"].unique())

    rows, cols = layout
    total_axes = rows * cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes_list = list(axes.flatten()) if total_axes > 1 else [axes]

    plot_col = "episode_plot" if use_log_x else "episode"
    if use_log_x:
        agg[plot_col] = agg["episode"] + 1.0
    else:
        agg[plot_col] = agg["episode"]

    for idx, family in enumerate(families_present):
        if idx >= total_axes:
            break
        ax = axes_list[idx]
        subset = agg[agg["family"] == family].copy()
        if smoothing > 1:
            subset["share"] = (
                subset.groupby(["scenario", "seed"])["share"]
                .transform(lambda x: x.rolling(window=smoothing, min_periods=1).mean())
            )
        mean_df = (
            subset.groupby(["scenario", "episode"], as_index=False)["share"]
            .mean(numeric_only=True)
            .rename(columns={"share": "mean_share"})
        )
        if use_log_x:
            mean_df[plot_col] = mean_df["episode"] + 1.0
        else:
            mean_df[plot_col] = mean_df["episode"]
        for scenario in scenarios:
            scenario_df = mean_df[mean_df["scenario"] == scenario]
            ax.plot(scenario_df[plot_col], scenario_df["mean_share"], label=scenario)
        ax.set_title(family, fontsize=14)
        ax.set_xlabel("Episode (log10 scale)" if use_log_x else "Episode")
        ax.set_ylabel("Share")
        if use_log_x:
            ax.set_xscale("log")

    for idx in range(len(families_present), total_axes):
        axes_list[idx].set_visible(False)

    if axes_list:
        handles, labels = axes_list[0].get_legend_handles_labels()
        if labels:
            fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_final_action_family_shares(
    frame: pd.DataFrame,
    *,
    families: Sequence[str] | None = None,
    family_rules: Mapping[str, Callable[[str], bool]] | None = None,
    tail_length: int = 1,
    figsize: tuple[int, int] = (10, 5),
    save_path: Path | str | None = None,
) -> None:
    """Bar plot summarising action-family share over the final ``tail_length`` episodes."""

    agg = _aggregate_action_families(frame, families=families, family_rules=family_rules)
    summary = (
        agg.sort_values("episode")
        .groupby(["scenario", "seed", "family"])
        .tail(tail_length)
        .groupby(["scenario", "family"], as_index=False)["share"]
        .mean(numeric_only=True)
    )
    plt.figure(figsize=figsize)
    sns.barplot(data=summary, x="scenario", y="share", hue="family")
    plt.title(f"Action family share (avg last {tail_length} episodes)")
    plt.ylabel("Share")
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


__all__ = [
    "load_experiment",
    "runs_to_frame",
    "plot_metric_over_episodes",
    "plot_final_metric",
    "plot_action_family_shares",
    "plot_final_action_family_shares",
]
