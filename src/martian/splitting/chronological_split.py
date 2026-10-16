"""Leak-audited chronological train/val/test splitting.

Reads `configs/splits.yaml` sol boundaries (train: MY1 sols 1-600; val: MY1
sols 601-668; test: MY2 sols 1-668 -- see that file). Assigns every cell-sol
row to a split, asserts no storm episode spans a split boundary, and
provides train-only standardisation-statistic fitting so no val/test
information ever leaks into a fitted scaler. Writes a split manifest (row
counts, sol ranges, positive-label prevalence per split) for
tests/test_chronological_split.py to consume.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_sources.schema import CONFIG_DIR

SPLIT_NAMES = ("train", "val", "test")


class EpisodeCrossesSplitBoundaryError(ValueError):
    """Raised when a storm episode's member cell-sols span more than one
    split -- this would leak future (or past) episode information across
    the boundary a model is supposed to respect."""


def load_splits_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "splits.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def assign_split(
    df: pd.DataFrame,
    splits_config: dict | None = None,
    *,
    mars_year_col: str = "mars_year",
    sol_col: str = "sol",
) -> pd.Series:
    splits_config = splits_config or load_splits_config()
    my = df[mars_year_col].to_numpy()
    sol = df[sol_col].to_numpy()

    train_lo, train_hi = splits_config["train_sol_range"]
    val_lo, val_hi = splits_config["val_sol_range"]
    test_lo, test_hi = splits_config["test_sol_range"]

    is_train = (my == splits_config["train_mars_year"]) & (sol >= train_lo) & (sol <= train_hi)
    is_val = (my == splits_config["val_mars_year"]) & (sol >= val_lo) & (sol <= val_hi)
    is_test = (my == splits_config["test_mars_year"]) & (sol >= test_lo) & (sol <= test_hi)

    out = np.full(len(df), "unassigned", dtype=object)
    out[is_train] = "train"
    out[is_val] = "val"
    out[is_test] = "test"
    return pd.Series(out, index=df.index, name="split")


def assert_no_episode_crosses_split_boundary(
    episode_catalogue: pd.DataFrame,
    splits_config: dict | None = None,
) -> None:
    """Given an episode catalogue (mars_year, onset_sol, end_sol columns --
    see labeling/connected_components.py::summarize_episodes), raise
    EpisodeCrossesSplitBoundaryError if any episode's onset and end sol fall
    in different splits."""
    crossing = find_episodes_crossing_split_boundary(episode_catalogue, splits_config)
    if len(crossing):
        raise EpisodeCrossesSplitBoundaryError(
            f"{len(crossing)} episode(s) span a split boundary (onset and end sol "
            f"fall in different splits): episode_id(s) "
            f"{crossing['episode_id'].tolist() if 'episode_id' in crossing.columns else crossing.index.tolist()}"
        )


def find_episodes_crossing_split_boundary(
    episode_catalogue: pd.DataFrame,
    splits_config: dict | None = None,
) -> pd.DataFrame:
    """Same check as `assert_no_episode_crosses_split_boundary`, but RETURNS
    the crossing subset instead of raising -- lets a caller (e.g.
    modeling/run_all_tasks_demo.py) exclude those episodes' member cell-sols
    before building modeling datasets, rather than simply crashing. This is
    the real, expected outcome in a genuine run: a storm episode straddling
    the train/val (or val/test) sol boundary is not a bug, but it DOES mean
    a train row's forward-looking horizon target could otherwise depend on a
    val-partition storm_episode value -- exactly the leakage this function
    exists to catch."""
    splits_config = splits_config or load_splits_config()
    if episode_catalogue.empty:
        return episode_catalogue.copy()

    onset_tag = assign_split(
        episode_catalogue.rename(columns={"onset_sol": "sol"}), splits_config
    )
    end_tag = assign_split(
        episode_catalogue.rename(columns={"end_sol": "sol"}), splits_config
    )
    return episode_catalogue.loc[onset_tag.to_numpy() != end_tag.to_numpy()]


def fit_standardization_stats(train_df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    """Mean/std computed ONLY from `train_df` -- caller must have already
    filtered to the train split. Returns {column: {"mean": ..., "std": ...}}."""
    stats = {}
    for col in columns:
        values = train_df[col].to_numpy(dtype=float)
        values = values[~np.isnan(values)]
        mean = float(np.mean(values)) if len(values) else 0.0
        std = float(np.std(values)) if len(values) else 1.0
        stats[col] = {"mean": mean, "std": std if std > 1e-12 else 1.0}
    return stats


def apply_standardization(df: pd.DataFrame, stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    out = df.copy()
    for col, s in stats.items():
        if col in out.columns:
            out[col] = (out[col] - s["mean"]) / s["std"]
    return out


def write_split_manifest(
    path: Path,
    df_with_split: pd.DataFrame,
    *,
    split_col: str = "split",
    label_cols: list[str] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    label_cols = label_cols or []

    manifest: dict = {"splits": {}}
    for split_name in (*SPLIT_NAMES, "unassigned"):
        subset = df_with_split.loc[df_with_split[split_col] == split_name]
        entry = {"n_rows": int(len(subset))}
        if len(subset):
            entry["mars_years"] = sorted(int(v) for v in subset["mars_year"].unique())
            entry["sol_min"] = int(subset["sol"].min())
            entry["sol_max"] = int(subset["sol"].max())
        for col in label_cols:
            if col in subset.columns and len(subset):
                entry[f"{col}_prevalence"] = float(subset[col].mean())
        manifest["splits"][split_name] = entry

    path.write_text(json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8")
    return path
