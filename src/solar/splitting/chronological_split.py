"""Leak-audited chronological train/val/test splitting for the
disk-integrated, SHARP-cadence timeline.

Reads `configs/splits.yaml`'s UTC date boundaries (train/val/test, strictly
time-ordered, non-overlapping -- validated at load time). Assigns every row
to a split by its `time_utc` timestamp, detects rows whose forward-looking
label horizon (`labeling/flare_labels.py`) reaches past its own split's
boundary into the next split (a purged/embargoed-time-series-CV concern,
not a feature-input leak -- `preprocessing/` is strictly causal, but a
label's horizon is deliberately not), and provides train-only
standardization-statistic fitting so no val/test information ever leaks
into a fitted scaler. Mirrors `martian/splitting/chronological_split.py`'s
design; see that module and `configs/splits.yaml`'s header for the
rationale behind each piece.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_sources.schema import CONFIG_DIR

SPLIT_NAMES = ("train", "val", "test")


class LabelHorizonCrossesSplitBoundaryError(ValueError):
    """Raised when a row's label horizon (time_utc, time_utc + horizon]
    reaches into a DIFFERENT split than the row itself -- this would let a
    train (or val) row's label be informed by observations that fall inside
    the next split, blurring the independence the split boundary is meant
    to guarantee."""


def load_splits_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "splits.yaml"
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    _validate_splits_config(config)
    return config


def _validate_splits_config(config: dict) -> None:
    train_start, train_end = pd.Timestamp(config["train_start"]), pd.Timestamp(config["train_end"])
    val_start, val_end = pd.Timestamp(config["val_start"]), pd.Timestamp(config["val_end"])
    test_start, test_end = pd.Timestamp(config["test_start"]), pd.Timestamp(config["test_end"])

    if not (train_start <= train_end < val_start <= val_end < test_start <= test_end):
        raise ValueError(
            "configs/splits.yaml: train/val/test date ranges must be non-overlapping "
            "and strictly chronologically ordered (train entirely before val entirely "
            f"before test); got train=[{train_start},{train_end}], val=[{val_start},{val_end}], "
            f"test=[{test_start},{test_end}]."
        )


def assign_split(
    df: pd.DataFrame,
    splits_config: dict | None = None,
    *,
    time_col: str = "time_utc",
) -> pd.Series:
    """Assigns each row to "train"/"val"/"test" by its time_col timestamp
    (UTC date ranges, both ends inclusive), or "unassigned" if it falls
    outside all three (e.g. before train_start or after test_end)."""
    splits_config = splits_config or load_splits_config()
    t = pd.to_datetime(df[time_col], utc=True)

    train_start = pd.Timestamp(splits_config["train_start"], tz="UTC")
    train_end = pd.Timestamp(splits_config["train_end"], tz="UTC") + pd.Timedelta(days=1)
    val_start = pd.Timestamp(splits_config["val_start"], tz="UTC")
    val_end = pd.Timestamp(splits_config["val_end"], tz="UTC") + pd.Timedelta(days=1)
    test_start = pd.Timestamp(splits_config["test_start"], tz="UTC")
    test_end = pd.Timestamp(splits_config["test_end"], tz="UTC") + pd.Timedelta(days=1)

    is_train = (t >= train_start) & (t < train_end)
    is_val = (t >= val_start) & (t < val_end)
    is_test = (t >= test_start) & (t < test_end)

    out = np.full(len(df), "unassigned", dtype=object)
    out[is_train.to_numpy()] = "train"
    out[is_val.to_numpy()] = "val"
    out[is_test.to_numpy()] = "test"
    return pd.Series(out, index=df.index, name="split")


def find_rows_with_label_horizon_crossing_split_boundary(
    df_with_split: pd.DataFrame,
    splits_config: dict | None = None,
    *,
    horizon_hours: float | None = None,
    time_col: str = "time_utc",
    split_col: str = "split",
) -> pd.DataFrame:
    """Returns the subset of df_with_split whose (time_utc, time_utc +
    horizon] window's END falls in a DIFFERENT split than the row's own
    (already-assigned) split. Rows already "unassigned" are never flagged --
    only a genuine train->val, train->test, or val->test crossing counts.
    Does not raise; lets a caller (e.g. a modeling dataset-builder) exclude
    these rows via `apply_embargo` before training, rather than crashing."""
    splits_config = splits_config or load_splits_config()
    horizon = pd.Timedelta(hours=horizon_hours if horizon_hours is not None else splits_config["embargo_hours"])
    if df_with_split.empty:
        return df_with_split.copy()

    horizon_end = pd.to_datetime(df_with_split[time_col], utc=True) + horizon
    end_split = assign_split(pd.DataFrame({time_col: horizon_end}), splits_config, time_col=time_col)

    own_split = df_with_split[split_col].to_numpy()
    end_split_arr = end_split.to_numpy()
    crosses = (
        np.isin(own_split, SPLIT_NAMES)
        & np.isin(end_split_arr, SPLIT_NAMES)
        & (own_split != end_split_arr)
    )
    return df_with_split.loc[crosses]


def assert_no_label_horizon_crosses_split_boundary(
    df_with_split: pd.DataFrame,
    splits_config: dict | None = None,
    *,
    horizon_hours: float | None = None,
    time_col: str = "time_utc",
    split_col: str = "split",
) -> None:
    crossing = find_rows_with_label_horizon_crossing_split_boundary(
        df_with_split, splits_config, horizon_hours=horizon_hours, time_col=time_col, split_col=split_col
    )
    if len(crossing):
        raise LabelHorizonCrossesSplitBoundaryError(
            f"{len(crossing)} row(s) have a label horizon crossing into a different "
            f"split -- apply_embargo() must be run first. First crossing timestamp(s): "
            f"{crossing[time_col].head(5).tolist()}"
        )


def apply_embargo(
    df_with_split: pd.DataFrame,
    splits_config: dict | None = None,
    *,
    horizon_hours: float | None = None,
    time_col: str = "time_utc",
    split_col: str = "split",
) -> pd.Series:
    """Returns a COPY of df_with_split[split_col] with every row whose label
    horizon crosses a split boundary reassigned to "embargo" (excluded from
    train/val/test entirely) -- the actual fix a caller applies before
    building a modeling dataset, so
    `assert_no_label_horizon_crosses_split_boundary` passes afterward."""
    crossing = find_rows_with_label_horizon_crossing_split_boundary(
        df_with_split, splits_config, horizon_hours=horizon_hours, time_col=time_col, split_col=split_col
    )
    out = df_with_split[split_col].copy()
    out.loc[crossing.index] = "embargo"
    return out


def fit_standardization_stats(train_df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    """Mean/std computed ONLY from train_df -- caller must have already
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
    time_col: str = "time_utc",
    label_cols: list[str] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    label_cols = label_cols or []

    manifest: dict = {"splits": {}}
    for split_name in (*SPLIT_NAMES, "embargo", "unassigned"):
        subset = df_with_split.loc[df_with_split[split_col] == split_name]
        entry = {"n_rows": int(len(subset))}
        if len(subset):
            t = pd.to_datetime(subset[time_col], utc=True)
            entry["time_utc_min"] = t.min().isoformat()
            entry["time_utc_max"] = t.max().isoformat()
        for col in label_cols:
            if col in subset.columns and len(subset):
                defined = subset[col].notna()
                entry[f"{col}_prevalence"] = float(subset.loc[defined, col].mean()) if defined.any() else None
        manifest["splits"][split_name] = entry

    path.write_text(json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8")
    return path
