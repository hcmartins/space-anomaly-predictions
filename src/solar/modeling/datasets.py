"""Joins `preprocessing/features.py`'s feature columns with
`labeling/flare_labels.py`'s label columns into one modeling-ready dataset,
keyed on `time_utc`. This is the only place features and labels are ever
combined -- `preprocessing/` stays strictly causal and never imports
`labeling/`, and this module is the sole bridge between the two (mirrors
`martian/modeling/datasets.py`'s equivalent role there).

`features_df` is built on the TRAILING-FILLED series
(`preprocessing/missing_data.py`'s output), so it can contain rows for
timestamps that were never genuinely observed (LOCF-filled gaps).
`labeled_df` is built directly on the pre-fill aligned series
(`labeling/flare_labels.py::build_labels` takes `alignment.py`'s
`aligned` DataFrame, not the filled one), so it only ever contains
GENUINELY OBSERVED timestamps. The inner join below therefore restricts the
final dataset to rows whose FORECAST ORIGIN is a real observation --
mirroring `martian/modeling/datasets.py`'s documented principle "you can
only forecast FROM a moment you actually measured" -- while the feature
HISTORY those origin rows draw on may still include trailing-filled
context, exactly as in martian.

Label columns are UNDEFINED (NaN) on rows with insufficient forward
coverage (`labeling/flare_labels.py`'s `label_coverage_fraction` gate) --
`drop_rows_with_undefined_label` is provided so a caller building a
specific classification task remembers to explicitly drop them (never
silently train against a fabricated 0), per
`planning/LABELING_DOCUMENTATION.md`.
"""
from __future__ import annotations

import pandas as pd

from labeling.flare_labels import LABEL_PREFIX
from preprocessing.features import feature_column_names, load_features_config
from splitting.chronological_split import apply_embargo, assign_split, load_splits_config

TARGET_COLUMNS = [
    f"{LABEL_PREFIX}flare_b_or_above",
    f"{LABEL_PREFIX}flare_c_or_above",
    f"{LABEL_PREFIX}flare_m_or_above",
    f"{LABEL_PREFIX}flare_x_or_above",
    f"{LABEL_PREFIX}primary",
]

# Carried on the assembled dataset for reporting/evaluation, but never
# treated as a model feature or a training target.
LABEL_METADATA_COLUMNS = [
    f"{LABEL_PREFIX}max_flare_class_in_horizon",
    f"{LABEL_PREFIX}coverage_fraction",
    f"{LABEL_PREFIX}horizon_hours",
]

_QC_METADATA_COLUMNS = ["is_observed", "interpolation_flag", "gap_length_steps"]


def build_flare_forecast_dataset(
    features_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    *,
    time_col: str = "time_utc",
    features_config: dict | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Inner-joins features_df with labeled_df's label_* columns on
    time_col. Returns (dataset, feature_manifest, target_columns) --
    dataset carries time_col, QC/label bookkeeping metadata,
    feature_manifest columns, and target_columns; feature_manifest/
    target_columns are the exact column-name lists a training loop should
    use to slice X/y, never inferred by prefix-guessing downstream."""
    features_config = features_config or load_features_config()
    if time_col not in features_df.columns:
        raise ValueError(f"features_df has no {time_col!r} column")
    if time_col not in labeled_df.columns:
        raise ValueError(f"labeled_df has no {time_col!r} column")

    left = features_df.copy()
    left[time_col] = pd.to_datetime(left[time_col], utc=True)
    right = labeled_df.copy()
    right[time_col] = pd.to_datetime(right[time_col], utc=True)

    label_cols = [c for c in right.columns if c.startswith(LABEL_PREFIX)]
    overlapping = set(left.columns) & set(label_cols)
    if overlapping:
        raise ValueError(
            f"features_df already contains column(s) {sorted(overlapping)}, which collide "
            f"with labeled_df's label_-prefixed columns -- preprocessing/features.py must "
            f"never produce a label_-prefixed column (see its module docstring's leakage "
            f"guarantee); this indicates a caller bug, not something to silently resolve."
        )

    merged = left.merge(right[[time_col] + label_cols], on=time_col, how="inner")

    raw_current_state_cols = [v for v in features_config["variables"] if v in merged.columns]
    feature_manifest = feature_column_names(merged) + raw_current_state_cols
    target_cols = [c for c in TARGET_COLUMNS if c in merged.columns]

    id_cols = (
        [time_col]
        + [c for c in _QC_METADATA_COLUMNS if c in merged.columns]
        + [c for c in LABEL_METADATA_COLUMNS if c in merged.columns]
    )
    dataset = merged[id_cols + feature_manifest + target_cols].copy()

    return dataset, feature_manifest, target_cols


def assign_split_with_embargo(
    dataset: pd.DataFrame,
    splits_config: dict | None = None,
    *,
    horizon_hours: float | None = None,
    time_col: str = "time_utc",
) -> pd.DataFrame:
    """Returns a COPY of dataset with a `split` column: `splitting.assign_split`
    followed immediately by `splitting.apply_embargo`, so a caller never gets
    an un-embargoed split by forgetting the second call (see
    `planning/SPLITTING_DOCUMENTATION.md` for why the embargo step exists --
    a row's label horizon can reach past its own split boundary into the
    next split).

    horizon_hours defaults to the dataset's OWN `label_horizon_hours` column
    (written by `labeling/flare_labels.py`, carried through by
    `build_flare_forecast_dataset`) rather than `configs/splits.yaml`'s
    static `embargo_hours` default -- this way the embargo always matches
    the horizon actually used to build this dataset's labels, even if
    `configs/labeling.yaml`'s `forecast_horizon_hours` has since been
    changed for a later run. Raises ValueError if the column is missing (a
    non-flare-forecast dataset was passed) or carries more than one distinct
    value (the embargo mechanism assumes a single horizon per dataset;
    passing horizon_hours explicitly bypasses this check)."""
    splits_config = splits_config or load_splits_config()
    if horizon_hours is None:
        horizon_col = f"{LABEL_PREFIX}horizon_hours"
        if horizon_col not in dataset.columns:
            raise ValueError(
                f"dataset has no {horizon_col!r} column to infer horizon_hours from -- "
                f"pass horizon_hours explicitly, or build this dataset via "
                f"build_flare_forecast_dataset() first."
            )
        distinct = dataset[horizon_col].dropna().unique()
        if len(distinct) != 1:
            raise ValueError(
                f"dataset's {horizon_col!r} column is not a single constant value "
                f"(got {sorted(distinct)}) -- pass horizon_hours explicitly."
            )
        horizon_hours = float(distinct[0])

    out = dataset.copy()
    out["split"] = assign_split(out, splits_config, time_col=time_col)
    out["split"] = apply_embargo(out, splits_config, horizon_hours=horizon_hours, time_col=time_col)
    return out


def drop_rows_with_undefined_label(dataset: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Drops rows where target_col is NaN (labeling/flare_labels.py's
    coverage-gated UNDEFINED marker) -- an undefined label must never be
    treated as a negative (0.0). Returns a COPY; does not mutate dataset."""
    if target_col not in dataset.columns:
        raise KeyError(f"{target_col!r} not in dataset -- did build_flare_forecast_dataset run first?")
    return dataset.loc[dataset[target_col].notna()].reset_index(drop=True)


def summarize_targets(dataset: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
    """Per-target n_total/n_defined/n_positive/positive_rate -- a quick
    sanity check before training (e.g. catching a target that ended up
    entirely UNDEFINED after a join gone wrong)."""
    rows = []
    for col in target_cols:
        defined = dataset[col].notna()
        n_defined = int(defined.sum())
        rows.append(
            {
                "target": col,
                "n_total": len(dataset),
                "n_defined": n_defined,
                "n_positive": int(dataset.loc[defined, col].sum()) if n_defined else 0,
                "positive_rate": float(dataset.loc[defined, col].mean()) if n_defined else float("nan"),
            }
        )
    return pd.DataFrame(rows)
