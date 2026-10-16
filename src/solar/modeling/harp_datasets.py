"""Per-HARP equivalent of modeling/datasets.py::build_flare_forecast_dataset
-- joins per-HARP features (preprocessing/features.py, applied per HARP
group) with per-HARP labels (labeling/harp_catalogue_labels.py), keyed on
(HARPNUM, time_utc) instead of time_utc alone.

Mirrors modeling/datasets.py's structure and principles exactly (inner
join restricting to genuinely-observed forecast origins, feature/target
column manifests never inferred by prefix-guessing downstream) -- see
that module's docstring for the full rationale, not repeated here.
"""
from __future__ import annotations

import pandas as pd

from labeling.harp_catalogue_labels import LABEL_PREFIX
from modeling.datasets import TARGET_COLUMNS
from preprocessing.features import feature_column_names, load_features_config

_QC_METADATA_COLUMNS = ["is_observed", "interpolation_flag", "gap_length_steps"]

# Deliberately NOT modeling.datasets.LABEL_METADATA_COLUMNS -- that lists
# the disk-integrated pipeline's column names
# (label_max_flare_class_in_horizon, label_coverage_fraction), which
# harp_catalogue_labels.py does not produce (it has no GOES-gap-based
# coverage fraction; its metadata column is named
# label_max_flare_class_in_harp_horizon to avoid ever being confused with
# the disk-integrated one if both pipelines' outputs are ever compared
# side by side).
HARP_LABEL_METADATA_COLUMNS = [
    f"{LABEL_PREFIX}max_flare_class_in_harp_horizon",
    f"{LABEL_PREFIX}horizon_hours",
]


def build_harp_flare_forecast_dataset(
    features_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    *,
    time_col: str = "time_utc",
    harp_col: str = "HARPNUM",
    features_config: dict | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Inner-joins features_df with labeled_df's label_* columns on
    (harp_col, time_col). Returns (dataset, feature_manifest,
    target_columns) -- same contract as
    modeling.datasets.build_flare_forecast_dataset, one join key wider."""
    features_config = features_config or load_features_config()
    for df, name in ((features_df, "features_df"), (labeled_df, "labeled_df")):
        for col in (time_col, harp_col):
            if col not in df.columns:
                raise ValueError(f"{name} has no {col!r} column")

    label_cols = [c for c in labeled_df.columns if c.startswith(LABEL_PREFIX)]
    overlapping = set(features_df.columns) & set(label_cols)
    if overlapping:
        raise ValueError(
            f"features_df already contains column(s) {sorted(overlapping)}, which collide "
            f"with labeled_df's label_-prefixed columns -- preprocessing/features.py must "
            f"never produce a label_-prefixed column; this indicates a caller bug, not "
            f"something to silently resolve."
        )

    raw_current_state_cols = [v for v in features_config["variables"] if v in features_df.columns]
    feature_manifest = feature_column_names(features_df) + raw_current_state_cols
    qc_cols = [c for c in _QC_METADATA_COLUMNS if c in features_df.columns]
    harp_label_meta_cols = [c for c in HARP_LABEL_METADATA_COLUMNS if c in labeled_df.columns]
    target_cols = [c for c in TARGET_COLUMNS if c in labeled_df.columns]

    # Trim to exactly the columns needed BEFORE merging, and merge directly
    # into the final result -- no separate final .copy()-of-a-subset
    # afterward. The original 3-step version (copy full width -> merge ->
    # copy a subset of the merge result again) hit a real
    # ArrayMemoryError at full per-HARP scale (~3.9M rows, ~260 columns,
    # trying to allocate a THIRD ~7GB array at the final .copy()) -- see
    # planning/FLARE_CATALOGUE_DOCUMENTATION.md. This keeps at most 2 large
    # objects (the trimmed `left` and the final merged `dataset`) alive at
    # once instead of 3. right's column list is harp_label_meta_cols +
    # target_cols (the curated TARGET_COLUMNS list), NOT the broader
    # `label_cols` computed above -- label_cols is only used for the
    # feature/label collision check; using it here too would duplicate
    # label_horizon_hours/label_max_flare_class_in_harp_horizon (both
    # match the label_ prefix AND are already in harp_label_meta_cols),
    # producing a same-named-column DataFrame instead of a Series
    # downstream (caught by a real run: `assign_split_with_embargo`
    # crashed with "'DataFrame' object has no attribute 'unique'").
    left = features_df[[harp_col, time_col] + qc_cols + feature_manifest].copy()
    left[time_col] = pd.to_datetime(left[time_col], utc=True)
    right = labeled_df[[harp_col, time_col] + harp_label_meta_cols + target_cols].copy()
    right[time_col] = pd.to_datetime(right[time_col], utc=True)

    dataset = left.merge(right, on=[harp_col, time_col], how="inner")

    return dataset, feature_manifest, target_cols
