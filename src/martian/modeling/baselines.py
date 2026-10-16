"""Baseline hierarchy: climatology, persistence, seasonal mean.

Each baseline is a small class implementing `.fit(train_df, y_train)` /
`.predict(apply_df)` / `.predict_proba(apply_df)` (for classification tasks)
so it can flow through the SAME iterate-fit-evaluate-record loop as a
classical model in modeling/train.py -- `uses_raw_dataframe = True` tells
that loop to pass the raw (un-imputed, un-numpy-ified) dataframe rather than
the imputed numeric feature matrix, since these baselines read specific
named columns (e.g. "predict the current value", "predict this season's
mean") rather than a generic feature vector.

All three read their configuration (which column to persist from, which
mode to use) from the calling task's own YAML config -- never hardcoded --
so "own configuration" (task requirement) extends to baselines too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from labeling.reporting import assign_mars_season


class PersistenceBaseline:
    """Predict the current sol's own value/state as the future value/state."""

    uses_raw_dataframe = True

    def __init__(self, *, source_column: str):
        self.source_column = source_column

    def get_params(self) -> dict:
        return {"source_column": self.source_column}

    def fit(self, train_df: pd.DataFrame, y_train) -> "PersistenceBaseline":
        return self  # nothing to learn -- purely a lookup of the current value

    def predict(self, apply_df: pd.DataFrame) -> np.ndarray:
        return apply_df[self.source_column].to_numpy(dtype=float)

    def predict_proba(self, apply_df: pd.DataFrame) -> np.ndarray:
        p = np.clip(self.predict(apply_df), 0.0, 1.0)
        return np.column_stack([1 - p, p])


class SeasonalMeanBaseline:
    """Predict the per-Ls-season mean of the target, fit on TRAIN ONLY.
    For a classification target (0/1), the "mean" IS the season's positive
    rate, so this class is reused unchanged for Task C."""

    uses_raw_dataframe = True

    def __init__(self, *, ls_column: str = "solar_longitude_ls"):
        self.ls_column = ls_column
        self.season_means_: pd.Series | None = None
        self.global_mean_: float | None = None

    def get_params(self) -> dict:
        return {"ls_column": self.ls_column}

    def fit(self, train_df: pd.DataFrame, y_train) -> "SeasonalMeanBaseline":
        season = assign_mars_season(train_df[self.ls_column])
        self.season_means_ = pd.Series(np.asarray(y_train, dtype=float), index=season).groupby(level=0).mean()
        self.global_mean_ = float(np.mean(y_train))
        return self

    def predict(self, apply_df: pd.DataFrame) -> np.ndarray:
        season = assign_mars_season(apply_df[self.ls_column])
        return np.array([self.season_means_.get(s, self.global_mean_) for s in season], dtype=float)

    def predict_proba(self, apply_df: pd.DataFrame) -> np.ndarray:
        p = np.clip(self.predict(apply_df), 0.0, 1.0)
        return np.column_stack([1 - p, p])


class ClimatologyBaseline:
    """Three modes, selected by the calling task's config:
      - "column": predict a named climatology-provider column directly
        (Task A: `dust_optical_depth_mu`).
      - "zero": predict the climatology-implied value of a Z-SCORE target,
        which is exactly 0 by construction (Task B).
      - "train_mean": predict the constant train-set base rate/mean --
        the standard "climatological average likelihood" baseline for a
        binary target (Task C).
    """

    uses_raw_dataframe = True

    def __init__(self, *, mode: str, column: str | None = None):
        if mode not in ("column", "zero", "train_mean"):
            raise ValueError(f"Unknown ClimatologyBaseline mode {mode!r}")
        if mode == "column" and not column:
            raise ValueError("ClimatologyBaseline mode='column' requires a `column` name.")
        self.mode = mode
        self.column = column
        self.constant_: float | None = None

    def get_params(self) -> dict:
        return {"mode": self.mode, "column": self.column}

    def fit(self, train_df: pd.DataFrame, y_train) -> "ClimatologyBaseline":
        if self.mode == "train_mean":
            self.constant_ = float(np.mean(y_train))
        return self

    def predict(self, apply_df: pd.DataFrame) -> np.ndarray:
        if self.mode == "column":
            return apply_df[self.column].to_numpy(dtype=float)
        if self.mode == "zero":
            return np.zeros(len(apply_df), dtype=float)
        return np.full(len(apply_df), self.constant_, dtype=float)

    def predict_proba(self, apply_df: pd.DataFrame) -> np.ndarray:
        p = np.clip(self.predict(apply_df), 0.0, 1.0)
        return np.column_stack([1 - p, p])


BASELINE_FACTORIES = {
    "climatology": ClimatologyBaseline,
    "persistence": PersistenceBaseline,
    "seasonal_mean": SeasonalMeanBaseline,
}


def build_baseline(name: str, hyperparameters: dict):
    if name not in BASELINE_FACTORIES:
        raise ValueError(f"Unknown baseline {name!r}. Registered baselines: {sorted(BASELINE_FACTORIES)}.")
    return BASELINE_FACTORIES[name](**hyperparameters)
