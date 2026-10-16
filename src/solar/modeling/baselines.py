"""Baseline classifiers for the flare-occurrence forecasting task.

Both implement `.fit(train_df, y_train)` / `.predict(apply_df)` /
`.predict_proba(apply_df)` so they can flow through the SAME
iterate-fit-evaluate loop as a classical model in `modeling/train.py` --
`uses_raw_dataframe = True` tells that loop to pass the raw dataframe
(reading specific named columns) rather than an imputed numeric feature
matrix. Direct port of `martian/modeling/baselines.py`'s
`ClimatologyBaseline`/`PersistenceBaseline` pattern (both are fully
generic, no martian-specific coupling); `SeasonalMeanBaseline` is not
ported here -- there is no per-row Mars-season-equivalent column in this
project's dataset (Solar Cycle phase would be the analogous seasonality,
but this project has not implemented a cycle-phase feature -- future work,
not fabricated here).  1

Only `ClimatologyBaseline(mode="train_mean")` is registered (the constant
train-set positive rate) -- `martian/modeling/baselines.py`'s "column" and
"zero" modes are regression-target concepts (a climatology-provider column,
a z-score's climatological value) with no analogue for a binary
flare-occurrence target.

`ClimatologyBaseline(mode="train_mean")` and `PersistenceBaseline` are
themselves recognized baselines in the published flare-forecasting
literature (a constant climatological rate, and "recent activity predicts
near-term activity"), not just carried over from martian/ for
consistency's sake.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class PersistenceBaseline:
    """Predicts using a configured, already-computed [0, 1]-ish proxy
    column read directly from apply_df -- e.g.
    `feat_persistence_elevated_flux_run_steps` (preprocessing/features.py),
    clipped to [0, 1]: 0 if there has been no recent run of elevated flux,
    1 as soon as there has been any. A simplified, explicitly-flagged proxy
    for the literature's "persistence" baseline ("if recently active,
    predict a further flare is likely"), not a literature-calibrated model
    -- see planning/MODELING_DOCUMENTATION.md."""

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


class ClimatologyBaseline:
    """mode="train_mean": predicts the constant train-set positive rate for
    every row -- the standard "climatological average likelihood" baseline
    for a binary target."""

    uses_raw_dataframe = True

    def __init__(self, *, mode: str = "train_mean"):
        if mode != "train_mean":
            raise ValueError(
                f"Unknown ClimatologyBaseline mode {mode!r}; only 'train_mean' is registered "
                f"for this binary flare-occurrence task (see module docstring)."
            )
        self.mode = mode
        self.constant_: float | None = None

    def get_params(self) -> dict:
        return {"mode": self.mode}

    def fit(self, train_df: pd.DataFrame, y_train) -> "ClimatologyBaseline":
        self.constant_ = float(np.mean(y_train))
        return self

    def predict(self, apply_df: pd.DataFrame) -> np.ndarray:
        return np.full(len(apply_df), self.constant_, dtype=float)

    def predict_proba(self, apply_df: pd.DataFrame) -> np.ndarray:
        p = np.clip(self.predict(apply_df), 0.0, 1.0)
        return np.column_stack([1 - p, p])


BASELINE_FACTORIES = {
    "climatology": ClimatologyBaseline,
    "persistence": PersistenceBaseline,
}


def build_baseline(name: str, hyperparameters: dict):
    if name not in BASELINE_FACTORIES:
        raise ValueError(f"Unknown baseline {name!r}. Registered baselines: {sorted(BASELINE_FACTORIES)}.")
    return BASELINE_FACTORIES[name](**hyperparameters)
