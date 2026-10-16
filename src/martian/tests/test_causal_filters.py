"""Tests proving the causal Savitzky-Golay filter never uses future
information, respects train/val/test boundaries, never touches labels, is
structurally separated from the descriptive-only centred variant, and reads
its parameters from config at runtime rather than hardcoding them."""
import numpy as np
import pandas as pd
import pytest
import yaml

from martian.preprocessing.causal_filters import (
    CAUSAL_COLUMN_SUFFIX,
    CENTERED_COLUMN_SUFFIX,
    DescriptiveOnlyColumnError,
    apply_causal_savgol,
    apply_centered_savgol_for_descriptive_comparison_only,
    assert_feature_safe,
    load_causal_filter_config,
)


def _make_df(sols, values, *, cell_id=0, mars_year=1, extra_cols=None) -> pd.DataFrame:
    n = len(sols)
    row = {
        "cell_id": [cell_id] * n,
        "mars_year": [mars_year] * n,
        "sol": sols,
        "dust_optical_depth": values,
    }
    if extra_cols:
        row.update(extra_cols)
    return pd.DataFrame(row)


def test_future_perturbation_cannot_alter_past_causal_output():
    sols = list(range(12))
    rng = np.random.default_rng(0)
    values = list(rng.normal(0.2, 0.02, size=12))
    df1 = _make_df(sols, values)
    perturbed = list(values)
    perturbed[-1] = perturbed[-1] + 100.0  # perturb only the LAST (future-most) point
    df2 = _make_df(sols, perturbed)

    out1 = apply_causal_savgol(df1)
    out2 = apply_causal_savgol(df2)

    col = f"dust_optical_depth{CAUSAL_COLUMN_SUFFIX}"
    # every row EXCEPT the perturbed one itself must be unaffected
    pd.testing.assert_series_equal(
        out1.loc[out1.index[:-1], col], out2.loc[out2.index[:-1], col]
    )
    # the perturbed row's own output legitimately changes (it now includes
    # the perturbed value as the trailing point of its own window)
    assert out1.loc[out1.index[-1], col] != out2.loc[out2.index[-1], col]


def test_filter_never_crosses_train_val_test_boundary():
    # Mirrors configs/splits.yaml's boundaries: MY sols 1-600 train,
    # sols 601-668 val, next MY test.
    rng = np.random.default_rng(1)
    train_sols = list(range(1, 30))          # small stand-in slice of "train"
    val_sols = list(range(601, 615))          # small stand-in slice of "val"
    test_sols = list(range(1, 15))            # MY2 "test", different mars_year

    train_df = _make_df(train_sols, list(rng.normal(0.2, 0.02, size=len(train_sols))), mars_year=1)
    val_df = _make_df(val_sols, list(rng.normal(0.2, 0.02, size=len(val_sols))), mars_year=1)
    test_df = _make_df(test_sols, list(rng.normal(0.2, 0.02, size=len(test_sols))), mars_year=2)

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    out_train_val_only = apply_causal_savgol(train_val_df)
    out_full = apply_causal_savgol(full_df)

    col = f"dust_optical_depth{CAUSAL_COLUMN_SUFFIX}"
    merged = out_train_val_only[["sol", "mars_year", col]].merge(
        out_full[["sol", "mars_year", col]], on=["sol", "mars_year"], suffixes=("_before", "_after")
    )
    # Appending test (MY2) rows must not change ANY train/val (MY1) causal
    # output -- proves the test partition, which lives entirely in a later
    # mars_year, cannot leak backward across the boundary.
    pd.testing.assert_series_equal(
        merged[f"{col}_before"], merged[f"{col}_after"], check_names=False
    )


def test_labels_are_never_touched_by_the_filter():
    df = _make_df(list(range(10)), list(np.linspace(0.1, 0.3, 10)), extra_cols={"label": list(range(10))})
    out = apply_causal_savgol(df)
    assert out["label"].tolist() == df["label"].tolist()
    assert f"label{CAUSAL_COLUMN_SUFFIX}" not in out.columns


def test_causal_and_centered_are_structurally_separated():
    sols = list(range(12))
    rng = np.random.default_rng(2)
    values = list(rng.normal(0.2, 0.05, size=12))
    df = _make_df(sols, values)

    causal_out = apply_causal_savgol(df)
    centered_out = apply_centered_savgol_for_descriptive_comparison_only(df)

    causal_col = f"dust_optical_depth{CAUSAL_COLUMN_SUFFIX}"
    centered_col = f"dust_optical_depth{CENTERED_COLUMN_SUFFIX}"

    assert causal_col in causal_out.columns
    assert centered_col not in causal_out.columns
    assert centered_col in centered_out.columns
    assert causal_col not in centered_out.columns

    # different coefficient positions on noisy (non-polynomial) data must give
    # genuinely different values, not aliases of the same computation
    joint = causal_out[["sol", causal_col]].merge(
        centered_out[["sol", centered_col]], on="sol"
    ).dropna()
    assert len(joint) > 0
    assert not np.allclose(joint[causal_col], joint[centered_col])


def test_assert_feature_safe_rejects_centered_column_only():
    with pytest.raises(DescriptiveOnlyColumnError):
        assert_feature_safe(["dust_optical_depth", f"dust_optical_depth{CENTERED_COLUMN_SUFFIX}"])
    assert_feature_safe(["dust_optical_depth", f"dust_optical_depth{CAUSAL_COLUMN_SUFFIX}"])  # no raise


def test_config_parameters_loaded_at_runtime(tmp_path):
    default_config = load_causal_filter_config()
    assert default_config["window_length"] == 7
    assert default_config["polyorder"] == 2

    custom_path = tmp_path / "custom_causal_filter.yaml"
    custom_config = dict(default_config)
    custom_config["window_length"] = 5
    custom_config["polyorder"] = 2
    custom_path.write_text(yaml.safe_dump(custom_config), encoding="utf-8")
    loaded = load_causal_filter_config(custom_path)
    assert loaded["window_length"] == 5

    df = _make_df(list(range(10)), list(np.linspace(0.1, 0.3, 10)))
    out_default = apply_causal_savgol(df, config=default_config)
    out_custom = apply_causal_savgol(df, config=loaded)

    col = f"dust_optical_depth{CAUSAL_COLUMN_SUFFIX}"
    # window_length=7 -> 6 leading NaNs; window_length=5 -> 4 leading NaNs.
    # This proves the parameter is actually READ and used, not hardcoded.
    assert out_default[col].isna().sum() == 6
    assert out_custom[col].isna().sum() == 4


def test_window_requires_sol_contiguity_no_look_across_gaps():
    sols = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]  # sol 8 missing
    values = [float(s) ** 2 for s in range(len(sols))]
    df = _make_df(sols, values)
    out = apply_causal_savgol(df, window_length=7, polyorder=2)
    col = f"dust_optical_depth{CAUSAL_COLUMN_SUFFIX}"
    by_sol = out.set_index("sol")[col]
    # windows spanning the sol=8 gap (sols 9..14, needing sol 8) must be NaN
    for s in [9, 10, 11, 12, 13, 14]:
        assert np.isnan(by_sol.loc[s]), f"sol {s} should be NaN (window crosses the gap)"
    # sol 15's window (9..15) is fully contiguous again -- must be valid
    assert not np.isnan(by_sol.loc[15])
