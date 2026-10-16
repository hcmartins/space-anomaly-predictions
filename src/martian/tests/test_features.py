"""Tests for shared causal feature engineering: lags, trailing stats,
first difference, slope, persistence-run counter, spatial-neighbour stats
(same sol only), and the structural target-prefix leakage guard."""
import numpy as np
import pandas as pd
import pytest

from martian.preprocessing.grid import GridDefinition, assign_cells
from martian.preprocessing.features import FEATURE_PREFIX, build_features, feature_column_names, load_features_config


@pytest.fixture
def grid() -> GridDefinition:
    return GridDefinition.from_resolution(5.0)


def _series_df(sols, dust, *, cell_id=0, lat=10.0, lon=100.0, atmospheric_anomaly=None) -> pd.DataFrame:
    n = len(sols)
    return pd.DataFrame(
        {
            "latitude": [lat] * n,
            "longitude": [lon] * n,
            "mars_year": [1] * n,
            "sol": sols,
            "solar_longitude_ls": [45.0] * n,
            "local_true_solar_time": [14.0] * n,
            "dust_optical_depth": dust,
            "dust_optical_depth_z": [0.0] * n,
            "surface_temperature_k": [200.0] * n,
            "surface_temperature_k_z": [0.0] * n,
            "atmospheric_anomaly": atmospheric_anomaly if atmospheric_anomaly is not None else [False] * n,
            "is_observed": [True] * n,
            "interpolation_flag": [False] * n,
            "low_support_flag": [False] * n,
            "observation_count": [1] * n,
        }
    )


def test_lags_are_exact_and_causal(grid):
    df = _series_df([0, 1, 2, 3], [10.0, 20.0, 30.0, 40.0])
    df = assign_cells(df, grid)
    out = build_features(df, grid)
    row3 = out.set_index("sol").loc[3]
    assert row3["feat_dust_optical_depth_lag1"] == 30.0
    assert row3["feat_dust_optical_depth_lag2"] == 20.0
    # sol 0 has no history at all
    row0 = out.set_index("sol").loc[0]
    assert np.isnan(row0["feat_dust_optical_depth_lag1"])


def test_lag3_present_when_history_exists(grid):
    df = _series_df([0, 1, 2, 3], [10.0, 20.0, 30.0, 40.0])
    df = assign_cells(df, grid)
    out = build_features(df, grid)
    row3 = out.set_index("sol").loc[3]
    assert row3["feat_dust_optical_depth_lag3"] == 10.0


def test_future_perturbation_never_changes_past_feature(grid):
    df1 = _series_df([0, 1, 2, 3], [10.0, 20.0, 30.0, 40.0])
    df1 = assign_cells(df1, grid)
    df2 = _series_df([0, 1, 2, 3], [10.0, 20.0, 30.0, 999.0])  # perturb only the LAST (future-most) value
    df2 = assign_cells(df2, grid)

    out1 = build_features(df1, grid)
    out2 = build_features(df2, grid)

    feat_cols = feature_column_names(out1)
    pd.testing.assert_frame_equal(
        out1.loc[out1["sol"] < 3, feat_cols].reset_index(drop=True),
        out2.loc[out2["sol"] < 3, feat_cols].reset_index(drop=True),
    )


def test_trailing_stats_only_use_past_values(grid):
    # Explicit no-threshold config: this test's purpose is the CAUSALITY
    # invariant (past values only), independent of the 2026-08-13
    # window-completeness default (0.6) -- see the dedicated
    # test_window_completeness_threshold_* tests below for that mechanism.
    df = _series_df([0, 1, 2], [10.0, 20.0, 30.0])
    df = assign_cells(df, grid)
    config = load_features_config()
    config["min_window_completeness_frac"] = None
    out = build_features(df, grid, config=config)
    row2 = out.set_index("sol").loc[2]
    # trailing mean at sol=2 should be mean(sol0, sol1) = 15.0, NOT include sol=2 itself
    assert row2["feat_dust_optical_depth_trailing5_mean"] == pytest.approx(15.0)


def test_window_completeness_threshold_suppresses_sparse_trailing_stats(grid):
    # window=5, only 2/5 slots ever have history by sol=2 -> 40% complete,
    # below a 60% threshold -> trailing stats must be withheld (NaN), not
    # computed from the 2 available points as the no-threshold default would.
    df = _series_df([0, 1, 2], [10.0, 20.0, 30.0])
    df = assign_cells(df, grid)
    config = load_features_config()
    config["min_window_completeness_frac"] = 0.6
    out = build_features(df, grid, config=config)
    row2 = out.set_index("sol").loc[2]
    assert np.isnan(row2["feat_dust_optical_depth_trailing5_mean"])


def test_window_completeness_threshold_allows_sufficiently_complete_window(grid):
    # window=5, sols 0-4 all present -> anchor at sol=4 has 4/5 = 80% complete
    # history (sols 0-3) -> meets an 0.6 threshold, trailing stats computed.
    df = _series_df([0, 1, 2, 3, 4], [10.0, 20.0, 30.0, 40.0, 50.0])
    df = assign_cells(df, grid)
    config = load_features_config()
    config["min_window_completeness_frac"] = 0.6
    out = build_features(df, grid, config=config)
    row4 = out.set_index("sol").loc[4]
    assert row4["feat_dust_optical_depth_trailing5_mean"] == pytest.approx(25.0)  # mean(10,20,30,40)


def test_shipped_default_completeness_threshold_matches_sensitivity_selection(grid):
    # The shipped configs/features.yaml primary (min_window_completeness_frac:
    # 0.6) is the real, sensitivity-selected value (2026-08-13; see that
    # file's own note and CHAPTER_4_PARAMETER_VALUES.md Section 11 item 7),
    # not an arbitrary default -- and `None` (no threshold) must still be a
    # valid, explicitly-settable option (test_trailing_stats_only_use_past_values
    # above exercises it), not removed.
    config = load_features_config()
    assert config["min_window_completeness_frac"] == pytest.approx(0.6)
    assert None in config["sensitivity_min_window_completeness_frac"]


def test_first_difference_and_slope(grid):
    df = _series_df([0, 1, 2], [10.0, 20.0, 30.0])
    df = assign_cells(df, grid)
    out = build_features(df, grid)
    row2 = out.set_index("sol").loc[2]
    assert row2["feat_dust_optical_depth_diff1"] == pytest.approx(10.0)  # 30 - 20
    assert row2["feat_dust_optical_depth_slope5"] == pytest.approx(10.0)  # +10/sol


def test_persistence_run_counter(grid):
    # anomaly pattern: F,F,T,T,T,F,T (sols 0..6)
    df = _series_df(
        list(range(7)), [1.0] * 7,
        atmospheric_anomaly=[False, False, True, True, True, False, True],
    )
    df = assign_cells(df, grid)
    out = build_features(df, grid)
    by_sol = out.set_index("sol")["feat_persistence_run_sols"]
    assert by_sol.loc[2] == 0  # sol1 (immediately prior) is False
    assert by_sol.loc[3] == 1  # sol2 True, sol1 False-stop
    assert by_sol.loc[4] == 2  # sol3,sol2 True, sol1 False-stop
    assert by_sol.loc[6] == 0  # sol5 is False


def test_spatial_neighbour_stats_same_sol_only(grid):
    # Two adjacent cells, cell A has a much higher dust value than cell B at
    # sol 0; cell B's own neighbour-mean feature at sol 0 must reflect A's
    # CURRENT (same sol) value, not any different-sol value.
    df_a = _series_df([0], [100.0], cell_id=0, lat=10.0, lon=100.0)
    df_b = _series_df([0], [1.0], cell_id=1, lat=10.0, lon=105.0)  # adjacent cell (lon +5deg)
    df = pd.concat([df_a, df_b], ignore_index=True)
    df = assign_cells(df, grid)
    out = build_features(df, grid)

    b_row = out.loc[out["longitude"] == 105.0].iloc[0]
    assert b_row["feat_dust_optical_depth_neighbour_mean"] == pytest.approx(100.0)


def test_spatial_neighbour_stats_never_reads_a_different_sol(grid):
    # Cell A has an anomalously large value at a LATER sol (sol=5) only;
    # cell B's neighbour stats at sol=0 must NOT see it (no matching same-sol
    # neighbour data exists at sol=0 for cell A).
    df_a = _series_df([5], [999.0], cell_id=0, lat=10.0, lon=100.0)
    df_b = _series_df([0], [1.0], cell_id=1, lat=10.0, lon=105.0)
    df = pd.concat([df_a, df_b], ignore_index=True)
    df = assign_cells(df, grid)
    out = build_features(df, grid)

    b_row = out.loc[out["longitude"] == 105.0].iloc[0]
    assert np.isnan(b_row["feat_dust_optical_depth_neighbour_mean"])


def test_no_feature_column_ever_uses_target_prefix(grid):
    df = _series_df([0, 1, 2], [10.0, 20.0, 30.0])
    df = assign_cells(df, grid)
    out = build_features(df, grid)
    assert not any(c.startswith("target_") for c in out.columns)


def test_config_loaded_at_runtime(tmp_path, grid):
    default_config = load_features_config()
    custom_config = dict(default_config)
    custom_config["lags_sols"] = [1]
    custom_path = tmp_path / "custom_features.yaml"
    import yaml

    custom_path.write_text(yaml.safe_dump(custom_config), encoding="utf-8")
    loaded = load_features_config(custom_path)
    assert loaded["lags_sols"] == [1]

    df = _series_df([0, 1, 2, 3], [10.0, 20.0, 30.0, 40.0])
    df = assign_cells(df, grid)
    out_default = build_features(df, grid, config=default_config)
    out_custom = build_features(df, grid, config=loaded)
    assert "feat_dust_optical_depth_lag2" in out_default.columns
    assert "feat_dust_optical_depth_lag2" not in out_custom.columns
