"""Tests for GOES sequence preprocessing (Chapter 3.4.1, Algorithm 3.1)."""
import numpy as np
import pandas as pd

from preprocessing.goes_preprocessing import preprocess_goes_sequence


def _config(**overrides):
    config = {
        "native_cadence_minutes": 1,
        "log_variables": ["xrsb_flux_w_m2"],
        "despike_kernel_minutes": 11,
        "despike_n_sigma": 5.0,
        "max_interpolated_gap_minutes": 10,
        "smoothing_window_minutes": 1,   # no-op smoothing unless overridden
        "background_percentile": 10.0,
        "background_window_minutes": 60,
        "resample_cadence_minutes": 1,   # no-op resample unless overridden
        "resample_aggregation": "mean",
    }
    config.update(overrides)
    return config


def _series(values, start="2018-01-01T00:00:00Z", freq_minutes=1):
    times = pd.date_range(start, periods=len(values), freq=f"{freq_minutes}min", tz="UTC")
    return pd.DataFrame({"time_utc": times, "xrsb_flux_w_m2": values})


def test_despike_removes_an_injected_spike_and_interpolates_it_back():
    quiet = [1e-7] * 40
    spiked = list(quiet)
    spiked[20] = 1e-4  # a huge, obviously anomalous single-minute spike
    df = _series(spiked)
    out = preprocess_goes_sequence(df, config=_config())
    # After despike+interpolation, the spike's bg-normalized value should
    # be close to the quiet-series baseline (~0), not an enormous outlier
    # reflecting the raw 1e-4 spike.
    spike_row_value = out.loc[out["time_utc"] == pd.Timestamp("2018-01-01T00:20:00Z"), "goes_seq_xrsb_flux_w_m2"]
    assert abs(spike_row_value.iloc[0]) < 0.5


def test_despike_leaves_a_clean_series_unflagged():
    quiet = [1e-7 + 1e-9 * i for i in range(40)]  # a gentle, non-anomalous trend
    df = _series(quiet)
    out = preprocess_goes_sequence(df, config=_config())
    assert not out["goes_seq_spike_removed_any"].any()


def test_gap_interpolation_respects_max_gap():
    values = [1e-7] * 30
    df = _series(values)
    # Remove a short run (<= max_interpolated_gap_minutes) and a long run.
    df_short_gap = df.drop(index=range(10, 12)).reset_index(drop=True)   # 2-min gap
    # background_window_minutes shrunk so the rolling-quantile warm-up
    # period (unrelated to gap interpolation) finishes before row 10.
    short_gap_config = _config(despike_n_sigma=100, background_window_minutes=8)
    out_short = preprocess_goes_sequence(df_short_gap, config=short_gap_config)
    assert out_short["goes_seq_xrsb_flux_w_m2"].iloc[5:].notna().all()
    assert out_short["goes_seq_interpolated_any"].sum() == 2

    df_long_gap = df.drop(index=range(10, 25)).reset_index(drop=True)   # 15-min gap > max 10
    out_long = preprocess_goes_sequence(df_long_gap, config=_config(despike_n_sigma=100, background_window_minutes=8))
    gap_rows = out_long.loc[out_long["time_utc"].between("2018-01-01T00:10:00Z", "2018-01-01T00:24:00Z")]
    assert gap_rows["goes_seq_xrsb_flux_w_m2"].isna().any()


def test_background_normalization_hand_computed():
    # A clean step series with no spikes/gaps/smoothing (window=1 step):
    # bg_normalized should equal log10(flux) - rolling 10th-percentile of
    # log10(flux) over the trailing window, computed by hand here.
    values = [1e-7] * 30 + [1e-6] * 30
    df = _series(values)
    config = _config(background_window_minutes=10, despike_n_sigma=100)
    out = preprocess_goes_sequence(df, config=config)

    ell = np.log10(pd.Series(values))
    expected_bg = ell.rolling(10, min_periods=max(1, 10 // 4)).quantile(0.10)
    expected = (ell - expected_bg).to_numpy()

    actual = out["goes_seq_xrsb_flux_w_m2"].to_numpy()
    np.testing.assert_allclose(actual, expected, equal_nan=True, atol=1e-9)


def test_resample_aggregates_within_bin_by_mean():
    values = [1e-7, 2e-7, 3e-7, 4e-7]  # 4 native-cadence minutes -> one 4-min bin
    df = _series(values)
    config = _config(resample_cadence_minutes=4, despike_n_sigma=100, background_window_minutes=1000)
    out = preprocess_goes_sequence(df, config=config)
    assert len(out) == 1

    ell = np.log10(pd.Series(values))
    expected_bg = ell.rolling(1000, min_periods=250).quantile(0.10)
    expected_mean = (ell - expected_bg).mean()
    np.testing.assert_allclose(out["goes_seq_xrsb_flux_w_m2"].iloc[0], expected_mean, atol=1e-9)


def test_perturbing_a_sufficiently_future_sample_does_not_change_a_past_bin():
    # Causality check: the only non-causal step is the CENTERED despike
    # window (module docstring explains why this is still safe -- a
    # one-time archive-cleaning pass, not a forecast-time feature). A
    # query point more than despike_kernel_minutes away from a future
    # perturbation must be completely unaffected by it.
    rng = np.random.default_rng(0)
    base_values = list(1e-7 + 1e-9 * rng.standard_normal(200))
    df_a = _series(base_values)

    perturbed_values = list(base_values)
    perturbed_values[150] = 1e-3   # a huge perturbation far in the "future"
    df_b = _series(perturbed_values)

    config = _config(background_window_minutes=60)
    out_a = preprocess_goes_sequence(df_a, config=config)
    out_b = preprocess_goes_sequence(df_b, config=config)

    query_time = pd.Timestamp("2018-01-01T00:50:00Z")  # 100 minutes before the perturbation
    val_a = out_a.loc[out_a["time_utc"] == query_time, "goes_seq_xrsb_flux_w_m2"].iloc[0]
    val_b = out_b.loc[out_b["time_utc"] == query_time, "goes_seq_xrsb_flux_w_m2"].iloc[0]
    assert val_a == val_b


def test_empty_input_returns_empty_frame_without_crashing():
    df = pd.DataFrame(columns=["time_utc", "xrsb_flux_w_m2"])
    out = preprocess_goes_sequence(df, config=_config())
    assert out.empty
    assert "goes_seq_xrsb_flux_w_m2" in out.columns
