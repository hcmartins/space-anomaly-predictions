"""Tests for per-HARP catalogue-attributed labeling, hand-computed."""
import numpy as np
import pandas as pd

from labeling.harp_catalogue_labels import build_harp_catalogue_labels


def _config(**overrides):
    from labeling.flare_labels import load_labeling_config

    config = load_labeling_config()
    config["forecast_horizon_hours"] = 24
    config.update(overrides)
    return config


def _aligned_df():
    # Two HARPs, one row each, both at the same t0.
    return pd.DataFrame([
        {"HARPNUM": 7115, "time_utc": pd.Timestamp("2017-09-04T00:00:00Z")},
        {"HARPNUM": 9001, "time_utc": pd.Timestamp("2017-09-04T00:00:00Z")},
    ])


_FLARE_COLUMNS = ["flare_id", "time_utc_peak", "xrsb_irrad_peak_w_m2", "harpnums"]


def _flares(rows):
    # Matches the real shape preprocessing.harp_attribution.attribute_flares_to_harps
    # always returns, even for zero rows (a "harpnums" column always exists).
    return pd.DataFrame(rows, columns=_FLARE_COLUMNS)


def test_real_attributed_positive_labels_correct_harp_only():
    # An M-class flare (1.8e-5 W/m^2) attributed to HARP 7115 only, peaking
    # 6h into the 24h window.
    flares = _flares([
        {"flare_id": "f1", "time_utc_peak": pd.Timestamp("2017-09-04T06:00:00Z"),
         "xrsb_irrad_peak_w_m2": 1.8e-5, "harpnums": [7115]},
    ])
    out = build_harp_catalogue_labels(_aligned_df(), flares, config=_config())
    row_7115 = out.loc[out["HARPNUM"] == 7115].iloc[0]
    row_9001 = out.loc[out["HARPNUM"] == 9001].iloc[0]
    assert row_7115["label_flare_m_or_above"] == 1.0
    assert row_9001["label_flare_m_or_above"] == 0.0  # not attributed to 9001 -- genuine negative


def test_no_flares_gives_genuine_negatives():
    out = build_harp_catalogue_labels(_aligned_df(), _flares([]), config=_config())
    assert (out["label_flare_b_or_above"] == 0.0).all()


def test_multi_harp_attribution_credits_every_harp():
    flares = _flares([
        {"flare_id": "f1", "time_utc_peak": pd.Timestamp("2017-09-04T06:00:00Z"),
         "xrsb_irrad_peak_w_m2": 1.8e-5, "harpnums": [7115, 9001]},
    ])
    out = build_harp_catalogue_labels(_aligned_df(), flares, config=_config())
    assert (out["label_flare_m_or_above"] == 1.0).all()


def test_unassigned_flare_masks_would_be_negatives_up_to_its_own_class():
    # A C-class (2e-6) unassigned flare (harpnums=[]) peaks in both HARPs'
    # windows. Neither HARP has an attributed flare of their own, so
    # THIS HARP's own B/C-or-above status is genuinely unknown -- the
    # unassigned event might belong to it, or to some other region
    # entirely. Only M/X-or-above stay definite negatives, since the
    # flux provably never reached that high regardless of attribution.
    flares = _flares([
        {"flare_id": "f1", "time_utc_peak": pd.Timestamp("2017-09-04T06:00:00Z"),
         "xrsb_irrad_peak_w_m2": 2.0e-6, "harpnums": []},
    ])
    out = build_harp_catalogue_labels(_aligned_df(), flares, config=_config())
    for _, row in out.iterrows():
        assert np.isnan(row["label_flare_b_or_above"])  # masked -- could have been this HARP's B-or-above event
        assert np.isnan(row["label_flare_c_or_above"])  # masked -- could have been this HARP's C-or-above event
        assert row["label_flare_m_or_above"] == 0.0  # NOT masked -- flux provably never reached M, regardless of region
        assert row["label_flare_x_or_above"] == 0.0


def test_unassigned_flare_never_overrides_a_genuine_positive():
    # HARP 7115 has its OWN real attributed M-class flare; a separate
    # unassigned C-class flare also peaks in the same window.
    flares = _flares([
        {"flare_id": "f1", "time_utc_peak": pd.Timestamp("2017-09-04T05:00:00Z"),
         "xrsb_irrad_peak_w_m2": 1.8e-5, "harpnums": [7115]},
        {"flare_id": "f2", "time_utc_peak": pd.Timestamp("2017-09-04T06:00:00Z"),
         "xrsb_irrad_peak_w_m2": 2.0e-6, "harpnums": []},
    ])
    out = build_harp_catalogue_labels(_aligned_df(), flares, config=_config())
    row_7115 = out.loc[out["HARPNUM"] == 7115].iloc[0]
    assert row_7115["label_flare_m_or_above"] == 1.0  # genuine positive, not masked to NaN
    assert row_7115["label_flare_c_or_above"] == 1.0  # also a genuine positive (M implies C-or-above)


def test_flare_outside_horizon_window_is_ignored():
    flares = _flares([
        {"flare_id": "f1", "time_utc_peak": pd.Timestamp("2017-09-05T01:00:00Z"),  # 25h later, outside 24h horizon
         "xrsb_irrad_peak_w_m2": 1.8e-5, "harpnums": [7115]},
    ])
    out = build_harp_catalogue_labels(_aligned_df(), flares, config=_config())
    row_7115 = out.loc[out["HARPNUM"] == 7115].iloc[0]
    assert row_7115["label_flare_m_or_above"] == 0.0


def test_label_primary_matches_configured_primary_target_class():
    flares = _flares([
        {"flare_id": "f1", "time_utc_peak": pd.Timestamp("2017-09-04T06:00:00Z"),
         "xrsb_irrad_peak_w_m2": 1.8e-5, "harpnums": [7115]},
    ])
    out = build_harp_catalogue_labels(_aligned_df(), flares, config=_config(primary_target_class="M"))
    row_7115 = out.loc[out["HARPNUM"] == 7115].iloc[0]
    assert row_7115["label_primary"] == row_7115["label_flare_m_or_above"] == 1.0
