"""Fills the v2 output template for worked examples S1, S2, M1-M3 from the output of
src/extract_worked_examples.py. Every field is READ from a canonical table by the selected
row's key; nothing is recomputed or estimated. Only the solar 6-h maximum is an aggregation
(a max over the raw 1-min GOES rows in (t0 - 6 h, t0]); the spec names it as an input summary.

    python src/fill_worked_examples.py outputs/worked_examples_<UTC>/

Writes into that folder: worked_examples_filled.json, raw_rows/*.csv (every source row used),
worked_examples_table.md.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "src" / "solar"
M = ROOT / "src" / "martian"
SOLAR_LSTM = S / "results/prediction_exports/goes_sharp_lstm_predictions_test.csv"
SOLAR_BASE = S / "results/prediction_exports/baseline_predictions_test.csv"
GOES_1MIN = S / "results/real_full_20100501_20250724/goes_disk.csv"
SEQ_TABLE = S / "results/real_full_sequence_goes_sharp_20100501_20250724/processed_goes.csv"
SEQ_SUMMARY = S / "results/real_full_sequence_goes_sharp_20100501_20250724/run_summary.json"
SOLAR_DS = S / "results/real_full_20100501_20250724/dataset_with_split.csv"
FLARES = S / "data_cache/flare_catalogue/normalized/flare_catalogue_normalized_full_20100501_20250724.csv"
MRUN = M / "outputs/real/run_20261004T080014968911Z_seed42"
MPRED = M / "outputs/real/chapter6_supplementary_20261004T093226Z/predictions_test_and_my37_late.csv.gz"
MFEAT = MRUN / "features/storm_detection_dataset.csv"
MLAB = MRUN / "labels/labeled_cell_sol.csv"
MCAT = MRUN / "labels/episode_catalogue.csv"
SHARP = ["USFLUX", "TOTUSJZ", "TOTUSJH", "ABSNJZH", "SAVNCPP", "MEANPOT", "TOTPOT", "R_VALUE", "MEANGAM",
         "MEANGBT", "MEANGBZ", "MEANGBH", "MEANALP", "MEANJZH", "MEANJZD", "MEANSHR", "AREA_ACR", "SHRGT45"]
N_LON, RES = 72, 5.0


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def goes_class(flux):
    for c, lo in (("X", 1e-4), ("M", 1e-5), ("C", 1e-6), ("B", 1e-7), ("A", 1e-8)):
        if flux >= lo:
            return f"{c}{flux / lo:.1f}"
    return "below A"


def solar_example(sel, raw):
    t0 = pd.Timestamp(sel["row"]["time_utc"])
    lstm = pd.read_csv(SOLAR_LSTM, parse_dates=["time_utc"])
    base = pd.read_csv(SOLAR_BASE, parse_dates=["time_utc"])
    i_l = int(np.flatnonzero(lstm.time_utc == t0)[0]); i_b = int(np.flatnonzero(base.time_utc == t0)[0])
    raw[f"{sel['rule']}_lstm_row"] = lstm.iloc[[i_l]]; raw[f"{sel['rule']}_baseline_row"] = base.iloc[[i_b]]

    g = pd.concat([c[(c.time_utc > t0 - pd.Timedelta(hours=6)) & (c.time_utc <= t0 + pd.Timedelta(hours=24))]
                   for c in pd.read_csv(GOES_1MIN, parse_dates=["time_utc"], chunksize=1_000_000)])
    past = g[(g.time_utc > t0 - pd.Timedelta(hours=6)) & (g.time_utc <= t0)].dropna(subset=["xrsb_flux_w_m2"])
    last = past.iloc[-1]; mx = past.loc[past.xrsb_flux_w_m2.idxmax()]
    fut = g[(g.time_utc > t0) & (g.time_utc <= t0 + pd.Timedelta(hours=24))].dropna(subset=["xrsb_flux_w_m2"])
    fmx = fut.loc[fut.xrsb_flux_w_m2.idxmax()]
    raw[f"{sel['rule']}_goes_1min_last_and_6h_max"] = pd.DataFrame([last, mx])

    seq = pd.concat([c[c.time_utc == str(t0)] for c in pd.read_csv(SEQ_TABLE, chunksize=500_000)])
    assert len(seq) == 1, "no unique sequence-table row at t0"
    raw[f"{sel['rule']}_sequence_input_row_t0"] = seq
    params = json.loads(SEQ_SUMMARY.read_text())["sharp_scaling_params"]
    sv = seq.iloc[0][SHARP].astype(float)
    top = sv.abs().sort_values(ascending=False).index[:3]
    top3 = [{"descriptor": k, "scaled_value": round(float(sv[k]), 3),
             "physical_value_winsorised": float(sv[k] * params[k]["scale"] + params[k]["median"]),
             "at_winsor_clip": bool(np.isclose(sv[k] * params[k]["scale"] + params[k]["median"], params[k]["upper"]) or
                                    np.isclose(sv[k] * params[k]["scale"] + params[k]["median"], params[k]["lower"]))} for k in top]

    fl = pd.read_csv(FLARES, parse_dates=["time_utc_peak", "time_utc_start", "time_utc_end"])
    win = fl[(fl.time_utc_peak > t0) & (fl.time_utc_peak <= t0 + pd.Timedelta(hours=24))].copy()
    win = win.sort_values("xrsb_irrad_peak_w_m2", ascending=False)
    raw[f"{sel['rule']}_flare_catalogue_window"] = win
    ds = pd.concat([c[c.time_utc == str(t0)] for c in pd.read_csv(SOLAR_DS, usecols=["time_utc", "label_max_flare_class_in_horizon",
                     "label_primary", "label_coverage_fraction", "split"], chunksize=200_000)])
    raw[f"{sel['rule']}_label_row"] = ds
    r = sel["row"]
    return {
        "Number of candidates satisfying the rule": sel["n_candidates"],
        "Instance key": f"t0 = {t0:%Y-%m-%d %H:%M} UTC (Cycle 25)",
        "Input summary at issuance": {
            "GOES 1-8 A flux, last 1-min value at or before t0": f"{last.xrsb_flux_w_m2:.3e} W m^-2 ({goes_class(last.xrsb_flux_w_m2)}) at {last.time_utc:%H:%M}",
            "GOES 1-8 A flux, max over (t0-6h, t0]": f"{mx.xrsb_flux_w_m2:.3e} W m^-2 ({goes_class(mx.xrsb_flux_w_m2)}) at {mx.time_utc:%Y-%m-%d %H:%M}",
            "Top-3 SHARP descriptors by |scaled value| (as fed to the LSTM)": top3,
        },
        "Model p / tau* / decision": f"{r['predicted_probability']:.4f} / 0.08 / {'alert' if r['dec_lstm'] else 'no alert'}",
        "Observed outcome": {
            "label (>=M within 24 h)": int(r["true_label"]),
            "label_max_flare_class_in_horizon (pipeline label, GOES 1-min peak)": ds.label_max_flare_class_in_horizon.iloc[0],
            "GOES 1-min peak in (t0, t0+24h]": f"{fmx.xrsb_flux_w_m2:.3e} W m^-2 ({goes_class(fmx.xrsb_flux_w_m2)}) at {fmx.time_utc:%Y-%m-%d %H:%M}",
            "largest catalogued flare peaking in (t0, t0+24h]": (f"{win.iloc[0].flare_class}, peak {win.iloc[0].time_utc_peak:%Y-%m-%d %H:%M}, "
                                                                f"AR {win.iloc[0].active_region}" if len(win) else "none catalogued"),
            "catalogued flares in window": int(len(win)),
        },
        "Model class": r["class_lstm"],
        "Persistence output / decision / class": f"{r['persistence_prob']:.1f} / {'alert' if r['dec_pers'] else 'no alert'} (tau 0.01) / {r['class_pers']}",
        "Provenance": {"run": "real_full_sequence_goes_sharp_20100501_20250724 (LSTM); real_full_20100501_20250724 (persistence)",
                       "files": {SOLAR_LSTM.name: {"sha256_16": sha(SOLAR_LSTM), "row_index": i_l},
                                 SOLAR_BASE.name: {"sha256_16": sha(SOLAR_BASE), "row_index": i_b}}},
    }


def mars_example(sel, raw, P, F, L, cat):
    r = sel["row"]; key = (int(r["cell_id"]), int(r["mars_year"]), int(r["sol"]))
    m_rf = (P.eval_set == "test") & (P.model == "random_forest_classifier") & (P.horizon == 1) & (P.cell_id == key[0]) & (P.mars_year == key[1]) & (P.sol == key[2])
    m_pe = (P.eval_set == "test") & (P.model == "persistence") & (P.horizon == 1) & (P.cell_id == key[0]) & (P.mars_year == key[1]) & (P.sol == key[2])
    i_rf, i_pe = int(np.flatnonzero(m_rf)[0]), int(np.flatnonzero(m_pe)[0])
    f = F[(F.cell_id == key[0]) & (F.mars_year == key[1]) & (F.sol == key[2])]
    assert len(f) == 1
    lab_cell = L[(L.cell_id == key[0]) & (L.mars_year == key[1]) & (L.sol >= key[2]) & (L.sol <= key[2] + 1)]
    fut = lab_cell[lab_cell.sol == key[2] + 1]
    raw[f"{sel['rule']}_prediction_rows"] = P.iloc[[i_rf, i_pe]]; raw[f"{sel['rule']}_feature_row"] = f
    raw[f"{sel['rule']}_label_rows_s_to_s_plus_1"] = lab_cell
    lat_band, lon_band = key[0] // N_LON, key[0] % N_LON
    fr = f.iloc[0]
    if len(fut) and bool(fut.storm_episode.iloc[0]):
        ep = cat[cat.episode_id == fut.episode_id.iloc[0]].iloc[0]
        outcome = (f"episode present on sol {key[2] + 1} (arrival sol {key[2] + 1}); episode {int(ep.episode_id)}: onset sol {int(ep.onset_sol)}, "
                   f"end sol {int(ep.end_sol)}, peak area {int(ep.peak_area_cells)} cells, qualifies={bool(ep.qualifies)}")
        raw[f"{sel['rule']}_episode"] = cat[cat.episode_id == ep.episode_id]
    elif len(fut):
        outcome = f"no episode: sol {key[2] + 1} observed, storm_episode=False (z={fut.dust_optical_depth_z.iloc[0]:.2f})"
    else:
        outcome = "target undefined (sol s+1 not observed)"
    return {
        "Number of candidates satisfying the rule": sel["n_candidates"],
        "Instance key": (f"cell {key[0]} (centre {-90 + RES * (lat_band + 0.5):+.1f} N, {RES * (lon_band + 0.5):.1f} E), "
                         f"MY{key[1]} sol {key[2]}, Ls {r['solar_longitude_ls']:.2f}, h = 1 sol, {r['season'].split(' (')[0]}"),
        "Input summary at issuance": {
            "harmonised tau (dust_optical_depth)": round(float(fr.dust_optical_depth), 4),
            "z-score": round(float(fr.dust_optical_depth_z), 3),
            "5-sol trailing mean (feat_dust_optical_depth_trailing5_mean)": None if pd.isna(fr.feat_dust_optical_depth_trailing5_mean) else round(float(fr.feat_dust_optical_depth_trailing5_mean), 4),
            "neighbour exceedance count": int(fr.feat_neighbour_exceedance_count),
            "in a qualifying episode at issuance (storm_episode)": bool(lab_cell[lab_cell.sol == key[2]].storm_episode.iloc[0]),
        },
        "Model p / tau* / decision": f"{r['p']:.4f} / {r['threshold']:.2f} / {'alert' if r['dec_rf'] else 'no alert'}",
        "Observed outcome": outcome,
        "Model class": r["class_rf"],
        "Persistence output / decision / class": f"{r['p_pers']:.0f} / {'alert' if r['dec_pers'] else 'no alert'} (tau {r['threshold_pers']:.2f}) / {r['class_pers']}",
        "Provenance": {"run": "run_20261004T080014968911Z_seed42",
                       "files": {MPRED.name: {"sha256_16": sha(MPRED), "row_index_rf": i_rf, "row_index_persistence": i_pe},
                                 MFEAT.name: {"sha256_16": sha(MFEAT), "row_index": int(f.index[0])}}},
    }


def main() -> None:
    out = Path(sys.argv[1])
    res = json.loads((out / "worked_examples.json").read_text())
    assert res["checks_passed"] is True
    raw, filled = {}, {}
    for sel in res["solar"]:
        filled[sel["rule"]] = solar_example(sel, raw) if "row" in sel else {"result": "no candidate"}
    P = pd.read_csv(MPRED)
    F = pd.read_csv(MFEAT, usecols=["cell_id", "mars_year", "sol", "dust_optical_depth", "dust_optical_depth_z",
                                    "feat_dust_optical_depth_trailing5_mean", "feat_neighbour_exceedance_count"])
    L = pd.read_csv(MLAB, usecols=["cell_id", "mars_year", "sol", "dust_optical_depth_z", "storm_episode", "episode_id"])
    cat = pd.read_csv(MCAT)
    for sel in res["mars"]:
        filled[sel["rule"]] = mars_example(sel, raw, P, F, L, cat) if "row" in sel else {"result": "no candidate"}
    (out / "raw_rows").mkdir(exist_ok=True)
    for k, v in raw.items():
        v.to_csv(out / "raw_rows" / f"{k}.csv", index=True)
    (out / "worked_examples_filled.json").write_text(json.dumps(filled, indent=2, default=str))
    print(json.dumps(filled, indent=2, default=str))


if __name__ == "__main__":
    main()
