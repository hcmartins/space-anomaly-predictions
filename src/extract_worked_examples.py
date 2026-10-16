"""
extract_worked_examples.py  --  deterministic extraction of worked examples S1, S2, M1, M2, M3
(Worked Prediction Examples -- Specification, v2, 2026-10-05).

PURPOSE
    Every value in a worked example must come from the canonical prediction files.  This script
    selects the examples by the specification's rules, refuses to run if the input files do not
    reproduce the published contingency counts, and writes every field with its provenance.
    Nothing is chosen by hand and nothing is estimated.

USE (from the repository root)
    python extract_worked_examples.py --out outputs/worked_examples_<UTC timestamp>/

BEFORE RUNNING
    The file paths and column names in CONFIG are the ones reported in the repository evidence
    (failure-case catalogue, newrun follow-ups, supplementary analyses).  If a column is named
    differently, change CONFIG only -- never the rules.  Every assumption is checked by an
    assertion; if an assertion fails, report the failure instead of editing the check.
"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- CONFIG
CONFIG = {
    # Solar: GOES+SHARP LSTM test predictions (37,598 rows) and disk-level baselines (211,594 rows)
    "solar_lstm_csv": "src/solar/results/prediction_exports/goes_sharp_lstm_predictions_test.csv",
    "solar_baseline_csv": "src/solar/results/prediction_exports/baseline_predictions_test.csv",
    "solar_baseline_thresholds": "src/solar/results/prediction_exports/baseline_thresholds.json",
    "solar_time_col": "time_utc",
    "solar_y_col": "true_label",        # >=M within 24 h
    "solar_lstm_p_col": "predicted_probability",
    "solar_persistence_p_col": "persistence_prob",
    "solar_lstm_tau": 0.08,             # Chapter 6, Table 6.4
    # Martian: per-row test predictions of the canonical run
    # canonical chapter6_supplementary_20261004T093226Z/predictions_test_and_my37_late.csv.gz test
    # rows plus per-row threshold and onset_like (src/martian/prepare_worked_example_inputs.py)
    "mars_pred_csv": "src/martian/outputs/real/worked_examples_inputs_20261005T082343Z/"
                     "mars_test_predictions_with_threshold_onset.csv.gz",
    "mars_model_col": "model", "mars_h_col": "horizon", "mars_split_col": "eval_set",
    "mars_cell_col": "cell_id", "mars_my_col": "mars_year", "mars_sol_col": "sol",
    "mars_ls_col": "solar_longitude_ls", "mars_y_col": "y", "mars_p_col": "p",
    "mars_tau_col": "threshold",
    "mars_rf_name": "random_forest_classifier", "mars_pers_name": "persistence",
    # Onset-like flag: must be the flag used by run_newrun_followups.py (cell NOT inside a
    # qualifying episode at the issuance sol).  Import it rather than re-deriving it.
    "mars_onset_flag_source": "src/martian/run_newrun_followups.py",
    "mars_onset_col": "onset_like",
}
# 2026-10-05: lower bound changed from 188.22 (the test partition's first Ls rounded to 2 dp;
# the true value is 188.2152, so 3 sol-386 rows fell outside) to the season boundary 180.0,
# the northern-autumn definition of Chapter 6 (Ls 180-270) behind Tables 6.9 and 6.12-6.13.
# The test partition has no row with Ls < 188.2152, so this is "all of test-partition autumn".
AUTUMN = (180.0, 270.0)    # northern autumn, Ls in [180, 270)
WINTER = (270.0, 360.0)    # northern winter, Ls in [270, 360]

# Published values that the inputs must reproduce before any example is selected.
CHECKS = {
    "solar_lstm_counts": dict(TP=9744, FP=10122, TN=16983, FN=749),          # Table 6.4
    "solar_common_n": 37598,
    "mars_rf_h1_tau": 0.09,                                                   # rerun report
    "mars_rf_h1_autumn": dict(n=10167, pos=37, tss=0.692),                    # Table 6.9
    "mars_rf_h1_winter": dict(n=3289, pos=455, tss=0.079),                    # Table 6.10
    "mars_rf_h1_autumn_onset": dict(n=10139, pos=31, tss=0.657),              # Table 6.12/6.13
}

# ----------------------------------------------------------------------------- helpers
def cls(y, d):
    return np.where((y == 1) & (d == 1), "TP", np.where((y == 0) & (d == 1), "FP",
           np.where((y == 0) & (d == 0), "TN", "FN")))

def counts(c):
    s = pd.Series(c).value_counts()
    return {k: int(s.get(k, 0)) for k in ("TP", "FP", "TN", "FN")}

def tss(cn):
    return cn["TP"] / (cn["TP"] + cn["FN"]) - cn["FP"] / (cn["FP"] + cn["TN"])

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]

def first(df, order, rule_id):
    if df.empty:
        return {"rule": rule_id, "result": "NO CANDIDATE"}   # a finding, not an error
    row = df.sort_values(order, kind="mergesort").iloc[0]
    return {"rule": rule_id, "n_candidates": int(len(df)), "row": row.to_dict()}

# ----------------------------------------------------------------------------- solar
def solar():
    C = CONFIG
    lstm = pd.read_csv(C["solar_lstm_csv"], parse_dates=[C["solar_time_col"]])
    base = pd.read_csv(C["solar_baseline_csv"], parse_dates=[C["solar_time_col"]])
    tau_p = json.load(open(C["solar_baseline_thresholds"]))["persistence"]
    assert abs(tau_p - 0.01) < 1e-12, f"persistence threshold {tau_p} != 0.01"
    m = lstm.merge(base[[C["solar_time_col"], C["solar_y_col"], C["solar_persistence_p_col"]]],
                   on=C["solar_time_col"], suffixes=("", "_base"), validate="one_to_one")
    assert len(m) == CHECKS["solar_common_n"], len(m)
    assert (m[C["solar_y_col"]] == m[C["solar_y_col"] + "_base"]).all(), "labels differ"
    y = m[C["solar_y_col"]].astype(int).to_numpy()
    m["dec_lstm"] = (m[C["solar_lstm_p_col"]] >= C["solar_lstm_tau"]).astype(int)
    m["dec_pers"] = (m[C["solar_persistence_p_col"]] >= tau_p).astype(int)
    m["class_lstm"], m["class_pers"] = cls(y, m["dec_lstm"]), cls(y, m["dec_pers"])
    got = counts(m["class_lstm"])
    assert got == CHECKS["solar_lstm_counts"], f"LSTM counts {got} do not reproduce Table 6.4"
    t = C["solar_time_col"]
    return [
        first(m[(m.class_lstm == "TP") & (m.class_pers == "FN")], [t], "S1"),
        first(m[(m.class_lstm == "FP") & (m.class_pers == "TN")], [t], "S2"),
    ]

# ----------------------------------------------------------------------------- Mars
def mars():
    C = CONFIG
    d = pd.read_csv(C["mars_pred_csv"])
    d = d[(d[C["mars_split_col"]] == "test") & (d[C["mars_h_col"]] == 1)]
    key = [C["mars_cell_col"], C["mars_my_col"], C["mars_sol_col"]]
    rf = d[d[C["mars_model_col"]] == C["mars_rf_name"]].copy()
    pe = d[d[C["mars_model_col"]] == C["mars_pers_name"]].copy()
    tau_rf = rf[C["mars_tau_col"]].unique()
    assert len(tau_rf) == 1 and abs(tau_rf[0] - CHECKS["mars_rf_h1_tau"]) < 1e-9, tau_rf
    m = rf.merge(pe[key + [C["mars_y_col"], C["mars_p_col"], C["mars_tau_col"]]], on=key,
                 suffixes=("", "_pers"), validate="one_to_one")
    assert (m[C["mars_y_col"]] == m[C["mars_y_col"] + "_pers"]).all(), "labels differ"
    y = m[C["mars_y_col"]].astype(int).to_numpy()
    m["dec_rf"] = (m[C["mars_p_col"]] >= tau_rf[0]).astype(int)
    m["dec_pers"] = (m[C["mars_p_col"] + "_pers"] >= m[C["mars_tau_col"] + "_pers"]).astype(int)
    m["class_rf"], m["class_pers"] = cls(y, m["dec_rf"]), cls(y, m["dec_pers"])
    ls = m[C["mars_ls_col"]]
    aut = m[(ls >= AUTUMN[0]) & (ls < AUTUMN[1])]
    win = m[(ls >= WINTER[0]) & (ls <= WINTER[1])]
    for name, sub in (("mars_rf_h1_autumn", aut), ("mars_rf_h1_winter", win)):
        cn, ref = counts(sub["class_rf"]), CHECKS[name]
        assert len(sub) == ref["n"] and cn["TP"] + cn["FN"] == ref["pos"], (name, len(sub), cn)
        assert abs(tss(cn) - ref["tss"]) < 5e-4, (name, tss(cn))
    onset = aut[aut[C["mars_onset_col"]].astype(bool)]
    cn, ref = counts(onset["class_rf"]), CHECKS["mars_rf_h1_autumn_onset"]
    assert len(onset) == ref["n"] and cn["TP"] + cn["FN"] == ref["pos"], (len(onset), cn)
    assert abs(tss(cn) - ref["tss"]) < 5e-4, tss(cn)
    # ordering: sol ascending, then cell_id ascending.  cell_id = lat_band*72 + lon_band with
    # band 0 southernmost and longitude 0-360 E, so this is latitude (S->N) then longitude.
    order = [C["mars_sol_col"], C["mars_cell_col"]]
    return [
        first(onset[(onset.class_rf == "TP") & (onset.class_pers == "FN")], order, "M1"),
        first(aut[aut.class_rf == "FP"], order, "M2"),
        first(win[win.class_rf == "FN"], order, "M3"),
    ]

# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    res = {"solar": solar(), "mars": mars(),
           "inputs_sha256_16": {k: sha(v) for k, v in CONFIG.items()
                                if isinstance(v, str) and v.endswith((".csv", ".gz", ".json"))},
           "checks_passed": True}
    json.dump(res, open(out / "worked_examples.json", "w"), indent=2, default=str)
    print(json.dumps(res, indent=2, default=str))
    # The input summaries (GOES flux, SHARP descriptors; tau_h, z, 5-sol trailing mean,
    # neighbour exceedance count) and the observed outcome details (largest GOES class and peak
    # time; sol of arrival) must then be read for the selected rows from the canonical feature
    # tables and flare catalogue by their keys, and appended -- not recomputed.