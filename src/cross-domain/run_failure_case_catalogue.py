"""Failure-case catalogue for the canonical 5-seed sweep, using the fixed
selection rule as implemented in evaluation/failure_cases.py and applied by
evaluation/comparison.py, UNCHANGED:

  selected = (y = 1 and p_fused < tau*_fused)      # every missed event at tau*
           | (y = 0 and p_fused > 0.5)             # every false alarm above 0.5

  (false_alarm_probability_threshold = 0.5, per comparison.py and C5R
  Section 5.6.3). comparison.py catalogues every configuration that has
  attention weights (A3, A4, A6, A7), each against the A1 unimodal baseline.

The function itself (catalogue_failure_cases) is called, so the rule is not
re-implemented; extra identifying and comparison columns are joined on the
test-row index afterwards.

Inputs
  sweep predictions   outputs/uncertainty_sweep_20261004T092140396722Z/predictions_*.csv
  attention weights   outputs/attention_recovery_20261005/alpha_*.npy (written only for
                      retrains that reproduced tau* and every test probability, < 1e-6;
                      see verification.json). Missing -> NaN, flagged.
  solar keys          src/solar/results/prediction_exports/goes_sharp_lstm_predictions_test.csv
                      (same rows/order; A1 must reproduce its probabilities exactly)
  solar persistence   src/solar/results/prediction_exports/baseline_predictions_test.csv
                      (persistence_prob) at baseline_thresholds.json["persistence"]
  Martian keys        canonical lstm_w3 h=1 test windows (A1 must reproduce them, < 1e-6)
  Martian persistence same file, model == persistence, horizon 1, joined on
                      (cell_id, mars_year, sol), at persistence_h1_evaluation.json's
                      validation-selected threshold

Outputs (outputs/failure_case_catalogue_<UTC>/)
  catalogue_all.csv            every selected forecast, configs A3/A4/A6/A7 x seeds 44-48
  catalogue_A7_seed44.csv      the primary-seed A7 subset
  selection_counts.csv         selected counts by config x domain x seed (x season)
  disagreement_counts.csv      decision disagreements vs A1 for A2-A7, by direction
                               and (Martian) season
  checks.txt
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.failure_cases import catalogue_failure_cases

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent
SWEEP = ROOT / "outputs" / "uncertainty_sweep_20261004T092140396722Z"
ABL = sorted((ROOT / "outputs").glob("ablation_run_20261004T09*"))
ALPHA_DIR = ROOT / "outputs" / "attention_recovery_20261005"
SOLAR_EXP = SRC / "solar" / "results" / "prediction_exports"
MRUN = SRC / "martian" / "outputs" / "real" / "run_20261004T080014968911Z_seed42"
MSUPP = SRC / "martian" / "outputs" / "real" / "chapter6_supplementary_20261004T093226Z"
SEEDS = [44, 45, 46, 47, 48]
CATALOGUED = ["A3", "A4", "A6", "A7"]          # configs with attention weights (comparison.py)
ALL_CFG = ["A2", "A3", "A4", "A5", "A6", "A7"]
FA_PROB = 0.5                                  # comparison.py's false_alarm_probability_threshold
GRID_RES, N_LON = 5.0, 72                      # src/martian/configs/grid.yaml
MARTIAN_HORIZON = 1


def vclass(y, d):
    return np.select([(y == 1) & (d == 1), (y == 0) & (d == 1), (y == 0) & (d == 0)], ["TP", "FP", "TN"], "FN")


def tss(y, d):
    y, d = y.astype(bool), d.astype(bool)
    return (y & d).sum() / y.sum() - (~y & d).sum() / (~y).sum()


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["transductive", "prototype"], default="transductive",
                    help="transductive: the original sweep forecasts (other domain's partition mean); prototype: the "
                         "primary non-transductive forecasts (fixed training prototype) for A3/A4/A6/A7")
    ap.add_argument("--proto-dir", type=Path, default=ROOT / "outputs" / "prototype_rescore_20261005T083742Z")
    ap.add_argument("--alpha-dir", type=Path, default=None)
    args = ap.parse_args()
    alpha_dir = args.alpha_dir or ALPHA_DIR
    tag = "failure_case_catalogue_" if args.source == "transductive" else "failure_case_catalogue_nontransductive_"
    out = ROOT / "outputs" / (tag + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True)
    checks = [f"source: {args.source}" + (f" ({args.proto_dir.name}; attention {alpha_dir.name})" if args.source == "prototype" else "")]

    def load(cfg, dom, seed):
        if args.source == "prototype" and cfg in CATALOGUED:
            return pd.read_csv(args.proto_dir / f"predictions_prototype_{cfg}_{dom}_seed{seed:02d}.csv")
        return pd.read_csv(SWEEP / f"predictions_{cfg}_{dom}_seed{seed:02d}.csv")

    # ---- instance keys + persistence -------------------------------------------------
    keys = {}
    lstm = pd.read_csv(SOLAR_EXP / "goes_sharp_lstm_predictions_test.csv")
    a1s = load("A1", "solar", 44)
    assert len(lstm) == len(a1s) and (lstm.true_label.values == a1s.y_true.values).all()
    assert np.abs(lstm.predicted_probability.values - a1s.y_prob.values).max() < 1e-6
    base = pd.read_csv(SOLAR_EXP / "baseline_predictions_test.csv")
    sk = lstm[["time_utc", "true_label"]].merge(base[["time_utc", "true_label", "persistence_prob"]], on="time_utc", how="left", suffixes=("", "_b"))
    assert len(sk) == len(lstm) and sk.persistence_prob.notna().all() and (sk.true_label == sk.true_label_b).all()
    tau_pers_s = json.loads((SOLAR_EXP / "baseline_thresholds.json").read_text())["persistence"]
    keys["solar"] = pd.DataFrame({"issuance_time_utc": sk.time_utc, "persistence_prob": sk.persistence_prob, "persistence_tau": tau_pers_s})
    checks.append(f"solar: {len(sk)} test rows keyed by goes_sharp_lstm log (A1 == log, max|dp| < 1e-6); persistence joined on time_utc for all rows, labels equal; tau_pers = {tau_pers_s}")

    P = pd.read_csv(MSUPP / "predictions_test_and_my37_late.csv.gz")
    q = P[(P.eval_set == "test") & (P.model == "lstm_w3") & (P.horizon == MARTIAN_HORIZON)].reset_index(drop=True)
    a1m = load("A1", "martian", 44)
    assert len(q) == len(a1m) and (q.y.values == a1m.y_true.values).all()
    assert np.abs(q.p.values - a1m.y_prob.values).max() < 1e-6
    pers = P[(P.eval_set == "test") & (P.model == "persistence") & (P.horizon == MARTIAN_HORIZON)][["cell_id", "mars_year", "sol", "y", "p"]]
    mk = q[["cell_id", "mars_year", "sol", "solar_longitude_ls", "season", "y"]].merge(pers, on=["cell_id", "mars_year", "sol"], how="left", suffixes=("", "_pers"))
    assert len(mk) == len(q) and mk.p.notna().all() and (mk.y == mk.y_pers).all()
    tau_pers_m = json.loads((MRUN / "metrics" / "storm_detection" / f"persistence_h{MARTIAN_HORIZON}_evaluation.json").read_text())["decision_threshold"]
    lat_band, lon_band = mk.cell_id // N_LON, mk.cell_id % N_LON
    # sanity: the cell's mean observed latitude lies inside the band the id implies
    feat = pd.read_csv(MRUN / "features" / "storm_detection_dataset.csv", usecols=["cell_id", "mars_year", "sol", "latitude"])
    chk = mk[["cell_id", "mars_year", "sol"]].merge(feat, on=["cell_id", "mars_year", "sol"], how="left")
    lo = -90 + GRID_RES * (chk.cell_id // N_LON)
    assert chk.latitude.notna().all() and ((chk.latitude >= lo - 1e-9) & (chk.latitude <= lo + GRID_RES + 1e-9)).all()
    keys["martian"] = pd.DataFrame({
        "cell_id": mk.cell_id, "cell_lat_min": -90 + GRID_RES * lat_band, "cell_lat_max": -90 + GRID_RES * (lat_band + 1),
        "cell_lat_centre": -90 + GRID_RES * (lat_band + 0.5), "cell_lon_min_E": GRID_RES * lon_band,
        "cell_lon_max_E": GRID_RES * (lon_band + 1), "cell_lon_centre_E": GRID_RES * (lon_band + 0.5),
        "mars_year": mk.mars_year, "sol": mk.sol, "Ls": mk.solar_longitude_ls, "horizon_sols": MARTIAN_HORIZON,
        "season": mk.season.str.replace(r" \(Ls.*\)", "", regex=True),
        "persistence_prob": mk.p, "persistence_tau": tau_pers_m})
    checks.append(f"martian: {len(mk)} test rows keyed by canonical lstm_w3 h={MARTIAN_HORIZON} windows (A1 == canonical, max|dp| < 1e-6); "
                  f"persistence joined on (cell_id, mars_year, sol) for all rows, labels equal; tau_pers = {tau_pers_m}; "
                  f"every row's observed latitude lies inside the cell's 5-degree band")

    # ---- pooled TSS reproduces tss_summary.json; A1 seed-invariant -------------------
    tss_json = {json.loads((d / "tss_summary.json").read_text())["seed"]: json.loads((d / "tss_summary.json").read_text())["tss"] for d in ABL}
    assert sorted(tss_json) == SEEDS
    for dom in ("solar", "martian"):
        a1_44 = load("A1", dom, 44)
        for seed in SEEDS:
            assert load("A1", dom, seed).equals(a1_44), "A1 not seed-invariant"
            for cfg in ["A1"] + ALL_CFG:
                d = load(cfg, dom, seed)
                v = tss(d.y_true.values, (d.y_prob >= d.threshold).values)
                if args.source == "prototype" and cfg in CATALOGUED:
                    PM = pd.read_csv(args.proto_dir / "metrics.csv")
                    ref = PM[(PM.config == cfg) & (PM.domain == dom) & (PM.seed == seed) & (PM.stratum == "ALL") & (PM["mode"] == "prototype")].tss.iloc[0]
                    assert abs(v - ref) < 1e-9
                else:
                    assert abs(v - tss_json[seed][cfg][dom]) < 1e-9
    checks.append("pooled TSS of every config x domain x seed reproduces its source (tss_summary.json; prototype metrics.csv for "
                  "A3/A4/A6/A7 in prototype mode) (< 1e-9); A1 CSVs identical across seeds")

    ver = json.loads((alpha_dir / "verification.json").read_text()) if (alpha_dir / "verification.json").exists() else {}

    # ---- catalogue -------------------------------------------------------------------
    cats, sel_counts = [], []
    for dom in ("solar", "martian"):
        a1 = load("A1", dom, 44)
        y = a1.y_true.to_numpy().astype(int)
        a1_dec = (a1.y_prob >= a1.threshold).to_numpy().astype(int)
        k = keys[dom]
        pers_dec = (k.persistence_prob >= k.persistence_tau).to_numpy().astype(int)
        for cfg in CATALOGUED:
            for seed in SEEDS:
                f = load(cfg, dom, seed)
                assert (f.y_true.to_numpy() == y).all()
                ap = alpha_dir / f"alpha_{cfg}_{dom}_seed{seed:02d}.npy"
                accepted = ap.exists() and ver.get(f"{cfg}_seed{seed}", {}).get(dom, {}).get("accepted", False)
                alpha = np.load(ap) if accepted else np.full((len(y), 2), np.nan)
                meta = pd.DataFrame({"row_index": np.arange(len(y))})
                c = catalogue_failure_cases(
                    y, f.y_prob.to_numpy(), a1.y_prob.to_numpy(), alpha,
                    threshold_fused=float(f.threshold.iloc[0]), threshold_unimodal=float(a1.threshold.iloc[0]),
                    false_alarm_probability_threshold=FA_PROB, metadata=meta)
                i = c.row_index.to_numpy()
                dec = (f.y_prob.to_numpy() >= f.threshold.iloc[0]).astype(int)
                row = pd.DataFrame({"domain": dom, "configuration": cfg, "seed": seed, "row_index": i})
                row = pd.concat([row, k.iloc[i].reset_index(drop=True).drop(columns=["persistence_prob", "persistence_tau"])], axis=1)
                row = row.assign(
                    failure_type=c.failure_type.values,
                    p_fused=f.y_prob.to_numpy()[i], tau_fused=float(f.threshold.iloc[0]), decision_fused=dec[i],
                    observed=y[i], class_fused=vclass(y[i], dec[i]),
                    p_A1=a1.y_prob.to_numpy()[i], tau_A1=float(a1.threshold.iloc[0]), decision_A1=a1_dec[i],
                    class_A1=vclass(y[i], a1_dec[i]), shared_with_unimodal=c.shared_with_unimodal.values,
                    p_persistence=k.persistence_prob.to_numpy()[i], tau_persistence=float(k.persistence_tau.iloc[0]),
                    decision_persistence=pers_dec[i], class_persistence=vclass(y[i], pers_dec[i]),
                    alpha_solar=c.alpha_solar_at_issuance.values, alpha_martian=c.alpha_martian_at_issuance.values,
                    alpha_auxiliary=(c.alpha_martian_at_issuance if dom == "solar" else c.alpha_solar_at_issuance).values,
                    alpha_verified=accepted)
                cats.append(row)
                strata = [("ALL", np.ones(len(row), bool))]
                if dom == "martian":
                    strata += [(s, (row.season == s).to_numpy()) for s in ("Northern Autumn", "Northern Winter")]
                for s, m in strata:
                    r = row[m]
                    sel_counts.append({"domain": dom, "configuration": cfg, "seed": seed, "season": s, "tau_fused": float(f.threshold.iloc[0]),
                                       "n_selected": len(r), "n_missed_event": int((r.failure_type == "missed_event").sum()),
                                       "n_false_alarm_gt_0.5": int((r.failure_type == "false_alarm").sum()),
                                       "n_shared_with_A1": int(r.shared_with_unimodal.sum()), "n_fusion_only": int((~r.shared_with_unimodal).sum()),
                                       "alpha_verified": accepted})
    cat = pd.concat(cats, ignore_index=True)
    cat = cat.astype({"cell_id": "Int64", "mars_year": "Int64", "sol": "Int64", "horizon_sols": "Int64"})
    cat.to_csv(out / "catalogue_all.csv", index=False)
    cat[(cat.configuration == "A7") & (cat.seed == 44)].to_csv(out / "catalogue_A7_seed44.csv", index=False)
    pd.DataFrame(sel_counts).to_csv(out / "selection_counts.csv", index=False)

    # ---- disagreement with A1 ----------------------------------------------------------
    dis = []
    for dom in ("solar", "martian"):
        a1 = load("A1", dom, 44)
        y = a1.y_true.to_numpy().astype(int)
        a1_dec = (a1.y_prob >= a1.threshold).to_numpy().astype(int)
        strata = [("ALL", np.ones(len(y), bool))]
        if dom == "martian":
            strata += [(s, (keys[dom].season == s).to_numpy()) for s in ("Northern Autumn", "Northern Winter")]
        for cfg in ALL_CFG:
            for seed in SEEDS:
                f = load(cfg, dom, seed)
                dec = (f.y_prob >= f.threshold).to_numpy().astype(int)
                for s, m in strata:
                    dd, aa, yy = dec[m], a1_dec[m], y[m]
                    diff = dd != aa
                    cfg_ok = diff & (dd == yy)
                    a1_ok = diff & (aa == yy)
                    dis.append({"domain": dom, "configuration": cfg, "seed": seed, "season": s, "n": int(m.sum()), "n_pos": int(yy.sum()),
                                "n_disagree": int(diff.sum()), "cfg_correct_A1_wrong": int(cfg_ok.sum()), "A1_correct_cfg_wrong": int(a1_ok.sum()),
                                "cfg_TP_A1_FN": int((cfg_ok & (yy == 1)).sum()), "cfg_TN_A1_FP": int((cfg_ok & (yy == 0)).sum()),
                                "A1_TP_cfg_FN": int((a1_ok & (yy == 1)).sum()), "A1_TN_cfg_FP": int((a1_ok & (yy == 0)).sum())})
    pd.DataFrame(dis).to_csv(out / "disagreement_counts.csv", index=False)

    nver = sum(1 for d in ("solar", "martian") for c in CATALOGUED for s in SEEDS
               if ver.get(f"{c}_seed{s}", {}).get(d, {}).get("accepted", False))
    checks.append(f"attention weights verified for {nver} of {2 * len(CATALOGUED) * len(SEEDS)} config x domain x seed cells (others NaN, alpha_verified = False)")
    checks.append(f"catalogue rows: {len(cat)}; A7 seed 44: {int(((cat.configuration == 'A7') & (cat.seed == 44)).sum())}")
    (out / "checks.txt").write_text("\n".join(checks) + "\n")
    print("\n".join(checks)); print(f"written to {out}")


if __name__ == "__main__":
    main()
