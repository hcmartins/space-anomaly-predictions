"""Physical-harmonisation and target-definition study (examiner review of
Chapter 4, rev 3). Answers three open findings WITHOUT retraining any model:

  F1  EMIRS taudust (column EXTINCTION optical depth at 1075 cm^-1, at the
      observed surface pressure) is standardised against the Montabone
      reference cdod610 (column ABSORPTION optical depth at 9.3 um = 1075 cm^-1,
      normalised to 610 Pa) with no conversion.
  F2  Horizon targets with no observed sol in (s, s+h] are labelled 0
      ("no episode") rather than unknown.
  F3  Roughly half of the positives are continuation of an episode already
      covering the cell at the issue sol, not new arrivals.

Step 0 (reproduction gate). Re-run the canonical ingestion -> QC -> gridding
-> aggregation -> labeling -> split-boundary exclusion -> catalogue
validation, and ABORT unless it reproduces the canonical run exactly
(142,608 cell-sols; labels; 156 MY36 qualifying episodes; POD 11/98;
precision 8/156) and the frozen models' saved test confusion matrices.

Step 1 (harmonised labels). Convert the OBSERVATION into the reference's
units, per retrieval, before aggregation:

      tau*_i = (tau_ext,i / f) * (610 Pa / p_s,i)

  f   extinction-to-absorption ratio at 9.3 um. Primary f = 1.3 (the value
      the reviewer cites; the author must supply its literature source), with
      a sensitivity sweep f in {1.2, 1.3, 1.4, 1.5}. Both wavenumbers are
      1075 cm^-1, so no spectral (wavelength) conversion is needed -- only
      extinction->absorption and pressure normalisation.
  p_s EMIRS l3atm `pres0`, in mbar (accepted-retrieval range 3.8-13.8), x100
      to Pa. NOTE: configs/real_emirs_schema.yaml maps pres0 to a column named
      `surface_pressure_pa` with "units ... assumed Pa"; the values are mbar.
      This script converts explicitly and never trusts that column name.
  dustuncert is scaled by the same per-retrieval factor, so inverse-variance
  weights stay consistent (QC rule 6 is scale-invariant, so the accepted set
  is unchanged). The Montabone mu/sigma are untouched: tau* is now in their
  units. Labels are rebuilt with the canonical functions and compared with the
  canonical labels (Cohen's kappa, Jaccard, prevalence, episode counts), and
  catalogue validation is recomputed on them.

Step 2 (re-scoring, no retraining). The frozen canonical storm-detection
models (all 13 x 3 horizons; verified per-row test predictions from
run_chapter6_supplementary.py) are re-scored against the canonical and the
harmonised labels on three subsets:
  all                 every test row (the canonical evaluation set);
  observed_horizon    rows with at least one OBSERVED sol of the same cell in
                      (s, s+h] -- removes the F2 "unknown" cases;
  non_episode_issue   rows whose cell is NOT inside a qualifying episode at
                      the issue sol -- every positive is then a new arrival
                      (F3), i.e. an onset-like task.
at two threshold policies: the canonical fixed 0.5 and the validation-selected
tau_star from run_chapter6_threshold_study.py (Eq. 6.1). TSS with 95%
cell-block bootstrap CIs (200 replicates) for the canonical and primary
harmonised labels.

Pre-registered materiality rule (fixed in this file BEFORE the study was run;
see MATERIALITY below): a full retraining rerun is required if EITHER
  (i)  Cohen's kappa between canonical and primary-harmonised dust_anomaly
       exceedances is < 0.80, OR
  (ii) the top-ranked model (by test TSS, 'all' subset) changes at any
       horizon under either threshold policy when the harmonised labels
       replace the canonical ones.

Outputs: outputs/real/harmonisation_target_study_<UTC>/
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from data_sources.real_documented_events import load_real_documented_events, load_real_documented_events_config
from data_sources.real_emirs_provider import RealEMIRSObservationProvider
from data_sources.real_mcd_provider import RealMCDClimatologyProvider
from labeling.anomaly_labels import load_anomaly_definition_config, run_full_labeling_pipeline
from labeling.catalogue_tolerance_tuning import load_catalogue_tolerance_tuning_config, tune_and_validate_tolerance
from labeling.catalogue_validation import load_catalogue_validation_config, run_catalogue_validation
from labeling.storm_taxonomy import classify_episode_catalogue, load_storm_taxonomy_config
from preprocessing.aggregation import aggregate_by_cell_sol, load_aggregation_config
from preprocessing.grid import assign_cells, load_primary_grid
from preprocessing.quality_control import load_qc_config, run_quality_control
from splitting.chronological_split import find_episodes_crossing_split_boundary, load_splits_config

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "outputs" / "real" / "run_20260813T195013457439Z_seed42"
KEYS = ["cell_id", "mars_year", "sol"]
DUST, UNC, PRES0_MBAR = "dust_optical_depth", "dust_optical_depth_uncertainty", "surface_pressure_pa"  # see docstring: values are mbar
HORIZONS = (1, 3, 5)
REF_PRESSURE_PA = 610.0
MBAR_TO_PA = 100.0
F_PRIMARY = 1.3
F_SWEEP = (1.2, 1.3, 1.4, 1.5)
SEED = 44
N_BOOT = 200
MATERIALITY = {"kappa_exceedance_min": 0.80, "top_model_must_not_change": True,
               "fixed_on": "2026-10-04, in code, before the study was run"}


# --------------------------------------------------------------------------- labels

def label_like_run_all(aggregated, mcd, grid, anomaly_cfg, splits_cfg):
    """mars/run_all.py step 8 exactly: labels, then split-boundary episode exclusion."""
    labeled, catalogue = run_full_labeling_pipeline(aggregated, mcd, grid, config=anomaly_cfg)
    catalogue = classify_episode_catalogue(catalogue, load_storm_taxonomy_config())  # as run_all.py step 8
    crossing = find_episodes_crossing_split_boundary(catalogue, splits_cfg)
    n_excl = 0
    if len(crossing):
        ids = set(crossing["episode_id"])
        labeled = labeled.loc[~labeled["episode_id"].isin(ids)].reset_index(drop=True)
        catalogue = catalogue.loc[~catalogue["episode_id"].isin(ids)].reset_index(drop=True)
        n_excl = len(ids)
    return labeled, catalogue, n_excl


def catalogue_metrics(labeled, catalogue, events, grid, cv_cfg, tt_cfg):
    design = run_catalogue_validation(labeled, catalogue, events, grid, config=cv_cfg)["summary"]
    tuned = tune_and_validate_tolerance(labeled, catalogue, events, grid, config=tt_cfg,
                                        unmatched_episode_label=cv_cfg.get("unmatched_episode_label", "false_positive_candidate"))
    t = tuned["test_validation"]["summary"]
    return {"n_qualifying_episodes_my36": design["n_qualifying_episodes"],
            "design_hits": design["n_hits"], "design_n_events": design["n_documented_events"], "design_pod": design["pod"],
            "design_matched_episodes": design["n_matched_episodes"], "design_precision": design["episode_precision"],
            "tuned_tolerance": tuned["selected_tolerance"],
            "heldout_hits": t["n_hits"], "heldout_n_events": t["n_documented_events"], "heldout_pod": t["pod"],
            "heldout_matched_episodes": t["n_matched_episodes"], "heldout_precision": t["episode_precision"]}


def harmonise(gridded: pd.DataFrame, f: float) -> pd.DataFrame:
    g = gridded.copy()
    p_pa = g[PRES0_MBAR].to_numpy(float) * MBAR_TO_PA
    factor = (1.0 / f) * (REF_PRESSURE_PA / p_pa)
    g[DUST] = g[DUST].to_numpy(float) * factor
    g[UNC] = g[UNC].to_numpy(float) * factor
    return g


def agreement(base: pd.DataFrame, other: pd.DataFrame) -> dict:
    cols = ["dust_anomaly", "storm_episode"] + [f"horizon_{h}sol" for h in HORIZONS]
    m = base[KEYS + cols].merge(other[KEYS + cols], on=KEYS, how="outer", suffixes=("_c", "_h"), indicator=True)
    both = m[m["_merge"] == "both"]
    out = {"n_rows_canonical": len(base), "n_rows_harmonised": len(other), "n_rows_common": len(both),
           "n_rows_only_canonical": int((m["_merge"] == "left_only").sum()),
           "n_rows_only_harmonised": int((m["_merge"] == "right_only").sum())}
    for c in cols:
        a, b = both[f"{c}_c"].astype(bool).to_numpy(), both[f"{c}_h"].astype(bool).to_numpy()
        out[f"{c}_kappa"] = float(cohen_kappa_score(a, b))
        out[f"{c}_jaccard_pos"] = float((a & b).sum() / max((a | b).sum(), 1))
        out[f"{c}_prev_canonical"] = float(a.mean()); out[f"{c}_prev_harmonised"] = float(b.mean())
        out[f"{c}_n_pos_canonical"] = int(a.sum()); out[f"{c}_n_pos_harmonised"] = int(b.sum())
        out[f"{c}_gained"] = int((~a & b).sum()); out[f"{c}_lost"] = int((a & ~b).sum())
    return out


# --------------------------------------------------------------------------- scoring

def observed_future_flag(observed_keys: pd.DataFrame, rows: pd.DataFrame, h: int) -> np.ndarray:
    """True iff the same cell+Mars year has >= 1 observed sol in (s, s+h]."""
    ok = observed_keys.sort_values(KEYS)
    gid = {k: i for i, k in enumerate(ok.groupby(["cell_id", "mars_year"]).groups.keys())}
    okey = np.array([gid[(c, y)] for c, y in zip(ok.cell_id, ok.mars_year)], dtype=np.int64) * 100_000 + ok.sol.to_numpy()
    okey.sort()
    rg = np.array([gid.get((c, y), -1) for c, y in zip(rows.cell_id, rows.mars_year)], dtype=np.int64)
    s = rows.sol.to_numpy().astype(np.int64)
    lo = np.searchsorted(okey, rg * 100_000 + s, side="right")
    hi = np.searchsorted(okey, rg * 100_000 + s + h, side="right")
    return (hi - lo > 0) & (rg >= 0)


def tss_counts(tp, fp, tn, fn):
    with np.errstate(invalid="ignore", divide="ignore"):
        return tp / (tp + fn) - fp / (fp + tn)


def score(y, p, t, cells, rng, with_ci):
    yh = p >= t; y = y.astype(bool)
    tp, fp, tn, fn = (y & yh), (~y & yh), (~y & ~yh), (y & ~yh)
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "prevalence": float(y.mean()) if len(y) else np.nan,
           "tp": int(tp.sum()), "fp": int(fp.sum()), "tn": int(tn.sum()), "fn": int(fn.sum()),
           "tss": float(tss_counts(tp.sum(), fp.sum(), tn.sum(), fn.sum())) if 0 < y.sum() < len(y) else np.nan}
    if with_ci and 0 < y.sum() < len(y):
        # cell-block bootstrap via per-cell confusion counts + multinomial cell weights (identical
        # to resampling cells with replacement, but vectorised)
        uc, inv = np.unique(cells, return_inverse=True)
        cnt = np.stack([np.bincount(inv, weights=a.astype(float), minlength=len(uc)) for a in (tp, fp, tn, fn)], axis=1)
        w = rng.multinomial(len(uc), np.full(len(uc), 1.0 / len(uc)), size=N_BOOT)
        rep = w @ cnt
        b = tss_counts(rep[:, 0], rep[:, 1], rep[:, 2], rep[:, 3])
        out["tss_ci95_lo"], out["tss_ci95_hi"] = (float(x) for x in np.nanpercentile(b, [2.5, 97.5]))
    return out


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gridded-cache", type=Path, default=None, help="optional pickle of QC-accepted gridded retrievals")
    args = ap.parse_args()
    t0 = time.perf_counter()
    out = RUN.parent / ("harmonisation_target_study_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True)
    (out / "materiality_rule.json").write_text(json.dumps(MATERIALITY, indent=2))
    rng = np.random.default_rng(SEED)
    grid, splits_cfg = load_primary_grid(), load_splits_config()
    anomaly_cfg, agg_cfg = load_anomaly_definition_config(), load_aggregation_config()
    cv_cfg, tt_cfg = load_catalogue_validation_config(), load_catalogue_tolerance_tuning_config()
    events = load_real_documented_events(load_real_documented_events_config())
    mcd = RealMCDClimatologyProvider()

    # ---- step 0: ingestion + reproduction gate ----
    if args.gridded_cache and args.gridded_cache.exists():
        gridded = pd.read_pickle(args.gridded_cache)
        gridded = gridded[0] if isinstance(gridded, tuple) else gridded
    else:
        raw = RealEMIRSObservationProvider().load(mars_year_range=(36, 37), seed=42)
        gridded = assign_cells(run_quality_control(raw, load_qc_config()).accepted, grid)
        del raw
        if args.gridded_cache:
            pd.to_pickle(gridded, args.gridded_cache)
    if PRES0_MBAR not in gridded.columns and "surface_pressure_mbar" in gridded.columns:
        gridded = gridded.rename(columns={"surface_pressure_mbar": PRES0_MBAR})  # schema renamed 2026-10-04; same mbar values
    p = gridded[PRES0_MBAR]
    assert p.notna().all() and (p > 0).all(), "every accepted retrieval needs a positive pres0"
    pres_summary = {"n_retrievals": len(gridded), "pres0_mbar_min": float(p.min()), "pres0_mbar_median": float(p.median()),
                    "pres0_mbar_max": float(p.max()),
                    "pressure_factor_610_over_ps_p05_p50_p95": [float(x) for x in np.percentile(REF_PRESSURE_PA / (p * MBAR_TO_PA), [5, 50, 95])]}

    def aggregate(g):
        return aggregate_by_cell_sol(g, min_obs_per_cell_sol=agg_cfg["primary"]["min_obs_per_cell_sol"],
                                     min_pressure_level_coverage_frac=agg_cfg["primary"]["min_pressure_level_coverage_frac"], config=agg_cfg)

    agg_c = aggregate(gridded)
    saved = pd.read_csv(RUN / "data" / "aggregated_cell_sol.csv", usecols=KEYS + [DUST])
    chk = agg_c[KEYS + [DUST]].merge(saved, on=KEYS, suffixes=("", "_s"))
    assert len(chk) == len(saved) == len(agg_c) and np.allclose(chk[DUST], chk[f"{DUST}_s"]), "aggregation not reproduced"
    lab_c, cat_c, nex_c = label_like_run_all(agg_c, mcd, grid, anomaly_cfg, splits_cfg)
    saved_lab = pd.read_csv(RUN / "labels" / "labeled_cell_sol.csv", usecols=KEYS + ["dust_anomaly", "storm_episode"])
    chk = lab_c[KEYS + ["dust_anomaly", "storm_episode"]].merge(saved_lab, on=KEYS, suffixes=("", "_s"))
    assert len(chk) == len(saved_lab) == len(lab_c), "label row set not reproduced"
    assert (chk.dust_anomaly == chk.dust_anomaly_s).all() and (chk.storm_episode == chk.storm_episode_s).all(), "labels not reproduced"
    cm_c = catalogue_metrics(lab_c, cat_c, events, grid, cv_cfg, tt_cfg)
    assert (cm_c["n_qualifying_episodes_my36"], cm_c["design_hits"], cm_c["design_matched_episodes"],
            cm_c["heldout_hits"], cm_c["heldout_matched_episodes"]) == (156, 11, 8, 8, 12), cm_c
    print(f"[0] reproduction gate passed ({time.perf_counter() - t0:.0f}s)", flush=True)

    # ---- step 1: harmonised labels, factor sweep ----
    versions = {"canonical": (lab_c, cat_c, nex_c, cm_c)}
    agree_rows, cat_rows = [], [{"label_version": "canonical", "f": np.nan, "n_qualifying_episodes_all": int(cat_c["qualifies"].sum()),
                                  "n_episodes_excluded_split_boundary": nex_c, **cm_c}]
    for f in F_SWEEP:
        agg_h = aggregate(harmonise(gridded, f))
        lab_h, cat_h, nex_h = label_like_run_all(agg_h, mcd, grid, anomaly_cfg, splits_cfg)
        cm_h = catalogue_metrics(lab_h, cat_h, events, grid, cv_cfg, tt_cfg)
        name = f"harmonised_f{f:.1f}"
        versions[name] = (lab_h, cat_h, nex_h, cm_h)
        agree_rows.append({"label_version": name, "f": f, **agreement(lab_c, lab_h),
                           "dust_tau_star_median": float(agg_h[DUST].median()), "dust_tau_ext_median": float(agg_c[DUST].median())})
        cat_rows.append({"label_version": name, "f": f, "n_qualifying_episodes_all": int(cat_h["qualifies"].sum()),
                         "n_episodes_excluded_split_boundary": nex_h, **cm_h})
        print(f"[1] {name} done ({time.perf_counter() - t0:.0f}s)", flush=True)
    A = pd.DataFrame(agree_rows); A.to_csv(out / "step1_label_agreement.csv", index=False)
    C = pd.DataFrame(cat_rows); C.to_csv(out / "step1_catalogue_validation.csv", index=False)

    # ---- step 2: re-scoring frozen models ----
    supp = sorted((RUN.parent).glob("chapter6_supplementary_*"))[-1]
    study = sorted((RUN.parent).glob("chapter6_threshold_study_*"))[-1]
    P = pd.read_csv(supp / "predictions_test_and_my37_late.csv.gz")
    P = P[P.eval_set == "test"]
    tau = pd.read_csv(study / "threshold_study.csv").set_index(["model", "horizon_sols"])["tau_star"].to_dict()
    observed = agg_c[KEYS]  # identical observed cell-sol set for every label version (every retrieval is harmonisable)
    rows = []
    for name, (lab, _, _, _) in versions.items():
        primary = name in ("canonical", f"harmonised_f{F_PRIMARY:.1f}")
        L = lab[KEYS + ["storm_episode"] + [f"horizon_{h}sol" for h in HORIZONS]]
        for (model, h), g in P.groupby(["model", "horizon"]):
            j = g.merge(L, on=KEYS, how="left")
            missing = j[f"horizon_{h}sol"].isna()
            j = j[~missing]
            y = j[f"horizon_{h}sol"].astype(bool).to_numpy()
            if name == "canonical":
                assert (y == j["y"].astype(bool).to_numpy()).all(), (model, h, "canonical target mismatch")
            obs = observed_future_flag(observed, j, h)
            in_ep = j["storm_episode"].astype(bool).to_numpy()
            subsets = {"all": np.ones(len(j), bool), "observed_horizon": obs, "non_episode_issue": ~in_ep}
            for sub, m in subsets.items():
                for pol, t in (("fixed_0.5", 0.5), ("val_tau_star", tau[(model, h)])):
                    r = score(y[m], j["p"].to_numpy()[m], t, j["cell_id"].to_numpy()[m], rng, with_ci=primary)
                    rows.append({"label_version": name, "model": model, "horizon_sols": h, "subset": sub,
                                 "threshold_policy": pol, "threshold": t, "n_rows_dropped_missing_label": int(missing.sum()), **r})
        print(f"[2] scored {name} ({time.perf_counter() - t0:.0f}s)", flush=True)
    S = pd.DataFrame(rows)
    # gate: canonical/all/fixed 0.5 must equal the saved evaluation JSONs
    for (model, h), g in S[(S.label_version == "canonical") & (S.subset == "all") & (S.threshold_policy == "fixed_0.5")].groupby(["model", "horizon_sols"]):
        mt = json.loads((RUN / "metrics" / "storm_detection" / f"{model}_h{h}_evaluation.json").read_text())["splits"]["test"]["metrics"]
        r = g.iloc[0]
        assert (r.tp, r.fp, r.tn, r.fn) == (mt["tp"], mt["fp"], mt["tn"], mt["fn"]), (model, h)
    S.to_csv(out / "step2_rescoring.csv", index=False)

    # ---- materiality decision ----
    kappa = float(A.loc[A.f == F_PRIMARY, "dust_anomaly_kappa"].iloc[0])
    top_changes = []
    hp = f"harmonised_f{F_PRIMARY:.1f}"
    for pol in ("fixed_0.5", "val_tau_star"):
        for h in HORIZONS:
            sel = S[(S.subset == "all") & (S.threshold_policy == pol) & (S.horizon_sols == h)]
            tc = sel[sel.label_version == "canonical"].sort_values("tss", ascending=False).iloc[0].model
            th = sel[sel.label_version == hp].sort_values("tss", ascending=False).iloc[0].model
            top_changes.append({"policy": pol, "horizon": h, "top_canonical": tc, "top_harmonised": th, "changed": tc != th})
    rerun = (kappa < MATERIALITY["kappa_exceedance_min"]) or any(x["changed"] for x in top_changes)
    decision = {"rule": MATERIALITY, "kappa_dust_anomaly_primary": kappa, "top_model_checks": top_changes,
                "full_rerun_required": bool(rerun), "pressure_summary": pres_summary,
                "runtime_seconds": time.perf_counter() - t0}
    (out / "materiality_decision.json").write_text(json.dumps(decision, indent=2, default=str))
    print(f"done: {out}  full_rerun_required={rerun}  kappa={kappa:.3f}", flush=True)


if __name__ == "__main__":
    main()
