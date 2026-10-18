"""Retrieval-uncertainty propagation analysis (examiner item, Chapter 4 --
new Section 4.4.5 "Uncertainty Propagation from Retrieval Errors";
CHAPTER_4_PARAMETER_VALUES.md Section 11 open item 2, "CDOD retrieval-
uncertainty Monte-Carlo label-stability analysis").

Question: how much does EMIRS dust-optical-depth (CDOD) retrieval error move
the Eq. 4.1-4.3 labels (dust_anomaly -> storm_episode -> horizon targets) and
the frozen canonical models' test scores?

Everything is read from the canonical run
(outputs/real/run_20261004T080014968911Z_seed42) or recomputed with this
package's own pipeline functions -- the same ingestion/QC/aggregation/
labeling/feature code mars/run_all.py uses. Before any perturbation, the
script REPRODUCES the canonical aggregated dust, labels and model test TSS
and aborts if they do not match, so every perturbed number is a controlled
change from the real canonical state.

Error scenarios (all perturb the cell-sol inverse-variance-mean CDOD x; the
climatology mu/sigma are held fixed; perturbed CDOD is clipped at 0 because
optical depth cannot be negative -- the same physical bound QC rule 4 enforces):

  S0+/S0-  examiner's stated retrieval error, fully systematic: x +/- 0.05
           everywhere (deterministic, one "draw" each).
  S1_indep independent per-retrieval errors (the assumption the inverse-
           variance aggregation itself makes): delta ~ N(0, u_comb), u_comb =
           sqrt(1/sum(1/u_i^2)) = the pipeline's own dust_optical_depth_uncertainty.
           Exactly equivalent to perturbing every retrieval by N(0, u_i) and
           re-aggregating.
  S2_corr  errors fully correlated WITHIN a cell-sol (same scene/orbit),
           independent between cell-sols: delta ~ N(0, u_corr), u_corr =
           sum(w_i u_i)/sum(w_i), w_i = 1/u_i^2 -- computed from the raw
           accepted retrievals. Exactly equivalent to one shared standard-
           normal draw per cell-sol scaling every retrieval's own u_i.
  S3_005   examiner's stated magnitude as random error: delta ~ N(0, 0.05),
           independent per cell-sol.
  S4_sol   spatially correlated errors: delta = u_comb * g(mars_year, sol),
           g ~ N(0,1) shared by EVERY cell observed on that sol (a
           planet-wide, per-sol retrieval bias -- worst case for the
           spatial-extent gate A_min=3).

Outputs go to a NEW directory, outputs/real/retrieval_uncertainty_propagation_<UTC>/
(never inside the canonical run directory, whose checksums.json would
otherwise be invalidated).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from data_sources.real_emirs_provider import RealEMIRSObservationProvider
from data_sources.real_mcd_provider import RealMCDClimatologyProvider
from labeling.anomaly_labels import (
    compute_exceedance,
    compute_storm_episodes,
    compute_z_scores,
    load_anomaly_definition_config,
    run_full_labeling_pipeline,
)
from modeling.datasets import build_storm_detection_dataset
from modeling.metrics import classification_metrics
from modeling.train import _horizon_frame, _predict
from preprocessing.aggregation import aggregate_by_cell_sol, load_aggregation_config
from preprocessing.features import build_features, load_features_config
from preprocessing.grid import assign_cells, load_primary_grid
from preprocessing.harmonisation import harmonise_dust_to_reference, load_harmonisation_config
from preprocessing.missing_data import apply_trailing_fill, load_missing_data_config
from preprocessing.quality_control import load_qc_config, run_quality_control
from splitting.chronological_split import assign_split, load_splits_config

MARTIAN_ROOT = Path(__file__).resolve().parent
# Canonical run (harmonised / masked / Eq. 6.1 tau*); override with --run-dir.
CANONICAL_RUN = MARTIAN_ROOT / "outputs" / "real" / "run_20261004T080014968911Z_seed42"
KEYS = ["cell_id", "mars_year", "sol"]
DUST = "dust_optical_depth"
HORIZONS = (1, 3, 5)
EXAMINER_SIGMA = 0.05
SEED = 20260926
# Frozen canonical storm-detection models re-scored under label/feature
# perturbation. Sequence models (LSTM/GRU) are excluded: re-scoring them needs
# the windowing + train-only imputation path of modeling/sequence_models.py.
MODELS = [
    ("baseline", "persistence"), ("baseline", "climatology"),
    ("classical", "logistic_regression"), ("classical", "random_forest_classifier"),
    ("classical", "gradient_boosting_classifier"), ("classical", "knn_classifier"),
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def fast_horizon_labels(df: pd.DataFrame, storm: np.ndarray) -> dict[int, np.ndarray]:
    """Vectorised equivalent of labeling.anomaly_labels.compute_horizon_labels
    (any storm_episode in (s, s+h], same cell + mars_year; NaN when no observed
    sol of the cell falls in (s, s+h], matching the 2026-10-04 masking policy).
    Verified equal to the library function on the canonical labels before use."""
    g = df.groupby(["cell_id", "mars_year"], sort=False).ngroup().to_numpy()
    sol = df["sol"].to_numpy().astype(np.int64)
    order = np.lexsort((sol, g))
    key = g[order].astype(np.int64) * 100_000 + sol[order]
    cs = np.concatenate([[0], np.cumsum(storm[order].astype(np.int64))])
    lo = np.arange(len(key)) + 1
    out = {}
    for h in HORIZONS:
        hi = np.searchsorted(key, key + h, side="right")
        res = np.empty(len(key), dtype=float)
        val = ((cs[hi] - cs[lo]) > 0).astype(float)
        val[(hi - lo) <= 0] = np.nan          # no observed sol in (s, s+h]: undefined target
        res[order] = val
        out[h] = res
    return out


def tss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = np.sum(y_true & y_pred); fn = np.sum(y_true & ~y_pred)
    fp = np.sum(~y_true & y_pred); tn = np.sum(~y_true & ~y_pred)
    pod = tp / (tp + fn) if (tp + fn) else np.nan
    pofd = fp / (fp + tn) if (fp + tn) else np.nan
    return float(pod - pofd)


def flip_stats(orig: np.ndarray, pert: np.ndarray, prefix: str) -> dict:
    orig = orig.astype(bool); pert = pert.astype(bool)
    n_pos = orig.sum()
    return {
        f"{prefix}_flip_rate_all": float(np.mean(orig != pert)),
        f"{prefix}_pos_to_neg_frac_of_pos": float(np.sum(orig & ~pert) / n_pos) if n_pos else np.nan,
        f"{prefix}_neg_to_pos_frac_of_neg": float(np.sum(~orig & pert) / (~orig).sum()),
        f"{prefix}_n_pos_rel_change": float((pert.sum() - n_pos) / n_pos) if n_pos else np.nan,
        f"{prefix}_jaccard_pos": float(np.sum(orig & pert) / np.sum(orig | pert)) if np.sum(orig | pert) else np.nan,
        f"{prefix}_label_noise_tss": tss(orig, pert),
    }


class Labeler:
    """Perturbed-CDOD -> Eq. 4.1-4.3 labels, holding mu/sigma fixed."""

    def __init__(self, aggregated: pd.DataFrame, mcd, grid, anomaly_config: dict):
        self.grid = grid
        self.cfg = anomaly_config
        z_df = compute_z_scores(aggregated, mcd, epsilon=anomaly_config["epsilon"])
        self.base = z_df.reset_index(drop=True)
        self.mu = self.base[f"{DUST}_mu"].to_numpy()
        self.sigma_eff = np.maximum(self.base[f"{DUST}_sigma"].to_numpy(), anomaly_config["epsilon"])
        self.x = self.base[DUST].to_numpy()

    def label(self, x_new: np.ndarray) -> tuple[pd.DataFrame, dict[int, np.ndarray], int]:
        df = self.base.copy()
        df[DUST] = x_new
        df[f"{DUST}_z"] = (x_new - self.mu) / self.sigma_eff  # NaN mu/sigma -> NaN z -> never exceeds, as in the pipeline
        df = compute_exceedance(df, tau=self.cfg["tau_primary"], config=self.cfg)
        df, catalogue = compute_storm_episodes(
            df, self.grid,
            persistence_sols=self.cfg["persistence_sols_primary"],
            min_area_cells=self.cfg["min_area_cells_primary"],
        )
        horizons = fast_horizon_labels(df, df["storm_episode"].to_numpy())
        for h, v in horizons.items():
            df[f"horizon_{h}sol"] = v
        n_qual = int(catalogue["qualifies"].sum()) if len(catalogue) else 0
        return df, horizons, n_qual


def summarise(draws: pd.DataFrame) -> pd.DataFrame:
    num = draws.drop(columns=["scenario", "draw"]).select_dtypes("number")
    rows = []
    for scen, grp in draws.groupby("scenario", sort=False):
        g = grp[num.columns]
        for col in num.columns:
            rows.append({"scenario": scen, "metric": col, "n_draws": len(g), "mean": g[col].mean(),
                         "p05": g[col].quantile(0.05), "p95": g[col].quantile(0.95)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    global CANONICAL_RUN
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-label-draws", type=int, default=100)
    ap.add_argument("--n-e2e-draws", type=int, default=8)
    ap.add_argument("--run-dir", type=Path, default=CANONICAL_RUN)
    ap.add_argument("--gridded-cache", type=Path, default=None,
                    help="optional pickle path: reuse QC-accepted gridded retrievals (+ raw dustuncert) across runs")
    args = ap.parse_args()
    CANONICAL_RUN = args.run_dir.resolve()
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    out_dir = MARTIAN_ROOT / "outputs" / "real" / (
        "retrieval_uncertainty_propagation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out_dir.mkdir(parents=True)
    summary: dict = {"canonical_run": CANONICAL_RUN.name, "seed": SEED,
                     "n_label_draws": args.n_label_draws, "n_e2e_draws": args.n_e2e_draws}

    # ---- 1. real ingestion + QC + gridding + aggregation (as mars/run_all.py steps 3-4) ----
    print("[1] ingestion + QC + gridding + aggregation ...", flush=True)
    grid = load_primary_grid()
    if args.gridded_cache and args.gridded_cache.exists():
        gridded, raw_unc = pd.read_pickle(args.gridded_cache)
    else:
        raw = RealEMIRSObservationProvider().load(mars_year_range=(36, 37), seed=42)
        qc_report = run_quality_control(raw, load_qc_config())
        gridded = assign_cells(qc_report.accepted, grid)
        raw_unc = raw[f"{DUST}_uncertainty"].copy()
        del raw, qc_report
        if args.gridded_cache:
            pd.to_pickle((gridded, raw_unc), args.gridded_cache)
    if "surface_pressure_pa" in gridded.columns:  # caches written before the 2026-10-04 schema rename (same mbar values)
        gridded = gridded.rename(columns={"surface_pressure_pa": "surface_pressure_mbar"})
    gridded_raw = gridded
    gridded, harm_summary = harmonise_dust_to_reference(gridded, load_harmonisation_config())  # as mars/run_all.py step 4
    summary["dust_harmonisation"] = harm_summary
    agg_cfg = load_aggregation_config()
    aggregated = aggregate_by_cell_sol(
        gridded, min_obs_per_cell_sol=agg_cfg["primary"]["min_obs_per_cell_sol"],
        min_pressure_level_coverage_frac=agg_cfg["primary"]["min_pressure_level_coverage_frac"], config=agg_cfg)
    print(f"    {time.perf_counter() - t0:.0f}s", flush=True)

    # per-retrieval dustuncert descriptive statistics
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    def desc(s: pd.Series) -> dict:
        s = s.dropna()
        return {"n": int(len(s)), "mean": float(s.mean()), **{f"p{int(p*100):02d}": float(s.quantile(p)) for p in q}}
    acc_raw = gridded_raw[[DUST, f"{DUST}_uncertainty"]].dropna()
    summary["per_retrieval_unharmonised_tau_ext"] = {
        "accepted_uncertainty_abs": desc(acc_raw[f"{DUST}_uncertainty"]),
        "accepted_dust": desc(acc_raw[DUST]),
        "accepted_frac_uncertainty_le_0.05": float((acc_raw[f"{DUST}_uncertainty"] <= EXAMINER_SIGMA).mean()),
    }
    acc = gridded[[DUST, f"{DUST}_uncertainty"]].dropna()
    summary["per_retrieval"] = {  # harmonised tau* units (what the labels are built from)
        "all_loaded_uncertainty_abs": desc(raw_unc),
        "accepted_uncertainty_abs": desc(acc[f"{DUST}_uncertainty"]),
        "accepted_uncertainty_rel": desc(acc[f"{DUST}_uncertainty"] / acc[DUST].clip(lower=1e-6)),
        "accepted_dust": desc(acc[DUST]),
        "accepted_frac_uncertainty_le_0.05": float((acc[f"{DUST}_uncertainty"] <= EXAMINER_SIGMA).mean()),
        "accepted_frac_rel_uncertainty_in_5_15pct": float(
            ((acc[f"{DUST}_uncertainty"] / acc[DUST]).between(0.05, 0.15)).mean()),
    }

    # u_corr per cell-sol (fully-correlated-within-cell-sol error of the inverse-variance mean)
    eps = agg_cfg["epsilon"]
    g = gridded.loc[gridded[DUST].notna() & gridded[f"{DUST}_uncertainty"].notna(), KEYS + [f"{DUST}_uncertainty"]].copy()
    g["_u"] = g[f"{DUST}_uncertainty"].clip(lower=eps)
    g["_w"] = 1.0 / g["_u"] ** 2
    g["_wu"] = g["_w"] * g["_u"]
    ucorr = g.groupby(KEYS)[["_w", "_wu"]].sum()
    ucorr = (ucorr["_wu"] / ucorr["_w"]).rename("u_corr").reset_index()
    aggregated = aggregated.merge(ucorr, on=KEYS, how="left")

    # ---- reproduction check 1: aggregated CDOD == canonical saved CSV ----
    saved_agg = pd.read_csv(CANONICAL_RUN / "data" / "aggregated_cell_sol.csv", usecols=KEYS + [DUST, f"{DUST}_uncertainty"])
    m = aggregated[KEYS + [DUST, f"{DUST}_uncertainty"]].merge(saved_agg, on=KEYS, suffixes=("", "_saved"))
    assert len(m) == len(saved_agg) == len(aggregated), (len(m), len(saved_agg), len(aggregated))
    assert np.allclose(m[DUST], m[f"{DUST}_saved"], rtol=1e-9, atol=1e-12)
    assert np.allclose(m[f"{DUST}_uncertainty"], m[f"{DUST}_uncertainty_saved"], rtol=1e-9, atol=1e-12)
    print(f"    reproduction OK: {len(aggregated)} cell-sols match canonical aggregated_cell_sol.csv", flush=True)

    # ---- 2. canonical labels via the library, then the fast Labeler ----
    print("[2] canonical labels ...", flush=True)
    mcd = RealMCDClimatologyProvider()
    anomaly_config = load_anomaly_definition_config()
    lib_labeled, _ = run_full_labeling_pipeline(aggregated, mcd, grid, config=anomaly_config)
    labeler = Labeler(aggregated, mcd, grid, anomaly_config)
    base_df, base_h, base_nqual = labeler.label(labeler.x.copy())
    chk = base_df[KEYS + ["dust_anomaly", "storm_episode"] + [f"horizon_{h}sol" for h in HORIZONS]].merge(
        lib_labeled[KEYS + ["dust_anomaly", "storm_episode"] + [f"horizon_{h}sol" for h in HORIZONS]],
        on=KEYS, suffixes=("", "_lib"))
    assert len(chk) == len(base_df)
    for c in ["dust_anomaly", "storm_episode"] + [f"horizon_{h}sol" for h in HORIZONS]:
        assert (chk[c].astype(float).fillna(-1) == chk[f"{c}_lib"].astype(float).fillna(-1)).all(), c
    saved_lab = pd.read_csv(CANONICAL_RUN / "labels" / "labeled_cell_sol.csv",
                            usecols=KEYS + ["dust_anomaly", "storm_episode"] + [f"horizon_{h}sol" for h in HORIZONS])
    chk2 = base_df[KEYS + ["dust_anomaly", "storm_episode"]].merge(saved_lab, on=KEYS, suffixes=("", "_saved"))
    assert len(chk2) == len(saved_lab)
    assert (chk2["dust_anomaly"] == chk2["dust_anomaly_saved"]).all()
    assert (chk2["storm_episode"] == chk2["storm_episode_saved"]).all()
    print(f"    reproduction OK: labels match library + canonical labeled_cell_sol.csv ({len(saved_lab)} rows)", flush=True)

    # evaluation rows = the canonical labeled rows (post split-boundary exclusion)
    eval_mask = base_df.set_index(KEYS).index.isin(saved_lab.set_index(KEYS).index)
    z0 = base_df[f"{DUST}_z"].to_numpy()
    u_comb = base_df[f"{DUST}_uncertainty"].to_numpy()
    u_corr = base_df["u_corr"].to_numpy()
    tau = anomaly_config["tau_primary"]
    u_z = u_comb / labeler.sigma_eff
    border_fixed = eval_mask & (np.abs(z0 - tau) < 0.5)
    border_1u = eval_mask & (np.abs(z0 - tau) <= u_z)
    border_005 = eval_mask & (np.abs(z0 - tau) <= EXAMINER_SIGMA / labeler.sigma_eff)
    summary["cell_sol_uncertainty"] = {
        "n_eval_cell_sols": int(eval_mask.sum()),
        "u_comb": desc(pd.Series(u_comb[eval_mask])),
        "u_corr": desc(pd.Series(u_corr[eval_mask])),
        "u_corr_over_u_comb": desc(pd.Series(u_corr[eval_mask] / u_comb[eval_mask])),
        "n_retrievals_per_cell_sol": desc(base_df.loc[eval_mask, f"{DUST}_n"]),
        "frac_single_retrieval_cell_sols": float((base_df.loc[eval_mask, f"{DUST}_n"] == 1).mean()),
        "u_z_u_comb_over_sigma_clim": desc(pd.Series(u_z[eval_mask])),
        "examiner_0.05_over_sigma_clim": desc(pd.Series(EXAMINER_SIGMA / labeler.sigma_eff[eval_mask])),
        "n_borderline_abs_z_minus_tau_lt_0.5": int(border_fixed.sum()),
        "n_borderline_within_1_u_comb": int(border_1u.sum()),
        "n_borderline_within_0.05": int(border_005.sum()),
        "canonical_positive_rates": {
            "dust_anomaly": float(base_df.loc[eval_mask, "dust_anomaly"].mean()),
            "storm_episode": float(base_df.loc[eval_mask, "storm_episode"].mean()),
            **{f"horizon_{h}sol": float(np.nanmean(base_h[h][eval_mask])) for h in HORIZONS},
            **{f"horizon_{h}sol_defined_frac": float(np.mean(~np.isnan(base_h[h][eval_mask]))) for h in HORIZONS},
        },
        "canonical_n_qualifying_episodes_pre_exclusion": base_nqual,
    }

    # ---- 3. frozen canonical models: reproduce test TSS, cache test predictions ----
    print("[3] frozen models ...", flush=True)
    ds = pd.read_csv(CANONICAL_RUN / "features" / "storm_detection_dataset.csv")
    fman = json.loads((CANONICAL_RUN / "features" / "storm_detection_feature_manifest.json").read_text())["feature_manifest"]
    splits_cfg = load_splits_config()
    key_index = pd.MultiIndex.from_frame(base_df[KEYS])
    model_cache = {}
    for h in HORIZONS:
        target = f"target_storm_horizon_h{h}"
        frame = _horizon_frame(ds, fman, target)
        frame["split"] = assign_split(frame, splits_cfg)
        train, test = frame.loc[frame["split"] == "train"], frame.loc[frame["split"] == "test"]
        imputer = SimpleImputer(strategy="median").fit(train[fman])
        pos = key_index.get_indexer(pd.MultiIndex.from_frame(test[KEYS]))
        assert (pos >= 0).all()
        assert (base_h[h][pos] == test[target].to_numpy()).all(), f"horizon {h} target mismatch"
        for kind, name in MODELS:
            model = joblib.load(CANONICAL_RUN / "models" / "storm_detection" / f"{name}_h{h}.joblib")
            p = _predict(model, test if kind == "baseline" else imputer.transform(test[fman]), task_type="classification")
            ev = json.loads((CANONICAL_RUN / "metrics" / "storm_detection" / f"{name}_h{h}_evaluation.json").read_text())
            reported = ev["splits"]["test"]["metrics"]["tss"]
            thr = float(ev.get("decision_threshold", 0.5))  # Eq. 6.1 tau* (fixed 0.5 for pre-2026-10-04 runs)
            mine = classification_metrics(test[target].to_numpy(), p, threshold=thr)["tss"]
            assert np.isclose(mine, reported, atol=1e-9), (name, h, mine, reported)
            ci = ev["splits"]["test"]["ci"]["tss"]
            ls = test["solar_longitude_ls"].to_numpy()
            model_cache[(name, h)] = {"pos": pos, "pred": p >= thr, "prob": p, "thr": thr, "tss": reported, "ci": ci,
                                      "imputer": imputer, "kind": kind, "model": model, "test_keys": test[KEYS],
                                      "autumn": (ls >= 180) & (ls < 270), "winter": ls >= 270}
    print("    reproduction OK: all frozen-model test TSS match evaluation JSONs", flush=True)

    # test TSS stratified by DUST retrieval uncertainty quartile (the pipeline's
    # own uncertainty_summary breakdown is dominated by surface-temperature
    # uncertainty in K, so it cannot answer this)
    strat_rows = []
    for (name, h), mc in model_cache.items():
        u_test = u_comb[mc["pos"]]
        y = base_h[h][mc["pos"]]
        bins = pd.qcut(u_test, q=4, duplicates="drop")
        for b in bins.categories:
            sel = np.asarray(bins == b)
            strat_rows.append({"model": name, "horizon": h, "dust_u_bin": str(b), "n": int(sel.sum()),
                               "prevalence": float(y[sel].mean()), "tss": tss(y[sel], mc["pred"][sel])})
    pd.DataFrame(strat_rows).to_csv(out_dir / "test_tss_by_dust_uncertainty_quartile.csv", index=False)
    summary["frozen_models"] = {f"{n}_h{h}": {"test_tss": v["tss"], "test_tss_ci": v["ci"], "threshold": v["thr"],
                                              "test_tss_autumn": tss(base_h[h][v["pos"]][v["autumn"]], v["pred"][v["autumn"]]),
                                              "test_tss_winter": tss(base_h[h][v["pos"]][v["winter"]], v["pred"][v["winter"]])}
                                for (n, h), v in model_cache.items()}

    # ---- 4. label-level Monte-Carlo ----
    print("[4] label-level Monte-Carlo ...", flush=True)
    x0 = labeler.x
    sol_groups = base_df.groupby(["mars_year", "sol"], sort=False).ngroup().to_numpy()
    n_sol_groups = sol_groups.max() + 1
    scenarios = {
        "S0_plus_0.05": lambda: np.full_like(x0, EXAMINER_SIGMA),
        "S0_minus_0.05": lambda: np.full_like(x0, -EXAMINER_SIGMA),
        "S1_indep": lambda: rng.standard_normal(len(x0)) * u_comb,
        "S2_corr_within_cell_sol": lambda: rng.standard_normal(len(x0)) * u_corr,
        "S3_indep_0.05": lambda: rng.standard_normal(len(x0)) * EXAMINER_SIGMA,
        "S4_sol_shared": lambda: rng.standard_normal(n_sol_groups)[sol_groups] * u_comb,
    }
    rows = []
    for scen, draw_fn in scenarios.items():
        n = 1 if scen.startswith("S0") else args.n_label_draws
        ts = time.perf_counter()
        for d in range(n):
            x_new = np.clip(x0 + draw_fn(), 0.0, None)
            df, hz, nqual = labeler.label(x_new)
            a0 = base_df["dust_anomaly"].to_numpy(); a1 = df["dust_anomaly"].to_numpy()
            row = {"scenario": scen, "draw": d, "n_qualifying_episodes": nqual}
            row.update(flip_stats(a0[eval_mask], a1[eval_mask], "dust_anomaly"))
            for name, bm in [("border_abs_lt_0.5", border_fixed), ("border_1u", border_1u), ("border_0.05", border_005)]:
                row[f"dust_anomaly_flip_rate_{name}"] = float(np.mean(a0[bm] != a1[bm])) if bm.any() else np.nan
            row.update(flip_stats(base_df["storm_episode"].to_numpy()[eval_mask], df["storm_episode"].to_numpy()[eval_mask], "storm_episode"))
            for h in HORIZONS:
                dm = eval_mask & ~np.isnan(base_h[h])  # defined targets (observation-based, identical in every draw)
                row.update(flip_stats(base_h[h][dm], hz[h][dm], f"horizon_{h}sol"))
            for (name, h), mc in model_cache.items():
                yy = hz[h][mc["pos"]]
                row[f"tss_label_only_{name}_h{h}"] = tss(yy, mc["pred"])
                row[f"tss_label_only_autumn_{name}_h{h}"] = tss(yy[mc["autumn"]], mc["pred"][mc["autumn"]])
                row[f"tss_label_only_winter_{name}_h{h}"] = tss(yy[mc["winter"]], mc["pred"][mc["winter"]])
            rows.append(row)
        print(f"    {scen}: {n} draws, {time.perf_counter() - ts:.0f}s", flush=True)
    label_draws = pd.DataFrame(rows)
    label_draws.to_csv(out_dir / "label_level_draws.csv", index=False)
    summarise(label_draws).to_csv(out_dir / "label_level_summary.csv", index=False)

    # ---- 5. end-to-end: perturbed CDOD -> fill -> features -> frozen models ----
    print("[5] end-to-end feature propagation ...", flush=True)
    missing_cfg = load_missing_data_config()
    features_cfg = load_features_config()
    e2e_rows = []
    agg_idx = pd.MultiIndex.from_frame(aggregated[KEYS])
    base_pos = agg_idx.get_indexer(key_index)
    for scen in ["S1_indep", "S2_corr_within_cell_sol", "S3_indep_0.05"]:
        for d in range(args.n_e2e_draws):
            ts = time.perf_counter()
            x_new = np.clip(x0 + scenarios[scen](), 0.0, None)
            df, hz, _ = labeler.label(x_new)
            agg_p = aggregated.copy()
            agg_p.loc[base_pos, DUST] = x_new
            filled = apply_trailing_fill(agg_p, max_gap_sols=missing_cfg["primary"]["max_trailing_fill_gap_sols"], config=missing_cfg)
            fz = compute_exceedance(compute_z_scores(filled, mcd, epsilon=anomaly_config["epsilon"]),
                                    tau=anomaly_config["tau_primary"], config=anomaly_config)
            feats = build_features(fz, grid, config=features_cfg)
            ds_p, _ = build_storm_detection_dataset(df, feats)
            ds_p = ds_p.set_index(KEYS)
            row = {"scenario": scen, "draw": d}
            for (name, h), mc in model_cache.items():
                test_p = ds_p.reindex(pd.MultiIndex.from_frame(mc["test_keys"]))
                X = test_p.reset_index() if mc["kind"] == "baseline" else mc["imputer"].transform(test_p[fman])
                p = _predict(mc["model"], X, task_type="classification")
                pred = p >= mc["thr"]
                y_pert = hz[h][mc["pos"]]; y_orig = base_h[h][mc["pos"]]
                a = mc["autumn"]
                row[f"tss_features_and_labels_autumn_{name}_h{h}"] = tss(y_pert[a], pred[a])
                row[f"pred_flip_rate_{name}_h{h}"] = float(np.mean(pred != mc["pred"]))
                row[f"prob_abs_change_mean_{name}_h{h}"] = float(np.nanmean(np.abs(p - mc["prob"])))
                row[f"tss_features_only_{name}_h{h}"] = tss(y_orig, pred)
                row[f"tss_features_and_labels_{name}_h{h}"] = tss(y_pert, pred)
            e2e_rows.append(row)
            print(f"    {scen} draw {d}: {time.perf_counter() - ts:.0f}s", flush=True)
    e2e = pd.DataFrame(e2e_rows)
    e2e.to_csv(out_dir / "end_to_end_draws.csv", index=False)
    summarise(e2e).to_csv(out_dir / "end_to_end_summary.csv", index=False)

    summary["runtime_seconds"] = time.perf_counter() - t0
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"done: {out_dir} ({summary['runtime_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
