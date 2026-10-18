"""Chapter 6 examiner-gap supplement (Martian storm detection): per-horizon
confusion matrices, per-season metrics with bootstrap CIs, calibration
(reliability bins, ECE, Murphy Brier decomposition), PR curves, and McNemar /
paired block-bootstrap significance tests.

Replays every frozen canonical storm-detection model (6 baseline/classical +
18 LSTM/GRU, x 3 horizons = 36 model/horizon pairs) from the canonical run
outputs/real/run_20261004T080014968911Z_seed42 (override with --run-dir) without retraining:
  * tabular models: joblib checkpoint + train-fitted median imputer (the same
    deterministic SimpleImputer modeling/train.py fits);
  * sequence models: state_dict checkpoint + the exact windowing
    (modeling/sequence_models.build_sequence_dataset) and train-only imputation
    (modeling/train._impute_sequence_train_only) of the original run, on a
    features_df rebuilt from the run's saved trailing_filled_cell_sol.csv.
Each replayed model's test TP/FP/TN/FN is asserted equal to its saved
*_evaluation.json before anything is written.

Seasons: labeling.reporting.assign_mars_season (Ls quadrants, northern-
hemisphere names). The canonical test split (MY37 sols 1-501) contains no
Northern Winter and only 14 Northern Spring cell-sols. MY37 sols 502-668
(Ls ~262-360) ARE in the canonical data but fall outside every split
(configs/splits.yaml's test upper bound predates their availability); they
were never used for training, early stopping, or selection, so they are scored
here as a clearly separated SUPPLEMENTARY out-of-window set ("my37_late").

Bootstrap: block = cell_id (configs/evaluation.yaml convention), 200 replicates,
both 90% (project convention) and 95% (alpha = 0.05) percentile intervals.
Decision threshold: each model's recorded operating point from its *_evaluation.json
(`decision_threshold`: Eq. 6.1 validation-selected tau* for runs from 2026-10-04;
runs without the field used the fixed 0.5). Run dir via --run-dir (default: the
2026-08-13 run, for reproducing the earlier chapter6.md numbers).

Outputs: outputs/real/chapter6_supplementary_<UTC>/
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest, chi2
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from data_sources.real_mcd_provider import RealMCDClimatologyProvider
from labeling.anomaly_labels import compute_exceedance, compute_z_scores, load_anomaly_definition_config
from labeling.reporting import assign_mars_season
from modeling.sequence_models import SequenceModel, build_sequence_dataset
from modeling.train import _horizon_frame, _impute_sequence_train_only, _predict
from preprocessing.features import build_features, load_features_config
from preprocessing.grid import load_primary_grid
from splitting.chronological_split import assign_split, load_splits_config

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "outputs" / "real" / "run_20261004T080014968911Z_seed42"
KEYS = ["cell_id", "mars_year", "sol"]
HORIZONS = (1, 3, 5)
THRESHOLD = 0.5
SEED = 44
N_BOOT = 200
N_BINS = 10
TABULAR = [("baseline", "persistence"), ("baseline", "climatology"), ("baseline", "seasonal_mean"),
           ("classical", "logistic_regression"), ("classical", "random_forest_classifier"),
           ("classical", "gradient_boosting_classifier"), ("classical", "knn_classifier")]
SEQUENCE = [f"{r}_w{w}" for r in ("lstm", "gru") for w in (3, 5, 10)]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def confusion(y, p, thr=THRESHOLD):
    yhat = p >= thr; y = y.astype(bool)
    return int((y & yhat).sum()), int((~y & yhat).sum()), int((~y & ~yhat).sum()), int((y & ~yhat).sum())


def tss_from(tp, fp, tn, fn):
    pod = tp / (tp + fn) if tp + fn else np.nan
    pofd = fp / (fp + tn) if fp + tn else np.nan
    return pod - pofd


def safe_auc(y, p, fn):
    return float(fn(y, p)) if 0 < y.sum() < len(y) else np.nan


def reliability(y, p):
    y = y.astype(float); edges = np.linspace(0, 1, N_BINS + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, N_BINS - 1)
    n = len(y); obar = y.mean(); rows = []; rel = res = ece = 0.0; mce = 0.0
    for b in range(N_BINS):
        m = idx == b; nb = int(m.sum())
        if nb == 0:
            rows.append({"bin_lo": edges[b], "bin_hi": edges[b + 1], "n": 0}); continue
        pb, ob = float(p[m].mean()), float(y[m].mean())
        rows.append({"bin_lo": edges[b], "bin_hi": edges[b + 1], "n": nb, "n_pos": int(y[m].sum()), "mean_pred": pb, "obs_freq": ob})
        rel += nb * (pb - ob) ** 2; res += nb * (ob - obar) ** 2; ece += nb * abs(pb - ob); mce = max(mce, abs(pb - ob))
    brier = float(np.mean((p - y) ** 2)); rel /= n; res /= n; ece /= n; unc = obar * (1 - obar)
    return pd.DataFrame(rows), {"brier": brier, "reliability": rel, "resolution": res, "uncertainty": unc,
                                "within_bin_residual": brier - (rel - res + unc), "ece": ece, "mce": mce,
                                "brier_skill_score_vs_base_rate": 1 - brier / unc if unc else np.nan}


def cell_bootstrap(y, p, cells, rng, thr=THRESHOLD):
    ub, inv = np.unique(cells, return_inverse=True)
    idx_by = [np.where(inv == k)[0] for k in range(len(ub))]
    out = {"tss": [], "roc_auc": [], "pr_auc": []}
    for _ in range(N_BOOT):
        ii = np.concatenate([idx_by[k] for k in rng.integers(0, len(ub), len(ub))])
        yy, pp = y[ii], p[ii]
        out["tss"].append(tss_from(*confusion(yy, pp, thr))); out["roc_auc"].append(safe_auc(yy, pp, roc_auc_score))
        out["pr_auc"].append(safe_auc(yy, pp, average_precision_score))
    return {k: np.array(v) for k, v in out.items()}


def ci(a, level):
    if np.all(np.isnan(a)):
        return (np.nan, np.nan)
    return tuple(np.nanpercentile(a, [(1 - level) / 2 * 100, (1 + level) / 2 * 100]))


def thr_of(model: str, h: int) -> float:
    """Each model's recorded operating point (Eq. 6.1 validation-selected tau*;
    the fixed 0.5 fallback applies only if the evaluation JSON records none)."""
    ev = json.loads((RUN / "metrics" / "storm_detection" / f"{model}_h{h}_evaluation.json").read_text())
    return float(ev.get("decision_threshold", THRESHOLD))


def main() -> None:
    import argparse
    global RUN
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=RUN)
    RUN = ap.parse_args().run_dir.resolve()
    t0 = time.perf_counter()
    out = ROOT / "outputs" / "real" / ("chapter6_supplementary_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True)
    rng = np.random.default_rng(SEED)
    splits_cfg = load_splits_config()
    grid = load_primary_grid()

    # ---- rebuild features_df exactly as mars/run_all.py steps 7 + 10 ----
    print("[0] rebuilding features_df ...", flush=True)
    filled = pd.read_csv(RUN / "data" / "trailing_filled_cell_sol.csv")
    anomaly_cfg = load_anomaly_definition_config()
    fz = compute_exceedance(compute_z_scores(filled, RealMCDClimatologyProvider(), epsilon=anomaly_cfg["epsilon"]),
                            tau=anomaly_cfg["tau_primary"], config=anomaly_cfg)
    features_df = build_features(fz, grid, config=load_features_config())
    print(f"    {time.perf_counter() - t0:.0f}s", flush=True)

    ds = pd.read_csv(RUN / "features" / "storm_detection_dataset.csv")
    fman = json.loads((RUN / "features" / "storm_detection_feature_manifest.json").read_text())["feature_manifest"]

    preds = []   # long table: eval_set, model, horizon, keys, y, p, Ls
    for h in HORIZONS:
        target = f"target_storm_horizon_h{h}"
        frame = _horizon_frame(ds, fman, target)
        frame["split"] = assign_split(frame, splits_cfg)
        late = (frame["mars_year"] == splits_cfg["test_mars_year"]) & (frame["sol"] > splits_cfg["test_sol_range"][1])
        frame.loc[late & (frame["split"] == "unassigned"), "split"] = "my37_late"
        parts = {s: frame.loc[frame["split"] == s] for s in ("train", "val", "test", "my37_late")}
        imputer = SimpleImputer(strategy="median").fit(parts["train"][fman])
        # my37_late exists only when the split leaves late MY37 unassigned (the
        # 2026-08-13 design); under the 2026-10-04 split those sols are test.
        eval_sets = ["test"] + (["my37_late"] if len(parts["my37_late"]) else [])

        def record(eval_set, model, df_meta, y, p):
            preds.append(pd.DataFrame({"eval_set": eval_set, "model": model, "horizon": h,
                                       "cell_id": df_meta["cell_id"].to_numpy(), "mars_year": df_meta["mars_year"].to_numpy(),
                                       "sol": df_meta["sol"].to_numpy(), "y": np.asarray(y).astype(int), "p": np.asarray(p, dtype=float)}))

        for kind, name in TABULAR:
            model = joblib.load(RUN / "models" / "storm_detection" / f"{name}_h{h}.joblib")
            for s in eval_sets:
                d = parts[s]
                p = _predict(model, d if kind == "baseline" else imputer.transform(d[fman]), task_type="classification")
                record(s, name, d, d[target], p)
        # sequence models
        seqs = {}
        for w in (3, 5, 10):
            built = {s: build_sequence_dataset(parts[s], features_df, fman, window_length_sols=w, target_col=target)
                     for s in ["train", "val"] + eval_sets}
            imputed = _impute_sequence_train_only(built["train"][0], *[built[s][0] for s in eval_sets])
            seqs[w] = {s: (imputed[i + 1], built[s][1], built[s][2]) for i, s in enumerate(eval_sets)}
        for name in SEQUENCE:
            rnn, w = name.split("_w"); w = int(w)
            prov = json.loads((RUN / "metrics" / "storm_detection" / f"{name}_h{h}_provenance.json").read_text())
            hp = {k: prov["exact_parameters"][k] for k in ("hidden_size", "num_layers", "dropout")}
            model = SequenceModel(input_size=len(fman), rnn_type=rnn, **hp)
            model.load_state_dict(torch.load(RUN / "models" / "storm_detection" / f"{name}_h{h}.pt", map_location="cpu"))
            model.eval()
            for s in eval_sets:
                X, y, meta = seqs[w][s]
                if len(X) == 0:
                    continue
                with torch.no_grad():
                    p = 1 / (1 + np.exp(-model(torch.tensor(X, dtype=torch.float32)).numpy().ravel()))
                record(s, name, meta, y, p)
        print(f"[h={h}] replayed ({time.perf_counter() - t0:.0f}s)", flush=True)

    P = pd.concat(preds, ignore_index=True)
    ls = ds[KEYS + ["solar_longitude_ls"]].drop_duplicates(KEYS)
    P = P.merge(ls, on=KEYS, how="left")
    P["season"] = assign_mars_season(P["solar_longitude_ls"])

    # ---- verification: replayed test confusion == saved evaluation JSON ----
    cm_rows = []
    for (model, h), g in P[P.eval_set == "test"].groupby(["model", "horizon"]):
        ev = json.loads((RUN / "metrics" / "storm_detection" / f"{model}_h{h}_evaluation.json").read_text())
        for split in ("train", "val", "test"):
            mt = ev["splits"][split]["metrics"]
            cm_rows.append({"model": model, "horizon_sols": h, "split": split, "threshold": thr_of(model, h),
                            "tp": int(mt["tp"]), "fp": int(mt["fp"]), "tn": int(mt["tn"]), "fn": int(mt["fn"]),
                            "n": int(ev["splits"][split]["n"]), "prevalence": mt["prevalence"], "pod": mt["pod"], "far": mt["far"],
                            "tss": mt["tss"], "hss": mt["hss"], "pr_auc": mt["pr_auc"], "roc_auc": mt["roc_auc"], "brier": mt["brier"],
                            "tss_ci90": ev["splits"][split]["ci"]["tss"]})
        mt = ev["splits"]["test"]["metrics"]
        mine = confusion(g["y"].to_numpy(), g["p"].to_numpy(), thr_of(model, h))
        assert mine == (mt["tp"], mt["fp"], mt["tn"], mt["fn"]), (model, h, mine, (mt["tp"], mt["fp"], mt["tn"], mt["fn"]))
    pd.DataFrame(cm_rows).to_csv(out / "confusion_matrices_all_models_horizons_splits.csv", index=False)
    print(f"[1] verification OK: all {P[P.eval_set=='test'].groupby(['model','horizon']).ngroups} test confusion matrices match evaluation JSONs", flush=True)
    P.to_csv(out / "predictions_test_and_my37_late.csv.gz", index=False, compression="gzip")

    # ---- overall + per-season metrics with cell-block bootstrap ----
    rows, rel_rows, pr_rows = [], [], []
    for (es, model, h), g in P.groupby(["eval_set", "model", "horizon"]):
        for season, gg in [("ALL", g)] + list(g.groupby("season")):
            y, p, cells = gg["y"].to_numpy(), gg["p"].to_numpy(), gg["cell_id"].to_numpy()
            t_m = thr_of(model, h)
            tp, fp, tn, fn = confusion(y, p, t_m)
            row = {"eval_set": es, "model": model, "horizon_sols": h, "season": season, "n": len(y), "n_pos": int(y.sum()),
                   "prevalence": float(y.mean()) if len(y) else np.nan, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                   "tss": tss_from(tp, fp, tn, fn), "roc_auc": safe_auc(y, p, roc_auc_score),
                   "pr_auc": safe_auc(y, p, average_precision_score)}
            if 0 < y.sum() < len(y) and len(np.unique(cells)) > 1:
                b = cell_bootstrap(y, p, cells, rng, t_m)
                for k in ("tss", "roc_auc", "pr_auc"):
                    row[f"{k}_ci90_lo"], row[f"{k}_ci90_hi"] = ci(b[k], 0.90)
                    row[f"{k}_ci95_lo"], row[f"{k}_ci95_hi"] = ci(b[k], 0.95)
                    row[f"{k}_n_nan_reps"] = int(np.isnan(b[k]).sum())
            if season == "ALL" and len(y):
                bins, cal = reliability(y, p); row.update(cal)
                bins.insert(0, "horizon_sols", h); bins.insert(0, "model", model); bins.insert(0, "eval_set", es)
                rel_rows.append(bins)
                if 0 < y.sum() < len(y):
                    prec, rec, _ = precision_recall_curve(y, p)
                    keep = np.unique(np.linspace(0, len(prec) - 1, min(len(prec), 300)).astype(int))
                    pr_rows.append(pd.DataFrame({"eval_set": es, "model": model, "horizon_sols": h, "recall": rec[keep], "precision": prec[keep]}))
            rows.append(row)
    M = pd.DataFrame(rows)
    M.to_csv(out / "metrics_overall_and_by_season.csv", index=False)
    pd.concat(rel_rows).to_csv(out / "reliability_bins.csv", index=False)
    pd.concat(pr_rows).to_csv(out / "pr_curves.csv", index=False)
    print(f"[2] metrics/seasons/calibration done ({time.perf_counter() - t0:.0f}s)", flush=True)

    # ---- significance: every model vs persistence, per horizon, on identical test rows ----
    sig_rows = []
    rng2 = np.random.default_rng(SEED + 1)
    T = P[P.eval_set == "test"]
    for h in HORIZONS:
        ref = T[(T.model == "persistence") & (T.horizon == h)].set_index(KEYS)
        for model in sorted(T.model.unique()):
            if model == "persistence":
                continue
            g = T[(T.model == model) & (T.horizon == h)].set_index(KEYS)
            j = ref.join(g, lsuffix="_a", rsuffix="_b", how="inner")
            y = j["y_a"].to_numpy(); pa = j["p_a"].to_numpy(); pb = j["p_b"].to_numpy()
            assert (j["y_a"] == j["y_b"]).all()
            ta, tb = thr_of("persistence", h), thr_of(model, h)
            ca = (pa >= ta) == y.astype(bool); cb = (pb >= tb) == y.astype(bool)
            n01 = int((~ca & cb).sum()); n10 = int((ca & ~cb).sum())
            stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if n01 + n10 else np.nan
            cells = j.index.get_level_values("cell_id").to_numpy()
            ub, inv = np.unique(cells, return_inverse=True); idx_by = [np.where(inv == k)[0] for k in range(len(ub))]
            diffs = []
            for _ in range(N_BOOT):
                ii = np.concatenate([idx_by[k] for k in rng2.integers(0, len(ub), len(ub))])
                diffs.append(tss_from(*confusion(y[ii], pa[ii], ta)) - tss_from(*confusion(y[ii], pb[ii], tb)))
            lo, hi = ci(np.array(diffs), 0.95)
            sig_rows.append({"horizon_sols": h, "model_a": "persistence", "model_b": model, "n_paired": len(y),
                             "tss_a": tss_from(*confusion(y, pa, ta)), "tss_b": tss_from(*confusion(y, pb, tb)),
                             "tss_diff_a_minus_b": tss_from(*confusion(y, pa, ta)) - tss_from(*confusion(y, pb, tb)),
                             "tss_diff_ci95_lo": lo, "tss_diff_ci95_hi": hi, "block_bootstrap_significant_at_0.05": bool(lo > 0 or hi < 0),
                             "mcnemar_n01_a_wrong_b_right": n01, "mcnemar_n10_a_right_b_wrong": n10, "mcnemar_chi2_cc": stat,
                             "mcnemar_log10_p": chi2.logsf(stat, 1) / np.log(10) if n01 + n10 else np.nan,
                             "mcnemar_exact_p": binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else np.nan})
    S = pd.DataFrame(sig_rows)
    order = np.argsort(S["mcnemar_log10_p"].fillna(0).to_numpy()); holm = np.zeros(len(S), bool)
    for rank, i in enumerate(order):
        if S.loc[i, "mcnemar_log10_p"] <= np.log10(0.05 / (len(S) - rank)): holm[i] = True
        else: break
    S["mcnemar_significant_holm_0.05"] = holm
    S.to_csv(out / "significance_vs_persistence.csv", index=False)
    print("[3] significance done", flush=True)

    # ---- figures ----
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
    show = ["persistence", "logistic_regression", "random_forest_classifier", "gradient_boosting_classifier",
            "knn_classifier", "lstm_w3", "gru_w3", "climatology"]
    color = dict(zip(show, PALETTE))
    PR = pd.concat(pr_rows)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="#fcfcfb", sharey=True)
    for ax, h in zip(axes, HORIZONS):
        ax.set_facecolor("#fcfcfb"); ax.grid(color="#e6e5e0", lw=0.6)
        for m in show:
            g = PR[(PR.eval_set == "test") & (PR.model == m) & (PR.horizon_sols == h)]
            if len(g) == 0: continue
            ap = M[(M.eval_set == "test") & (M.model == m) & (M.horizon_sols == h) & (M.season == "ALL")].pr_auc.iloc[0]
            ax.plot(g.recall, g.precision, color=color[m], lw=1.8, label=f"{m} ({ap:.3f})")
        prev = M[(M.eval_set == "test") & (M.model == "persistence") & (M.horizon_sols == h) & (M.season == "ALL")].prevalence.iloc[0]
        ax.axhline(prev, color="#52514e", ls="--", lw=1, label=f"no-skill ({prev:.3f})")
        ax.set_title(f"horizon {h} sol", fontsize=9); ax.set_xlabel("Recall (POD)"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.legend(fontsize=6, frameon=False, title="PR-AUC", title_fontsize=6)
    axes[0].set_ylabel("Precision")
    fig.suptitle("Mars storm detection: precision–recall curves, canonical test split (MY37 sols 1–501)", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "figure_pr_curves_test.png", dpi=200); plt.close(fig)

    R = pd.concat(rel_rows)
    fig = plt.figure(figsize=(11, 7.4), facecolor="#fcfcfb")
    gs = fig.add_gridspec(4, 4, height_ratios=[3, 1, 3, 1], hspace=0.55, wspace=0.3)
    for k, m in enumerate(show):
        r, c = 2 * (k // 4), k % 4
        g = R[(R.eval_set == "test") & (R.model == m) & (R.horizon_sols == 1) & (R.n > 0)]
        ax = fig.add_subplot(gs[r, c]); axn = fig.add_subplot(gs[r + 1, c], sharex=ax)
        for a_ in (ax, axn): a_.set_facecolor("#fcfcfb"); a_.grid(color="#e6e5e0", lw=0.6)
        ax.plot([0, 1], [0, 1], color="#52514e", ls="--", lw=1)
        ax.plot(g.mean_pred, g.obs_freq, color=color[m], lw=2, marker="o", ms=4)
        ece = M[(M.eval_set == "test") & (M.model == m) & (M.horizon_sols == 1) & (M.season == "ALL")].ece.iloc[0]
        ax.set_title(f"{m}  (ECE {ece:.3f})", fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.tick_params(labelbottom=False)
        axn.bar((g.bin_lo + g.bin_hi) / 2, g.n, width=0.09, color=color[m], alpha=0.55); axn.set_yscale("log"); axn.tick_params(labelsize=6)
        if c == 0: ax.set_ylabel("Observed frequency"); axn.set_ylabel("bin n", fontsize=7)
        if r == 2: axn.set_xlabel("Mean predicted probability")
    fig.suptitle("Mars storm detection, horizon 1 sol: reliability diagrams (10 equal-width bins) with bin counts, test split", fontsize=9)
    fig.savefig(out / "figure_reliability_test_h1.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"done: {out} ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
