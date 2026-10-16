"""Chapter 6 examiner-gap supplement (solar): confusion matrices, PR curves,
calibration (reliability bins, ECE, Murphy Brier decomposition), 95% day-block
bootstrap intervals, and McNemar tests at alpha = 0.05.

Never retrains or re-scores anything new: every number is computed from the
already-verified per-instance prediction exports in results/prediction_exports/
(produced by export_baseline_predictions.py / export_lstm_predictions.py, whose
reconstructed confusion matrices were checked against the original run's
metrics.json before being written -- CHAPTER_3_PARAMETER_VALUES.md Section 12.8).
This script re-checks that again (test-split TP/FP/TN/FN from the exports vs
metrics.json) and aborts on any mismatch.

Two evaluation sets are reported:
  * "own_test"      -- each model's own maximum-available test set (the Table 3.6
                       / Appendix numbers): 211,594 rows for the 6 tabular models,
                       107,489 GOES-only LSTM, 37,598 GOES+SHARP LSTM.
  * "common_subset" -- the 37,598-row intersection (identical to the GOES+SHARP
                       LSTM test set), the paired comparison of Section 12.9.

Also writes per-class / per-split prevalence of the >=C / >=M / >=X 24-hour
labels already present in dataset_with_split.csv (evidence for the scope
decision; no new model is trained).

Outputs: results/prediction_exports/chapter6_supplementary/
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "results" / "prediction_exports"
OUT = EXP / "chapter6_supplementary"
SEED = 44
N_BOOT = 200
N_BINS = 10
ALPHA = 0.05
TABULAR = ["climatology", "persistence", "logistic_regression", "random_forest_classifier",
           "gradient_boosting_classifier", "knn_classifier"]
ALL_MODELS = TABULAR + ["goes_lstm", "goes_sharp_lstm"]
# dataviz reference palette, fixed categorical order (never cycled)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
COLOR = dict(zip(ALL_MODELS, PALETTE))


def confusion(y, p, thr):
    yhat = p >= thr
    y = y.astype(bool)
    return int((y & yhat).sum()), int((~y & yhat).sum()), int((~y & ~yhat).sum()), int((y & ~yhat).sum())


def tss_from(tp, fp, tn, fn):
    pod = tp / (tp + fn) if tp + fn else np.nan
    pofd = fp / (fp + tn) if fp + tn else np.nan
    return pod - pofd


def safe_auc(y, p, fn):
    return float(fn(y, p)) if 0 < y.sum() < len(y) else np.nan


def reliability(y, p, n_bins=N_BINS):
    """Equal-width bins on [0,1]; returns bin table + ECE/MCE + Murphy decomposition.
    Binned decomposition (Murphy 1973): BS = REL - RES + UNC + residual, where the
    residual is the within-bin forecast-variance term that binning discards."""
    y = y.astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    n = len(y); obar = y.mean()
    rows, rel, res, ece, mce = [], 0.0, 0.0, 0.0, 0.0
    for b in range(n_bins):
        m = idx == b
        nb = int(m.sum())
        if nb == 0:
            rows.append({"bin_lo": edges[b], "bin_hi": edges[b + 1], "n": 0, "mean_pred": np.nan, "obs_freq": np.nan})
            continue
        pb, ob = float(p[m].mean()), float(y[m].mean())
        rows.append({"bin_lo": edges[b], "bin_hi": edges[b + 1], "n": nb, "mean_pred": pb, "obs_freq": ob,
                     "n_pos": int(y[m].sum())})
        rel += nb * (pb - ob) ** 2
        res += nb * (ob - obar) ** 2
        ece += nb * abs(pb - ob)
        mce = max(mce, abs(pb - ob))
    brier = float(np.mean((p - y) ** 2))
    rel, res, ece = rel / n, res / n, ece / n
    unc = obar * (1 - obar)
    return pd.DataFrame(rows), {
        "brier": brier, "reliability": rel, "resolution": res, "uncertainty": unc,
        "within_bin_residual": brier - (rel - res + unc), "ece": ece, "mce": mce,
        "brier_skill_score_vs_climatology": 1 - brier / unc if unc else np.nan,
        "prevalence": obar, "n": n,
    }


def block_bootstrap(y, p, thr, blocks, rng, n_boot=N_BOOT):
    ub, inv = np.unique(blocks, return_inverse=True)
    idx_by = [np.where(inv == k)[0] for k in range(len(ub))]
    out = {"tss": [], "roc_auc": [], "pr_auc": []}
    for _ in range(n_boot):
        pick = rng.integers(0, len(ub), len(ub))
        ii = np.concatenate([idx_by[k] for k in pick])
        yy, pp = y[ii], p[ii]
        out["tss"].append(tss_from(*confusion(yy, pp, thr)))
        out["roc_auc"].append(safe_auc(yy, pp, roc_auc_score))
        out["pr_auc"].append(safe_auc(yy, pp, average_precision_score))
    return {k: np.array(v) for k, v in out.items()}


def ci(a, level):
    return tuple(np.nanpercentile(a, [(1 - level) / 2 * 100, (1 + level) / 2 * 100]))


def mcnemar(y, pa, ta, pb, tb):
    ca = (pa >= ta) == y.astype(bool)
    cb = (pb >= tb) == y.astype(bool)
    n01 = int((~ca & cb).sum()); n10 = int((ca & ~cb).sum())
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if n01 + n10 else np.nan
    log_p = chi2.logsf(stat, df=1) if n01 + n10 else np.nan
    exact = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else np.nan
    return n01, n10, stat, log_p / np.log(10), exact


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---- load exports + thresholds ----
    base = pd.read_csv(EXP / "baseline_predictions_test.csv")
    thr = json.loads((EXP / "baseline_thresholds.json").read_text())
    for m in ("goes_lstm", "goes_sharp_lstm"):
        thr[m] = json.loads((EXP / f"{m}_threshold.json").read_text())["threshold"]
    lstm = {"goes_lstm": pd.read_csv(EXP / "goes_lstm_predictions_test.csv"),
            "goes_sharp_lstm": pd.read_csv(EXP / "goes_sharp_lstm_predictions_test.csv")}
    own = {m: (base["true_label"].to_numpy(), base[f"{m}_prob"].to_numpy(), base["time_utc"].to_numpy()) for m in TABULAR}
    for m, df in lstm.items():
        own[m] = (df["true_label"].to_numpy(), df["predicted_probability"].to_numpy(), df["time_utc"].to_numpy())
    common_ts = set(lstm["goes_sharp_lstm"]["time_utc"])
    common = {}
    for m, (y, p, t) in own.items():
        mask = np.isin(t, list(common_ts))
        order = np.argsort(t[mask])
        common[m] = (y[mask][order], p[mask][order], t[mask][order])
    ref_t = common["goes_sharp_lstm"][2]
    for m in ALL_MODELS:
        assert (common[m][2] == ref_t).all() and (common[m][0] == common["goes_sharp_lstm"][0]).all(), m

    # ---- 1. confusion matrices: every split from metrics.json, test re-checked vs exports ----
    runs = {
        "disk_classical": ROOT / "results/real_full_20100501_20250724/metrics.json",
        "per_harp_classical": ROOT / "results/real_full_per_harp_20100501_20250724/metrics.json",
        "goes_lstm": ROOT / "results/real_full_sequence_20100501_20250724/metrics.json",
        "goes_sharp_lstm": ROOT / "results/real_full_sequence_goes_sharp_20100501_20250724/metrics.json",
    }
    cm_rows = []
    for run, path in runs.items():
        d = json.loads(path.read_text())
        models = d["targets"]["label_primary"]["models"] if "targets" in d else {"lstm": d["sequence_models"]["lstm"]}
        for name, rec in models.items():
            label = run if run.endswith("lstm") else name
            for split in ("train", "val", "test"):
                mt = rec["metrics"][split]
                cm_rows.append({"run": run, "model": label, "split": split, "threshold": rec.get("decision_threshold"),
                                "threshold_source": rec.get("decision_threshold_source"),
                                "tp": mt["tp"], "fp": mt["fp"], "tn": mt["tn"], "fn": mt["fn"],
                                "n": mt["tp"] + mt["fp"] + mt["tn"] + mt["fn"],
                                "prevalence": (mt["tp"] + mt["fn"]) / (mt["tp"] + mt["fp"] + mt["tn"] + mt["fn"]),
                                "pod": mt["pod"], "far": mt["far"], "tss": mt["tss"], "hss": mt["hss"],
                                "pr_auc": mt["pr_auc"], "roc_auc": mt["roc_auc"], "brier": mt["brier"]})
    cm = pd.DataFrame(cm_rows)
    for m in ALL_MODELS:
        y, p, _ = own[m]
        tp, fp, tn, fn = confusion(y, p, thr[m])
        run = m if m.endswith("lstm") else "disk_classical"
        r = cm[(cm.run == run) & (cm.model == m) & (cm.split == "test")].iloc[0]
        assert (tp, fp, tn, fn) == (r.tp, r.fp, r.tn, r.fn), (m, (tp, fp, tn, fn), tuple(r[["tp", "fp", "tn", "fn"]]))
    cm.to_csv(OUT / "confusion_matrices_all_runs_all_splits.csv", index=False)
    cc = []
    for m in ALL_MODELS:
        y, p, _ = common[m]
        tp, fp, tn, fn = confusion(y, p, thr[m])
        cc.append({"model": m, "threshold": thr[m], "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": len(y)})
    pd.DataFrame(cc).to_csv(OUT / "confusion_matrices_common_subset.csv", index=False)
    print("[1] confusion matrices OK (exports == metrics.json)", flush=True)

    # ---- 2+3. PR curves, calibration, bootstrap CIs ----
    metric_rows, rel_rows, pr_rows = [], [], []
    boots = {}
    for setname, data in (("own_test", own), ("common_subset", common)):
        for m in ALL_MODELS:
            y, p, t = data[m]
            blocks = pd.to_datetime(pd.Series(t), utc=True).dt.strftime("%Y-%m-%d").to_numpy()
            tp, fp, tn, fn = confusion(y, p, thr[m])
            bins, cal = reliability(y, p)
            bins.insert(0, "model", m); bins.insert(0, "eval_set", setname)
            rel_rows.append(bins)
            prec, rec, _ = precision_recall_curve(y, p)
            keep = np.unique(np.linspace(0, len(prec) - 1, min(len(prec), 400)).astype(int))
            pr_rows.append(pd.DataFrame({"eval_set": setname, "model": m, "recall": rec[keep], "precision": prec[keep]}))
            b = block_bootstrap(y, p, thr[m], blocks, rng)
            boots[(setname, m)] = (b, blocks)
            row = {"eval_set": setname, "model": m, "n": len(y), "n_days": len(np.unique(blocks)), "threshold": thr[m],
                   "tp": tp, "fp": fp, "tn": tn, "fn": fn, "tss": tss_from(tp, fp, tn, fn),
                   "roc_auc": safe_auc(y, p, roc_auc_score), "pr_auc": safe_auc(y, p, average_precision_score),
                   "pr_auc_no_skill": float(y.mean()), **cal}
            for k in ("tss", "roc_auc", "pr_auc"):
                row[f"{k}_ci90_lo"], row[f"{k}_ci90_hi"] = ci(b[k], 0.90)
                row[f"{k}_ci95_lo"], row[f"{k}_ci95_hi"] = ci(b[k], 0.95)
                row[f"{k}_n_nan_reps"] = int(np.isnan(b[k]).sum())
            metric_rows.append(row)
            print(f"    {setname} {m}: done", flush=True)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "metrics_with_bootstrap_and_calibration.csv", index=False)
    pd.concat(rel_rows).to_csv(OUT / "reliability_bins.csv", index=False)
    pd.concat(pr_rows).to_csv(OUT / "pr_curves.csv", index=False)
    print("[2/3] PR, calibration, bootstrap OK", flush=True)

    # ---- 4. significance on the common subset: McNemar + paired day-block TSS diff ----
    y = common["goes_sharp_lstm"][0]
    blocks = boots[("common_subset", "goes_sharp_lstm")][1]
    ub, inv = np.unique(blocks, return_inverse=True)
    idx_by = [np.where(inv == k)[0] for k in range(len(ub))]
    pairs = [(a, b) for a in ALL_MODELS for b in ALL_MODELS if a < b and "climatology" not in (a, b)]
    sig_rows = []
    rng_pair = np.random.default_rng(SEED + 1)
    for a, b in pairs:
        pa, pb = common[a][1], common[b][1]
        n01, n10, stat, log10p, exact_p = mcnemar(y, pa, thr[a], pb, thr[b])
        diffs = []
        for _ in range(N_BOOT):
            ii = np.concatenate([idx_by[k] for k in rng_pair.integers(0, len(ub), len(ub))])
            diffs.append(tss_from(*confusion(y[ii], pa[ii], thr[a])) - tss_from(*confusion(y[ii], pb[ii], thr[b])))
        diffs = np.array(diffs)
        lo, hi = ci(diffs, 1 - ALPHA)
        sig_rows.append({"model_a": a, "model_b": b,
                         "tss_a": tss_from(*confusion(y, pa, thr[a])), "tss_b": tss_from(*confusion(y, pb, thr[b])),
                         "tss_diff_a_minus_b": tss_from(*confusion(y, pa, thr[a])) - tss_from(*confusion(y, pb, thr[b])),
                         "tss_diff_ci95_lo": lo, "tss_diff_ci95_hi": hi, "block_bootstrap_significant_at_0.05": bool(lo > 0 or hi < 0),
                         "mcnemar_n01_a_wrong_b_right": n01, "mcnemar_n10_a_right_b_wrong": n10,
                         "mcnemar_chi2_cc": stat, "mcnemar_log10_p": log10p, "mcnemar_exact_p": exact_p})
    sig = pd.DataFrame(sig_rows)
    # Holm correction across all pairs (on the chi-square p-values, in log space)
    order = np.argsort(sig["mcnemar_log10_p"].to_numpy())
    m_tests = len(sig)
    holm_sig = np.zeros(m_tests, dtype=bool)
    for rank, i in enumerate(order):
        if sig.loc[i, "mcnemar_log10_p"] <= np.log10(ALPHA / (m_tests - rank)):
            holm_sig[i] = True
        else:
            break
    sig["mcnemar_significant_holm_0.05"] = holm_sig
    sig.to_csv(OUT / "significance_common_subset.csv", index=False)
    print("[4] significance OK", flush=True)

    # ---- 5. per-class / per-split prevalence of the existing 24h labels ----
    ds = pd.read_csv(ROOT / "results/real_full_20100501_20250724/dataset_with_split.csv",
                     usecols=["split", "label_horizon_hours", "label_flare_c_or_above", "label_flare_m_or_above", "label_flare_x_or_above"])
    prev = []
    for split, g in ds.groupby("split"):
        for cls in ("c", "m", "x"):
            col = g[f"label_flare_{cls}_or_above"].dropna()
            prev.append({"split": split, "class": f">={cls.upper()}", "horizon_hours": sorted(g["label_horizon_hours"].dropna().unique().tolist()),
                         "n_defined": int(len(col)), "n_positive": int(col.sum()), "prevalence": float(col.mean()) if len(col) else np.nan})
    pd.DataFrame(prev).to_csv(OUT / "label_prevalence_by_class_and_split.csv", index=False)
    print("[5] prevalence OK", flush=True)

    # ---- 6. figures (static, light surface, fixed palette order) ----
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False, "axes.spines.right": False})
    pr = pd.concat(pr_rows)
    fig, ax = plt.subplots(figsize=(6.4, 4.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for m in ALL_MODELS:
        g = pr[(pr.eval_set == "common_subset") & (pr.model == m)]
        ap = metrics[(metrics.eval_set == "common_subset") & (metrics.model == m)].pr_auc.iloc[0]
        ax.plot(g.recall, g.precision, color=COLOR[m], lw=2, label=f"{m} (PR-AUC {ap:.3f})")
    base_rate = common["goes_sharp_lstm"][0].mean()
    ax.axhline(base_rate, color="#52514e", lw=1, ls="--", label=f"no-skill (prevalence {base_rate:.3f})")
    ax.set_xlabel("Recall (POD)"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(color="#e6e5e0", lw=0.6); ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.set_title("Solar ≥M-class, 24 h: precision–recall, common test subset (n = 37,598)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "figure_pr_curves_common_subset.png", dpi=200); plt.close(fig)

    relb = pd.concat(rel_rows)
    # one y-axis per panel: reliability curve on top, bin counts in its own panel below
    fig = plt.figure(figsize=(11, 7.4), facecolor="#fcfcfb")
    gs = fig.add_gridspec(4, 4, height_ratios=[3, 1, 3, 1], hspace=0.55, wspace=0.3)
    for k, m in enumerate(ALL_MODELS):
        r, c = 2 * (k // 4), k % 4
        g = relb[(relb.eval_set == "common_subset") & (relb.model == m) & (relb.n > 0)]
        ax = fig.add_subplot(gs[r, c]); axn = fig.add_subplot(gs[r + 1, c], sharex=ax)
        for a_ in (ax, axn):
            a_.set_facecolor("#fcfcfb"); a_.grid(color="#e6e5e0", lw=0.6)
        ax.plot([0, 1], [0, 1], color="#52514e", lw=1, ls="--")
        ax.plot(g.mean_pred, g.obs_freq, color=COLOR[m], lw=2, marker="o", ms=4)
        ece = metrics[(metrics.eval_set == "common_subset") & (metrics.model == m)].ece.iloc[0]
        ax.set_title(f"{m}  (ECE {ece:.3f})", fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.tick_params(labelbottom=False)
        axn.bar((g.bin_lo + g.bin_hi) / 2, g.n, width=0.09, color=COLOR[m], alpha=0.55)
        axn.set_yscale("log"); axn.tick_params(labelsize=6)
        if c == 0:
            ax.set_ylabel("Observed frequency"); axn.set_ylabel("bin n", fontsize=7)
        if r == 2:
            axn.set_xlabel("Mean predicted probability")
    fig.suptitle("Solar ≥M-class, 24 h: reliability diagrams (10 equal-width bins) with bin counts, common test subset (n = 37,598)", fontsize=9)
    fig.savefig(OUT / "figure_reliability_common_subset.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"done: {OUT}", flush=True)


if __name__ == "__main__":
    main()
