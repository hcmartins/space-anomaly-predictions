"""Thesis-ready empirical figures (PNG at 300 dpi + vector PDF) from the canonical
2026-10-04 outputs. Every figure is drawn from verified per-row predictions or
the CSVs computed from them; nothing is simulated.

Writes to <repo>/thesis_figures/ plus FIGURE_CAPTIONS.md.

  mars_pr_curves_{pooled,autumn}.{png,pdf}          precision-recall, h = 1/3/5
  mars_reliability_{pooled,autumn}_h{1,3,5}.{png,pdf} reliability + bin counts
  solar_pr_curves_common_subset.{png,pdf}
  solar_reliability_common_subset.{png,pdf}
  fusion_within_season_tss.{png,pdf}                5 seeds per config, pooled / autumn / winter
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from chapter6_figures import INK2, PALETTE, SURFACE, _style, pr_panel, reliability_grid

SRC = Path(__file__).resolve().parent
OUT = SRC.parent / "thesis_figures"
MRUN = SRC / "martian/outputs/real/run_20261004T080014968911Z_seed42"
MSUPP = SRC / "martian/outputs/real/chapter6_supplementary_20261004T093226Z"
SSUPP = SRC / "solar/results/prediction_exports/chapter6_supplementary"
FSUPP = sorted((SRC / "cross-domain/outputs").glob("fusion_supplementary_*"))[-1]
MODELS = ["persistence", "random_forest_classifier", "knn_classifier", "gradient_boosting_classifier",
          "logistic_regression", "lstm_w3", "lstm_w5", "gru_w3"]
COLOR = dict(zip(MODELS, PALETTE))
AUT = "Northern Autumn (Ls 180-270)"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def reliability_bins(y, p, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({"bin_lo": edges[b], "bin_hi": edges[b + 1], "n": int(m.sum()),
                     "mean_pred": float(p[m].mean()) if m.any() else np.nan, "obs_freq": float(y[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    P = pd.read_csv(MSUPP / "predictions_test_and_my37_late.csv.gz")
    P = P[(P.eval_set == "test") & P.model.isin(MODELS)]
    thr = {}
    for f in (MRUN / "metrics/storm_detection").glob("*_evaluation.json"):
        m, h = f.name.replace("_evaluation.json", "").rsplit("_h", 1)
        thr[(m, int(h))] = json.loads(f.read_text())["decision_threshold"]

    captions = []
    for tag, sub in (("pooled", P), ("autumn", P[P.season == AUT])):
        label = "canonical test split (MY37 sols 386–668, autumn + winter)" if tag == "pooled" else "northern autumn only (MY37 Ls 188–270)"
        # PR curves
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), facecolor=SURFACE, sharey=True)
        for ax, h in zip(axes, (1, 3, 5)):
            g = sub[sub.horizon == h]
            prs, aps = [], {}
            for m in MODELS:
                gm = g[g.model == m]
                if gm.y.sum() == 0:
                    continue
                pr, rc, _ = precision_recall_curve(gm.y, gm.p)
                keep = np.unique(np.linspace(0, len(pr) - 1, min(len(pr), 400)).astype(int))
                prs.append(pd.DataFrame({"model": m, "recall": rc[keep], "precision": pr[keep]}))
                aps[m] = average_precision_score(gm.y, gm.p)
            prev = g[g.model == "persistence"].y.mean()
            pr_panel(ax, pd.concat(prs), [m for m in MODELS if m in aps], COLOR, aps, float(prev))
            ax.set_title(f"horizon {h} sol", fontsize=9)
            ax.legend(fontsize=6, frameon=False, title="PR-AUC", title_fontsize=6, loc="upper right")
        axes[0].set_ylabel("Precision")
        fig.suptitle(f"Mars dust-anomaly-episode detection: precision–recall, {label}", fontsize=9)
        fig.tight_layout(); save(fig, f"mars_pr_curves_{tag}")
        captions.append((f"mars_pr_curves_{tag}", f"Precision–recall curves of the frozen Martian dust-anomaly-episode detection models, {label}, at horizons of 1, 3 and 5 sols. Step curves (the construction average precision integrates); persistence is a single operating point; dashed line = prevalence (no skill). Sequence models are scored on their window-feasible subsets."))
        # reliability
        for h in (1, 3, 5):
            g = sub[sub.horizon == h]
            rel, ece = [], {}
            for m in MODELS:
                gm = g[g.model == m]
                b = reliability_bins(gm.y.to_numpy().astype(float), gm.p.to_numpy())
                b.insert(0, "model", m); rel.append(b)
                ece[m] = float(np.nansum(b.n * np.abs(b.mean_pred - b.obs_freq)) / b.n.sum())
            title = f"Mars dust-anomaly-episode detection, horizon {h} sol: reliability (10 equal-width bins) with bin counts,\n{label}"
            for ext in ("png", "pdf"):
                reliability_grid(pd.concat(rel), MODELS, COLOR, ece, title, OUT / f"mars_reliability_{tag}_h{h}.{ext}")
            captions.append((f"mars_reliability_{tag}_h{h}", f"Reliability diagrams (10 equal-width probability bins, empty bins omitted) of the frozen Martian models at horizon {h} sol, {label}. Lower panels: number of forecasts per bin (log scale). ECE in each title."))

    # solar (common subset), re-rendered at thesis resolution from the solar CSVs
    M = pd.read_csv(SSUPP / "metrics_with_bootstrap_and_calibration.csv"); M = M[M.eval_set == "common_subset"].set_index("model")
    smodels = ["climatology", "persistence", "logistic_regression", "random_forest_classifier",
               "gradient_boosting_classifier", "knn_classifier", "goes_lstm", "goes_sharp_lstm"]
    scolor = dict(zip(smodels, PALETTE))
    pr = pd.read_csv(SSUPP / "pr_curves.csv"); pr = pr[pr.eval_set == "common_subset"]
    fig, ax = plt.subplots(figsize=(8.2, 4.6), facecolor=SURFACE)
    pr_panel(ax, pr, smodels, scolor, M.pr_auc.to_dict(), float(M.pr_auc_no_skill.iloc[0]))
    ax.set_ylabel("Precision")
    ax.legend(fontsize=7, frameon=False, title="PR-AUC", title_fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title("Solar ≥M-class, 24 h: precision–recall, common test subset (n = 37,598)", fontsize=9)
    fig.tight_layout(); save(fig, "solar_pr_curves_common_subset")
    rel = pd.read_csv(SSUPP / "reliability_bins.csv"); rel = rel[rel.eval_set == "common_subset"]
    for ext in ("png", "pdf"):
        reliability_grid(rel, smodels, scolor, M.ece.to_dict(),
                         "Solar ≥M-class, 24 h: reliability (10 equal-width bins) with bin counts, common test subset (n = 37,598)",
                         OUT / f"solar_reliability_common_subset.{ext}")
    captions += [("solar_pr_curves_common_subset", "Precision–recall curves of all eight solar ≥M-class models on the 37,598-instance common test subset (paired comparison; prevalence 0.279, dashed)."),
                 ("solar_reliability_common_subset", "Reliability diagrams of the eight solar models on the common test subset, with per-bin forecast counts.")]

    # fusion: within-season TSS per configuration, 5 seeds
    F = pd.read_csv(FSUPP / "metrics_per_seed.csv"); F = F[F.domain == "martian"]
    cfgs = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), facecolor=SURFACE, sharey=True)
    for ax, (st, lab) in zip(axes, (("ALL", "pooled (autumn + winter)"), ("Northern Autumn", "northern autumn"), ("Northern Winter", "northern winter (out of season)"))):
        _style(ax)
        for i, c in enumerate(cfgs):
            v = F[(F.config == c) & (F.stratum == st)].tss.to_numpy()
            ax.scatter(np.full(len(v), i), v, s=28, color=PALETTE[0], edgecolor=SURFACE, linewidth=1.2, zorder=3)
            ax.hlines(np.nanmean(v), i - 0.3, i + 0.3, color=INK2, lw=2, zorder=2)
        ax.axhline(0, color=INK2, lw=0.8, ls="--")
        ax.set_xticks(range(len(cfgs))); ax.set_xticklabels(cfgs); ax.set_title(lab, fontsize=9)
        n_pos = int(F[(F.config == "A1") & (F.stratum == st)].n_pos.iloc[0])
        ax.text(0.98, 0.02, f"{n_pos} test positives", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=INK2)
    axes[0].set_ylabel("Martian test TSS")
    fig.suptitle("Cross-domain fusion ablation, Martian domain: test TSS by configuration (dots = seeds 44–48, bar = mean)", fontsize=9)
    fig.tight_layout(); save(fig, "fusion_within_season_tss")
    captions.append(("fusion_within_season_tss", "Martian test TSS of the seven fusion-ablation configurations over five seeds, pooled and within northern autumn and winter. A1 = unimodal baseline; A7 = full fusion model. Autumn rests on only 21 positive test windows."))

    with open(OUT / "FIGURE_CAPTIONS.md", "w", encoding="utf-8") as fh:
        fh.write("# Thesis figures — captions and provenance\n\nAll figures are **empirical**: drawn from the canonical runs' verified per-row "
                 "predictions (Mars: `run_20261004T080014968911Z_seed42` via `chapter6_supplementary_20261004T093226Z`; solar: "
                 "`results/prediction_exports/chapter6_supplementary`; fusion: `" + FSUPP.name + "`). Regenerate with "
                 "`python src/make_thesis_figures.py`. Each figure is provided as PNG (300 dpi) and PDF (vector).\n\n")
        for name, cap in captions:
            fh.write(f"**`{name}`** — {cap} *(Empirical.)*\n\n")
    print("written to", OUT)


if __name__ == "__main__":
    main()
