"""Shared static-figure helpers for the Chapter 6 supplements (solar + Martian).

Redraws PR curves and reliability diagrams from the CSVs written by
solar/analyze_chapter6_supplementary.py and martian/run_chapter6_supplementary.py,
so figures can be regenerated without recomputing any metric.

PR curves are drawn as right-continuous STEP functions (the same construction
average precision / PR-AUC integrates); linear interpolation between PR points
is not valid in PR space. Models whose scores take <= 2 distinct values
(climatology, persistence) are drawn as operating points, not curves.
One y-axis per panel (reliability curve above, its bin counts below).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import pandas as pd

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK2, GRID = "#fcfcfb", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def _style(ax):
    ax.set_facecolor(SURFACE); ax.grid(color=GRID, lw=0.6)


def pr_panel(ax, pr: pd.DataFrame, models, color, pr_auc: dict, prevalence: float):
    _style(ax)
    for m in models:
        g = pr[pr.model == m].sort_values("recall")
        if g.empty:
            continue
        lab = f"{m} ({pr_auc[m]:.3f})"
        if g[["recall", "precision"]].drop_duplicates().shape[0] <= 3:   # <=2 distinct scores -> operating points
            pts = g[(g.recall > 0) & ~((g.recall == 1) & (g.precision == prevalence) & (len(g) > 2))]
            ax.plot(pts.recall, pts.precision, ls="none", marker="o", ms=6, color=color[m], label=lab,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        else:
            ax.step(g.recall, g.precision, where="post", color=color[m], lw=1.8, label=lab)
    ax.axhline(prevalence, color=INK2, ls="--", lw=1, label=f"no-skill ({prevalence:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_xlabel("Recall (POD)")


def reliability_grid(rel: pd.DataFrame, models, color, ece: dict, title: str, path):
    fig = plt.figure(figsize=(11, 7.6), facecolor=SURFACE)
    gs = fig.add_gridspec(4, 4, height_ratios=[3, 1.1, 3, 1.1], hspace=0.6, wspace=0.32)
    for k, m in enumerate(models):
        r, c = 2 * (k // 4), k % 4
        g = rel[(rel.model == m) & (rel.n > 0)]
        ax = fig.add_subplot(gs[r, c]); axn = fig.add_subplot(gs[r + 1, c], sharex=ax)
        _style(ax); _style(axn)
        ax.plot([0, 1], [0, 1], color=INK2, ls="--", lw=1)
        ax.plot(g.mean_pred, g.obs_freq, color=color[m], lw=0 if len(g) <= 2 else 2, marker="o", ms=5 if len(g) <= 2 else 4)
        ax.set_title(f"{m}  (ECE {ece[m]:.3f})", fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.tick_params(labelbottom=False)
        axn.bar((g.bin_lo + g.bin_hi) / 2, g.n, width=0.09, color=color[m], alpha=0.6)
        axn.set_yscale("log")
        axn.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
        axn.yaxis.set_minor_formatter(NullFormatter())
        axn.tick_params(labelsize=6)
        if c == 0:
            ax.set_ylabel("Observed frequency"); axn.set_ylabel("bin n", fontsize=7)
        if r == 2:
            axn.set_xlabel("Mean predicted probability")
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE); plt.close(fig)


def solar_figures(out):
    from pathlib import Path
    out = Path(out)
    models = ["climatology", "persistence", "logistic_regression", "random_forest_classifier",
              "gradient_boosting_classifier", "knn_classifier", "goes_lstm", "goes_sharp_lstm"]
    color = dict(zip(models, PALETTE))
    M = pd.read_csv(out / "metrics_with_bootstrap_and_calibration.csv"); M = M[M.eval_set == "common_subset"].set_index("model")
    pr = pd.read_csv(out / "pr_curves.csv"); pr = pr[pr.eval_set == "common_subset"]
    rel = pd.read_csv(out / "reliability_bins.csv"); rel = rel[rel.eval_set == "common_subset"]
    fig, ax = plt.subplots(figsize=(8.2, 4.6), facecolor=SURFACE)
    pr_panel(ax, pr, models, color, M.pr_auc.to_dict(), float(M.pr_auc_no_skill.iloc[0]))
    ax.set_ylabel("Precision")
    ax.legend(fontsize=7, frameon=False, title="PR-AUC", title_fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title("Solar ≥M-class, 24 h: precision–recall, common test subset (n = 37,598)", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "figure_pr_curves_common_subset.png", dpi=200, facecolor=SURFACE); plt.close(fig)
    reliability_grid(rel, models, color, M.ece.to_dict(),
                     "Solar ≥M-class, 24 h: reliability diagrams (10 equal-width bins) with bin counts, common test subset (n = 37,598)",
                     out / "figure_reliability_common_subset.png")


def martian_figures(out):
    from pathlib import Path
    out = Path(out)
    models = ["persistence", "logistic_regression", "random_forest_classifier", "gradient_boosting_classifier",
              "knn_classifier", "lstm_w3", "gru_w3", "climatology"]
    color = dict(zip(models, PALETTE))
    M = pd.read_csv(out / "metrics_overall_and_by_season.csv")
    M = M[(M.eval_set == "test") & (M.season == "ALL")]
    pr = pd.read_csv(out / "pr_curves.csv"); pr = pr[pr.eval_set == "test"]
    rel = pd.read_csv(out / "reliability_bins.csv"); rel = rel[rel.eval_set == "test"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), facecolor=SURFACE, sharey=True)
    for ax, h in zip(axes, (1, 3, 5)):
        mh = M[M.horizon_sols == h].set_index("model")
        pr_panel(ax, pr[pr.horizon_sols == h], models, color, mh.pr_auc.to_dict(), float(mh.prevalence.iloc[0]))
        ax.set_title(f"horizon {h} sol", fontsize=9)
        ax.legend(fontsize=6, frameon=False, title="PR-AUC", title_fontsize=6, loc="upper right")
    axes[0].set_ylabel("Precision")
    fig.suptitle("Mars storm detection: precision–recall, canonical test split (MY37 sols 1–501)", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "figure_pr_curves_test.png", dpi=200, facecolor=SURFACE); plt.close(fig)
    for h in (1, 3, 5):
        mh = M[M.horizon_sols == h].set_index("model")
        reliability_grid(rel[rel.horizon_sols == h], models, color, mh.ece.to_dict(),
                         f"Mars storm detection, horizon {h} sol: reliability diagrams (10 equal-width bins) with bin counts, test split",
                         out / f"figure_reliability_test_h{h}.png")


if __name__ == "__main__":
    import sys
    {"solar": solar_figures, "martian": martian_figures}[sys.argv[1]](sys.argv[2])
