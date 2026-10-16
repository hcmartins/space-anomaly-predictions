"""Required regression figures: observed-versus-predicted plot and residual
analysis (residuals vs. predicted, residual distribution). Each returns
(figure, source_dataframe) so the caller can save both via
evaluation.figure_provenance.save_figure_with_provenance.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def observed_vs_predicted_figure(y_true: np.ndarray, y_pred: np.ndarray):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=14)
    lims = [float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))]
    ax.plot(lims, lims, color="gray", linestyle="--", linewidth=1, label="1:1 line")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title("Observed vs. predicted")
    ax.legend(fontsize=8)

    source_df = pd.DataFrame({"observed": y_true, "predicted": y_pred})
    return fig, source_df


def residual_analysis_figure(y_true: np.ndarray, y_pred: np.ndarray):
    """Two panels: residuals (predicted - observed) vs. predicted value,
    and a histogram of residuals."""
    residuals = y_pred - y_true

    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(11, 5))
    ax_scatter.scatter(y_pred, residuals, alpha=0.4, s=14)
    ax_scatter.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax_scatter.set_xlabel("Predicted")
    ax_scatter.set_ylabel("Residual (predicted - observed)")
    ax_scatter.set_title("Residuals vs. predicted")

    ax_hist.hist(residuals, bins=30, color="steelblue", edgecolor="black")
    ax_hist.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    ax_hist.set_xlabel("Residual (predicted - observed)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(f"Residual distribution (mean={np.mean(residuals):.4g}, std={np.std(residuals):.4g})")

    source_df = pd.DataFrame({"observed": y_true, "predicted": y_pred, "residual": residuals})
    return fig, source_df
