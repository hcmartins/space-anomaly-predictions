"""Common-subset comparison + uncertainty analysis for Chapter 3 Section 3.8
(examiner round-12 follow-up), run on the real per-instance prediction
exports from export_baseline_predictions.py and export_lstm_predictions.py.

Implements exactly the four things requested:
  1. Intersect the three models' test timestamps into one common,
     identically-evaluated subset; recompute TSS/HSS/ROC-AUC/PR-AUC/Brier
     for all 8 models (6 baselines + 2 LSTMs) on that subset, at each
     model's own already-selected (validation-selected) threshold.
  2. Paired per-instance metric differences on the common subset for the
     narratively relevant pairs (best baseline vs each LSTM; LSTM vs LSTM).
  3. Time-block-clustered bootstrap (block = calendar day, matching the
     documented within-window/within-day dependence from overlapping
     24h-horizon instances) for per-model TSS/ROC-AUC intervals AND paired
     TSS-difference intervals -- resampling days with replacement, not
     instances, exactly the correction Section 3.8 says any future interval
     must use.
  4. McNemar's test on paired thresholded decisions for the same pairs
     (reported with the same within-block-dependence caveat as the
     bootstrap: a standard instance-level McNemar is not cluster-robust,
     stated explicitly rather than silently presented as exact).
  5. Reliability-bin counts (10 equal-width probability bins, empty bins
     omitted) for every model on the common subset.

200 replicates, 90% percentile interval, seed=44 -- matching this project's
seed convention and the same replicate count/interval level already used
for the real Martian block-bootstrap results (Chapter 6).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

SOLAR_ROOT = Path(__file__).resolve().parent
IN_DIR = SOLAR_ROOT / "results" / "prediction_exports"
OUT_DIR = IN_DIR
SEED = 44
N_REPLICATES = 200
INTERVAL = 0.90  # 90% percentile interval -> 5th/95th percentiles

BASELINE_MODELS = [
    "climatology", "persistence", "logistic_regression",
    "random_forest_classifier", "gradient_boosting_classifier", "knn_classifier",
]


def load_all() -> tuple[pd.DataFrame, dict[str, float]]:
    base = pd.read_csv(IN_DIR / "baseline_predictions_test.csv", parse_dates=["time_utc"])
    base_thr = json.loads((IN_DIR / "baseline_thresholds.json").read_text(encoding="utf-8"))

    goes = pd.read_csv(IN_DIR / "goes_lstm_predictions_test.csv", parse_dates=["time_utc"])
    goes = goes.rename(columns={"predicted_probability": "goes_lstm_prob", "true_label": "true_label_goes"})
    goes_thr = json.loads((IN_DIR / "goes_lstm_threshold.json").read_text(encoding="utf-8"))["threshold"]

    combo = pd.read_csv(IN_DIR / "goes_sharp_lstm_predictions_test.csv", parse_dates=["time_utc"])
    combo = combo.rename(columns={"predicted_probability": "goes_sharp_lstm_prob", "true_label": "true_label_combo"})
    combo_thr = json.loads((IN_DIR / "goes_sharp_lstm_threshold.json").read_text(encoding="utf-8"))["threshold"]

    merged = base.merge(goes, on="time_utc", how="inner").merge(combo, on="time_utc", how="inner")
    # Sanity: the true label must agree across all three exports for any
    # shared timestamp -- if it doesn't, the three pipelines disagree on
    # ground truth for this row and the merge must not proceed silently.
    assert (merged["true_label"] == merged["true_label_goes"]).all(), "true_label mismatch: baseline vs GOES LSTM"
    assert (merged["true_label"] == merged["true_label_combo"]).all(), "true_label mismatch: baseline vs GOES+SHARP LSTM"
    merged = merged.drop(columns=["true_label_goes", "true_label_combo"])

    thresholds = dict(base_thr)
    thresholds["goes_lstm"] = goes_thr
    thresholds["goes_sharp_lstm"] = combo_thr

    prob_cols = {name: f"{name}_prob" for name in BASELINE_MODELS}
    prob_cols["goes_lstm"] = "goes_lstm_prob"
    prob_cols["goes_sharp_lstm"] = "goes_sharp_lstm_prob"

    print(f"Baseline test rows: {len(base)}; GOES-LSTM test rows: {len(goes)}; "
          f"GOES+SHARP-LSTM test rows: {len(combo)}; COMMON SUBSET rows: {len(merged)}")
    return merged, thresholds, prob_cols


def tss_hss(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> tuple[float, float, int, int, int, int]:
    pred = (y_prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    pod = tp / (tp + fn) if (tp + fn) else float("nan")
    pofd = fp / (fp + tn) if (fp + tn) else float("nan")
    tss = pod - pofd
    hss_num = 2 * (tp * tn - fp * fn)
    hss_den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = hss_num / hss_den if hss_den else float("nan")
    return tss, hss, tp, fp, tn, fn


def point_metrics(df: pd.DataFrame, name: str, prob_col: str, threshold: float) -> dict:
    y_true = df["true_label"].to_numpy()
    y_prob = df[prob_col].to_numpy()
    tss, hss, tp, fp, tn, fn = tss_hss(y_true, y_prob, threshold)
    n_classes = len(np.unique(y_true))
    roc_auc = roc_auc_score(y_true, y_prob) if n_classes == 2 else float("nan")
    pr_auc = average_precision_score(y_true, y_prob) if n_classes == 2 else float("nan")
    brier = float(np.mean((y_prob - y_true) ** 2))
    return {
        "model": name, "threshold": threshold, "n": len(df),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "tss": tss, "hss": hss, "roc_auc": roc_auc, "pr_auc": pr_auc, "brier": brier,
    }


def make_day_blocks(df: pd.DataFrame) -> np.ndarray:
    """Calendar-day (UTC) block id per row -- the resampling unit for the
    block bootstrap. A day is >= the 24h label horizon, so it is at least
    as coarse as the overlapping-window dependence documented in Section
    3.2.1; days, not instances, are resampled with replacement."""
    return df["time_utc"].dt.floor("D").to_numpy()


def block_bootstrap_single(df: pd.DataFrame, prob_col: str, threshold: float, blocks: np.ndarray,
                            unique_blocks: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Returns (tss_replicates, roc_auc_replicates) over N_REPLICATES
    day-block-resampled pseudo-datasets."""
    tss_reps = np.empty(N_REPLICATES)
    auc_reps = np.empty(N_REPLICATES)
    block_to_rows = {b: np.where(blocks == b)[0] for b in unique_blocks}
    y_true_all = df["true_label"].to_numpy()
    y_prob_all = df[prob_col].to_numpy()
    for r in range(N_REPLICATES):
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        idx = np.concatenate([block_to_rows[b] for b in sampled_blocks])
        y_true = y_true_all[idx]
        y_prob = y_prob_all[idx]
        tss, _, _, _, _, _ = tss_hss(y_true, y_prob, threshold)
        tss_reps[r] = tss
        auc_reps[r] = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    return tss_reps, auc_reps


def block_bootstrap_paired_diff(df: pd.DataFrame, prob_col_a: str, thr_a: float,
                                 prob_col_b: str, thr_b: float, blocks: np.ndarray,
                                 unique_blocks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Paired TSS(A) - TSS(B) replicate distribution, resampling the SAME
    day blocks for both models in each replicate (paired bootstrap)."""
    diffs = np.empty(N_REPLICATES)
    block_to_rows = {b: np.where(blocks == b)[0] for b in unique_blocks}
    y_true_all = df["true_label"].to_numpy()
    prob_a_all = df[prob_col_a].to_numpy()
    prob_b_all = df[prob_col_b].to_numpy()
    for r in range(N_REPLICATES):
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        idx = np.concatenate([block_to_rows[b] for b in sampled_blocks])
        y_true = y_true_all[idx]
        tss_a, _, _, _, _, _ = tss_hss(y_true, prob_a_all[idx], thr_a)
        tss_b, _, _, _, _, _ = tss_hss(y_true, prob_b_all[idx], thr_b)
        diffs[r] = tss_a - tss_b
    return diffs


def mcnemar(df: pd.DataFrame, prob_col_a: str, thr_a: float, prob_col_b: str, thr_b: float) -> dict:
    """Standard (instance-level, continuity-corrected) McNemar's test on
    paired thresholded correctness. NOT cluster-robust -- overlapping
    24h-horizon windows mean adjacent instances are not independent, so
    this p-value is reported as approximate/anti-conservative, per
    Section 3.8's own documented limitation, not as an exact test."""
    y_true = df["true_label"].to_numpy()
    pred_a = (df[prob_col_a].to_numpy() >= thr_a).astype(int)
    pred_b = (df[prob_col_b].to_numpy() >= thr_b).astype(int)
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    n01 = int(((~correct_a) & correct_b).sum())  # A wrong, B right
    n10 = int((correct_a & (~correct_b)).sum())  # A right, B wrong
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if (n01 + n10) > 0 else 0.0
    from scipy.stats import chi2
    p_value = float(1 - chi2.cdf(stat, df=1)) if (n01 + n10) > 0 else 1.0
    return {"n01_A_wrong_B_right": n01, "n10_A_right_B_wrong": n10, "statistic": float(stat), "p_value": p_value}


def reliability_bins(df: pd.DataFrame, prob_col: str, n_bins: int = 10) -> list[dict]:
    y_true = df["true_label"].to_numpy()
    y_prob = df[prob_col].to_numpy()
    edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append({
            "bin_lower": float(edges[b]), "bin_upper": float(edges[b + 1]),
            "n": n, "mean_predicted_prob": float(y_prob[mask].mean()),
            "observed_frequency": float(y_true[mask].mean()),
        })
    return rows


def main() -> None:
    df, thresholds, prob_cols = load_all()
    blocks = make_day_blocks(df)
    unique_blocks = np.unique(blocks)
    print(f"Common subset spans {len(unique_blocks)} distinct UTC calendar days.")

    # 1. Point metrics on the common subset, all 8 models
    point_rows = [
        point_metrics(df, name, col, thresholds[name])
        for name, col in prob_cols.items()
    ]
    point_df = pd.DataFrame(point_rows).sort_values("tss", ascending=False)
    point_df.to_csv(OUT_DIR / "common_subset_point_metrics.csv", index=False)
    print("\n=== Common-subset point metrics (all 8 models) ===")
    print(point_df.to_string(index=False))

    # 2 & 3. Per-model block-bootstrap intervals + paired comparisons
    rng = np.random.default_rng(SEED)
    interval_rows = []
    for name, col in prob_cols.items():
        tss_reps, auc_reps = block_bootstrap_single(df, col, thresholds[name], blocks, unique_blocks, rng)
        lo, hi = np.nanpercentile(tss_reps, [(1 - INTERVAL) / 2 * 100, (1 + INTERVAL) / 2 * 100])
        auc_lo, auc_hi = np.nanpercentile(auc_reps, [(1 - INTERVAL) / 2 * 100, (1 + INTERVAL) / 2 * 100])
        interval_rows.append({
            "model": name, "tss_point": point_df.set_index("model").loc[name, "tss"],
            "tss_ci90_lo": lo, "tss_ci90_hi": hi,
            "roc_auc_point": point_df.set_index("model").loc[name, "roc_auc"],
            "roc_auc_ci90_lo": auc_lo, "roc_auc_ci90_hi": auc_hi,
        })
    interval_df = pd.DataFrame(interval_rows)
    interval_df.to_csv(OUT_DIR / "common_subset_bootstrap_intervals.csv", index=False)
    print(f"\n=== Per-model day-block bootstrap ({N_REPLICATES} reps, {int(INTERVAL*100)}% interval) ===")
    print(interval_df.to_string(index=False))

    pairs = [
        ("persistence", "goes_lstm"),
        ("persistence", "goes_sharp_lstm"),
        ("random_forest_classifier", "goes_lstm"),
        ("random_forest_classifier", "goes_sharp_lstm"),
        ("goes_lstm", "goes_sharp_lstm"),
    ]
    paired_rows = []
    mcnemar_rows = []
    for a, b in pairs:
        diffs = block_bootstrap_paired_diff(
            df, prob_cols[a], thresholds[a], prob_cols[b], thresholds[b], blocks, unique_blocks, rng
        )
        lo, hi = np.percentile(diffs, [(1 - INTERVAL) / 2 * 100, (1 + INTERVAL) / 2 * 100])
        point_diff = point_df.set_index("model").loc[a, "tss"] - point_df.set_index("model").loc[b, "tss"]
        paired_rows.append({
            "model_a": a, "model_b": b, "tss_diff_point": point_diff,
            "tss_diff_ci90_lo": lo, "tss_diff_ci90_hi": hi,
            "excludes_zero": bool(lo > 0 or hi < 0),
        })
        mc = mcnemar(df, prob_cols[a], thresholds[a], prob_cols[b], thresholds[b])
        mc["model_a"] = a
        mc["model_b"] = b
        mcnemar_rows.append(mc)

    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(OUT_DIR / "common_subset_paired_tss_diff.csv", index=False)
    print(f"\n=== Paired TSS-difference day-block bootstrap ({N_REPLICATES} reps, {int(INTERVAL*100)}% interval) ===")
    print(paired_df.to_string(index=False))

    mcnemar_df = pd.DataFrame(mcnemar_rows)[["model_a", "model_b", "n01_A_wrong_B_right", "n10_A_right_B_wrong", "statistic", "p_value"]]
    mcnemar_df.to_csv(OUT_DIR / "common_subset_mcnemar.csv", index=False)
    print("\n=== McNemar's test (instance-level, continuity-corrected; NOT cluster-robust -- see caveat) ===")
    print(mcnemar_df.to_string(index=False))

    # 4. Reliability bins for every model
    reliability_all = {}
    for name, col in prob_cols.items():
        reliability_all[name] = reliability_bins(df, col)
    (OUT_DIR / "common_subset_reliability_bins.json").write_text(json.dumps(reliability_all, indent=2), encoding="utf-8")
    print("\nWrote common_subset_point_metrics.csv, common_subset_bootstrap_intervals.csv, "
          "common_subset_paired_tss_diff.csv, common_subset_mcnemar.csv, common_subset_reliability_bins.json")


if __name__ == "__main__":
    main()
