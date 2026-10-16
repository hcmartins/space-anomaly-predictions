"""Post-hoc sensitivity of the canonical northern-autumn Task 2 result to the local time and
observational support of the TARGET observation (and, secondarily, of the issuance observation).

No model is retrained. Inputs are the saved canonical test predictions
(chapter6_supplementary_20261004T093226Z/predictions_test_and_my37_late.csv.gz), the canonical
thresholds (run_20261004T080014968911Z_seed42/metrics/storm_detection/*_evaluation.json), and the
canonical aggregated / labelled cell-sol tables of the same run.

Pre-specified definitions (fixed in this file before any stratum result was computed):
  * Target observation (h = 1): the observed cell-sol at s + 1, the only sol that defines the
    h = 1 target (labeling/anomaly_labels.py::compute_horizon_labels).
    h = 3, 5 (secondary): the observed cell-sols of the cell in (s, s + h]; strata use
    "all target-window observations day" vs "at least one night observation".
  * Day / night: local true solar time LTST in [6, 18) h = day, otherwise night. This is the
    6-h binning of evaluation/failure_analysis.py (bins 6-12 and 12-18 = day; 18-24 and 0-6 = night).
  * Support: observation_count of the target cell-sol: 1 = single retrieval, >= 2 = multiple.
  * Uncertainty: dust_optical_depth_uncertainty of the target cell-sol (the uncertainty of the
    label variable itself; uncertainty_summary is not used because it averages uncertainties of
    different variables). Split at the median over all TRAINING-partition (MY36) observed
    cell-sols: <= median = lower, > median = higher. The cut is independent of test outcomes.
  * High-risk observation: single retrieval AND higher uncertainty AND night.
  * Season: northern autumn, Ls in [180, 270) (labeling/reporting.py::assign_mars_season).
  * Thresholds: the canonical validation-selected tau* of each model (RF h1 0.09, persistence 0.01).
  * Paired RF - persistence test: identical cases; cell blocks (multinomial(n_cells, 1/n_cells)
    weights, the Table 6.24 unit), B = 1000, a fresh np.random.default_rng(44) per stratum,
    percentile 95 % intervals, two-sided bootstrap p; ROC-AUC difference on the same weights.
    Sol-block sensitivity (MY-sol groups resampled with replacement, fresh default_rng(20260818)).
    Holm adjustment within each family (one metric x one family of strata).
  * Reliability rule: a stratum with fewer than 10 positives is reported but labelled
    "too few positives for reliable inference"; its comparison is not used for a conclusion.

Gates (asserted before any stratum value is written):
  * RF and persistence, h = 1/3/5, northern autumn and winter: n, positives and TSS reproduce the
    canonical newrun_followups onset_like_subset_scores.csv ("all" subset) to < 1e-6.
  * Every stratification partitions the autumn population exactly (counts reconcile).

    cd src/martian && python run_local_time_sensitivity.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from labeling.reporting import assign_split

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "outputs/real/run_20261004T080014968911Z_seed42"
SUPP = ROOT / "outputs/real/chapter6_supplementary_20261004T093226Z/predictions_test_and_my37_late.csv.gz"
FOLLOW = ROOT / "outputs/real/newrun_followups_20261004T165854Z/onset_like_subset_scores.csv"
KEYS = ["cell_id", "mars_year", "sol"]
MODELS = ["random_forest_classifier", "persistence", "gradient_boosting_classifier", "logistic_regression",
          "knn_classifier", "climatology", "lstm_w3"]
B, MIN_POS = 1000, 10
AUT = (180.0, 270.0)


def night(ltst):
    return ~((ltst >= 6) & (ltst < 18))


def metrics(y, p, t):
    y = y.astype(bool); d = p >= t
    tp, fp, tn, fn = int((y & d).sum()), int((~y & d).sum()), int((~y & ~d).sum()), int((y & ~d).sum())
    two = 0 < y.sum() < len(y)
    pod = tp / (tp + fn) if tp + fn else np.nan
    pofd = fp / (fp + tn) if fp + tn else np.nan
    den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return {"n": len(y), "n_pos": int(y.sum()), "prevalence": y.mean() if len(y) else np.nan, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "tss": pod - pofd if two else np.nan, "pod": pod, "pofd": pofd, "precision": tp / (tp + fp) if tp + fp else np.nan,
            "hss": 2 * (tp * tn - fn * fp) / den if den else np.nan,
            "roc_auc": roc_auc_score(y, p) if two else np.nan, "pr_auc": average_precision_score(y, p) if two else np.nan,
            "brier": float(np.mean((p - y) ** 2)) if len(y) else np.nan}


def tss_tab(C):
    return C[..., 0] / (C[..., 0] + C[..., 3]) - C[..., 1] / (C[..., 1] + C[..., 2])


def pval(d):
    d = d[np.isfinite(d)]
    return min(1.0, 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1))) if len(d) else np.nan


def holm(p):
    p = np.asarray(p, float); ok = np.isfinite(p); adj = np.full(len(p), np.nan)
    q = p[ok]; o = np.argsort(q); m = len(q); run = 0.0; a = np.empty(m)
    for r, i in enumerate(o):
        run = max(run, (m - r) * q[i]); a[i] = min(1.0, run)
    adj[ok] = a
    return adj


def paired(sub, ta, tb):
    """RF (a) minus persistence (b) on identical rows; cell blocks + sol-block sensitivity."""
    y = sub.y.to_numpy().astype(bool); pa, pb = sub.p_rf.to_numpy(), sub.p_pe.to_numpy()
    def tab(p, t):
        d = p >= t
        return np.stack([y & d, ~y & d, ~y & ~d, y & ~d], 1).astype(float)
    A, Bt = tab(pa, ta), tab(pb, tb)
    uc, inv = np.unique(sub.cell_id.to_numpy(), return_inverse=True)
    pA = np.stack([np.bincount(inv, weights=A[:, k], minlength=len(uc)) for k in range(4)], 1)
    pB = np.stack([np.bincount(inv, weights=Bt[:, k], minlength=len(uc)) for k in range(4)], 1)
    rng = np.random.default_rng(44)
    W = rng.multinomial(len(uc), np.full(len(uc), 1 / len(uc)), size=B)
    dt = tss_tab(W @ pA) - tss_tab(W @ pB)
    da = np.full(B, np.nan)
    for i in range(B):
        w = W[i][inv].astype(float); k = w > 0
        if 0 < y[k].sum() < k.sum():
            da[i] = roc_auc_score(y[k], pa[k], sample_weight=w[k]) - roc_auc_score(y[k], pb[k], sample_weight=w[k])
    # sol-block sensitivity (TSS)
    sols = (sub.mars_year.astype(str) + "_" + sub.sol.astype(str)).to_numpy()
    us, sinv = np.unique(sols, return_inverse=True); members = [np.flatnonzero(sinv == j) for j in range(len(us))]
    rng2 = np.random.default_rng(20260818); ds = np.empty(B)
    for i in range(B):
        idx = np.concatenate([members[j] for j in rng2.integers(0, len(us), len(us))])
        yy = y[idx]
        if 0 < yy.sum() < len(yy):
            ra, rb = pa[idx] >= ta, pb[idx] >= tb
            ds[i] = (ra[yy].mean() - ra[~yy].mean()) - (rb[yy].mean() - rb[~yy].mean())
        else:
            ds[i] = np.nan
    est_t = tss_tab(A.sum(0)) - tss_tab(Bt.sum(0))
    est_a = roc_auc_score(y, pa) - roc_auc_score(y, pb) if 0 < y.sum() < len(y) else np.nan
    lo, hi = np.nanpercentile(dt, [2.5, 97.5]); alo, ahi = np.nanpercentile(da, [2.5, 97.5]) if np.isfinite(da).any() else (np.nan, np.nan)
    slo, shi = np.nanpercentile(ds, [2.5, 97.5]) if np.isfinite(ds).any() else (np.nan, np.nan)
    return {"n_cells": len(uc), "n_sols": len(us), "tss_diff": est_t, "tss_ci_lo": lo, "tss_ci_hi": hi, "tss_p": pval(dt),
            "auc_diff": est_a, "auc_ci_lo": alo, "auc_ci_hi": ahi, "auc_p": pval(da),
            "tss_ci_lo_solblock": slo, "tss_ci_hi_solblock": shi, "tss_p_solblock": pval(ds)}


def main() -> None:
    out = ROOT / "outputs/real" / ("local_time_sensitivity_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True)
    checks = []
    P = pd.read_csv(SUPP); P = P[(P.eval_set == "test") & P.model.isin(MODELS)]
    thr = {}
    for f in (RUN / "metrics/storm_detection").glob("*_evaluation.json"):
        m, h = f.name.replace("_evaluation.json", "").rsplit("_h", 1)
        thr[(m, int(h))] = float(json.loads(f.read_text())["decision_threshold"])
    A = pd.read_csv(RUN / "data/aggregated_cell_sol.csv", usecols=KEYS + ["local_true_solar_time", "observation_count", "dust_optical_depth_uncertainty"])
    A["part"] = assign_split(A).to_numpy()
    unc_cut = float(A.loc[A.part == "train", "dust_optical_depth_uncertainty"].median())
    A["night"] = night(A.local_true_solar_time.to_numpy())
    A["single"] = A.observation_count == 1
    A["high_unc"] = A.dust_optical_depth_uncertainty > unc_cut
    A["high_risk"] = A.night & A.single & A.high_unc
    checks.append(f"pre-specified uncertainty cut = training-partition median of dust_optical_depth_uncertainty = {unc_cut:.6f} "
                  f"(n = {int((A.part == 'train').sum())} MY36 cell-sols)")
    Aidx = A.set_index(KEYS)

    # ---- gates: reproduce canonical autumn / winter scores ----
    F = pd.read_csv(FOLLOW); F = F[F.subset == "all"]
    ls = P.solar_longitude_ls % 360
    P["season"] = np.where((ls >= 180) & (ls < 270), "autumn", np.where(ls >= 270, "winter", "other"))
    for m in ("random_forest_classifier", "persistence"):
        for h in (1, 3, 5):
            for sea, lab in (("autumn", "Northern Autumn (Ls 180-270)"), ("winter", "Northern Winter (Ls 270-360)")):
                g = P[(P.model == m) & (P.horizon == h) & (P.season == sea)]
                r = metrics(g.y.to_numpy(), g.p.to_numpy(), thr[(m, h)]); ref = F[(F.model == m) & (F.horizon_sols == h) & (F.season == lab)].iloc[0]
                assert r["n"] == ref.n and r["n_pos"] == ref.n_pos and abs(r["tss"] - ref.tss) < 1e-6, (m, h, sea, r, ref.to_dict())
    checks.append("gate: RF and persistence n, positives and TSS at h = 1/3/5 in autumn and winter reproduce newrun_followups (all subset); "
                  "RF h1 autumn n 10,167, 37 positives, TSS 0.6920 (Table 6.9)")

    # ---- target / issuance attributes per issuance row ----
    obs_sols = A.groupby(["cell_id", "mars_year"]).sol.apply(lambda s: np.sort(s.to_numpy())).to_dict()
    def attrs(df, h):
        rows = []
        for c, my, s in df[KEYS].itertuples(index=False):
            ss = obs_sols[(c, my)]; tgt = ss[(ss > s) & (ss <= s + h)]
            ta = Aidx.loc[[(c, my, int(x)) for x in tgt]]
            ia = Aidx.loc[(c, my, s)]
            rows.append({"n_target_obs": len(tgt), "t_night": bool(ta.night.iloc[0]) if h == 1 else bool(ta.night.any()),
                         "t_all_night": bool(ta.night.all()), "t_single": bool(ta.single.iloc[0]) if h == 1 else bool(ta.single.any()),
                         "t_high_unc": bool(ta.high_unc.iloc[0]) if h == 1 else bool(ta.high_unc.any()),
                         "t_high_risk": bool(ta.high_risk.any()),
                         "t_ltst": float(ta.local_true_solar_time.iloc[0]) if h == 1 else np.nan,
                         "t_obs_count": int(ta.observation_count.iloc[0]) if h == 1 else np.nan,
                         "t_unc": float(ta.dust_optical_depth_uncertainty.iloc[0]) if h == 1 else np.nan,
                         "i_night": bool(ia.night), "i_single": bool(ia.single), "i_high_unc": bool(ia.high_unc), "i_high_risk": bool(ia.high_risk)})
        return pd.DataFrame(rows, index=df.index)

    results, pairs, rowfiles = [], [], {}
    for h in (1, 3, 5):
        rf = P[(P.model == "random_forest_classifier") & (P.horizon == h) & (P.season == "autumn")]
        pe = P[(P.model == "persistence") & (P.horizon == h) & (P.season == "autumn")]
        J = rf[KEYS + ["solar_longitude_ls", "y", "p"]].merge(pe[KEYS + ["y", "p"]], on=KEYS, suffixes=("", "_pe"), validate="one_to_one")
        assert len(J) == len(rf) == len(pe) and (J.y == J.y_pe).all()
        J = J.rename(columns={"p": "p_rf"}); J = pd.concat([J, attrs(J, h)], axis=1)
        assert (J.n_target_obs >= 1).all(), "a defined target without an observed target sol"
        rowfiles[h] = J
        if h == 1:
            strata = {
                "all autumn": np.ones(len(J), bool),
                "target day": ~J.t_night, "target night": J.t_night,
                "target single retrieval": J.t_single, "target >=2 retrievals": ~J.t_single,
                "target lower uncertainty": ~J.t_high_unc, "target higher uncertainty": J.t_high_unc,
                "target day x single": ~J.t_night & J.t_single, "target day x multiple": ~J.t_night & ~J.t_single,
                "target night x single": J.t_night & J.t_single, "target night x multiple": J.t_night & ~J.t_single,
                "issuance day": ~J.i_night, "issuance night": J.i_night,
                "excl. high-risk target": ~J.t_high_risk,
                "excl. high-risk target or issuance": ~(J.t_high_risk | J.i_high_risk),
                "target day and >=2 retrievals (cleanest)": ~J.t_night & ~J.t_single,
            }
            fam = {"all autumn": "reference", "target day": "A target single-factor", "target night": "A target single-factor",
                   "target single retrieval": "A target single-factor", "target >=2 retrievals": "A target single-factor",
                   "target lower uncertainty": "A target single-factor", "target higher uncertainty": "A target single-factor",
                   "target day x single": "B crossed", "target day x multiple": "B crossed", "target night x single": "B crossed",
                   "target night x multiple": "B crossed", "issuance day": "D issuance", "issuance night": "D issuance",
                   "excl. high-risk target": "C sensitivity", "excl. high-risk target or issuance": "C sensitivity",
                   "target day and >=2 retrievals (cleanest)": "C sensitivity"}
            # reconciliation
            for a, b in (("target day", "target night"), ("target single retrieval", "target >=2 retrievals"),
                         ("target lower uncertainty", "target higher uncertainty"), ("issuance day", "issuance night")):
                assert int(strata[a].sum()) + int(strata[b].sum()) == len(J) == 10167
                assert int(J.y[strata[a]].sum()) + int(J.y[strata[b]].sum()) == 37
            cross = ["target day x single", "target day x multiple", "target night x single", "target night x multiple"]
            assert sum(int(strata[k].sum()) for k in cross) == 10167 and sum(int(J.y[strata[k]].sum()) for k in cross) == 37
            checks.append("reconciliation: every h = 1 stratification partitions the 10,167 autumn cases and 37 positives exactly")
        else:
            strata = {"all autumn": np.ones(len(J), bool), "target window all day": ~J.t_night, "target window >=1 night": J.t_night,
                      "excl. windows with a high-risk target obs": ~J.t_high_risk}
            fam = {"all autumn": "reference", "target window all day": f"E h{h} target window", "target window >=1 night": f"E h{h} target window",
                   "excl. windows with a high-risk target obs": f"E h{h} target window"}
            assert int(strata["target window all day"].sum()) + int(strata["target window >=1 night"].sum()) == len(J)

        # per-model metrics on identical rows for the classical models; LSTM on its own window subset
        for name, msk in strata.items():
            sub = J[np.asarray(msk)]
            for m in MODELS:
                g = P[(P.model == m) & (P.horizon == h) & (P.season == "autumn")].merge(sub[KEYS], on=KEYS)
                r = metrics(g.y.to_numpy(), g.p.to_numpy(), thr[(m, h)])
                results.append({"horizon": h, "stratum": name, "family": fam[name], "model": m, "tau": thr[(m, h)],
                                "rows": "identical to RF" if m != "lstm_w3" else "lstm_w3 window-feasible subset", **r})
            pr = {"horizon": h, "stratum": name, "family": fam[name], "n": len(sub), "n_pos": int(sub.y.sum()),
                  "reliable": bool(sub.y.sum() >= MIN_POS)}
            pr.update(paired(sub, thr[("random_forest_classifier", h)], thr[("persistence", h)]) if 0 < sub.y.sum() < len(sub) else {})
            pairs.append(pr)
            print(f"h{h} {name}: n={len(sub)} pos={int(sub.y.sum())}", flush=True)

    R = pd.DataFrame(results); Q = pd.DataFrame(pairs)
    for col in ("tss_p", "auc_p", "tss_p_solblock"):
        Q[col + "_holm"] = np.nan
        for fname, g in Q[Q.family != "reference"].groupby("family"):
            Q.loc[g.index, col + "_holm"] = holm(g[col].to_numpy())
    R.to_csv(out / "stratum_metrics.csv", index=False)
    Q.to_csv(out / "paired_rf_minus_persistence.csv", index=False)
    for h, J in rowfiles.items():
        J.to_csv(out / f"rows_autumn_h{h}.csv", index=False)
    (out / "checks.txt").write_text("\n".join(checks) + "\n")
    (out / "run_record.json").write_text(json.dumps({
        "script": "src/martian/run_local_time_sensitivity.py", "run": RUN.name, "predictions": str(SUPP.relative_to(ROOT)),
        "inputs": ["data/aggregated_cell_sol.csv", "metrics/storm_detection/*_evaluation.json", str(FOLLOW.relative_to(ROOT))],
        "uncertainty_cut_train_median": unc_cut, "day_definition": "LTST in [6, 18) h", "season": "Ls in [180, 270)",
        "bootstrap": {"unit": "cell (multinomial weights)", "B": B, "seed": 44, "ci": 0.95, "sensitivity_unit": "MY-sol", "sensitivity_seed": 20260818},
        "multiplicity": "Holm within family (A target single-factor, B crossed, C sensitivity, D issuance, E h3/h5 window), per metric",
        "min_positives_for_inference": MIN_POS}, indent=2))
    print("\n".join(checks)); print("written to", out)


if __name__ == "__main__":
    main()
