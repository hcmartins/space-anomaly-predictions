"""Paired A5 - A7 comparison (TSS, ROC-AUC, Brier) on the canonical 2026-10-04
fusion sweep's saved test predictions. Each configuration uses its own
validation-selected tau* (the `threshold` column). No training.

(a) Pooled test set, seed 44, both domains. Uses exactly the Chapter 6 seed-44
    paired procedure: uncertainty_analysis.paired_bootstrap_difference, i.e.
    moving blocks (solar 120 rows, Martian 1 row), B = 1000, a fresh
    np.random.default_rng(20260818) per comparison. ROC-AUC and Brier are
    computed on the identical index draws (same rng sequence). The TSS
    distribution is asserted equal to paired_bootstrap_difference's own output.
(b) Martian, seeds 44-48, within northern autumn and northern winter. Uses the
    cell-block paired test from run_fusion_supplementary.py
    (paired_within_season.csv): a cell-level multinomial(n_cells, 1/n_cells)
    weight vector per replicate, B = 1000, NumPy default_rng(44). First, that
    script's whole random stream is replayed and every saved
    paired_within_season.csv row is asserted to be reproduced. The A5 - A7 family
    then uses a fresh default_rng(44), drawn in the order seed 44..48 x (autumn,
    winter). ROC-AUC and Brier use the same weights as row sample weights.
    Solar has no seasons, so (b) is Martian only.

p-values: two-sided bootstrap p = min(1, 2 * min((#d<=0 + 1)/(B+1), (#d>=0 + 1)/(B+1))),
computed from the same replicates. Holm adjustment within each family, where a
family = one metric within one part: (a) 2 tests (solar, martian); (b) 10 tests
(5 seeds x 2 seasons).

Acceptance gate: before any new value is written, the script asserts that
  - seed-44 pooled TSS of A7 and A5 equals tss_summary.json and rounds to
    Chapter 6 section 8 (solar 0.321 / 0.532; Mars 0.593 / 0.645);
  - A5 - A7 point estimates and CIs reproduce uncertainty_report.md
    (Mars 0.0521 [0.0125, 0.0957]; solar 0.2112 [0.1730, 0.2499]);
  - every within-season A5/A7 TSS, ROC-AUC and Brier value reproduces
    fusion_supplementary_*/metrics_per_seed.csv.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from uncertainty_analysis import DEFAULT_BLOCK_LENGTHS, block_indices, paired_bootstrap_difference, percentile_interval

ROOT = Path(__file__).resolve().parent
SWEEP = ROOT / "outputs" / "uncertainty_sweep_20261004T092140396722Z"
ABL = sorted((ROOT / "outputs").glob("ablation_run_20261004T09*"))
FSUPP = ROOT / "outputs" / "fusion_supplementary_20261004T170020Z"
MSUPP = ROOT.parent / "martian/outputs/real/chapter6_supplementary_20261004T093226Z"
B = 1000
SEEDS = [44, 45, 46, 47, 48]
CH6_SEED44 = {("solar", "A7"): 0.321, ("solar", "A5"): 0.532, ("martian", "A7"): 0.593, ("martian", "A5"): 0.645}
REPORT_A5_A7 = {"martian": (0.0521, 0.0125, 0.0957), "solar": (0.2112, 0.1730, 0.2499)}
AUT, WIN = "Northern Autumn", "Northern Winter"


def load(cfg, dom, seed):
    d = pd.read_csv(SWEEP / f"predictions_{cfg}_{dom}_seed{seed}.csv")
    return d.y_true.to_numpy().astype(int), d.y_prob.to_numpy(), float(d.threshold.iloc[0])


def tss(y, p, t, w=None):
    w = np.ones(len(y)) if w is None else w
    yh = p >= t
    pos, neg = w[y == 1].sum(), w[y == 0].sum()
    return (w[(y == 1) & yh].sum() / pos - w[(y == 0) & yh].sum() / neg) if pos > 0 and neg > 0 else np.nan


def roc(y, p, w=None):
    keep = np.ones(len(y), bool) if w is None else w > 0
    yy = y[keep]
    return roc_auc_score(yy, p[keep], sample_weight=None if w is None else w[keep]) if 0 < yy.sum() < len(yy) else np.nan


def brier(y, p, w=None):
    return float(np.average((p - y) ** 2, weights=w))


def pval(d):
    d = d[np.isfinite(d)]
    return min(1.0, 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1)))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def season_of(ls):
    ls = np.mod(np.asarray(ls, float), 360)
    return np.select([ls < 90, ls < 180, ls < 270], ["Northern Spring", "Northern Summer", AUT], WIN)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    tss_json = {json.loads((d / "tss_summary.json").read_text())["seed"]: json.loads((d / "tss_summary.json").read_text())["tss"] for d in ABL}
    checks = []

    # ---------- gate 1: seed-44 pooled TSS vs tss_summary.json and Chapter 6 ----------
    for (dom, cfg), ch6 in CH6_SEED44.items():
        y, p, t = load(cfg, dom, 44)
        v = tss(y, p, t)
        assert abs(v - tss_json[44][cfg][dom]) < 1e-9, (dom, cfg, v)
        assert round(v, 3) == ch6, f"DISCREPANCY: {dom} {cfg} seed-44 TSS {v:.4f} != Chapter 6 {ch6}"
        checks.append(f"seed-44 pooled TSS {dom} {cfg} = {v:.6f} -> {round(v, 3)} (Chapter 6 section 8: {ch6}) OK")

    rows = []
    # ---------- (a) pooled, seed 44, Chapter 6 paired moving-block procedure ----------
    for dom in ("solar", "martian"):
        a = pd.read_csv(SWEEP / f"predictions_A5_{dom}_seed44.csv"); b = pd.read_csv(SWEEP / f"predictions_A7_{dom}_seed44.csv")
        L = DEFAULT_BLOCK_LENGTHS[dom]
        ref = paired_bootstrap_difference(a, b, L, B)          # the report's own function, seed 20260818
        y = a.y_true.to_numpy().astype(int); pa, pb = a.y_prob.to_numpy(), b.y_prob.to_numpy()
        ta, tb = float(a.threshold.iloc[0]), float(b.threshold.iloc[0])
        rng = np.random.default_rng(20260818)
        D = np.empty((B, 3))
        for i in range(B):
            idx = block_indices(len(y), L, rng)
            D[i] = (tss(y[idx], pa[idx], ta) - tss(y[idx], pb[idx], tb), roc(y[idx], pa[idx]) - roc(y[idx], pb[idx]),
                    brier(y[idx], pa[idx]) - brier(y[idx], pb[idx]))
        assert np.allclose(D[:, 0], ref, equal_nan=True, atol=1e-12), "index replay diverged from paired_bootstrap_difference"
        pt, lo, hi = tss(y, pa, ta) - tss(y, pb, tb), *percentile_interval(ref)
        rp, rlo, rhi = REPORT_A5_A7[dom]
        assert abs(round(pt, 4) - rp) < 1e-9 and abs(round(lo, 4) - rlo) < 1e-9 and abs(round(hi, 4) - rhi) < 1e-9, \
            f"DISCREPANCY vs uncertainty_report.md: {dom} {pt:.4f} [{lo:.4f}, {hi:.4f}] != {REPORT_A5_A7[dom]}"
        checks.append(f"(a) {dom} A5-A7 TSS {pt:.4f} [{lo:.4f}, {hi:.4f}] reproduces uncertainty_report.md {REPORT_A5_A7[dom]} OK")
        est = {"TSS": (tss(y, pa, ta), tss(y, pb, tb)), "ROC-AUC": (roc(y, pa), roc(y, pb)), "Brier": (brier(y, pa), brier(y, pb))}
        for k, metric in enumerate(("TSS", "ROC-AUC", "Brier")):
            l, h = percentile_interval(D[:, k])
            rows.append({"part": "a", "domain": dom, "season": "pooled", "seed": 44, "metric": metric,
                         "A5": est[metric][0], "A7": est[metric][1], "difference_A5_minus_A7": est[metric][0] - est[metric][1],
                         "ci95_lo": l, "ci95_hi": h, "p_value": pval(D[:, k]), "tau_A5": ta, "tau_A7": tb,
                         "n": len(y), "n_pos": int(y.sum()),
                         "block_unit": f"moving block of {L} consecutive test row(s)" + (" (= one cell-sol window)" if dom == "martian" else " (24 h at 12 min)"),
                         "replicates": B, "random_seed": 20260818})

    # ---------- (b) within season, Martian, cell-block paired test ----------
    P = pd.read_csv(MSUPP / "predictions_test_and_my37_late.csv.gz")
    q = P[(P.eval_set == "test") & (P.model == "lstm_w3") & (P.horizon == 1)].reset_index(drop=True)
    y1, p1, _ = load("A1", "martian", 44)
    assert (q.y.values == y1).all() and np.abs(q.p.values - p1).max() < 1e-6
    season = season_of(q.solar_longitude_ls.to_numpy()); cells = q.cell_id.to_numpy()
    preds = {(c, s): load(c, "martian", s) for c in ("A1", "A2", "A5", "A7") for s in SEEDS}
    f = lambda C: C[..., 0] / (C[..., 0] + C[..., 3]) - C[..., 1] / (C[..., 1] + C[..., 2])

    def cell_tables(msk, p, t, y):
        yb = y[msk].astype(bool); yh = p[msk] >= t
        c = np.stack([yb & yh, ~yb & yh, ~yb & ~yh, yb & ~yh], 1).astype(float)
        uc, inv = np.unique(cells[msk], return_inverse=True)
        return np.stack([np.bincount(inv, weights=c[:, k], minlength=len(uc)) for k in range(4)], 1), uc, inv

    # replay run_fusion_supplementary.py's stream and check every saved row
    saved = pd.read_csv(FSUPP / "paired_within_season.csv")
    rng = np.random.default_rng(44); k = 0
    for a, b in (("A7", "A1"), ("A7", "A2"), ("A5", "A1")):
        for s in SEEDS:
            ya, pa, ta = preds[(a, s)]; _, pb, tb = preds[(b, s)]
            for sea in ("ALL", AUT, WIN):
                msk = np.ones(len(ya), bool) if sea == "ALL" else season == sea
                A, uc, _ = cell_tables(msk, pa, ta, ya); Bt, _, _ = cell_tables(msk, pb, tb, ya)
                W = rng.multinomial(len(uc), np.full(len(uc), 1 / len(uc)), size=B)
                diff = f(W @ A) - f(W @ Bt); lo, hi = np.nanpercentile(diff, [2.5, 97.5])
                r = saved.iloc[k]; k += 1
                assert (r.comparison, r.seed, r.season) == (f"{a} - {b}", s, sea)
                assert abs(r["diff"] - (f(A.sum(0)) - f(Bt.sum(0)))) < 1e-12 and abs(r.ci95_lo - lo) < 1e-12 and abs(r.ci95_hi - hi) < 1e-12, (a, b, s, sea)
    checks.append(f"(b) replayed run_fusion_supplementary.py cell-block stream: all {k} saved paired_within_season.csv rows reproduced exactly OK")

    MP = pd.read_csv(FSUPP / "metrics_per_seed.csv")
    rng = np.random.default_rng(44)
    for s in SEEDS:
        ya, pa, ta = preds[("A5", s)]; _, pb, tb = preds[("A7", s)]
        for sea in (AUT, WIN):
            msk = season == sea
            A, uc, inv = cell_tables(msk, pa, ta, ya); Bt, _, _ = cell_tables(msk, pb, tb, ya)
            W = rng.multinomial(len(uc), np.full(len(uc), 1 / len(uc)), size=B)
            y, qa, qb = ya[msk], pa[msk], pb[msk]
            D = np.empty((B, 3)); D[:, 0] = f(W @ A) - f(W @ Bt)
            for i in range(B):
                w = W[i][inv].astype(float)
                D[i, 1] = roc(y, qa, w) - roc(y, qb, w)
                D[i, 2] = brier(y, qa, w) - brier(y, qb, w)
            est = {"TSS": (f(A.sum(0)), f(Bt.sum(0))), "ROC-AUC": (roc(y, qa), roc(y, qb)), "Brier": (brier(y, qa), brier(y, qb))}
            for cfg, j in (("A5", 0), ("A7", 1)):
                ref = MP[(MP.domain == "martian") & (MP.config == cfg) & (MP.seed == s) & (MP.stratum == sea)].iloc[0]
                for metric, col in (("TSS", "tss"), ("ROC-AUC", "roc_auc"), ("Brier", "brier")):
                    assert abs(est[metric][j] - ref[col]) < 1e-9, f"DISCREPANCY vs metrics_per_seed.csv: {cfg} {s} {sea} {metric}"
            for kk, metric in enumerate(("TSS", "ROC-AUC", "Brier")):
                lo, hi = np.nanpercentile(D[:, kk], [2.5, 97.5])
                rows.append({"part": "b", "domain": "martian", "season": sea.replace("Northern ", "").lower(), "seed": s, "metric": metric,
                             "A5": est[metric][0], "A7": est[metric][1], "difference_A5_minus_A7": est[metric][0] - est[metric][1],
                             "ci95_lo": lo, "ci95_hi": hi, "p_value": pval(D[:, kk]), "tau_A5": ta, "tau_A7": tb,
                             "n": int(msk.sum()), "n_pos": int(y.sum()), "block_unit": f"cell (cell_id; {len(uc)} cells, multinomial weights)",
                             "replicates": B, "random_seed": 44})
    checks.append("(b) every within-season A5/A7 TSS, ROC-AUC and Brier value reproduces fusion_supplementary metrics_per_seed.csv OK")

    R = pd.DataFrame(rows)
    R["holm_family"] = R.part + ":" + R.metric
    R["p_holm"] = np.nan
    for fam, g in R.groupby("holm_family"):
        R.loc[g.index, "p_holm"] = holm(g.p_value.to_numpy())
    R["holm_family_size"] = R.groupby("holm_family").p_value.transform("size")
    out = ROOT / "outputs" / ("a5_vs_a7_paired_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True)
    R.to_csv(out / "a5_vs_a7_paired.csv", index=False)
    (out / "checks.txt").write_text("\n".join(checks) + "\n", encoding="utf-8")
    print("\n".join(checks)); print(out)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(R.drop(columns=["block_unit", "holm_family"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
