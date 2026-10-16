"""Paired comparisons for the non-transductive (prototype) A7 forecasts (P13 follow-up).

A7 test forecasts: outputs/prototype_rescore_20261005T083742Z/predictions_prototype_A7_*_seed*.csv
(training-set prototype, tau* re-selected on validation with the same prototype).
A1, A2, A5 have no auxiliary pathway, so their canonical sweep forecasts are used unchanged.

(a) Pooled test set, seed 44, both domains:
    solar   - uncertainty_analysis.paired_bootstrap_difference (moving blocks of 120 rows,
              B = 1000, fresh default_rng(20260818) per comparison); ROC-AUC on the same draws.
    Martian - sol blocks (whole MY-sol groups resampled with replacement), exactly the paired
              procedure of run_fusion_code_audit_paired.py (fresh default_rng(20260818) per
              comparison), which Table 6.A3 identifies as the defensible unit; ROC-AUC on the same draws.
(b) Martian, seeds 44-48, within northern autumn and winter: the cell-block paired test of
    Table 6.24 (run_fusion_supplementary.py / run_a5_vs_a7_paired.py): cell-level
    multinomial(n_cells, 1/n_cells) weights, B = 1000, fresh default_rng(44) per comparison,
    drawn in the order seed 44..48 x (autumn, winter); ROC-AUC uses the same weights.
p-values: two-sided bootstrap p (run_a5_vs_a7_paired.pval). Holm within each family:
(a) one metric across 3 comparisons x 2 domains = 6 tests; (b) one comparison x one metric
across 5 seeds x 2 seasons = 10 tests.

Gates, all before any new value:
  1. every prototype TSS recomputed from predictions_prototype_*.csv equals metrics.csv
     (per seed) and summary.csv (seed mean) of the prototype re-score;
  2. the solar procedure reproduces uncertainty_report.md's canonical A5 - A7 interval
     (paired_bootstrap_difference on the sweep files);
  3. the Martian sol-block procedure reproduces the saved canonical A7 - A1 sol-block interval
     (fusion_code_audit_20261004T203942Z/P1_12_martian_paired_differences_by_block_unit.csv);
  4. the cell-block procedure reproduces every saved paired_within_season.csv row
     (replay of run_fusion_supplementary.py's stream).
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from run_a5_vs_a7_paired import holm, pval, roc, season_of, tss
from uncertainty_analysis import block_indices, paired_bootstrap_difference, percentile_interval

ROOT = Path(__file__).resolve().parent
SWEEP = ROOT / "outputs" / "uncertainty_sweep_20261004T092140396722Z"
PROTO = ROOT / "outputs" / "prototype_rescore_20261005T083742Z"
FSUPP = ROOT / "outputs" / "fusion_supplementary_20261004T170020Z"
AUDIT = ROOT / "outputs" / "fusion_code_audit_20261004T203942Z" / "P1_12_martian_paired_differences_by_block_unit.csv"
MSUPP = ROOT.parent / "martian/outputs/real/chapter6_supplementary_20261004T093226Z/predictions_test_and_my37_late.csv.gz"
B, SEEDS = 1000, [44, 45, 46, 47, 48]
AUT, WIN = "Northern Autumn", "Northern Winter"
COMPARATORS = ["A1", "A2", "A5"]


def load_canon(cfg, dom, seed):
    return pd.read_csv(SWEEP / f"predictions_{cfg}_{dom}_seed{seed:02d}.csv")


def load_proto(dom, seed):
    return pd.read_csv(PROTO / f"predictions_prototype_A7_{dom}_seed{seed:02d}.csv")


def gidx(k, rng):                       # run_fusion_code_audit_paired.py, verbatim logic
    u, inv = np.unique(k, return_inverse=True); m = [np.where(inv == i)[0] for i in range(len(u))]
    return np.concatenate([m[i] for i in rng.integers(0, len(u), len(u))])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    out = ROOT / "outputs" / ("prototype_paired_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True)
    checks = []
    P = pd.read_csv(MSUPP)
    q = P[(P.eval_set == "test") & (P.model == "lstm_w3") & (P.horizon == 1)].reset_index(drop=True)
    assert (q.y.values == load_canon("A1", "martian", 44).y_true.values).all()
    season = season_of(q.solar_longitude_ls.to_numpy()); cells = q.cell_id.to_numpy()
    sols = (q.mars_year.astype(str) + "_" + q.sol.astype(str)).to_numpy()
    strata = {"solar": {"ALL": None}, "martian": {"ALL": None, AUT: season == AUT, WIN: season == WIN}}

    # ---- gate 1: prototype TSS reproduces metrics.csv and summary.csv ----
    M = pd.read_csv(PROTO / "metrics.csv"); S = pd.read_csv(PROTO / "summary.csv")
    n = 0
    for cfg in ("A3", "A4", "A6", "A7"):
        for dom in ("solar", "martian"):
            vals = {st: [] for st in strata[dom]}
            for s in SEEDS:
                d = pd.read_csv(PROTO / f"predictions_prototype_{cfg}_{dom}_seed{s:02d}.csv")
                for st, msk in strata[dom].items():
                    m = np.ones(len(d), bool) if msk is None else msk
                    v = tss(d.y_true.to_numpy().astype(int)[m], d.y_prob.to_numpy()[m], float(d.threshold.iloc[0]))
                    ref = M[(M.config == cfg) & (M.domain == dom) & (M.seed == s) & (M.stratum == st) & (M["mode"] == "prototype")].tss.iloc[0]
                    assert abs(v - ref) < 1e-12, (cfg, dom, s, st, v, ref); vals[st].append(v); n += 1
            for st, v in vals.items():
                ref = S[(S.config == cfg) & (S.domain == dom) & (S.stratum == st) & (S["mode"] == "prototype")].tss_mean.iloc[0]
                assert abs(np.mean(v) - ref) < 1e-12, (cfg, dom, st)
    checks.append(f"gate 1: {n} prototype TSS values recomputed from predictions_prototype_*.csv equal metrics.csv; seed means equal summary.csv")

    # ---- gate 2: solar moving-block procedure reproduces uncertainty_report.md (A5 - A7 canonical) ----
    ref = paired_bootstrap_difference(load_canon("A5", "solar", 44), load_canon("A7", "solar", 44), 120, B)
    lo, hi = percentile_interval(ref)
    assert (round(lo, 4), round(hi, 4)) == (0.1730, 0.2499), (lo, hi)
    checks.append(f"gate 2: solar moving-block (120 rows) canonical A5-A7 interval [{lo:.4f}, {hi:.4f}] = uncertainty_report.md [0.1730, 0.2499]")

    # ---- gate 3: Martian sol-block procedure reproduces the saved canonical A7 - A1 sol interval ----
    a, b = load_canon("A7", "martian", 44), load_canon("A1", "martian", 44)
    y = a.y_true.to_numpy(); pa, pb = a.y_prob.to_numpy(), b.y_prob.to_numpy(); ta, tb = a.threshold[0], b.threshold[0]
    rng = np.random.default_rng(20260818)
    dist = np.array([(lambda i: tss(y[i], pa[i], ta) - tss(y[i], pb[i], tb))(gidx(sols, rng)) for _ in range(B)])
    sav = pd.read_csv(AUDIT).set_index("comparison").loc["A7-A1"]
    l3, h3 = percentile_interval(dist)
    assert abs(l3 - sav.lo_sol) < 1e-12 and abs(h3 - sav.hi_sol) < 1e-12, (l3, h3, sav.lo_sol, sav.hi_sol)
    checks.append(f"gate 3: Martian sol-block canonical A7-A1 interval [{l3:.4f}, {h3:.4f}] = saved P1_12 table [{sav.lo_sol:.4f}, {sav.hi_sol:.4f}]")

    # ---- gate 4: cell-block replay of run_fusion_supplementary.py ----
    f = lambda C: C[..., 0] / (C[..., 0] + C[..., 3]) - C[..., 1] / (C[..., 1] + C[..., 2])

    def cell_tables(msk, p, t, yy):
        yb = yy[msk].astype(bool); yh = p[msk] >= t
        c = np.stack([yb & yh, ~yb & yh, ~yb & ~yh, yb & ~yh], 1).astype(float)
        uc, inv = np.unique(cells[msk], return_inverse=True)
        return np.stack([np.bincount(inv, weights=c[:, k], minlength=len(uc)) for k in range(4)], 1), uc, inv

    saved = pd.read_csv(FSUPP / "paired_within_season.csv")
    rng = np.random.default_rng(44); k = 0
    for ca, cb in (("A7", "A1"), ("A7", "A2"), ("A5", "A1")):
        for s in SEEDS:
            da, db = load_canon(ca, "martian", s), load_canon(cb, "martian", s)
            yy = da.y_true.to_numpy().astype(int)
            for sea in ("ALL", AUT, WIN):
                msk = np.ones(len(yy), bool) if sea == "ALL" else season == sea
                A, uc, _ = cell_tables(msk, da.y_prob.to_numpy(), da.threshold[0], yy); Bt, _, _ = cell_tables(msk, db.y_prob.to_numpy(), db.threshold[0], yy)
                W = rng.multinomial(len(uc), np.full(len(uc), 1 / len(uc)), size=B)
                d = f(W @ A) - f(W @ Bt); l, h = np.nanpercentile(d, [2.5, 97.5]); r = saved.iloc[k]; k += 1
                assert abs(r.ci95_lo - l) < 1e-12 and abs(r.ci95_hi - h) < 1e-12, (ca, cb, s, sea)
    checks.append(f"gate 4: cell-block stream replayed; all {k} saved paired_within_season.csv rows reproduced")

    rows = []
    # ---- (a) pooled, seed 44 ----
    for dom in ("solar", "martian"):
        A7 = load_proto(dom, 44); y = A7.y_true.to_numpy().astype(int); p7, t7 = A7.y_prob.to_numpy(), float(A7.threshold.iloc[0])
        for cb in COMPARATORS:
            C = load_canon(cb, dom, 44); assert (C.y_true.to_numpy() == y).all()
            pc, tc = C.y_prob.to_numpy(), float(C.threshold.iloc[0])
            rng = np.random.default_rng(20260818); D = np.empty((B, 2))
            for i in range(B):
                idx = block_indices(len(y), 120, rng) if dom == "solar" else gidx(sols, rng)
                D[i] = (tss(y[idx], p7[idx], t7) - tss(y[idx], pc[idx], tc), roc(y[idx], p7[idx]) - roc(y[idx], pc[idx]))
            if dom == "solar":
                assert np.allclose(D[:, 0], paired_bootstrap_difference(A7, C, 120, B), equal_nan=True, atol=1e-12)
            est = {"TSS": (tss(y, p7, t7), tss(y, pc, tc)), "ROC-AUC": (roc(y, p7), roc(y, pc))}
            for j, met in enumerate(("TSS", "ROC-AUC")):
                l, h = percentile_interval(D[:, j])
                rows.append({"part": "a", "comparison": f"A7(prototype) - {cb}", "domain": dom, "season": "pooled", "seed": 44, "metric": met,
                             "A7_prototype": est[met][0], "comparator": est[met][1], "difference": est[met][0] - est[met][1],
                             "ci95_lo": l, "ci95_hi": h, "p_value": pval(D[:, j]), "tau_A7": t7, "tau_comparator": tc, "n": len(y), "n_pos": int(y.sum()),
                             "block_unit": "moving block, 120 rows" if dom == "solar" else f"sol ({len(np.unique(sols))} sols)",
                             "replicates": B, "random_seed": 20260818})
    checks.append("(a) solar TSS replicates equal paired_bootstrap_difference on the same files")

    # ---- (b) within season, Martian, cell blocks ----
    for cb in COMPARATORS:
        rng = np.random.default_rng(44)
        for s in SEEDS:
            A7, C = load_proto("martian", s), load_canon(cb, "martian", s)
            yy = A7.y_true.to_numpy().astype(int); assert (C.y_true.to_numpy() == yy).all()
            p7, t7, pc, tc = A7.y_prob.to_numpy(), float(A7.threshold.iloc[0]), C.y_prob.to_numpy(), float(C.threshold.iloc[0])
            for sea in (AUT, WIN):
                msk = season == sea
                A, uc, inv = cell_tables(msk, p7, t7, yy); Bt, _, _ = cell_tables(msk, pc, tc, yy)
                W = rng.multinomial(len(uc), np.full(len(uc), 1 / len(uc)), size=B)
                y, q7, qc = yy[msk], p7[msk], pc[msk]
                D = np.empty((B, 2)); D[:, 0] = f(W @ A) - f(W @ Bt)
                for i in range(B):
                    w = W[i][inv].astype(float); D[i, 1] = roc(y, q7, w) - roc(y, qc, w)
                est = {"TSS": (f(A.sum(0)), f(Bt.sum(0))), "ROC-AUC": (roc(y, q7), roc(y, qc))}
                for j, met in enumerate(("TSS", "ROC-AUC")):
                    l, h = np.nanpercentile(D[:, j], [2.5, 97.5])
                    rows.append({"part": "b", "comparison": f"A7(prototype) - {cb}", "domain": "martian", "season": sea.split()[1].lower(), "seed": s,
                                 "metric": met, "A7_prototype": est[met][0], "comparator": est[met][1], "difference": est[met][0] - est[met][1],
                                 "ci95_lo": l, "ci95_hi": h, "p_value": pval(D[:, j]), "tau_A7": t7, "tau_comparator": tc,
                                 "n": int(msk.sum()), "n_pos": int(y.sum()), "block_unit": f"cell ({len(uc)} cells, multinomial weights)",
                                 "replicates": B, "random_seed": 44})

    R = pd.DataFrame(rows)
    R["holm_family"] = np.where(R.part == "a", "a:" + R.metric, "b:" + R.comparison + ":" + R.metric)
    R["p_holm"] = np.nan
    for _, g in R.groupby("holm_family"):
        R.loc[g.index, "p_holm"] = holm(g.p_value.to_numpy())
    R["holm_family_size"] = R.groupby("holm_family").p_value.transform("size")
    R.to_csv(out / "prototype_paired.csv", index=False)
    (out / "checks.txt").write_text("\n".join(checks) + "\n")
    print("\n".join(checks))
    print(R.drop(columns=["block_unit", "holm_family", "replicates", "random_seed"]).round(4).to_string(index=False))
    print("written to", out)


if __name__ == "__main__":
    main()
