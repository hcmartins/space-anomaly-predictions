"""Fusion supplement for the 2026-10-04 5-seed sweep: within-season Martian
fusion scores, contingency counts, and threshold-independent metrics.

Inputs: the per-row test predictions of every configuration x domain x seed
(outputs/uncertainty_sweep_20261004T092140396722Z/predictions_*.csv; columns
y_true, y_prob, threshold, regime). Martian test rows are mapped to
(cell_id, mars_year, sol, Ls, season) through the canonical lstm_w3_h1 test
windows (same windowing, same order), and the mapping is VERIFIED: A1 (the
frozen unimodal Martian encoder + head) must reproduce the canonical
lstm_w3_h1 probabilities row for row. Every configuration's pooled TSS is also
asserted equal to its tss_summary.json before anything is written.

Outputs (outputs/fusion_supplementary_<UTC>/):
  metrics_per_seed.csv       contingency counts + TSS/POD/POFD/precision +
                             ROC-AUC, PR-AUC (average precision), Brier, per
                             domain x config x seed x stratum (Martian: ALL /
                             autumn / winter; solar: ALL / each year)
  metrics_seed_summary.csv   mean / sd / min / max over the 5 seeds
  paired_within_season.csv   Martian paired comparisons (A7-A1, A7-A2, A5-A1)
                             per seed and season, cell-block bootstrap 95% CI
  steps_per_epoch.json       steps per epoch computed with train_fusion.py's
                             own rule from the actual export arrays
"""
from __future__ import annotations

import glob
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent
SWEEP = ROOT / "outputs" / "uncertainty_sweep_20261004T092140396722Z"
ABL = sorted((ROOT / "outputs").glob("ablation_run_20261004T09*"))
MRUN = SRC / "martian" / "outputs" / "real" / "run_20261004T080014968911Z_seed42"
MSUPP = SRC / "martian" / "outputs" / "real" / "chapter6_supplementary_20261004T093226Z"
SOLAR_EXPORT = SRC / "solar" / "results" / "real_full_sequence_goes_sharp_20100501_20250724" / "fusion_export"
SEED = 44
N_BOOT = 1000


def season_of(ls):
    ls = np.mod(np.asarray(ls, float), 360)
    return np.select([ls < 90, ls < 180, ls < 270], ["Northern Spring", "Northern Summer", "Northern Autumn"], "Northern Winter")


def metrics(y, p, t):
    y = y.astype(bool); yh = p >= t
    tp, fp, tn, fn = int((y & yh).sum()), int((~y & yh).sum()), int((~y & ~yh).sum()), int((y & ~yh).sum())
    pod = tp / (tp + fn) if tp + fn else np.nan
    pofd = fp / (fp + tn) if fp + tn else np.nan
    two = 0 < y.sum() < len(y)
    return {"n": len(y), "n_pos": int(y.sum()), "prevalence": float(y.mean()) if len(y) else np.nan,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "pod": pod, "pofd": pofd,
            "precision": tp / (tp + fp) if tp + fp else np.nan, "tss": pod - pofd if two else np.nan,
            "roc_auc": float(roc_auc_score(y, p)) if two else np.nan,
            "pr_auc": float(average_precision_score(y, p)) if two else np.nan,
            "brier": float(np.mean((p - y) ** 2))}


def main() -> None:
    out = ROOT / "outputs" / ("fusion_supplementary_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True)
    rng = np.random.default_rng(SEED)

    # Martian row keys via the canonical lstm_w3_h1 test windows
    P = pd.read_csv(MSUPP / "predictions_test_and_my37_late.csv.gz")
    q = P[(P.eval_set == "test") & (P.model == "lstm_w3") & (P.horizon == 1)].reset_index(drop=True)
    a1 = pd.read_csv(SWEEP / "predictions_A1_martian_seed44.csv")
    assert len(q) == len(a1) and (q.y.values == a1.y_true.values).all()
    assert np.abs(q.p.values - a1.y_prob.values).max() < 1e-6, "row alignment with canonical lstm_w3_h1 failed"
    mkeys = q[["cell_id", "mars_year", "sol", "solar_longitude_ls"]].copy()
    mkeys["season"] = season_of(mkeys.solar_longitude_ls)

    tss_json = {}
    for d in ABL:
        j = json.loads((d / "tss_summary.json").read_text()); tss_json[j["seed"]] = j["tss"]

    rows, preds = [], {}
    for f in sorted(glob.glob(str(SWEEP / "predictions_*.csv"))):
        cfg, dom, seed = re.match(r"predictions_(A\d)_(\w+)_seed(\d+)\.csv", os.path.basename(f)).groups()
        seed = int(seed)
        d = pd.read_csv(f)
        y, p, t = d.y_true.to_numpy(), d.y_prob.to_numpy(), float(d.threshold.iloc[0])
        m_all = metrics(y, p, t)
        assert abs(m_all["tss"] - tss_json[seed][cfg][dom]) < 1e-9, (cfg, dom, seed, m_all["tss"], tss_json[seed][cfg][dom])
        strata = [("ALL", np.ones(len(d), bool))]
        if dom == "martian":
            strata += [(s, (mkeys.season == s).to_numpy()) for s in ("Northern Autumn", "Northern Winter")]
            preds[(cfg, seed)] = (y, p, t)
        else:
            strata += [(f"year {r}", (d.regime.astype(str) == str(r)).to_numpy()) for r in sorted(d.regime.unique())]
        for name, msk in strata:
            rows.append({"domain": dom, "config": cfg, "seed": seed, "stratum": name, "threshold": t, **metrics(y[msk], p[msk], t)})
    M = pd.DataFrame(rows)
    M.to_csv(out / "metrics_per_seed.csv", index=False)
    S = (M.groupby(["domain", "config", "stratum"])
         .agg(n=("n", "first"), n_pos=("n_pos", "first"),
              tss_mean=("tss", "mean"), tss_sd=("tss", "std"), tss_min=("tss", "min"), tss_max=("tss", "max"),
              roc_auc_mean=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std"),
              pr_auc_mean=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"),
              brier_mean=("brier", "mean"), precision_mean=("precision", "mean"),
              tp_mean=("tp", "mean"), fp_mean=("fp", "mean"), tn_mean=("tn", "mean"), fn_mean=("fn", "mean"))
         .reset_index())
    S.to_csv(out / "metrics_seed_summary.csv", index=False)

    # paired within-season comparisons (Martian), cell-block bootstrap
    cells = mkeys.cell_id.to_numpy()
    pair_rows = []
    for a, b in (("A7", "A1"), ("A7", "A2"), ("A5", "A1")):
        for seed in sorted(tss_json):
            ya, pa, ta = preds[(a, seed)]; yb, pb, tb = preds[(b, seed)]
            assert (ya == yb).all()
            for season in ("ALL", "Northern Autumn", "Northern Winter"):
                msk = np.ones(len(ya), bool) if season == "ALL" else (mkeys.season == season).to_numpy()
                y = ya[msk].astype(bool)
                ca = np.stack([(y & (pa[msk] >= ta)), (~y & (pa[msk] >= ta)), (~y & (pa[msk] < ta)), (y & (pa[msk] < ta))], 1).astype(float)
                cb = np.stack([(y & (pb[msk] >= tb)), (~y & (pb[msk] >= tb)), (~y & (pb[msk] < tb)), (y & (pb[msk] < tb))], 1).astype(float)
                uc, inv = np.unique(cells[msk], return_inverse=True)
                A = np.stack([np.bincount(inv, weights=ca[:, k], minlength=len(uc)) for k in range(4)], 1)
                B = np.stack([np.bincount(inv, weights=cb[:, k], minlength=len(uc)) for k in range(4)], 1)
                W = rng.multinomial(len(uc), np.full(len(uc), 1 / len(uc)), size=N_BOOT)
                f = lambda C: C[..., 0] / (C[..., 0] + C[..., 3]) - C[..., 1] / (C[..., 1] + C[..., 2])
                diff = f(W @ A) - f(W @ B)
                lo, hi = np.nanpercentile(diff, [2.5, 97.5])
                pair_rows.append({"comparison": f"{a} - {b}", "seed": seed, "season": season, "n": int(msk.sum()), "n_pos": int(y.sum()),
                                  "tss_a": float(f(A.sum(0))), "tss_b": float(f(B.sum(0))), "diff": float(f(A.sum(0)) - f(B.sum(0))),
                                  "ci95_lo": float(lo), "ci95_hi": float(hi),
                                  "verdict": "a better" if lo > 0 else ("b better" if hi < 0 else "n.s.")})
    pd.DataFrame(pair_rows).to_csv(out / "paired_within_season.csv", index=False)

    # steps per epoch, train_fusion.py rule: max(1, min(len(Xs)//n_S, len(Xm)//n_M))
    cfg = yaml.safe_load((ROOT / "configs" / "fusion.yaml").read_text())
    ns = np.load(SOLAR_EXPORT / "X_train.npy", mmap_mode="r").shape[0]
    nm = np.load(MRUN / "fusion_export" / "X_train.npy", mmap_mode="r").shape[0]
    steps = max(1, min(ns // cfg["batch_size_solar"], nm // cfg["batch_size_martian"]))
    (out / "steps_per_epoch.json").write_text(json.dumps({
        "rule": "max(1, min(len(X_solar_train) // batch_size_solar, len(X_martian_train) // batch_size_martian))  (training/train_fusion.py line 65)",
        "n_solar_train": int(ns), "n_martian_train": int(nm), "batch_size_solar": cfg["batch_size_solar"],
        "batch_size_martian": cfg["batch_size_martian"], "steps_per_epoch": int(steps),
        "binding_domain": "martian" if nm // cfg["batch_size_martian"] < ns // cfg["batch_size_solar"] else "solar",
        "note": "computed from the actual export arrays with the training code's own formula; train_fusion.py does not log it"}, indent=2))
    print(f"done: {out}")


if __name__ == "__main__":
    main()
