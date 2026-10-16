"""Follow-up to run_chapter6_threshold_study.py: paired significance vs
persistence and per-season metrics at each model's validation-selected tau*
(Eq. 6.1), re-using the verified test predictions written by
run_chapter6_supplementary.py. Identical bootstrap/McNemar code and seeds as
that script; only the thresholds differ. Outputs go into the threshold-study
folder."""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.metrics import average_precision_score, roc_auc_score

from labeling.reporting import assign_mars_season
from run_chapter6_supplementary import KEYS, N_BOOT, SEED, ci, confusion, safe_auc, tss_from

ROOT = Path(__file__).resolve().parent / "outputs" / "real"
SUPP = sorted(ROOT.glob("chapter6_supplementary_*"))[-1]
STUDY = sorted(ROOT.glob("chapter6_threshold_study_*"))[-1]


def main() -> None:
    P = pd.read_csv(SUPP / "predictions_test_and_my37_late.csv.gz")
    P = P[P.eval_set == "test"]
    tau = pd.read_csv(STUDY / "threshold_study.csv").set_index(["model", "horizon_sols"])["tau_star"].to_dict()
    rng = np.random.default_rng(SEED + 1)
    sig = []
    for h in (1, 3, 5):
        ref = P[(P.model == "persistence") & (P.horizon == h)].set_index(KEYS)
        ta = tau[("persistence", h)]
        for m in sorted(P.model.unique()):
            if m == "persistence":
                continue
            j = ref.join(P[(P.model == m) & (P.horizon == h)].set_index(KEYS), lsuffix="_a", rsuffix="_b", how="inner")
            y, pa, pb, tb = j.y_a.to_numpy(), j.p_a.to_numpy(), j.p_b.to_numpy(), tau[(m, h)]
            ca = (pa >= ta) == y.astype(bool); cb = (pb >= tb) == y.astype(bool)
            n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
            stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if n01 + n10 else np.nan
            cells = j.index.get_level_values("cell_id").to_numpy()
            ub, inv = np.unique(cells, return_inverse=True); idx = [np.where(inv == k)[0] for k in range(len(ub))]
            d = []
            for _ in range(N_BOOT):
                ii = np.concatenate([idx[k] for k in rng.integers(0, len(ub), len(ub))])
                d.append(tss_from(*confusion(y[ii], pa[ii], ta)) - tss_from(*confusion(y[ii], pb[ii], tb)))
            lo, hi = ci(np.array(d), 0.95)
            sig.append({"horizon_sols": h, "model_b": m, "tau_b": tb, "n_paired": len(y),
                        "tss_persistence": tss_from(*confusion(y, pa, ta)), "tss_b": tss_from(*confusion(y, pb, tb)),
                        "tss_diff_b_minus_persistence": tss_from(*confusion(y, pb, tb)) - tss_from(*confusion(y, pa, ta)),
                        "ci95_lo": -hi, "ci95_hi": -lo, "significant_at_0.05": bool(lo > 0 or hi < 0),
                        "mcnemar_chi2_cc": stat, "mcnemar_log10_p": chi2.logsf(stat, 1) / np.log(10) if n01 + n10 else np.nan,
                        "mcnemar_exact_p": binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else np.nan})
    pd.DataFrame(sig).to_csv(STUDY / "significance_vs_persistence_at_tau_star.csv", index=False)

    ds = pd.read_csv(SUPP.parent / "run_20260813T195013457439Z_seed42" / "features" / "storm_detection_dataset.csv",
                     usecols=KEYS + ["solar_longitude_ls"]).drop_duplicates(KEYS)
    P = P.drop(columns=[c for c in ("solar_longitude_ls", "season") if c in P.columns]).merge(ds, on=KEYS, how="left")
    P["season"] = assign_mars_season(P["solar_longitude_ls"])
    rng = np.random.default_rng(SEED)
    rows = []
    for (m, h), g in P.groupby(["model", "horizon"]):
        t = tau[(m, h)]
        for season, gg in [("ALL", g)] + list(g.groupby("season")):
            y, p, cells = gg.y.to_numpy(), gg.p.to_numpy(), gg.cell_id.to_numpy()
            row = {"model": m, "horizon_sols": h, "tau_star": t, "season": season, "n": len(y), "n_pos": int(y.sum()),
                   "tss": tss_from(*confusion(y, p, t))}
            if 0 < y.sum() < len(y):
                ub, inv = np.unique(cells, return_inverse=True); idx = [np.where(inv == k)[0] for k in range(len(ub))]
                b = []
                for _ in range(N_BOOT):
                    ii = np.concatenate([idx[k] for k in rng.integers(0, len(ub), len(ub))])
                    b.append(tss_from(*confusion(y[ii], p[ii], t)))
                row["tss_ci95_lo"], row["tss_ci95_hi"] = ci(np.array(b), 0.95)
            rows.append(row)
    pd.DataFrame(rows).to_csv(STUDY / "metrics_by_season_at_tau_star.csv", index=False)
    print("done", STUDY)


if __name__ == "__main__":
    main()
