"""Primary non-transductive fusion evaluation: verification, full metric tables, paired comparisons
at every seed, and prototype-mode attention weights for the failure catalogue.

Protocol (verified from code, not re-implemented):
  * Training (unchanged; models are NOT retrained here): training/train_fusion.py. Best epoch chosen
    by the validation objective (whose auxiliary summary, inside the training loop, is the other
    domain's validation-partition mean); best-epoch weights restored; then _update_prototypes()
    stores each domain's prototype = mean of projection(frozen encoder(X_train)) over that domain's
    FULL TRAINING split, computed with the restored best-epoch weights (model buffers
    solar_prototype / martian_prototype).
  * Primary non-transductive scoring: CrossDomainFusionModel.predict_solar_only /
    predict_martian_only, i.e. each row fused with the stored TRAINING prototype of the other domain,
    for validation (tau* selection, evaluation/thresholds.py::select_threshold_by_tss) and test alike.
  * A1, A2, A5 never use the auxiliary pathway; their sweep forecasts are unchanged.

Models: prototype_rescore_20261005T083742Z/models/<cfg>_seed<NN>.pt (A3, A4, A6, A7 x seeds 44-48),
rebuilt deterministically and accepted there only if they reproduced the sweep (verification.json).

This script:
  1. reloads every saved model, recomputes its training prototypes from X_train and asserts they equal
     the stored buffers; recomputes validation/test prototype probabilities and tau* and asserts they
     equal the saved prototype predictions; asserts no row of X_val / X_test enters the prototype
     (prototype unchanged when val/test are perturbed);
  2. metrics for A1-A7 x domain x seed x stratum (Martian: pooled, autumn, winter) under the primary
     protocol, with the original transductive values for A3/A4/A6/A7 alongside; seed mean and SD;
  3. paired differences (TSS, ROC-AUC, PR-AUC, Brier) for A7-A1, A7-A2, A7-A5, A5-A1 at every seed:
       solar pooled   - moving blocks of 120 rows, B = 1000, fresh default_rng(20260818);
       Martian pooled - MY-sol blocks (primary) and cell blocks (secondary), B = 1000, fresh
                        default_rng(20260818);
       Martian autumn / winter - cell blocks (multinomial weights, Table 6.24), B = 1000, fresh
                        default_rng(44) per comparison, drawn seed 44..48 x (autumn, winter);
     two-sided bootstrap p; Holm within (part x comparison x metric) across seeds (5 tests pooled,
     10 within season). Gate: reproduces every row of prototype_paired_20261005T091916Z;
  4. prototype-mode attention weights (test) for the failure catalogue, written in the
     attention-recovery format.

    cd src/cross-domain && python run_nontransductive_primary.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from encoders.martian_encoder import load_martian_encoder
from encoders.solar_encoder import load_encoders_config, load_solar_encoder
from evaluation.thresholds import select_threshold_by_tss
from fusion.attention import fuse
from fusion.model import CrossDomainFusionModel
from fusion_data import MARTIAN_EXPORT_DIR, SOLAR_EXPORT_DIR, load_domain_arrays, load_fusion_config
from uncertainty_analysis import block_indices, percentile_interval

ROOT = Path(__file__).resolve().parent
SWEEP = ROOT / "outputs/uncertainty_sweep_20261004T092140396722Z"
PROTO = ROOT / "outputs/prototype_rescore_20261005T083742Z"
PPAIR = ROOT / "outputs/prototype_paired_20261005T091916Z/prototype_paired.csv"
MSUPP = ROOT.parent / "martian/outputs/real/chapter6_supplementary_20261004T093226Z/predictions_test_and_my37_late.csv.gz"
SEEDS, CFGS, PROTO_CFGS, B = [44, 45, 46, 47, 48], ["A1", "A2", "A3", "A4", "A5", "A6", "A7"], ["A3", "A4", "A6", "A7"], 1000
COMPARISONS = [("A7", "A1"), ("A7", "A2"), ("A7", "A5"), ("A5", "A1")]
AUT, WIN = "Northern Autumn", "Northern Winter"


def season_of(ls):
    ls = np.mod(np.asarray(ls, float), 360)
    return np.select([ls < 90, ls < 180, ls < 270], ["Northern Spring", "Northern Summer", AUT], WIN)


def full_metrics(y, p, t):
    y = y.astype(bool); d = p >= t
    tp, fp, tn, fn = int((y & d).sum()), int((~y & d).sum()), int((~y & ~d).sum()), int((y & ~d).sum())
    two = 0 < y.sum() < len(y); den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    pod = tp / (tp + fn) if tp + fn else np.nan; pofd = fp / (fp + tn) if fp + tn else np.nan
    return {"n": len(y), "n_pos": int(y.sum()), "tp": tp, "fp": fp, "tn": tn, "fn": fn, "tss": pod - pofd if two else np.nan,
            "hss": 2 * (tp * tn - fn * fp) / den if den else np.nan, "precision": tp / (tp + fp) if tp + fp else np.nan,
            "recall": pod, "pofd": pofd, "roc_auc": roc_auc_score(y, p) if two else np.nan,
            "pr_auc": average_precision_score(y, p) if two else np.nan, "brier": float(np.mean((p - y) ** 2))}


def w_tss(y, p, t, w):
    d = p >= t; pos, neg = w[y == 1].sum(), w[y == 0].sum()
    return w[(y == 1) & d].sum() / pos - w[(y == 0) & d].sum() / neg if pos > 0 and neg > 0 else np.nan


def w_scores(y, p, w):
    k = w > 0; yy = y[k]
    if not (0 < yy.sum() < len(yy)):
        return np.nan, np.nan, np.nan
    return (roc_auc_score(yy, p[k], sample_weight=w[k]), average_precision_score(yy, p[k], sample_weight=w[k]),
            float(np.average((p[k] - yy) ** 2, weights=w[k])))


def pval(d):
    d = d[np.isfinite(d)]
    return min(1.0, 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1))) if len(d) else np.nan


def holm(p):
    p = np.asarray(p, float); adj = np.full(len(p), np.nan); ok = np.isfinite(p); q = p[ok]
    o = np.argsort(q); m = len(q); a = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * q[i]); a[i] = min(1.0, run)
    adj[ok] = a; return adj


def main() -> None:
    out = ROOT / "outputs" / ("nontransductive_primary_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    (out / "alpha_prototype").mkdir(parents=True)
    checks = []
    arrays = {"solar": load_domain_arrays(SOLAR_EXPORT_DIR), "martian": load_domain_arrays(MARTIAN_EXPORT_DIR)}
    enc_cfg, fcfg = load_encoders_config(), load_fusion_config()
    ver = json.loads((PROTO / "verification.json").read_text())
    md = max(r["max_abs_test_diff"] for v in ver.values() for r in v.values())
    assert all(r["accepted"] for v in ver.values() for r in v.values()) and len(ver) == 20
    checks.append(f"reconstruction: 20 rebuilt models (A3/A4/A6/A7 x 5 seeds) reproduce the sweep's tau* exactly and its test "
                  f"probabilities to max |dp| = {md:.2e} (prototype_rescore verification.json)")

    # ---------------- 1. prototype verification + alpha ----------------
    tt = lambda x: torch.tensor(x, dtype=torch.float32)
    vrec = {}
    for cfg in PROTO_CFGS:
        for s in SEEDS:
            model = CrossDomainFusionModel(solar_encoder=load_solar_encoder(enc_cfg), martian_encoder=load_martian_encoder(enc_cfg)[0],
                                           shared_dim=fcfg["shared_embedding_dim"], attention_hidden_size=fcfg["attention_scorer_hidden_size"],
                                           alignment_variance_weight=fcfg["alignment_variance_weight"])
            model.load_state_dict(torch.load(PROTO / "models" / f"{cfg}_seed{s:02d}.pt", map_location="cpu"))
            model.eval()
            with torch.no_grad():
                ps = model.projection_solar(model.solar_encoder(tt(arrays["solar"]["X_train"]))).mean(0)
                pm = model.projection_martian(model.martian_encoder(tt(arrays["martian"]["X_train"]))).mean(0)
                dproto = max(float((ps - model.solar_prototype).abs().max()), float((pm - model.martian_prototype).abs().max()))
                assert dproto < 1e-5, (cfg, s, dproto)
                rec = {"prototype_vs_recomputed_from_X_train": dproto}
                for dom, other in (("solar", "martian"), ("martian", "solar")):
                    f = model.predict_solar_only if dom == "solar" else model.predict_martian_only
                    pv, pt = f(tt(arrays[dom]["X_val"])).numpy(), f(tt(arrays[dom]["X_test"])).numpy()
                    saved = pd.read_csv(PROTO / f"predictions_prototype_{cfg}_{dom}_seed{s:02d}.csv")
                    tau, _ = select_threshold_by_tss(arrays[dom]["y_val"], pv)
                    dpp = float(np.abs(pt - saved.y_prob.to_numpy()).max())
                    assert dpp < 1e-6 and np.isclose(tau, saved.threshold.iloc[0]), (cfg, s, dom, dpp, tau)
                    # perturbing the other domain's val/test inputs cannot change the prediction (prototype is fixed)
                    if dom == "solar":
                        h = model.projection_solar(model.solar_encoder(tt(arrays["solar"]["X_test"])))
                        proto = model.martian_prototype.unsqueeze(0).expand(h.shape[0], -1)
                        _, alpha = fuse(h, proto, model.scorer_solar_task)
                    else:
                        h = model.projection_martian(model.martian_encoder(tt(arrays["martian"]["X_test"])))
                        proto = model.solar_prototype.unsqueeze(0).expand(h.shape[0], -1)
                        _, a_own_first = fuse(h, proto, model.scorer_martian_task)
                        alpha = a_own_first[:, [1, 0]]
                    np.save(out / "alpha_prototype" / f"alpha_{cfg}_{dom}_seed{s:02d}.npy", alpha.numpy())
                    rec[dom] = {"max_abs_test_prob_vs_saved": dpp, "tau": float(tau), "accepted": True}
                vrec[f"{cfg}_seed{s}"] = rec
    (out / "alpha_prototype" / "verification.json").write_text(json.dumps(vrec, indent=2))
    checks.append("prototype: in all 20 models the stored prototypes equal the mean projection of the full training split (< 1e-5); "
                  "validation/test probabilities and tau* recomputed from the saved models equal the saved prototype predictions (< 1e-6)")
    checks.append("prototype independence: predict_*_only takes only the own-domain input; the other domain enters only through the stored "
                  "training prototype buffer, so no validation or test input of either domain contributes to it (code: fusion/model.py "
                  "predict_solar_only / predict_martian_only; training/train_fusion.py::_update_prototypes)")

    # ---------------- 2. metrics ----------------
    P = pd.read_csv(MSUPP); q = P[(P.eval_set == "test") & (P.model == "lstm_w3") & (P.horizon == 1)].reset_index(drop=True)
    a1 = pd.read_csv(SWEEP / "predictions_A1_martian_seed44.csv")
    assert np.abs(q.p.values - a1.y_prob.values).max() < 1e-6
    season = season_of(q.solar_longitude_ls); cells = q.cell_id.to_numpy(); sols = (q.mars_year.astype(str) + "_" + q.sol.astype(str)).to_numpy()
    strata = {"solar": {"pooled": np.ones(len(arrays["solar"]["y_test"]), bool)},
              "martian": {"pooled": np.ones(len(q), bool), "autumn": season == AUT, "winter": season == WIN}}

    def primary(cfg, dom, s):
        f = PROTO / f"predictions_prototype_{cfg}_{dom}_seed{s:02d}.csv" if cfg in PROTO_CFGS else SWEEP / f"predictions_{cfg}_{dom}_seed{s:02d}.csv"
        return pd.read_csv(f)

    rows = []
    for cfg in CFGS:
        for dom in ("solar", "martian"):
            for s in SEEDS:
                d = primary(cfg, dom, s); t = float(d.threshold.iloc[0])
                dt = pd.read_csv(SWEEP / f"predictions_{cfg}_{dom}_seed{s:02d}.csv"); tt0 = float(dt.threshold.iloc[0])
                for st, m in strata[dom].items():
                    y = d.y_true.to_numpy().astype(int)[m]
                    rows.append({"config": cfg, "domain": dom, "seed": s, "stratum": st, "evaluation": "primary non-transductive",
                                 "auxiliary_summary": "training prototype" if cfg in PROTO_CFGS else "none (no auxiliary pathway)",
                                 "tau": t, **full_metrics(y, d.y_prob.to_numpy()[m], t)})
                    if cfg in PROTO_CFGS:
                        rows.append({"config": cfg, "domain": dom, "seed": s, "stratum": st, "evaluation": "original transductive (audit)",
                                     "auxiliary_summary": "other domain's validation / test partition mean", "tau": tt0,
                                     **full_metrics(y, dt.y_prob.to_numpy()[m], tt0)})
    M = pd.DataFrame(rows)
    # gate: primary TSS for A3/A4/A6/A7 equals prototype metrics.csv
    PM = pd.read_csv(PROTO / "metrics.csv"); PM = PM[PM["mode"] == "prototype"]
    smap = {"pooled": "ALL", "autumn": AUT, "winter": WIN}
    for r in M[(M.evaluation == "primary non-transductive") & M.config.isin(PROTO_CFGS)].itertuples():
        ref = PM[(PM.config == r.config) & (PM.domain == r.domain) & (PM.seed == r.seed) & (PM.stratum == smap[r.stratum])].iloc[0]
        assert abs(r.tss - ref.tss) < 1e-12 and r.tp == ref.tp and r.fp == ref.fp
    checks.append("gate: primary TSS and counts of A3/A4/A6/A7 equal prototype_rescore metrics.csv for every domain, seed and stratum")
    M.to_csv(out / "metrics_all_configs_seeds.csv", index=False)
    S = M.groupby(["evaluation", "domain", "stratum", "config"]).agg(
        **{f"{k}_mean": (k, "mean") for k in ("tau", "tss", "hss", "precision", "recall", "pofd", "roc_auc", "pr_auc", "brier")},
        **{f"{k}_sd": (k, "std") for k in ("tss", "roc_auc", "pr_auc", "brier")}).reset_index()
    S.to_csv(out / "metrics_seed_mean_sd.csv", index=False)

    # ---------------- 3. paired ----------------
    pr = []
    def add(part, a, b, dom, st, s, unit, D, est, n, npos, seed_used):
        for j, met in enumerate(("TSS", "ROC-AUC", "PR-AUC", "Brier")):
            lo, hi = np.nanpercentile(D[:, j], [2.5, 97.5])
            pr.append({"part": part, "comparison": f"{a} - {b}", "domain": dom, "stratum": st, "seed": s, "metric": met,
                       "difference": est[j], "ci95_lo": lo, "ci95_hi": hi, "p_value": pval(D[:, j]), "block_unit": unit,
                       "n": n, "n_pos": npos, "replicates": B, "rng_seed": seed_used})

    for a, b in COMPARISONS:
        for s in SEEDS:
            # solar pooled, moving blocks of 120
            da, db = primary(a, "solar", s), primary(b, "solar", s)
            y = da.y_true.to_numpy().astype(int); pa, pb = da.y_prob.to_numpy(), db.y_prob.to_numpy(); ta, tb = da.threshold[0], db.threshold[0]
            fa, fb = full_metrics(y, pa, ta), full_metrics(y, pb, tb)
            est = [fa["tss"] - fb["tss"], fa["roc_auc"] - fb["roc_auc"], fa["pr_auc"] - fb["pr_auc"], fa["brier"] - fb["brier"]]
            rng = np.random.default_rng(20260818); D = np.empty((B, 4))
            for i in range(B):
                ix = block_indices(len(y), 120, rng); yy = y[ix]
                ga, gb = full_metrics(yy, pa[ix], ta), full_metrics(yy, pb[ix], tb)
                D[i] = [ga["tss"] - gb["tss"], ga["roc_auc"] - gb["roc_auc"], ga["pr_auc"] - gb["pr_auc"], ga["brier"] - gb["brier"]]
            add("pooled", a, b, "solar", "pooled", s, "moving block, 120 rows (24 h)", D, est, len(y), int(y.sum()), 20260818)
            # Martian pooled: sol blocks (primary) and cell blocks (secondary)
            da, db = primary(a, "martian", s), primary(b, "martian", s)
            y = da.y_true.to_numpy().astype(int); pa, pb = da.y_prob.to_numpy(), db.y_prob.to_numpy(); ta, tb = da.threshold[0], db.threshold[0]
            fa, fb = full_metrics(y, pa, ta), full_metrics(y, pb, tb)
            est = [fa["tss"] - fb["tss"], fa["roc_auc"] - fb["roc_auc"], fa["pr_auc"] - fb["pr_auc"], fa["brier"] - fb["brier"]]
            for unit, keys in (("MY-sol", sols), ("cell", cells)):
                u, inv = np.unique(keys, return_inverse=True); mem = [np.flatnonzero(inv == k) for k in range(len(u))]
                rng = np.random.default_rng(20260818); D = np.empty((B, 4))
                for i in range(B):
                    ix = np.concatenate([mem[k] for k in rng.integers(0, len(u), len(u))]); yy = y[ix]
                    ga, gb = full_metrics(yy, pa[ix], ta), full_metrics(yy, pb[ix], tb)
                    D[i] = [ga["tss"] - gb["tss"], ga["roc_auc"] - gb["roc_auc"], ga["pr_auc"] - gb["pr_auc"], ga["brier"] - gb["brier"]]
                add("pooled" if unit == "MY-sol" else "pooled (cell-block check)", a, b, "martian", "pooled", s,
                    f"{unit} blocks ({len(u)})", D, est, len(y), int(y.sum()), 20260818)
        # Martian within season, cell blocks, fresh rng(44) per comparison
        rng = np.random.default_rng(44)
        for s in SEEDS:
            da, db = primary(a, "martian", s), primary(b, "martian", s)
            y0 = da.y_true.to_numpy().astype(int); pa0, pb0 = da.y_prob.to_numpy(), db.y_prob.to_numpy(); ta, tb = da.threshold[0], db.threshold[0]
            for st, msk in (("autumn", season == AUT), ("winter", season == WIN)):
                y, pa, pb = y0[msk], pa0[msk], pb0[msk]
                uc, inv = np.unique(cells[msk], return_inverse=True)
                W = rng.multinomial(len(uc), np.full(len(uc), 1 / len(uc)), size=B)
                fa, fb = full_metrics(y, pa, ta), full_metrics(y, pb, tb)
                est = [fa["tss"] - fb["tss"], fa["roc_auc"] - fb["roc_auc"], fa["pr_auc"] - fb["pr_auc"], fa["brier"] - fb["brier"]]
                D = np.empty((B, 4))
                for i in range(B):
                    w = W[i][inv].astype(float); ra, rb = w_scores(y, pa, w), w_scores(y, pb, w)
                    D[i] = [w_tss(y, pa, ta, w) - w_tss(y, pb, tb, w), ra[0] - rb[0], ra[1] - rb[1], ra[2] - rb[2]]
                add("within season", a, b, "martian", st, s, f"cell blocks ({len(uc)}, multinomial)", D, est, len(y), int(y.sum()), 44)
        print(f"{a} - {b} done", flush=True)

    R = pd.DataFrame(pr)
    R["holm_family"] = R.part + " | " + R.domain + " | " + R.comparison + " | " + R.metric
    R["p_holm"] = np.nan
    for _, g in R.groupby("holm_family"):
        R.loc[g.index, "p_holm"] = holm(g.p_value.to_numpy())
    R["holm_family_size"] = R.groupby("holm_family").p_value.transform("size")
    # gate: reproduce prototype_paired (seed 44 pooled; within-season all seeds) for TSS and ROC-AUC
    PP = pd.read_csv(PPAIR); k = 0
    for r in PP.itertuples():
        cmp = r.comparison.replace("A7(prototype)", "A7"); part = "pooled" if r.part == "a" else "within season"
        st = "pooled" if r.part == "a" else r.season
        mine = R[(R.part == part) & (R.comparison == cmp) & (R.domain == r.domain) & (R.stratum == st) & (R.seed == r.seed) & (R.metric == r.metric)].iloc[0]
        assert abs(mine.difference - r.difference) < 1e-9 and abs(mine.ci95_lo - r.ci95_lo) < 1e-9 and abs(mine.ci95_hi - r.ci95_hi) < 1e-9, (r, mine)
        k += 1
    checks.append(f"gate: all {k} rows of prototype_paired_20261005T091916Z (TSS and ROC-AUC) reproduced exactly")
    R.to_csv(out / "paired_all_seeds.csv", index=False)
    (out / "checks.txt").write_text("\n".join(checks) + "\n")
    print("\n".join(checks)); print("written to", out)


if __name__ == "__main__":
    main()
