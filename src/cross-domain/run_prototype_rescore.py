"""Non-transductive re-scoring of the canonical fusion sweep (P13).

The canonical test forecasts (uncertainty_sweep_20261004T092140396722Z) were made by
run_ablation.py::_predict_joint, where each row's auxiliary summary is the mean projection of
the OTHER domain's whole validation / test partition. Here the same models are re-scored
with the single-domain API (CrossDomainFusionModel.predict_solar_only / predict_martian_only),
whose auxiliary summary is the stored TRAINING-set prototype (_update_prototypes, computed
with the restored best-epoch weights). tau* is re-selected on validation with the same
prototype (evaluation/thresholds.py::select_threshold_by_tss, unchanged).

Models. attention_recovery_20261005 kept only attention arrays, not weights, so each model is
rebuilt with the same verified deterministic retrain (run_ablation.py helpers at the sweep
seed). It is ACCEPTED only if its joint-mode validation tau* and every joint-mode test
probability reproduce the sweep CSV (< 1e-6) in both domains. Its state_dict, prototypes
included, is then saved, so later analyses need no rebuild.

A5 has no fusion pathway (heads read their own projection), so its forecasts do not depend on
the auxiliary summary; it is reported from the sweep unchanged, not passed through the
single-domain API (which would invoke scorers A5 never used). A3's noise and A4's shuffle
transform are not applied by the single-domain API: the prototype is used as is.

Outputs (outputs/prototype_rescore_<UTC>/): models/<cfg>_seed<NN>.pt, verification.json,
metrics.csv (canonical joint vs prototype, per config x domain x seed x stratum),
summary.csv (mean / min / max over seeds).

    cd src/cross-domain && python run_prototype_rescore.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from encoders.solar_encoder import CROSS_DOMAIN_ROOT, load_encoders_config
from evaluation.ablation import get_configuration
from evaluation.thresholds import select_threshold_by_tss
from fusion_data import MARTIAN_EXPORT_DIR, SOLAR_EXPORT_DIR, load_domain_arrays, load_fusion_config
from run_ablation import DOMAINS, _base_training_config, _predict_joint, _tensor, _train_a7, _train_fusion_variant

SRC = CROSS_DOMAIN_ROOT.parent
SWEEP = CROSS_DOMAIN_ROOT / "outputs" / "uncertainty_sweep_20261004T092140396722Z"
MSUPP = SRC / "martian" / "outputs" / "real" / "chapter6_supplementary_20261004T093226Z" / "predictions_test_and_my37_late.csv.gz"
SEEDS = [44, 45, 46, 47, 48]
RESCORED = ["A7", "A3", "A4", "A6"]
TOL = 1e-6


def season_of(ls):
    ls = np.mod(np.asarray(ls, float), 360)
    return np.select([ls < 90, ls < 180, ls < 270], ["Northern Spring", "Northern Summer", "Northern Autumn"], "Northern Winter")


def scores(y, p, t):
    y = y.astype(bool); d = p >= t
    tp, fp, tn, fn = int((y & d).sum()), int((~y & d).sum()), int((~y & ~d).sum()), int((y & ~d).sum())
    two = 0 < y.sum() < len(y)
    return {"n": len(y), "n_pos": int(y.sum()), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "tss": tp / (tp + fn) - fp / (fp + tn) if two else np.nan,
            "roc_auc": float(roc_auc_score(y, p)) if two else np.nan}


def main() -> None:
    out = CROSS_DOMAIN_ROOT / "outputs" / ("prototype_rescore_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    (out / "models").mkdir(parents=True)
    arrays = {"solar": load_domain_arrays(SOLAR_EXPORT_DIR), "martian": load_domain_arrays(MARTIAN_EXPORT_DIR)}
    enc_cfg = load_encoders_config()
    base_cfg = _base_training_config(load_fusion_config())

    # Martian season per test row (same alignment as run_fusion_supplementary.py)
    P = pd.read_csv(MSUPP)
    q = P[(P.eval_set == "test") & (P.model == "lstm_w3") & (P.horizon == 1)].reset_index(drop=True)
    a1 = pd.read_csv(SWEEP / "predictions_A1_martian_seed44.csv")
    assert len(q) == len(a1) and np.abs(q.p.values - a1.y_prob.values).max() < 1e-6
    season = season_of(q.solar_longitude_ls)
    strata = {"solar": [("ALL", np.ones(len(arrays["solar"]["y_test"]), bool))],
              "martian": [("ALL", np.ones(len(q), bool))] + [(s, season == s) for s in ("Northern Autumn", "Northern Winter")]}

    rows, ver = [], {}
    for seed in SEEDS:
        for cfg_id in ["A1", "A2", "A5"]:                      # no auxiliary pathway: canonical only
            for dom in DOMAINS:
                c = pd.read_csv(SWEEP / f"predictions_{cfg_id}_{dom}_seed{seed:02d}.csv")
                for st, m in strata[dom]:
                    s = scores(c.y_true.to_numpy()[m], c.y_prob.to_numpy()[m], c.threshold.iloc[0])
                    rows.append({"config": cfg_id, "domain": dom, "seed": seed, "stratum": st, "mode": "canonical_joint",
                                 "tau": float(c.threshold.iloc[0]), **s})
        for cfg_id in RESCORED:
            cfg = get_configuration(cfg_id)
            model, _ = (_train_a7(arrays, enc_cfg, base_cfg, seed) if cfg_id == "A7"
                        else _train_fusion_variant(cfg_id, arrays, enc_cfg, base_cfg, seed))
            model.eval()
            vj = _predict_joint(model, _tensor(arrays["solar"]["X_val"]), _tensor(arrays["martian"]["X_val"]), cfg=cfg, seed=seed)
            tj = _predict_joint(model, _tensor(arrays["solar"]["X_test"]), _tensor(arrays["martian"]["X_test"]), cfg=cfg, seed=seed + 1)
            rec = {}
            for dom in DOMAINS:
                c = pd.read_csv(SWEEP / f"predictions_{cfg_id}_{dom}_seed{seed:02d}.csv")
                tau_j, _ = select_threshold_by_tss(arrays[dom]["y_val"], vj[dom]["y_prob"])
                diff = float(np.abs(tj[dom]["y_prob"] - c.y_prob.to_numpy()).max())
                rec[dom] = {"tau_joint_retrained": float(tau_j), "tau_sweep": float(c.threshold.iloc[0]), "max_abs_test_diff": diff,
                            "accepted": bool(np.isclose(tau_j, c.threshold.iloc[0]) and diff < TOL)}
            ver[f"{cfg_id}_seed{seed}"] = rec
            (out / "verification.json").write_text(json.dumps(ver, indent=2))
            assert all(r["accepted"] for r in rec.values()), f"{cfg_id} seed {seed} does not reproduce the sweep: {rec}"
            torch.save(model.state_dict(), out / "models" / f"{cfg_id}_seed{seed:02d}.pt")

            with torch.no_grad():
                proto = {"solar": {"val": model.predict_solar_only(_tensor(arrays["solar"]["X_val"])).numpy(),
                                   "test": model.predict_solar_only(_tensor(arrays["solar"]["X_test"])).numpy()},
                         "martian": {"val": model.predict_martian_only(_tensor(arrays["martian"]["X_val"])).numpy(),
                                     "test": model.predict_martian_only(_tensor(arrays["martian"]["X_test"])).numpy()}}
            for dom in DOMAINS:
                c = pd.read_csv(SWEEP / f"predictions_{cfg_id}_{dom}_seed{seed:02d}.csv")
                tau_p, src = select_threshold_by_tss(arrays[dom]["y_val"], proto[dom]["val"])
                y = arrays[dom]["y_test"]
                pd.DataFrame({"y_true": y, "y_prob": proto[dom]["test"], "threshold": tau_p}).to_csv(
                    out / f"predictions_prototype_{cfg_id}_{dom}_seed{seed:02d}.csv", index=False)
                for st, m in strata[dom]:
                    for mode, p, t in (("canonical_joint", c.y_prob.to_numpy(), float(c.threshold.iloc[0])),
                                       ("prototype", proto[dom]["test"], float(tau_p))):
                        rows.append({"config": cfg_id, "domain": dom, "seed": seed, "stratum": st, "mode": mode, "tau": t,
                                     **scores(y[m], p[m], t)})
            print(f"{cfg_id} seed {seed} done", flush=True)
            pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)

    M = pd.DataFrame(rows)
    M.to_csv(out / "metrics.csv", index=False)
    S = M.groupby(["config", "domain", "stratum", "mode"]).agg(
        tss_mean=("tss", "mean"), tss_min=("tss", "min"), tss_max=("tss", "max"),
        auc_mean=("roc_auc", "mean"), auc_min=("roc_auc", "min"), auc_max=("roc_auc", "max")).reset_index()
    S.to_csv(out / "summary.csv", index=False)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
