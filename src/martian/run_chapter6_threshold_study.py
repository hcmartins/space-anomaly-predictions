"""Chapter 6 threshold-protocol study (Martian storm detection).

The canonical Martian run scores every classifier at a FIXED 0.5 threshold
(configs/task_storm_detection.yaml `decision_threshold: 0.5`), whereas the solar
pipeline (solar/modeling/metrics.py::select_threshold) and the cross-domain
fusion (cross-domain/evaluation/thresholds.py::select_threshold_by_tss) both
select tau* by maximising TSS on the VALIDATION split only (Eq. 6.1). This
script applies that same Eq. 6.1 protocol to the frozen canonical Martian models,
changing nothing else:

  * replays val + test predictions exactly as run_chapter6_supplementary.py does
    (tabular: joblib + train-fitted median imputer; sequence: state_dict +
    build_sequence_dataset + train-only imputation on a rebuilt features_df),
    asserting both splits' 0.5-threshold confusion matrices equal the saved
    *_evaluation.json values;
  * tau* = argmax_t TSS_val(t) over np.linspace(0.01, 0.99, 99) -- the identical
    grid used in solar and cross-domain; the test split plays no role;
  * reports test TSS at 0.5 and at tau*, with cell-block bootstrap intervals
    (200 replicates, 90% and 95%), plus test TP/FP/TN/FN at tau*.

No model is retrained; thresholds are the only thing that changes.
Outputs: outputs/real/chapter6_threshold_study_<UTC>/
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer

from data_sources.real_mcd_provider import RealMCDClimatologyProvider
from labeling.anomaly_labels import compute_exceedance, compute_z_scores, load_anomaly_definition_config
from modeling.sequence_models import SequenceModel, build_sequence_dataset
from modeling.train import _horizon_frame, _impute_sequence_train_only, _predict
from preprocessing.features import build_features, load_features_config
from preprocessing.grid import load_primary_grid
from splitting.chronological_split import assign_split, load_splits_config

from run_chapter6_supplementary import (HORIZONS, RUN, SEQUENCE, TABULAR, ci, confusion,
                                        tss_from)

GRID = np.linspace(0.01, 0.99, 99)
SEED = 44
N_BOOT = 200


def tss_at(y, p, t):
    return tss_from(*confusion(y, p, t))


def cell_boot_tss(y, p, t, cells, rng):
    ub, inv = np.unique(cells, return_inverse=True)
    idx_by = [np.where(inv == k)[0] for k in range(len(ub))]
    out = []
    for _ in range(N_BOOT):
        ii = np.concatenate([idx_by[k] for k in rng.integers(0, len(ub), len(ub))])
        out.append(tss_at(y[ii], p[ii], t))
    return np.array(out)


def main() -> None:
    t0 = time.perf_counter()
    out = RUN.parent / ("chapter6_threshold_study_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True)
    rng = np.random.default_rng(SEED)
    splits_cfg = load_splits_config()
    filled = pd.read_csv(RUN / "data" / "trailing_filled_cell_sol.csv")
    anomaly_cfg = load_anomaly_definition_config()
    fz = compute_exceedance(compute_z_scores(filled, RealMCDClimatologyProvider(), epsilon=anomaly_cfg["epsilon"]),
                            tau=anomaly_cfg["tau_primary"], config=anomaly_cfg)
    features_df = build_features(fz, load_primary_grid(), config=load_features_config())
    ds = pd.read_csv(RUN / "features" / "storm_detection_dataset.csv")
    fman = json.loads((RUN / "features" / "storm_detection_feature_manifest.json").read_text())["feature_manifest"]
    print(f"[0] features rebuilt ({time.perf_counter() - t0:.0f}s)", flush=True)

    rows = []
    for h in HORIZONS:
        target = f"target_storm_horizon_h{h}"
        frame = _horizon_frame(ds, fman, target)
        frame["split"] = assign_split(frame, splits_cfg)
        parts = {s: frame.loc[frame["split"] == s] for s in ("train", "val", "test")}
        imputer = SimpleImputer(strategy="median").fit(parts["train"][fman])
        preds = {}
        for kind, name in TABULAR:
            model = joblib.load(RUN / "models" / "storm_detection" / f"{name}_h{h}.joblib")
            preds[name] = {s: (parts[s][target].to_numpy(),
                               _predict(model, parts[s] if kind == "baseline" else imputer.transform(parts[s][fman]), task_type="classification"),
                               parts[s]["cell_id"].to_numpy()) for s in ("val", "test")}
        for w in (3, 5, 10):
            built = {s: build_sequence_dataset(parts[s], features_df, fman, window_length_sols=w, target_col=target)
                     for s in ("train", "val", "test")}
            _, Xva, Xte = _impute_sequence_train_only(built["train"][0], built["val"][0], built["test"][0])
            X = {"val": Xva, "test": Xte}
            for rnn in ("lstm", "gru"):
                name = f"{rnn}_w{w}"
                prov = json.loads((RUN / "metrics" / "storm_detection" / f"{name}_h{h}_provenance.json").read_text())
                model = SequenceModel(input_size=len(fman), rnn_type=rnn,
                                      **{k: prov["exact_parameters"][k] for k in ("hidden_size", "num_layers", "dropout")})
                model.load_state_dict(torch.load(RUN / "models" / "storm_detection" / f"{name}_h{h}.pt", map_location="cpu"))
                model.eval()
                preds[name] = {}
                for s in ("val", "test"):
                    with torch.no_grad():
                        p = 1 / (1 + np.exp(-model(torch.tensor(X[s], dtype=torch.float32)).numpy().ravel()))
                    preds[name][s] = (np.asarray(built[s][1]), p, built[s][2]["cell_id"].to_numpy())

        for name, sp in preds.items():
            ev = json.loads((RUN / "metrics" / "storm_detection" / f"{name}_h{h}_evaluation.json").read_text())["splits"]
            for s in ("val", "test"):
                mt = ev[s]["metrics"]
                assert confusion(sp[s][0], sp[s][1], 0.5) == (mt["tp"], mt["fp"], mt["tn"], mt["fn"]), (name, h, s)
            yv, pv, _ = sp["val"]; yt, pt, ct = sp["test"]
            val_tss = np.array([tss_at(yv, pv, t) for t in GRID])
            if np.all(np.isnan(val_tss)):
                t_star, src = 0.5, "fixed_fallback (val TSS undefined)"
            else:
                i = int(np.nanargmax(val_tss)); t_star, src = float(GRID[i]), "validation_selected (maximizing tss)"
            b05 = cell_boot_tss(yt, pt, 0.5, ct, rng); bst = cell_boot_tss(yt, pt, t_star, ct, rng)
            tp, fp, tn, fn = confusion(yt, pt, t_star)
            rows.append({"model": name, "horizon_sols": h, "n_val": len(yv), "n_val_pos": int(yv.sum()),
                         "n_test": len(yt), "n_test_pos": int(yt.sum()),
                         "val_tss_at_0.5": tss_at(yv, pv, 0.5), "tau_star": t_star, "tau_star_source": src,
                         "val_tss_at_tau_star": float(np.nanmax(val_tss)) if not np.all(np.isnan(val_tss)) else np.nan,
                         "tau_star_at_grid_edge": t_star in (GRID[0], GRID[-1]),
                         "test_tss_at_0.5": tss_at(yt, pt, 0.5),
                         "test_tss_0.5_ci95_lo": ci(b05, 0.95)[0], "test_tss_0.5_ci95_hi": ci(b05, 0.95)[1],
                         "test_tss_at_tau_star": tss_at(yt, pt, t_star),
                         "test_tss_tau_ci90_lo": ci(bst, 0.90)[0], "test_tss_tau_ci90_hi": ci(bst, 0.90)[1],
                         "test_tss_tau_ci95_lo": ci(bst, 0.95)[0], "test_tss_tau_ci95_hi": ci(bst, 0.95)[1],
                         "test_tp_at_tau_star": tp, "test_fp_at_tau_star": fp, "test_tn_at_tau_star": tn, "test_fn_at_tau_star": fn})
        print(f"[h={h}] done ({time.perf_counter() - t0:.0f}s)", flush=True)

    pd.DataFrame(rows).to_csv(out / "threshold_study.csv", index=False)
    print(f"done: {out} ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
