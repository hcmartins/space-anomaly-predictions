"""Recover per-instance test-set attention weights for the canonical 5-seed
sweep (outputs/uncertainty_sweep_20261004T092140396722Z), which saved
predictions but neither attention weights nor fusion model weights.

Each requested fusion configuration (A3, A4, A6, A7) is retrained with
run_ablation.py's own helpers at the sweep seed (train_fusion seeds itself
with torch.manual_seed(seed)), then predicted exactly as run_ablation.py
does (_predict_joint, seed for val, seed + 1 for test). The retrain is
ACCEPTED only if
  * the validation-selected threshold equals the sweep CSV's `threshold`, and
  * every test probability matches the sweep CSV to < 1e-6, in both domains.
Only accepted runs write their attention arrays; a mismatch is recorded in
verification.json and no alpha is written for that (config, seed).

Outputs (outputs/attention_recovery_<UTC>/):
  alpha_<config>_<domain>_seed<NN>.npy   (n_test, 2) [alpha_solar, alpha_martian]
  verification.json

    cd src/cross-domain
    python run_recover_attention.py --seeds 44 45 46 47 48 --configs A7 A3 A4 A6
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.ablation import get_configuration
from evaluation.thresholds import select_threshold_by_tss
from fusion_data import MARTIAN_EXPORT_DIR, SOLAR_EXPORT_DIR, load_domain_arrays, load_fusion_config
from run_ablation import DOMAINS, _base_training_config, _predict_joint, _tensor, _train_a7, _train_fusion_variant
from encoders.solar_encoder import CROSS_DOMAIN_ROOT, load_encoders_config

SWEEP = CROSS_DOMAIN_ROOT / "outputs" / "uncertainty_sweep_20261004T092140396722Z"
TOL = 1e-6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[44, 45, 46, 47, 48])
    parser.add_argument("--configs", nargs="+", default=["A7", "A3", "A4", "A6"])
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out = args.out_dir or CROSS_DOMAIN_ROOT / "outputs" / ("attention_recovery_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True, exist_ok=True)
    ver_path = out / "verification.json"
    verification = json.loads(ver_path.read_text()) if ver_path.exists() else {}

    arrays = {"solar": load_domain_arrays(SOLAR_EXPORT_DIR), "martian": load_domain_arrays(MARTIAN_EXPORT_DIR)}
    encoders_config = load_encoders_config()
    base_training_config = _base_training_config(load_fusion_config())

    for seed in args.seeds:
        for config_id in args.configs:
            cfg = get_configuration(config_id)
            if not cfg["use_fusion_pathway"]:
                raise ValueError(f"{config_id} has no attention pathway")
            if config_id == "A7":
                model, _ = _train_a7(arrays, encoders_config, base_training_config, seed)
            else:
                model, _ = _train_fusion_variant(config_id, arrays, encoders_config, base_training_config, seed)
            model.eval()
            val_out = _predict_joint(model, _tensor(arrays["solar"]["X_val"]), _tensor(arrays["martian"]["X_val"]), cfg=cfg, seed=seed)
            test_out = _predict_joint(model, _tensor(arrays["solar"]["X_test"]), _tensor(arrays["martian"]["X_test"]), cfg=cfg, seed=seed + 1)

            rec = {}
            for domain in DOMAINS:
                saved = pd.read_csv(SWEEP / f"predictions_{config_id}_{domain}_seed{seed:02d}.csv")
                tau, tau_src = select_threshold_by_tss(arrays[domain]["y_val"], val_out[domain]["y_prob"])
                max_diff = float(np.abs(test_out[domain]["y_prob"] - saved.y_prob.to_numpy()).max())
                ok = bool(np.isclose(tau, saved.threshold.iloc[0]) and max_diff < TOL and (saved.y_true.to_numpy() == arrays[domain]["y_test"]).all())
                rec[domain] = {"threshold_retrained": float(tau), "threshold_sweep": float(saved.threshold.iloc[0]),
                               "max_abs_test_prob_diff": max_diff, "accepted": ok}
                if ok:
                    np.save(out / f"alpha_{config_id}_{domain}_seed{seed:02d}.npy", test_out[domain]["alpha"])
            verification[f"{config_id}_seed{seed}"] = rec
            ver_path.write_text(json.dumps(verification, indent=2))
            print(f"{config_id} seed {seed}: {rec}", flush=True)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
