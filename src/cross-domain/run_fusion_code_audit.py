"""Computed evidence for the Chapter 5 code audit (P1.7, P1.11, P1.12) on the
canonical 2026-10-04 fusion sweep. No training.

P1.7  A2 trainable-parameter counts, by instantiating CapacityMatchedUnimodalModel
      with the real frozen encoders and the real A7 per-domain targets, counted
      as sum(p.numel() for p in model.parameters() if p.requires_grad); also the
      A7 total and per-module breakdown. Cross-checked against history_A2.json.
P1.11 L_align per epoch for every fusion configuration and seed, read from
      ablation_run_20261004T09*/history_A*.json -> CSV + line plot.
P1.12 Martian primary-seed (44) pooled intervals under three resampling units,
      B = 1000, seed 20260818 (the canonical report's seed):
        one-cell-sol blocks  = uncertainty_analysis.block_indices(n, 1, rng)
                               (reproduces uncertainty_report.md exactly -- asserted)
        cell blocks          = resample whole cell_id groups with replacement
        sol blocks           = resample whole (mars_year, sol) groups
      for TSS (at the configuration's own validation-selected threshold),
      ROC-AUC and PR-AUC. Test windows are mapped to (cell_id, mars_year, sol)
      through the canonical lstm_w3_h1 test windows; the mapping is verified by
      A1 reproducing the canonical probabilities row for row.
"""
from __future__ import annotations

import glob
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from encoders.martian_encoder import load_martian_encoder
from encoders.solar_encoder import load_encoders_config, load_solar_encoder
from fusion.model import CrossDomainFusionModel
from fusion.unimodal import CapacityMatchedUnimodalModel, domain_share_of_trainable_params
from fusion_data import load_fusion_config
from uncertainty_analysis import block_indices, percentile_interval

ROOT = Path(__file__).resolve().parent
SWEEP = ROOT / "outputs" / "uncertainty_sweep_20261004T092140396722Z"
ABL = sorted((ROOT / "outputs").glob("ablation_run_20261004T09*"))
MSUPP = ROOT.parent / "martian/outputs/real/chapter6_supplementary_20261004T093226Z"
BOOT_SEED = 20260818
B = 1000


def count_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main() -> None:
    out = ROOT / "outputs" / ("fusion_code_audit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True)
    res = {}

    # ---------------- P1.7 ----------------
    enc_cfg, fus = load_encoders_config(), load_fusion_config()
    s_enc = load_solar_encoder(enc_cfg); m_enc, _ = load_martian_encoder(enc_cfg)
    a7 = CrossDomainFusionModel(solar_encoder=s_enc, martian_encoder=m_enc, shared_dim=fus["shared_embedding_dim"],
                                attention_hidden_size=fus["attention_scorer_hidden_size"])
    parts = {n: count_trainable(getattr(a7, n)) for n in ("projection_solar", "projection_martian", "scorer_solar_task",
                                                           "scorer_martian_task", "head_solar", "head_martian")}
    targets = {"solar": domain_share_of_trainable_params(projection=a7.projection_solar, scorer=a7.scorer_solar_task, head=a7.head_solar),
               "martian": domain_share_of_trainable_params(projection=a7.projection_martian, scorer=a7.scorer_martian_task, head=a7.head_martian)}
    a2 = {}
    for dom, enc in (("solar", s_enc), ("martian", m_enc)):
        m = CapacityMatchedUnimodalModel(enc, shared_dim=fus["shared_embedding_dim"], target_param_count=targets[dom])
        a2[dom] = {"target": targets[dom], "trainable_params": count_trainable(m), "encoder_hidden_size": enc.hidden_size,
                   "projection": f"Linear({enc.hidden_size}, {fus['shared_embedding_dim']}) = {count_trainable(m.projection)}",
                   "head": f"Linear({fus['shared_embedding_dim']}, {m.head_hidden_size}) -> Tanh -> Linear({m.head_hidden_size}, 1) = {count_trainable(m.head)}",
                   "head_hidden_size": m.head_hidden_size,
                   "frozen_encoder_params_excluded": sum(p.numel() for p in enc.parameters())}
    hist_counts = {d.name: {k: (v["target_param_count"], v["actual_param_count"]) for k, v in json.loads((d / "history_A2.json").read_text()).items()} for d in ABL}
    res["P1.7"] = {"A7_total_trainable": count_trainable(a7), "A7_parts": parts, "A2": a2, "history_A2_target_actual": hist_counts}

    # ---------------- P1.11 ----------------
    rows = []
    for d in ABL:
        seed = json.loads((d / "tss_summary.json").read_text())["seed"]
        for cfg in ("A3", "A4", "A5", "A6", "A7"):
            h = json.loads((d / f"history_{cfg}.json").read_text())
            for e in h["history"]:
                rows.append({"seed": seed, "config": cfg, "epoch": e["epoch"], "alignment_loss": e["alignment_loss"],
                             "total": e["total"], "loss_solar": e["loss_solar"], "loss_martian": e["loss_martian"],
                             "balance_loss": e["balance_loss"], "val_score": e["val_score"], "convergence_status": h["convergence_status"]})
    AL = pd.DataFrame(rows)
    AL.to_csv(out / "P1_11_alignment_loss_per_epoch.csv", index=False)
    pal = {"A5": "#2a78d6", "A6": "#eb6834", "A7": "#1baf7a", "A3": "#eda100", "A4": "#e87ba4"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True, facecolor="#fcfcfb")
    for ax, cfgs, ttl in zip(axes, (("A7",), ("A5",), ("A6",)), ("A7 full model (λ_align = 0.1)", "A5 alignment only (λ_align = 0.1)", "A6 no alignment (λ_align = 0, L_align logged only)")):
        ax.set_facecolor("#fcfcfb"); ax.grid(color="#e6e5e0", lw=0.6)
        for cfg in cfgs:
            for seed, g in AL[AL.config == cfg].groupby("seed"):
                ax.plot(g.epoch, g.alignment_loss, color=pal[cfg], lw=1.6, alpha=0.85, marker="o", ms=3)
                ax.annotate(str(seed), (g.epoch.iloc[-1], g.alignment_loss.iloc[-1]), fontsize=6, color="#52514e", xytext=(3, 0), textcoords="offset points")
        ax.set_title(ttl, fontsize=9); ax.set_xlabel("epoch"); ax.set_yscale("log")
    axes[0].set_ylabel("training L_align (epoch mean, log scale)")
    fig.suptitle("Alignment loss per epoch, every seed (44–48), canonical 2026-10-04 sweep; labels = seed", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "P1_11_alignment_loss.png", dpi=200, facecolor="#fcfcfb"); fig.savefig(out / "P1_11_alignment_loss.pdf"); plt.close(fig)

    # ---------------- P1.12 ----------------
    P = pd.read_csv(MSUPP / "predictions_test_and_my37_late.csv.gz")
    q = P[(P.eval_set == "test") & (P.model == "lstm_w3") & (P.horizon == 1)].reset_index(drop=True)
    a1 = pd.read_csv(SWEEP / "predictions_A1_martian_seed44.csv")
    assert (q.y.values == a1.y_true.values).all() and np.abs(q.p.values - a1.y_prob.values).max() < 1e-6
    cells = q.cell_id.to_numpy(); sols = (q.mars_year.astype(str) + "_" + q.sol.astype(str)).to_numpy()
    rep = (SWEEP / "uncertainty_report.md").read_text(encoding="utf-8").split("**martian**")[1]

    def group_idx(keys, rng):
        u, inv = np.unique(keys, return_inverse=True)
        members = [np.where(inv == k)[0] for k in range(len(u))]
        pick = rng.integers(0, len(u), len(u))
        return np.concatenate([members[k] for k in pick])

    def stats(y, p, t, idx):
        yy, pp = y[idx], p[idx]
        tp = np.sum((yy == 1) & (pp >= t)); fn = np.sum((yy == 1) & (pp < t)); fp = np.sum((yy == 0) & (pp >= t)); tn = np.sum((yy == 0) & (pp < t))
        tss = tp / (tp + fn) - fp / (fp + tn) if (tp + fn) and (fp + tn) else np.nan
        two = 0 < yy.sum() < len(yy)
        return tss, (roc_auc_score(yy, pp) if two else np.nan), (average_precision_score(yy, pp) if two else np.nan)

    rows = []
    for f in sorted(glob.glob(str(SWEEP / "predictions_A*_martian_seed44.csv"))):
        cfg = re.search(r"predictions_(A\d)_", os.path.basename(f)).group(1)
        d = pd.read_csv(f); y = d.y_true.to_numpy().astype(int); p = d.y_prob.to_numpy(); t = float(d.threshold.iloc[0])
        est = stats(y, p, t, np.arange(len(y)))
        dists = {}
        for scheme in ("one_cell_sol", "cell", "sol"):
            rng = np.random.default_rng(BOOT_SEED)
            dd = np.empty((B, 3))
            for b in range(B):
                idx = block_indices(len(y), 1, rng) if scheme == "one_cell_sol" else group_idx(cells if scheme == "cell" else sols, rng)
                dd[b] = stats(y, p, t, idx)
            dists[scheme] = dd
        # reproduction check against uncertainty_report.md (TSS, one-cell-sol blocks)
        m = re.search(rf"\| {cfg} \| 5 \| [\d.-]+ \| [\d.-]+ \| \[([\d.-]+), ([\d.-]+)\] \|", rep)
        lo, hi = percentile_interval(dists["one_cell_sol"][:, 0])
        assert m and abs(lo - float(m.group(1))) < 5e-5 and abs(hi - float(m.group(2))) < 5e-5, (cfg, lo, hi, m.groups() if m else None)
        for k, metric in enumerate(("TSS", "ROC-AUC", "PR-AUC")):
            r = {"configuration": cfg, "domain": "martian", "seed": 44, "metric": metric, "estimate": est[k], "threshold": t if metric == "TSS" else np.nan}
            for scheme in ("one_cell_sol", "cell", "sol"):
                r[f"ci95_lo_{scheme}"], r[f"ci95_hi_{scheme}"] = percentile_interval(dists[scheme][:, k])
            rows.append(r)
    IV = pd.DataFrame(rows)
    IV.to_csv(out / "P1_12_martian_intervals_by_block_unit.csv", index=False)
    res["P1.12"] = {"bootstrap_seed": BOOT_SEED, "resamples": B, "n_test_windows": int(len(q)), "n_cells": int(len(np.unique(cells))),
                    "n_sols": int(len(np.unique(sols))), "reproduced_report_one_cell_sol_TSS_intervals": True}
    (out / "audit_results.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print("done", out)


if __name__ == "__main__":
    main()
