"""Follow-up analyses on the canonical Martian run (2026-10-04):

1. Continuation-versus-arrival analysis of the storm-detection target (the
   horizon target is MEMBERSHIP: a cell inside a continuing episode is
   positive). For every defined positive target, classify whether the cell
   was already inside a qualifying episode at the issue sol ("continuation")
   or not ("new arrival"); also count how many episode cell-sols are the
   episode's onset sol. Then re-score every frozen storm-detection model on the
   NON-EPISODE-ISSUE subset (cell not in an episode at the issue sol, so every
   positive is a new arrival) -- pooled, autumn and winter -- with cell-block
   bootstrap CIs and paired comparisons against persistence.
2. New-run checks: exact Ls limits of every split; defined targets and
   positives per split at h = 1, 3, 5 (tabular rows) and sequence-window
   counts; catalogue matching temporal offsets.

Inputs: the canonical run and the verified per-row test predictions written by
run_chapter6_supplementary.py (every model's test confusion matrix at its
recorded tau* was asserted equal to the saved evaluation JSON there).
Outputs: outputs/real/newrun_followups_<UTC>/
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from labeling.reporting import assign_mars_season
from splitting.chronological_split import assign_split, load_splits_config

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "outputs" / "real" / "run_20261004T080014968911Z_seed42"
SUPP = ROOT / "outputs" / "real" / "chapter6_supplementary_20261004T093226Z"
KEYS = ["cell_id", "mars_year", "sol"]
HORIZONS = (1, 3, 5)
SEED = 44
N_BOOT = 200
AUT, WIN = "Northern Autumn (Ls 180-270)", "Northern Winter (Ls 270-360)"


def counts(y, yh):
    y = y.astype(bool); yh = yh.astype(bool)
    return np.stack([(y & yh), (~y & yh), (~y & ~yh), (y & ~yh)], axis=1).astype(float)


def tss_c(c):
    tp, fp, tn, fn = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    with np.errstate(invalid="ignore", divide="ignore"):
        return tp / (tp + fn) - fp / (fp + tn)


def cell_boot(c_rows, cells, rng):
    uc, inv = np.unique(cells, return_inverse=True)
    per = np.stack([np.bincount(inv, weights=c_rows[:, k], minlength=len(uc)) for k in range(4)], axis=1)
    w = rng.multinomial(len(uc), np.full(len(uc), 1.0 / len(uc)), size=N_BOOT)
    return tss_c(w @ per), per, w


def main() -> None:
    t0 = time.perf_counter()
    out = RUN.parent / ("newrun_followups_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True)
    rng = np.random.default_rng(SEED)
    splits_cfg = load_splits_config()
    L = pd.read_csv(RUN / "labels" / "labeled_cell_sol.csv")
    L["split"] = assign_split(L, splits_cfg)
    L["season"] = assign_mars_season(L["solar_longitude_ls"])
    cat = pd.read_csv(RUN / "labels" / "episode_catalogue.csv")
    onset = cat.loc[cat["qualifies"]].set_index("episode_id")["onset_sol"]

    # ---------------- 1a. continuation vs arrival (label-level) ----------------
    Ls = L.sort_values(KEYS).reset_index(drop=True)
    rows = []
    for split in ("train", "val", "test", "ALL"):
        S = Ls if split == "ALL" else Ls[Ls.split == split]
        for season in ("ALL", AUT, WIN):
            T = S if season == "ALL" else S[S.season == season]
            for h in HORIZONS:
                d = T[T[f"horizon_{h}sol"].notna()]
                pos = d[d[f"horizon_{h}sol"] == 1]
                if len(d) == 0:
                    continue
                rows.append({"split": split, "season": season, "horizon_sols": h, "n_defined": len(d), "n_pos": len(pos),
                             "prevalence": len(pos) / len(d),
                             "n_continuation": int(pos.storm_episode.sum()), "n_new_arrival": int((~pos.storm_episode).sum()),
                             "continuation_share": float(pos.storm_episode.mean()) if len(pos) else np.nan})
    CA = pd.DataFrame(rows)
    CA.to_csv(out / "continuation_vs_arrival_targets.csv", index=False)
    ep = L[L.storm_episode].copy()
    ep["is_onset_sol"] = ep.episode_id.map(onset) == ep.sol
    ep["first_sol_in_cell"] = ep.groupby(["cell_id", "mars_year", "episode_id"]).sol.transform("min") == ep.sol
    ep_summary = {"n_episode_cell_sols": int(len(ep)), "share_episode_onset_sol": float(ep.is_onset_sol.mean()),
                  "share_first_arrival_in_cell": float(ep.first_sol_in_cell.mean())}

    # ---------------- 1b. frozen models on the onset-like (non-episode-issue) subset ----------------
    P = pd.read_csv(SUPP / "predictions_test_and_my37_late.csv.gz")
    P = P[P.eval_set == "test"].merge(L[KEYS + ["storm_episode"]], on=KEYS, how="left")
    assert P.storm_episode.notna().all()
    thr = {}
    for f in (RUN / "metrics" / "storm_detection").glob("*_evaluation.json"):
        m, h = f.name.replace("_evaluation.json", "").rsplit("_h", 1)
        thr[(m, int(h))] = float(json.loads(f.read_text())["decision_threshold"])
    sc_rows, pair_rows = [], []
    for h in HORIZONS:
        ref = P[(P.model == "persistence") & (P.horizon == h)].set_index(KEYS)
        for model in sorted(P.model.unique()):
            g = P[(P.model == model) & (P.horizon == h)]
            t = thr[(model, h)]
            for subset in ("all", "non_episode_issue"):
                gs = g if subset == "all" else g[~g.storm_episode.astype(bool)]
                for season in ("ALL", AUT, WIN):
                    gg = gs if season == "ALL" else gs[gs.season == season]
                    y = gg.y.to_numpy().astype(bool)
                    r = {"model": model, "horizon_sols": h, "threshold": t, "subset": subset, "season": season,
                         "n": len(gg), "n_pos": int(y.sum())}
                    if 0 < y.sum() < len(y):
                        c = counts(y, gg.p.to_numpy() >= t)
                        b, _, _ = cell_boot(c, gg.cell_id.to_numpy(), rng)
                        r.update(tss=float(tss_c(c.sum(0))), tss_ci95_lo=float(np.nanpercentile(b, 2.5)), tss_ci95_hi=float(np.nanpercentile(b, 97.5)))
                    sc_rows.append(r)
            if model == "persistence":
                continue
            j = ref.join(g.set_index(KEYS)[["p", "y"]], lsuffix="_a", rsuffix="_b", how="inner")
            j = j[~j.storm_episode.astype(bool)]
            for season in ("ALL", AUT, WIN):
                jj = j if season == "ALL" else j[j.season == season]
                y = jj.y_a.to_numpy().astype(bool)
                if not (0 < y.sum() < len(y)) or y.sum() < 5:
                    pair_rows.append({"horizon_sols": h, "model_b": model, "season": season, "n_paired": len(jj), "n_pos": int(y.sum()), "verdict": "too few positives"})
                    continue
                ca = counts(y, jj.p_a.to_numpy() >= thr[("persistence", h)]); cb = counts(y, jj.p_b.to_numpy() >= t)
                cells = jj.index.get_level_values("cell_id").to_numpy()
                uc, inv = np.unique(cells, return_inverse=True)
                pa = np.stack([np.bincount(inv, weights=ca[:, k], minlength=len(uc)) for k in range(4)], 1)
                pb = np.stack([np.bincount(inv, weights=cb[:, k], minlength=len(uc)) for k in range(4)], 1)
                w = rng.multinomial(len(uc), np.full(len(uc), 1.0 / len(uc)), size=N_BOOT)
                d = tss_c(w @ pb) - tss_c(w @ pa)
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                pair_rows.append({"horizon_sols": h, "model_b": model, "season": season, "n_paired": len(jj), "n_pos": int(y.sum()),
                                  "tss_persistence": float(tss_c(ca.sum(0))), "tss_b": float(tss_c(cb.sum(0))),
                                  "diff_b_minus_persistence": float(tss_c(cb.sum(0)) - tss_c(ca.sum(0))),
                                  "ci95_lo": float(lo), "ci95_hi": float(hi),
                                  "verdict": "B better" if lo > 0 else ("persistence better" if hi < 0 else "n.s.")})
    pd.DataFrame(sc_rows).to_csv(out / "onset_like_subset_scores.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(out / "onset_like_vs_persistence.csv", index=False)

    # ---------------- 2. new-run checks ----------------
    ds = pd.read_csv(RUN / "features" / "storm_detection_dataset.csv", usecols=KEYS + ["solar_longitude_ls"] + [f"target_storm_horizon_h{h}" for h in HORIZONS])
    ds["split"] = assign_split(ds, splits_cfg)
    ls_rows = []
    for split, g in ds.groupby("split"):
        ls_rows.append({"split": split, "mars_year": sorted(g.mars_year.unique().tolist()), "sol_min": int(g.sol.min()), "sol_max": int(g.sol.max()),
                        "ls_min": float(g.solar_longitude_ls.min()), "ls_max": float(g.solar_longitude_ls.max()), "n_rows": len(g),
                        **{f"h{h}_defined": int(g[f"target_storm_horizon_h{h}"].notna().sum()) for h in HORIZONS},
                        **{f"h{h}_positives": int(g[f"target_storm_horizon_h{h}"].sum()) for h in HORIZONS}})
    LSR = pd.DataFrame(ls_rows); LSR.to_csv(out / "split_ls_limits_and_counts.csv", index=False)
    seq_rows = []
    for f in sorted((RUN / "metrics" / "storm_detection").glob("*_provenance.json")):
        pr = json.loads(f.read_text())
        if pr.get("model_kind") != "sequence":
            continue
        ev = json.loads((RUN / "metrics" / "storm_detection" / f.name.replace("_provenance", "_evaluation")).read_text())["splits"]
        m = ev["test"]["metrics"]; v = ev["val"]["metrics"]
        seq_rows.append({"model": pr["model_name"], "horizon_sols": pr["horizon_sols"], "n_train_windows": pr["n_train"], "n_val_windows": pr["n_val"],
                         "n_test_windows": ev["test"]["n"], "val_positives": int(v["tp"] + v["fn"]), "test_positives": int(m["tp"] + m["fn"])})
    pd.DataFrame(seq_rows).to_csv(out / "sequence_window_counts.csv", index=False)
    mp = pd.read_csv(RUN / "labels" / "catalogue_matched_pairs.csv")
    rep = (RUN / "reports" / "final_report.md").read_text(encoding="utf-8")
    checks = {"episode_cell_sols": ep_summary,
              "catalogue_matched_pairs_design_tolerance": {"n_pairs": int(len(mp)), "n_events": int(mp.event_id.nunique()),
                                                            "n_episodes": int(mp.episode_id.nunique()),
                                                            "temporal_offset_sols": mp.temporal_offset_sols.tolist(),
                                                            "mean_offset_sols": float(mp.temporal_offset_sols.mean()),
                                                            "median_offset_sols": float(mp.temporal_offset_sols.median())},
              "final_report_offsets": [l.strip() for l in rep.splitlines() if "mean_temporal_offset_sols" in l],
              "runtime_seconds": time.perf_counter() - t0}
    (out / "checks.json").write_text(json.dumps(checks, indent=2, default=str), encoding="utf-8")
    print(f"done: {out}", flush=True)


if __name__ == "__main__":
    main()
