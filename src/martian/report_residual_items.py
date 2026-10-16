"""Residual Martian items for the canonical run (run_20261004T080014968911Z_seed42):

  (a) rows affected by trailing-fill imputation, per partition
  (b) climatological z statistics per partition
  (c) candidate anomalous cell-sols failing the persistence rule, the extent
      rule, or both

Reads only the run's saved tables; nothing is recomputed from raw data, and
no model is retrained. Partitions follow labeling/reporting.py::assign_split
(configs/splits.yaml); "unassigned" = the MY37 sol 381-385 embargo.

Before counting, the script rebuilds the saved features it reasons about and
asserts they match the run's own feature table:
  - feat_dust_optical_depth_lag1..3 equal the filled series at s-1..s-3
  - feat_persistence_run_sols equals the run length recomputed with imputed
    sols treated as non-anomalous (their z is NaN: latitude/longitude are not
    carried forward, so the climatology query returns NaN)
  - the h=1/3/5 LSTM window counts (w = 3, 5, 10) reproduce the canonical
    test-set sizes in the chapter-6 supplementary predictions file

    cd src/martian
    python report_residual_items.py

Writes outputs/real/<run>/residual_items/*.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from labeling.reporting import assign_split
from preprocessing.grid import bands_from_cell_id, cell_id as grid_cell_id, load_primary_grid, neighbours

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "outputs" / "real" / "run_20261004T080014968911Z_seed42"
SUPP = ROOT / "outputs" / "real" / "chapter6_supplementary_20261004T093226Z" / "predictions_test_and_my37_late.csv.gz"
PARTS = ["train", "val", "unassigned", "test"]
QUANTILES = [0.01, 0.05, 0.50, 0.95, 0.99]


def main() -> None:
    out = RUN / "residual_items"
    out.mkdir(exist_ok=True)
    anomaly_cfg = yaml.safe_load((ROOT / "configs" / "anomaly_definition.yaml").read_text())
    feat_cfg = yaml.safe_load((ROOT / "configs" / "features.yaml").read_text())
    window, lags, min_compl = feat_cfg["trailing_window_sols"], feat_cfg["lags_sols"], feat_cfg["min_window_completeness_frac"]
    grid = load_primary_grid()

    filled = pd.read_csv(RUN / "data" / "trailing_filled_cell_sol.csv")
    labeled = pd.read_csv(RUN / "labels" / "labeled_cell_sol.csv")
    cat = pd.read_csv(RUN / "labels" / "episode_catalogue.csv")
    feats = pd.read_csv(RUN / "features" / "storm_detection_dataset.csv",
                        usecols=["cell_id", "mars_year", "sol", "feat_dust_optical_depth_lag1", "feat_dust_optical_depth_lag2",
                                 "feat_dust_optical_depth_lag3", "feat_persistence_run_sols", "feat_dust_optical_depth_neighbour_mean",
                                 "target_storm_horizon_h1", "target_storm_horizon_h3", "target_storm_horizon_h5"])
    filled["part"] = assign_split(filled).to_numpy()
    labeled["part"] = assign_split(labeled).to_numpy()
    checks = []

    # ---------------------------------------------------------------- (a)
    imp = filled.interpolation_flag.astype(bool).to_numpy()
    assert (imp == ~filled.is_observed.astype(bool).to_numpy()).all()
    assert int((~imp).sum()) == len(labeled)
    key = {(int(c), int(m), int(s)): i for i, (c, m, s) in enumerate(filled[["cell_id", "mars_year", "sol"]].itertuples(index=False))}
    dust = filled.dust_optical_depth.to_numpy(float)
    # anomaly flag on the filled series as step 7 produces it: observed rows carry their labelled z;
    # imputed rows have z = NaN, so z > tau is False
    z_lab = labeled.set_index(["cell_id", "mars_year", "sol"]).dust_optical_depth_z
    anom = np.zeros(len(filled), bool)
    obs_idx = np.flatnonzero(~imp)
    zk = filled.loc[obs_idx, ["cell_id", "mars_year", "sol"]].apply(tuple, axis=1)
    anom[obs_idx] = (z_lab.reindex(pd.MultiIndex.from_tuples(zk)).to_numpy() > anomaly_cfg["tau_primary"])

    lat_b, lon_b = bands_from_cell_id(filled.cell_id.to_numpy(), grid)
    base = filled.loc[~imp, ["cell_id", "mars_year", "sol", "part"]].reset_index()
    rec = {k: [] for k in ["lag_imp", "win_imp", "trail_enabled_by_imp", "nbr_imp", "pers_cut_by_imp", "lag1", "lag2", "lag3", "run", "nbr_mean"]}
    for i, c, m, s in base[["index", "cell_id", "mars_year", "sol"]].itertuples(index=False):
        hist = [key.get((c, m, s - k)) for k in range(1, window + 1)]
        exists = np.array([h is not None for h in hist])
        himp = np.array([h is not None and imp[h] for h in hist])
        rec["lag_imp"].append(bool(himp[[l - 1 for l in lags]].any()))
        rec["win_imp"].append(bool(himp.any()))
        n_valid, n_obs = exists.sum(), (exists & ~himp).sum()
        rec["trail_enabled_by_imp"].append(bool(n_valid / window >= min_compl and not n_obs / window >= min_compl))
        for l in lags:
            rec[f"lag{l}"].append(dust[hist[l - 1]] if hist[l - 1] is not None else np.nan)
        nb = [key.get((int(grid_cell_id(a, b, grid)), m, s)) for a, b in neighbours(int(lat_b[i]), int(lon_b[i]), grid)]
        nb = [n for n in nb if n is not None]
        rec["nbr_imp"].append(any(imp[n] for n in nb))
        rec["nbr_mean"].append(float(np.mean(dust[nb])) if nb else np.nan)
        run, k = 0, 1
        while True:                      # features.py persistence counter, replayed
            p = key.get((c, m, s - k))
            if p is None:
                cut = False
                break
            if anom[p]:
                run += 1; k += 1
            else:
                cut = bool(imp[p])       # the run stopped on an imputed sol
                break
        rec["run"].append(run); rec["pers_cut_by_imp"].append(cut)
    for k, v in rec.items():
        base[k] = v

    chk = base.merge(feats, on=["cell_id", "mars_year", "sol"], how="left", validate="1:1")
    for l in lags:
        a, b = chk[f"lag{l}"].to_numpy(), chk[f"feat_dust_optical_depth_lag{l}"].to_numpy()
        assert np.allclose(a, b, equal_nan=True), f"lag{l} replay mismatch"
    assert (chk.run == chk.feat_persistence_run_sols).all(), "persistence replay mismatch"
    assert np.allclose(chk.nbr_mean, chk.feat_dust_optical_depth_neighbour_mean, equal_nan=True), "neighbour replay mismatch"
    checks.append("dust lag1-3, dust neighbour mean and feat_persistence_run_sols replayed exactly for all 142,608 observed rows "
                  "(imputed sols treated as non-anomalous)")

    # sequence windows containing >= 1 imputed step, per h-defined anchor set
    supp = pd.read_csv(SUPP, usecols=["eval_set", "model", "horizon"])
    seq_rows = []
    for w in (3, 5, 10):
        steps = [[key.get((c, m, s - k)) for k in range(w)] for c, m, s in chk[["cell_id", "mars_year", "sol"]].itertuples(index=False)]
        feasible = np.array([all(x is not None for x in st) for st in steps])
        has_imp = np.array([f and any(imp[x] for x in st) for f, st in zip(feasible, steps)])
        for h in (1, 3, 5):
            defined = chk[f"target_storm_horizon_h{h}"].notna().to_numpy()
            n_canon = int(((supp.eval_set == "test") & (supp.model == f"lstm_w{w}") & (supp.horizon == h)).sum())
            assert int((feasible & defined & (chk.part == "test")).sum()) == n_canon, f"w{w} h{h} window count mismatch"
            for p in PARTS:
                mk = (chk.part == p).to_numpy() & defined & feasible
                seq_rows.append({"window_sols": w, "horizon_sols": h, "partition": p, "n_windows": int(mk.sum()),
                                 "n_windows_with_imputed_step": int((mk & has_imp).sum())})
    checks.append("sequence window counts (w = 3/5/10 x h = 1/3/5) reproduce the canonical LSTM test-set sizes")
    pd.DataFrame(seq_rows).to_csv(out / "a_sequence_windows_with_imputed_steps.csv", index=False)

    src_part = np.full(len(filled), None, dtype=object)
    for j in np.flatnonzero(imp):
        p = key.get((int(filled.cell_id[j]), int(filled.mars_year[j]), int(filled.sol[j]) - 1))
        src_part[j] = filled.part[p]
    a_rows = []
    for p in PARTS:
        fp = (filled.part == p).to_numpy()
        b = base[base.part == p]
        a_rows.append({"partition": p, "observed_rows": int((fp & ~imp).sum()), "imputed_rows_created": int((fp & imp).sum()),
                       "imputed_from_other_partition": int(sum(1 for j in np.flatnonzero(fp & imp) if src_part[j] != p)),
                       "obs_rows_dust_lag1to3_from_imputed": int(b.lag_imp.sum()),
                       "obs_rows_trailing5_window_contains_imputed": int(b.win_imp.sum()),
                       "obs_rows_trailing_stats_exist_only_because_of_imputation": int(b.trail_enabled_by_imp.sum()),
                       "obs_rows_neighbour_stats_include_imputed": int(b.nbr_imp.sum()),
                       "obs_rows_persistence_run_stopped_by_imputed": int(b.pers_cut_by_imp.sum()),
                       "obs_rows_any_feature_touched": int((b.win_imp | b.nbr_imp | b.pers_cut_by_imp).sum())})
    a = pd.DataFrame(a_rows)
    a.loc[len(a)] = ["ALL"] + a.drop(columns="partition").sum().tolist()
    a.to_csv(out / "a_trailing_fill_per_partition.csv", index=False)

    # ---------------------------------------------------------------- (b)
    b_rows = []
    for p in PARTS + ["ALL"]:
        z = labeled.dust_optical_depth_z if p == "ALL" else labeled.loc[labeled.part == p, "dust_optical_depth_z"]
        q = z.quantile(QUANTILES)
        b_rows.append({"partition": p, "n": int(z.notna().sum()), "n_nan": int(z.isna().sum()), "mean": z.mean(), "sd": z.std(ddof=1),
                       **{f"q{int(round(k * 100)):02d}": q[k] for k in QUANTILES},
                       "frac_gt_tau": float((z > anomaly_cfg["tau_primary"]).mean())})
    pd.DataFrame(b_rows).to_csv(out / "b_z_statistics_per_partition.csv", index=False)

    # ---------------------------------------------------------------- (c)
    P, A = anomaly_cfg["persistence_sols_primary"], anomaly_cfg["min_area_cells_primary"]
    cat["persistence_ok"] = cat.duration_sols >= P
    cat["extent_ok"] = cat.peak_area_cells >= A
    assert (cat.qualifies == (cat.persistence_ok & cat.extent_ok)).all()
    cat["outcome"] = np.select([cat.persistence_ok & cat.extent_ok, ~cat.persistence_ok & cat.extent_ok, cat.persistence_ok & ~cat.extent_ok],
                               ["qualifies", "fails persistence only", "fails extent only"], "fails both")
    cand = labeled[labeled.atmospheric_anomaly.astype(bool)].merge(cat[["episode_id", "outcome", "duration_sols", "peak_area_cells"]],
                                                                    on="episode_id", how="left", validate="m:1")
    assert cand.outcome.notna().all() and len(cand) == int(labeled.atmospheric_anomaly.sum())
    assert int((cand.outcome == "qualifies").sum()) == int(labeled.storm_episode.sum())
    ep_part = labeled.dropna(subset=["episode_id"]).groupby("episode_id").part.agg(lambda s: s.unique().tolist())
    assert ep_part.map(len).max() == 1, "an episode spans two partitions"
    cat["partition"] = cat.episode_id.map(ep_part.map(lambda x: x[0]))
    order = ["qualifies", "fails persistence only", "fails extent only", "fails both"]
    c_cs = pd.crosstab(cand.part, cand.outcome).reindex(index=PARTS, columns=order, fill_value=0)
    c_cs.loc["ALL"] = c_cs.sum()
    c_ep = pd.crosstab(cat.partition, cat.outcome).reindex(index=PARTS, columns=order, fill_value=0)
    c_ep.loc["ALL"] = c_ep.sum()
    c_cs.to_csv(out / "c_candidate_cell_sols_by_rule.csv")
    c_ep.to_csv(out / "c_candidate_episodes_by_rule.csv")
    checks.append(f"episode 'qualifies' == (duration >= {P}) & (peak_area >= {A}) for all {len(cat)} candidates; "
                  f"{len(cand)} candidate cell-sols; qualifying cell-sols == storm_episode count; no episode spans two partitions")

    (out / "checks.txt").write_text("\n".join(checks) + "\n")
    print(a.to_string(index=False)); print(pd.DataFrame(seq_rows).query("partition=='test'").to_string(index=False))
    print(pd.DataFrame(b_rows).to_string(index=False)); print(c_cs); print(c_ep); print("\n".join(checks))


if __name__ == "__main__":
    main()
