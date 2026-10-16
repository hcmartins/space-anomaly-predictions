"""Exports real martian sequence tensors (X, y per split) for Chapter 5's
cross-domain fusion training (src/cross-domain/run_fusion.py).

Reuses an existing `mars.run_all` run's persisted, real intermediate
artifacts -- never re-runs QC/gridding/aggregation/trailing-fill, only
reconstructs the one intermediate (the dense `features_df`) that
mars/run_all.py never persisted, then builds the windowed sequence tensors
that modeling/train.py::run_task builds in-memory and never persists.

    cd src/martian
    python -m export_fusion_sequences --run-dir outputs/real/<run_id>

Writes <run-dir>/fusion_export/{X,y}_{train,val,test}.npy and
regime_test.npy (real solar_longitude_ls per test-split row -- a simple,
non-fabricated seasonal grouping for cross-domain's post-hoc
negative-transfer reporting only; never fed into any training step).

NO DEFAULT --run-dir: this pipeline is real-data-only (data_sources/simulated_*
were removed) and no real `mars.run_all` run exists yet -- RealEMIRSObservationProvider
requires real downloaded EMIRS FITS files that have not been obtained (see
configs/real_emirs_ingestion.yaml). --run-dir must point at a real completed
run once one exists; there is no simulated run left to fall back to.

Also builds a val sequence set (sol 601-667 of MY36, per configs/splits.yaml)
in addition to train/test.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data_sources.real_mcd_provider import RealMCDClimatologyProvider
from labeling.anomaly_labels import compute_exceedance, load_anomaly_definition_config
from labeling.climatology_deviation import compute_z_scores
from modeling.datasets import TASK_TARGET_PREFIXES, build_storm_detection_dataset
from modeling.sequence_models import build_sequence_dataset
from modeling.train import _horizon_frame, _impute_sequence_train_only
from preprocessing.features import build_features, load_features_config
from preprocessing.grid import load_primary_grid
from splitting.chronological_split import assign_split, load_splits_config

MARTIAN_ROOT = Path(__file__).resolve().parent
WINDOW_LENGTH_SOLS = 3   # matches the real checkpoint's lstm_w3_h1.pt window
HORIZON_SOLS = 1         # matches the real checkpoint's h1 horizon


def export_fusion_sequences(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    out_dir = run_dir / "fusion_export"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reconstruct the never-persisted dense features_df, deterministically,
    # from the run's own already-persisted real trailing-filled data --
    # mirrors mars/run_all.py's steps 7 and 10 exactly.
    filled = pd.read_csv(run_dir / "data" / "trailing_filled_cell_sol.csv")
    grid = load_primary_grid()
    mcd = RealMCDClimatologyProvider()
    anomaly_config = load_anomaly_definition_config()
    filled_z = compute_z_scores(filled, mcd, epsilon=anomaly_config["epsilon"])
    filled_exceedance = compute_exceedance(filled_z, tau=anomaly_config["tau_primary"], config=anomaly_config)
    features_config = load_features_config()
    features_df = build_features(filled_exceedance, grid, config=features_config)

    labeled_df = pd.read_csv(run_dir / "labels" / "labeled_cell_sol.csv")
    dataset_df, feature_manifest = build_storm_detection_dataset(labeled_df, features_df, horizons_sols=(HORIZON_SOLS,))

    target_col = f"{TASK_TARGET_PREFIXES['storm_detection']}{HORIZON_SOLS}"
    frame = _horizon_frame(dataset_df, feature_manifest, target_col)
    splits_config = load_splits_config()
    frame = frame.assign(split=assign_split(frame, splits_config))

    ls_lookup = {
        (int(cid), int(my), int(sol)): ls
        for cid, my, sol, ls in frame[["cell_id", "mars_year", "sol", "solar_longitude_ls"]].itertuples(index=False, name=None)
    }

    raw_X: dict[str, np.ndarray] = {}
    raw_y: dict[str, np.ndarray] = {}
    meta_by_split: dict[str, pd.DataFrame] = {}
    counts: dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        split_frame = frame.loc[frame["split"] == split_name].reset_index(drop=True)
        X, y, meta = build_sequence_dataset(
            split_frame, features_df, feature_manifest,
            window_length_sols=WINDOW_LENGTH_SOLS, target_col=target_col,
        )
        raw_X[split_name], raw_y[split_name], meta_by_split[split_name] = X, y, meta
        counts[split_name] = len(y)

    # Train-only median imputation (real, tested logic), applied to val and
    # test using train-fit statistics -- same function called twice (each
    # call independently refits identical medians from the same raw
    # X_train, so the returned imputed X_train is identical both times).
    X_train_imp, X_val_imp = _impute_sequence_train_only(raw_X["train"], raw_X["val"])
    X_train_imp_check, X_test_imp = _impute_sequence_train_only(raw_X["train"], raw_X["test"])
    assert np.array_equal(X_train_imp, X_train_imp_check), "train-fit medians were not deterministic across calls"
    imputed_X = {"train": X_train_imp, "val": X_val_imp, "test": X_test_imp}

    for split_name in ("train", "val", "test"):
        np.save(out_dir / f"X_{split_name}.npy", imputed_X[split_name].astype(np.float32))
        np.save(out_dir / f"y_{split_name}.npy", raw_y[split_name].astype(np.float32))

    ls_test = np.array(
        [
            ls_lookup.get((int(cid), int(my), int(sol)), np.nan)
            for cid, my, sol in meta_by_split["test"][["cell_id", "mars_year", "sol"]].itertuples(index=False, name=None)
        ],
        dtype=np.float64,
    )
    # Quartile-bucketed real solar_longitude_ls -- a coarse, real "season"
    # proxy (4 groups) rather than raw continuous Ls, which would give one
    # near-singleton "regime" per test row and make negative-transfer-by-
    # regime analysis meaningless regardless of sample size.
    regime_test = pd.qcut(ls_test, q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop").astype(str)
    np.save(out_dir / "regime_test.npy", np.asarray(regime_test))

    print(f"Exported real martian fusion sequences to {out_dir}: {counts}")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export real martian sequence tensors for cross-domain fusion training.")
    parser.add_argument("--run-dir", required=True, help="A real, completed mars.run_all output directory (outputs/real/<run_id>).")
    args = parser.parse_args(argv)
    try:
        export_fusion_sequences(MARTIAN_ROOT / args.run_dir)
    except Exception as exc:  # noqa: BLE001 -- top-level CLI boundary
        print(f"EXPORT FAILED: {exc}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
