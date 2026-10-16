"""Input preparation for src/extract_worked_examples.py (Worked Prediction Examples, v2).

extract_worked_examples.py reads one Martian per-row table that must carry a per-row
decision threshold and the onset-like flag. The canonical prediction file
(chapter6_supplementary_20261004T093226Z/predictions_test_and_my37_late.csv.gz) has neither:
  - thresholds are stored per model x horizon in the canonical run's
    metrics/storm_detection/<model>_h<h>_evaluation.json ("decision_threshold");
  - the onset-like flag is computed in memory by run_newrun_followups.py, never saved:
        P = P[P.eval_set == "test"].merge(L[KEYS + ["storm_episode"]], on=KEYS, how="left")
        ... g[~g.storm_episode.astype(bool)]          # "non_episode_issue" subset
This script adds exactly those two columns, by the same join, and changes nothing else:
the prediction rows, their order and every original column are asserted unchanged.

    python src/martian/prepare_worked_example_inputs.py
writes src/martian/outputs/real/worked_examples_inputs_<UTC>/mars_test_predictions_with_threshold_onset.csv.gz
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "outputs" / "real" / "run_20261004T080014968911Z_seed42"
SUPP = ROOT / "outputs" / "real" / "chapter6_supplementary_20261004T093226Z" / "predictions_test_and_my37_late.csv.gz"
KEYS = ["cell_id", "mars_year", "sol"]          # run_newrun_followups.py's KEYS


def main() -> None:
    out = ROOT / "outputs" / "real" / ("worked_examples_inputs_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True)
    P0 = pd.read_csv(SUPP)
    P = P0[P0.eval_set == "test"].reset_index(drop=True)
    L = pd.read_csv(RUN / "labels" / "labeled_cell_sol.csv", usecols=KEYS + ["storm_episode"])
    Q = P.merge(L, on=KEYS, how="left", validate="many_to_one")
    assert Q.storm_episode.notna().all()
    assert Q[P.columns].equals(P), "join altered the prediction rows"
    thr = {}
    for f in (RUN / "metrics" / "storm_detection").glob("*_evaluation.json"):
        m, h = f.name.replace("_evaluation.json", "").rsplit("_h", 1)
        thr[(m, int(h))] = float(json.loads(f.read_text())["decision_threshold"])
    Q["threshold"] = [thr[(m, int(h))] for m, h in zip(Q.model, Q.horizon)]
    Q["onset_like"] = ~Q.storm_episode.astype(bool)
    Q = Q.drop(columns="storm_episode")
    Q.to_csv(out / "mars_test_predictions_with_threshold_onset.csv.gz", index=False)
    print(f"{len(Q)} rows; columns {list(Q.columns)}\nwritten to {out}")


if __name__ == "__main__":
    main()
