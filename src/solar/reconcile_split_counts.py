"""Reconciles the disk-level solar partition counts quoted in the thesis:
Table 3.2 (split sizes: train 237,879 / test 211,671) against the counts the
trained classical models actually used (train 237,012 / test 211,594, as in
run_summary.json's task_results and Chapter 6 Table 6.2).

For each split it counts the rows whose primary label is UNDEFINED (NaN),
which modeling/train.py::_prepare_frame drops before fitting/scoring, and
checks that this alone accounts for the whole difference. It also reports the
test-split class counts for the trained target (>=M within 24 h) and by
maximum flare class in the horizon.

    cd src/solar
    python reconcile_split_counts.py

Writes results/real_full_20100501_20250724/split_count_reconciliation/*.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

SOLAR_ROOT = Path(__file__).resolve().parent
RUN = SOLAR_ROOT / "results" / "real_full_20100501_20250724"
TARGET = "label_primary"


def main() -> None:
    out = RUN / "split_count_reconciliation"
    out.mkdir(exist_ok=True)
    summary = json.loads((RUN / "run_summary.json").read_text())
    min_cov = 1.0 - yaml.safe_load((SOLAR_ROOT / "configs" / "labeling.yaml").read_text())["max_allowed_missing_coverage_fraction"]
    cols = ["time_utc", "split", TARGET, "label_flare_c_or_above", "label_flare_m_or_above", "label_flare_x_or_above",
            "label_max_flare_class_in_horizon", "label_coverage_fraction", "is_observed"]
    d = pd.read_csv(RUN / "dataset_with_split.csv", usecols=cols)
    d["time_utc"] = pd.to_datetime(d.time_utc, utc=True)
    assert (d[TARGET].fillna(-1) == d.label_flare_m_or_above.fillna(-1)).all(), "label_primary is not >=M"

    # 1. split sizes vs trained counts
    trained = summary["task_results"]["targets"][TARGET]
    rows = []
    for split, n_key in (("train", "n_train"), ("val", "n_val"), ("test", "n_test")):
        s = d[d.split == split]
        undefined = s[TARGET].isna()
        assert len(s) == summary["split_sizes"][split]
        assert len(s) - undefined.sum() == trained[n_key], f"{split}: undefined labels do not explain the gap"
        # every undefined row is exactly the coverage gate, nothing else
        assert ((s.label_coverage_fraction < min_cov) == undefined).all()
        rows.append({"split": split, "split_size_table_3_2": len(s), "undefined_label": int(undefined.sum()),
                     "defined_label_trained_or_scored": int((~undefined).sum()), "run_summary_n": trained[n_key],
                     "n_ge_m": int(s.label_flare_m_or_above.sum()),
                     "prev_ge_m_over_split_size": s.label_flare_m_or_above.sum() / len(s),
                     "prev_ge_m_over_defined": s.label_flare_m_or_above.sum() / (~undefined).sum(),
                     "undefined_coverage_max": float(s.loc[undefined, "label_coverage_fraction"].max()) if undefined.any() else None,
                     "defined_coverage_min": float(s.loc[~undefined, "label_coverage_fraction"].min()),
                     "undefined_inputs_observed": int(s.loc[undefined, "is_observed"].sum())})
    pd.DataFrame(rows).to_csv(out / "split_vs_trained_counts.csv", index=False)

    # 2. where the undefined rows are (date clusters)
    u = d[d[TARGET].isna()]
    (u.groupby(["split", u.time_utc.dt.date]).agg(n=("time_utc", "size"), first=("time_utc", "min"), last=("time_utc", "max"),
                                                  coverage_min=("label_coverage_fraction", "min"), coverage_max=("label_coverage_fraction", "max"))
      .reset_index().rename(columns={"time_utc": "date"}).to_csv(out / "undefined_label_rows_by_date.csv", index=False))

    # 3. test class counts (scored rows)
    t = d[(d.split == "test") & d[TARGET].notna()]
    binary = pd.DataFrame([{"target": f">={c}", "positive": int(t[col].sum()), "negative": int((t[col] == 0).sum()), "n": len(t)}
                           for c, col in (("C", "label_flare_c_or_above"), ("M", "label_flare_m_or_above"), ("X", "label_flare_x_or_above"))])
    binary.to_csv(out / "test_binary_class_counts.csv", index=False)
    maxcls = t.label_max_flare_class_in_horizon.fillna("below B").value_counts().reindex(["below B", "B", "C", "M", "X"]).fillna(0).astype(int)
    maxcls.rename_axis("max_class_in_24h").reset_index(name="n").to_csv(out / "test_max_class_counts.csv", index=False)

    print(pd.DataFrame(rows).to_string(index=False)); print(binary.to_string(index=False)); print(maxcls.to_string())
    print(f"written to {out}")


if __name__ == "__main__":
    main()
