"""Step 12: runtime leakage checks against THIS run's actual data --
distinct from (and complementary to) the static pytest leakage suite
(tests/test_datasets_no_leakage.py, tests/test_chronological_split.py),
which proves the MECHANISM is correct in isolation on fixtures. This module
re-checks the same invariants against the real, freshly-built datasets a
run just produced, so a run never silently ships leaked artifacts even if
some future code change broke an invariant the static fixtures didn't
happen to exercise.
"""
from __future__ import annotations

import pandas as pd

from modeling.datasets import TASK_TARGET_PREFIXES
from splitting.chronological_split import (
    EpisodeCrossesSplitBoundaryError,
    assert_no_episode_crosses_split_boundary,
)


class LeakageCheckFailed(RuntimeError):
    """Raised when a runtime leakage check fails against this run's actual
    data -- the run must stop, never ship a leaked artifact."""


def run_leakage_checks(
    *,
    episode_catalogue: pd.DataFrame,
    splits_config: dict,
    task_datasets: dict,  # task_name -> (dataset_df, feature_manifest)
) -> dict:
    """Returns a summary dict on success; raises LeakageCheckFailed on any
    violation (collecting every problem found, not just the first)."""
    problems: list[str] = []

    try:
        assert_no_episode_crosses_split_boundary(episode_catalogue, splits_config)
        episode_boundary_ok = True
    except EpisodeCrossesSplitBoundaryError as exc:
        episode_boundary_ok = False
        problems.append(f"Episode(s) cross a split boundary: {exc}")

    all_target_prefixes = list(TASK_TARGET_PREFIXES.values())
    feature_manifest_ok = True
    for task_name, (dataset_df, feature_manifest) in task_datasets.items():
        for col in feature_manifest:
            if any(col.startswith(prefix) for prefix in all_target_prefixes):
                feature_manifest_ok = False
                problems.append(f"Task {task_name!r}'s feature manifest contains a target column: {col!r}")

        own_targets = {c for c in dataset_df.columns if c.startswith("target_")}
        for other_name, (_, other_manifest) in task_datasets.items():
            if other_name == task_name:
                continue
            leaked = own_targets & set(other_manifest)
            if leaked:
                feature_manifest_ok = False
                problems.append(
                    f"Task {task_name!r}'s target column(s) {sorted(leaked)} appear in "
                    f"task {other_name!r}'s feature manifest"
                )

    if problems:
        raise LeakageCheckFailed(
            f"{len(problems)} leakage check(s) failed:\n- " + "\n- ".join(problems)
        )

    return {
        "episode_boundary_check": "passed" if episode_boundary_ok else "failed",
        "feature_manifest_target_leakage_check": "passed" if feature_manifest_ok else "failed",
        "tasks_checked": list(task_datasets.keys()),
    }
