"""Fixed split config shared by split/training mechanics tests (see conftest.py)."""

# Fixed split config for tests of split / training MECHANICS. These tests build
# synthetic data around these windows and must not depend on the live
# configs/splits.yaml, which is a real-data design decision (it changed on
# 2026-10-04 to train = MY36, val = early MY37, test = later MY37). The live
# config is validated separately by
# tests/test_chronological_split.py::test_val_split_boundary_is_non_empty_and_causally_after_train.
MECHANICS_SPLITS = {
    "train_mars_year": 36, "train_sol_range": [1, 431],
    "val_mars_year": 36, "val_sol_range": [432, 668],
    "test_mars_year": 37, "test_sol_range": [1, 501],
}


def mechanics_splits() -> dict:
    import copy
    return copy.deepcopy(MECHANICS_SPLITS)
