"""scikit-learn classifier wrappers. Hyperparameters and the random seed are
always read from the CALLING task's own config -- never a shared global
default -- so each task run is independently seeded and independently
tunable. Classification-only: this delivery has exactly one modeling task
(binary flare-occurrence forecasting), unlike martian/ which mixes
regression and classification tasks -- see `registry.py` for why that
means no regressor/classifier namespace split is needed here.1
"""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def make_random_forest_classifier(hyperparameters: dict, seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(random_state=seed, **hyperparameters)


def make_gradient_boosting_classifier(hyperparameters: dict, seed: int) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(random_state=seed, **hyperparameters)


def make_hist_gradient_boosting_classifier(hyperparameters: dict, seed: int) -> HistGradientBoostingClassifier:
    # Added for the per-HARP pipeline (run_pipeline_per_harp.py) specifically:
    # sklearn's GradientBoostingClassifier took 2947s (49min) on the
    # disk-integrated run's 237k training rows; the per-HARP dataset is
    # ~8.6x more rows, which would put plain GradientBoostingClassifier at
    # many hours. HistGradientBoostingClassifier (histogram-binned, same
    # boosted-tree family) is orders of magnitude faster at this scale for
    # the same task -- NOT used for the disk-integrated
    # task_flare_forecast.yaml run, whose real, already-reported numbers
    # must not change algorithm out from under them.
    return HistGradientBoostingClassifier(random_state=seed, **hyperparameters)


def make_logistic_regression(hyperparameters: dict, seed: int) -> LogisticRegression:
    return LogisticRegression(random_state=seed, **hyperparameters)


def make_knn_classifier(hyperparameters: dict, seed: int) -> KNeighborsClassifier:
    # KNeighborsClassifier has no random_state parameter (deterministic given
    # its data and hyperparameters) -- seed is accepted for interface
    # uniformity with the other factories but intentionally unused.
    del seed
    return KNeighborsClassifier(**hyperparameters)
