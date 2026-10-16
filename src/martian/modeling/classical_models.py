"""scikit-learn model wrappers. Hyperparameters and the random seed are
always read from the CALLING task's own config -- never a shared global
default -- so each task's model is independently seeded and independently
tunable.

K-nearest neighbours is registered for CLASSIFICATION ONLY ("where
appropriate" -- the original plan names `KNeighborsClassifier` but not a
KNN regressor; a KNN regressor is not registered here, so requesting one
raises `TaskModelMismatchError` via the registry rather than silently
substituting a different model family)."""
from __future__ import annotations

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def make_random_forest_regressor(hyperparameters: dict, seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(random_state=seed, **hyperparameters)


def make_random_forest_classifier(hyperparameters: dict, seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(random_state=seed, **hyperparameters)


def make_gradient_boosting_regressor(hyperparameters: dict, seed: int) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(random_state=seed, **hyperparameters)


def make_gradient_boosting_classifier(hyperparameters: dict, seed: int) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(random_state=seed, **hyperparameters)


def make_logistic_regression(hyperparameters: dict, seed: int) -> LogisticRegression:
    return LogisticRegression(random_state=seed, **hyperparameters)


def make_knn_classifier(hyperparameters: dict, seed: int) -> KNeighborsClassifier:
    # KNeighborsClassifier has no random_state parameter (deterministic given
    # its data and hyperparameters) -- seed is accepted for interface
    # uniformity with the other factories but intentionally unused.
    del seed
    return KNeighborsClassifier(**hyperparameters)
