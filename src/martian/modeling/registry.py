"""Model factory that structurally prevents task cross-contamination: a
regressor can never be requested for a classification task and vice versa,
and a model family not registered for a task_type (e.g. KNN regression,
deliberately not registered -- see classical_models.py) is rejected rather
than silently substituted.
"""
from __future__ import annotations

from .classical_models import (
    make_gradient_boosting_classifier,
    make_gradient_boosting_regressor,
    make_knn_classifier,
    make_logistic_regression,
    make_random_forest_classifier,
    make_random_forest_regressor,
)

REGRESSOR_FACTORIES = {
    "random_forest_regressor": make_random_forest_regressor,
    "gradient_boosting_regressor": make_gradient_boosting_regressor,
}

CLASSIFIER_FACTORIES = {
    "random_forest_classifier": make_random_forest_classifier,
    "gradient_boosting_classifier": make_gradient_boosting_classifier,
    "logistic_regression": make_logistic_regression,
    "knn_classifier": make_knn_classifier,
}


class TaskModelMismatchError(ValueError):
    """Raised when a model name's namespace doesn't match the requested
    task_type -- e.g. requesting a classifier for a regression task, or a
    KNN regressor (not registered for any task_type)."""


def get_model(task_type: str, name: str, hyperparameters: dict, seed: int):
    if task_type == "regression":
        if name not in REGRESSOR_FACTORIES:
            raise TaskModelMismatchError(
                f"Model {name!r} is not a registered regressor and cannot be used "
                f"for a regression task. Registered regressors: {sorted(REGRESSOR_FACTORIES)}."
            )
        return REGRESSOR_FACTORIES[name](hyperparameters, seed)

    if task_type == "classification":
        if name not in CLASSIFIER_FACTORIES:
            raise TaskModelMismatchError(
                f"Model {name!r} is not a registered classifier and cannot be used "
                f"for a classification task. Registered classifiers: {sorted(CLASSIFIER_FACTORIES)}."
            )
        return CLASSIFIER_FACTORIES[name](hyperparameters, seed)

    raise ValueError(f"Unknown task_type {task_type!r}; expected 'regression' or 'classification'.")
