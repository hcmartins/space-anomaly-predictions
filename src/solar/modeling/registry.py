"""Classifier factory. Unlike `martian/modeling/registry.py`, there is no
task_type parameter or regressor namespace: this delivery has exactly one
modeling task (binary flare-occurrence forecasting -- see
`configs/flare_forecast_task.yaml`), so every registered model is a
classifier. If a regression task is added later (e.g. forecasting peak
flux magnitude directly, rather than class-threshold occurrence), split
this into REGRESSOR_FACTORIES/CLASSIFIER_FACTORIES with a `get_model(task_type,
name, ...)` signature then, mirroring martian's structure -- not
speculatively built now for a task that does not exist yet..
"""
from __future__ import annotations

from .classical_models import (
    make_gradient_boosting_classifier,
    make_hist_gradient_boosting_classifier,
    make_knn_classifier,
    make_logistic_regression,
    make_random_forest_classifier,
)

CLASSIFIER_FACTORIES = {
    "random_forest_classifier": make_random_forest_classifier,
    "gradient_boosting_classifier": make_gradient_boosting_classifier,
    "hist_gradient_boosting_classifier": make_hist_gradient_boosting_classifier,
    "logistic_regression": make_logistic_regression,
    "knn_classifier": make_knn_classifier,
}


class UnknownModelError(ValueError):
    """Raised when a model name is not registered -- never silently
    substitutes a different model family."""


def get_model(name: str, hyperparameters: dict, seed: int):
    if name not in CLASSIFIER_FACTORIES:
        raise UnknownModelError(
            f"Model {name!r} is not a registered classifier. Registered classifiers: "
            f"{sorted(CLASSIFIER_FACTORIES)}."
        )
    return CLASSIFIER_FACTORIES[name](hyperparameters, seed)
