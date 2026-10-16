"""Tests for the model registry's task-namespace enforcement."""
import pytest
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from martian.modeling.registry import TaskModelMismatchError, get_model


def test_get_model_regressor_for_regression_task():
    model = get_model("regression", "random_forest_regressor", {"n_estimators": 5}, seed=1)
    assert isinstance(model, RandomForestRegressor)
    assert model.random_state == 1


def test_get_model_classifier_for_classification_task():
    model = get_model("classification", "random_forest_classifier", {"n_estimators": 5}, seed=2)
    assert isinstance(model, RandomForestClassifier)
    assert model.random_state == 2


def test_get_model_rejects_classifier_for_regression_task():
    with pytest.raises(TaskModelMismatchError):
        get_model("regression", "random_forest_classifier", {}, seed=1)


def test_get_model_rejects_regressor_for_classification_task():
    with pytest.raises(TaskModelMismatchError):
        get_model("classification", "random_forest_regressor", {}, seed=1)


def test_get_model_unknown_task_type_raises():
    with pytest.raises(ValueError):
        get_model("clustering", "random_forest_regressor", {}, seed=1)


def test_get_model_unknown_name_raises():
    with pytest.raises(TaskModelMismatchError):
        get_model("regression", "not_a_real_model", {}, seed=1)


def test_get_model_gradient_boosting_regressor():
    model = get_model("regression", "gradient_boosting_regressor", {"n_estimators": 5}, seed=3)
    assert isinstance(model, GradientBoostingRegressor)
    assert model.random_state == 3


def test_get_model_gradient_boosting_classifier():
    model = get_model("classification", "gradient_boosting_classifier", {"n_estimators": 5}, seed=4)
    assert isinstance(model, GradientBoostingClassifier)
    assert model.random_state == 4


def test_get_model_logistic_regression_classification_only():
    model = get_model("classification", "logistic_regression", {"max_iter": 100}, seed=5)
    assert isinstance(model, LogisticRegression)
    with pytest.raises(TaskModelMismatchError):
        get_model("regression", "logistic_regression", {}, seed=5)


def test_get_model_knn_classifier_registered_classification_only():
    model = get_model("classification", "knn_classifier", {"n_neighbors": 3}, seed=6)
    assert isinstance(model, KNeighborsClassifier)


def test_get_model_knn_regressor_not_registered_anywhere():
    # KNN is registered "where appropriate" -- classification only
    # (KNeighborsClassifier, no regressor counterpart). Requesting it for
    # regression must fail rather than silently substitute a different
    # model family.
    with pytest.raises(TaskModelMismatchError):
        get_model("regression", "knn_classifier", {}, seed=6)
    with pytest.raises(TaskModelMismatchError):
        get_model("regression", "knn_regressor", {}, seed=6)
