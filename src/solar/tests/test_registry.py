"""Tests for the classifier factory registry."""
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from modeling.registry import CLASSIFIER_FACTORIES, UnknownModelError, get_model


def test_get_model_returns_correctly_seeded_classifier():
    model = get_model("logistic_regression", {"max_iter": 200}, seed=7)
    assert isinstance(model, LogisticRegression)
    assert model.random_state == 7
    assert model.max_iter == 200


def test_hist_gradient_boosting_classifier_is_registered_and_seeded():
    # Added for run_pipeline_per_harp.py's scale-driven substitution for
    # gradient_boosting_classifier -- see modeling/classical_models.py's
    # make_hist_gradient_boosting_classifier docstring.
    model = get_model("hist_gradient_boosting_classifier", {"max_iter": 50}, seed=3)
    assert isinstance(model, HistGradientBoostingClassifier)
    assert model.random_state == 3
    assert model.max_iter == 50


def test_get_model_unknown_name_raises():
    with pytest.raises(UnknownModelError):
        get_model("not_a_model", {}, seed=1)


def test_all_registered_classifiers_are_constructible():
    for name in CLASSIFIER_FACTORIES:
        model = get_model(name, {}, seed=1)
        assert hasattr(model, "fit") and hasattr(model, "predict_proba")


def test_knn_classifier_ignores_seed_without_error():
    model = get_model("knn_classifier", {"n_neighbors": 3}, seed=99)
    assert model.n_neighbors == 3
