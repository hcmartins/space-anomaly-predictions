"""Tests for classical-model coefficient/feature-importance interpretability
(Chapter 3.8's documented requirement)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from modeling.interpretability import extract_interpretability


def test_linear_model_reports_coefficients_ranked_by_magnitude():
    X = np.array([[0.0, 5.0], [1.0, 5.0], [0.0, 5.1], [1.0, 4.9]])
    y = np.array([0, 1, 0, 1])
    model = LogisticRegression().fit(X, y)
    result = extract_interpretability(model, ["feat_a", "feat_b"])
    assert result["type"] == "coefficients"
    features_in_order = [r["feature"] for r in result["ranked_features"]]
    # feat_a is the one driving the label in this constructed example, so
    # its |coefficient| must rank first.
    assert features_in_order[0] == "feat_a"


def test_tree_model_reports_feature_importances():
    X = np.array([[0.0, 5.0], [1.0, 5.0], [0.0, 5.1], [1.0, 4.9]] * 5)
    y = np.array([0, 1, 0, 1] * 5)
    model = DecisionTreeClassifier(random_state=0).fit(X, y)
    result = extract_interpretability(model, ["feat_a", "feat_b"])
    assert result["type"] == "feature_importance"
    assert {r["feature"] for r in result["ranked_features"]} == {"feat_a", "feat_b"}


def test_model_with_neither_attribute_returns_none():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = KNeighborsClassifier(n_neighbors=1).fit(X, y)
    assert extract_interpretability(model, ["feat_a"]) is None
