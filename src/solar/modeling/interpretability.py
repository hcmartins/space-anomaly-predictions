"""Coefficient / feature-importance interpretability for classical models
-- Chapter 3.8's documented requirement: "For baseline and traditional
machine-learning models, interpretability is examined through established
techniques such as coefficient analysis and feature importance rankings."

Deliberately scoped to what a fitted scikit-learn estimator exposes
natively (`coef_` for linear models, `feature_importances_` for tree
ensembles) -- no SHAP/attribution tooling for the (not-yet-built) deep
models; see planning/MODELING_DOCUMENTATION.md for that boundary.
"""
from __future__ import annotations

import numpy as np


def extract_interpretability(model, feature_manifest: list[str]) -> dict | None:
    """Returns a ranked (by magnitude) feature list for a fitted classical
    model, or None if the model exposes neither `coef_` nor
    `feature_importances_` (e.g. KNeighborsClassifier has no native
    per-feature attribution -- correctly reported as unavailable, not
    silently omitted or fabricated)."""
    if hasattr(model, "coef_"):
        coefs = np.ravel(model.coef_)
        ranked = sorted(zip(feature_manifest, coefs), key=lambda kv: abs(kv[1]), reverse=True)
        return {
            "type": "coefficients",
            "ranked_features": [{"feature": f, "coefficient": float(c)} for f, c in ranked],
        }
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        ranked = sorted(zip(feature_manifest, importances), key=lambda kv: kv[1], reverse=True)
        return {
            "type": "feature_importance",
            "ranked_features": [{"feature": f, "importance": float(v)} for f, v in ranked],
        }
    return None
