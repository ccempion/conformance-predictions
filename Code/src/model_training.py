"""
Module for training and applying predictive models (HL-08, HL-09).
Implements scikit-learn-style objects for preprocessing and prediction.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any, Tuple, Optional

from src.config import RegressionAlgorithm, ClassificationAlgorithm


def predict(model: Pipeline, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies the fitted model to generate predictions.

    Args:
        model: A fitted scikit-learn Pipeline.
        X: Feature matrix for which to generate predictions.

    Returns:
        A tuple containing:
            - Hard predictions (0/1)
            - Predicted probabilities for the positive class
    """
    y_pred = model.predict(X)

    # Check if the classifier supports probability estimates
    if hasattr(model.named_steps['classifier'], "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 1:
            y_prob = proba[:, 0]       # degenerate: only one class seen in training
        elif proba.shape[1] == 2:
            y_prob = proba[:, 1]       # binary: probability of the positive class (1D)
        else:
            y_prob = proba             # multi-class: full (n_samples, n_classes) matrix
    else:
        # Fallback for models without predict_proba (not expected for DecisionTree)
        y_prob = y_pred.astype(float)

    return y_pred, y_prob


# ------------------------------------------
# Classification algorithm menu (PROCESS_OUTCOME / FUTURE_DEVIATIONS)
# ------------------------------------------

<<<<<<< HEAD
# Per-algorithm defaults. Decision Tree caps depth at 4 to keep the tree readable; the
=======
# Per-algorithm defaults. Decision Tree caps depth at 4 so the tree stays readable; the
>>>>>>> 75-additional-ml-algorithm
# others use sklearn defaults plus a fixed seed / enough iterations to converge.
_CLASSIFIER_DEFAULTS: Dict[ClassificationAlgorithm, Dict[str, Any]] = {
    ClassificationAlgorithm.DECISION_TREE: {"max_depth": 4, "random_state": 42},
    ClassificationAlgorithm.LOGISTIC_REGRESSION: {"max_iter": 1000},
    ClassificationAlgorithm.GRADIENT_BOOSTING: {"random_state": 42},
}


def create_classification_pipeline(
    algorithm: ClassificationAlgorithm = ClassificationAlgorithm.DECISION_TREE,
    ml_params: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """
    Build a StandardScaler + classifier pipeline for deviation/outcome risk prediction.

    Counterpart to create_regression_pipeline. Every classifier here exposes predict_proba,
<<<<<<< HEAD
    so the existing predict() yields a risk probability. The estimator step is named
    'classifier' so predict() and visualize_decision_tree keep working unchanged.
=======
    so predict() yields a risk probability. The estimator step is named 'classifier' so
    predict() and visualize_decision_tree keep working unchanged.
>>>>>>> 75-additional-ml-algorithm
    """
    defaults = dict(_CLASSIFIER_DEFAULTS.get(algorithm, {}))
    if ml_params:
        defaults.update(ml_params)

    if algorithm == ClassificationAlgorithm.LOGISTIC_REGRESSION:
        clf = LogisticRegression(**defaults)
    elif algorithm == ClassificationAlgorithm.GRADIENT_BOOSTING:
        clf = GradientBoostingClassifier(**defaults)
    else:  # DECISION_TREE (default)
        clf = DecisionTreeClassifier(**defaults)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf),
    ])


def train_classification_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    algorithm: ClassificationAlgorithm = ClassificationAlgorithm.DECISION_TREE,
    ml_params: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """Fit a classification pipeline (deviation/outcome risk) to the training data."""
    classes = np.unique(y_train)
    if len(classes) < 2:
        raise ValueError(
            f"Training labels contain only one class ({classes.tolist()}), so the "
            f"classifier has nothing to discriminate. For FUTURE_DEVIATIONS this usually "
            f"means the deviation cost threshold T is too high and left no positive cases "
            f"-- lower T (default 0 = 'any real deviation') and re-run. For PROCESS_OUTCOME "
            f"it means every case in the training split shares the same outcome."
        )
    pipeline = create_classification_pipeline(algorithm, ml_params)
    pipeline.fit(X_train, y_train)
    return pipeline


# ------------------------------------------
# Regression (REMAINING_TIME target)
# ------------------------------------------

# Per-algorithm defaults. Decision Tree mirrors the classification depth cap (4) so its
# tree stays readable; the ensembles use sklearn defaults plus a fixed seed.
_REGRESSOR_DEFAULTS: Dict[RegressionAlgorithm, Dict[str, Any]] = {
    RegressionAlgorithm.DECISION_TREE: {"max_depth": 4, "random_state": 42},
    RegressionAlgorithm.RANDOM_FOREST: {"n_estimators": 100, "random_state": 42},
    RegressionAlgorithm.GRADIENT_BOOSTING: {"random_state": 42},
}


def create_regression_pipeline(
    algorithm: RegressionAlgorithm = RegressionAlgorithm.DECISION_TREE,
    ml_params: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """
    Build a StandardScaler + regressor pipeline for remaining-time prediction.

<<<<<<< HEAD
    Counterpart to create_classification_pipeline: selects a regressor from the
    chosen RegressionAlgorithm. ml_params overrides the per-algorithm defaults. The
    classifier step in the classification pipeline is named 'classifier'; here it is
    named 'regressor' so the two are unambiguous downstream.
=======
    Counterpart to create_classification_pipeline: selects a regressor from the chosen
    RegressionAlgorithm. ml_params overrides the per-algorithm defaults. The classifier
    step in the classification pipeline is named 'classifier'; here it is named
    'regressor' so the two are unambiguous downstream.
>>>>>>> 75-additional-ml-algorithm
    """
    defaults = dict(_REGRESSOR_DEFAULTS.get(algorithm, {}))
    if ml_params:
        defaults.update(ml_params)

    if algorithm == RegressionAlgorithm.RANDOM_FOREST:
        regressor = RandomForestRegressor(**defaults)
    elif algorithm == RegressionAlgorithm.GRADIENT_BOOSTING:
        regressor = GradientBoostingRegressor(**defaults)
    else:  # DECISION_TREE (default)
        regressor = DecisionTreeRegressor(**defaults)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", regressor),
    ])


def train_regression_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    algorithm: RegressionAlgorithm = RegressionAlgorithm.DECISION_TREE,
    ml_params: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """Fit a regression pipeline to the training data (remaining time in seconds)."""
    pipeline = create_regression_pipeline(algorithm, ml_params)
    pipeline.fit(X_train, y_train)
    return pipeline


def predict_regression(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Apply a fitted regression pipeline; returns continuous predictions (no proba)."""
    return model.predict(X)
