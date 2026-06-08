"""
Module for training and applying predictive models using scikit-learn. Supported are 
1. Regression: 
    - Random Forest
    - Decision Tree
    - Gradient Boosting
2. Regression: 
    - Logistic regression
    - Decision Tree
    - Gradient Boosting
predictors. Preprocessed training data is used to train these predictors and targets can be predicted using feature data from the test split.
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
    """Applies the fitted model to generate predictions.

    Args:
        model (Pipeline): A fitted scikit-learn Pipeline.
        X (pd.DataFrame): Feature matrix for which to generate predictions.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing hard predictions (0/1), probabilities for the positive class. 
    """
    y_pred = model.predict(X)

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


# Per-algorithm defaults. Decision Tree caps depth at 4 to keep the tree readable; the
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
    """Build a StandardScaler + classifier pipeline for deviation/outcome risk prediction.

    Args:
        algorithm (ClassificationAlgorithm, optional): User specified classification algorithm (Logistic Regression / Gradient Boosting / Decision Tree). Defaults to ClassificationAlgorithm.DECISION_TREE.
        ml_params (Optional[Dict[str, Any]], optional): Parameters that are to be passed through to the specified classification algorithm. Defaults to None.

    Returns:
        Pipeline: A Pipeline object modified with the given parameters.
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
    """Creates a pipeline by calling @create_classification_pipeline and then fits the classification pipeline using the provided data.

    Args:
        X_train (pd.DataFrame): Training data: feature dataframe.
        y_train (np.ndarray): Training data: target vector.
        algorithm (ClassificationAlgorithm, optional): User specified classification algorithm. Defaults to ClassificationAlgorithm.DECISION_TREE.
        ml_params (Optional[Dict[str, Any]], optional): User specified parameters for the provided classification algorithm. Defaults to None.

    Raises:
        ValueError: Raises a ValueError if parameters or algorithm does not fit.

    Returns:
        Pipeline: A fitted sklearn pipeline object.
    """
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


# Regression: per-algorithm defaults. Decision Tree mirrors the classification depth cap (4) so its
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
    """Build a StandardScaler + regressor pipeline for remaining-time prediction.

    Args:
        algorithm (RegressionAlgorithm, optional): User specified regression algorithm. Defaults to RegressionAlgorithm.DECISION_TREE.
        ml_params (Optional[Dict[str, Any]], optional): Corresponding regression algorithm parameters. Defaults to None.

    Returns:
        Pipeline: Sklearn pipeline object modified given the provided parameters. 
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
    """Fit a regression pipeline to the training data (remaining time in seconds).

    Args:
        X_train (pd.DataFrame): Training data: feature dataframe. 
        y_train (np.ndarray): Training data: target vector. 
        algorithm (RegressionAlgorithm, optional): User specified regression algorithm. Defaults to RegressionAlgorithm.DECISION_TREE.
        ml_params (Optional[Dict[str, Any]], optional): Corresponding regression algorithm parameters. Defaults to None.

    Returns:
        Pipeline: Sklearn pipeline object fitted to provided training data.
    """
    pipeline = create_regression_pipeline(algorithm, ml_params)
    pipeline.fit(X_train, y_train)
    return pipeline


def predict_regression(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Apply a fitted regression pipeline; returns continuous predictions (no proba).

    Args:
        model (Pipeline): Sklearn pipeline object (specified and fitted).
        X (pd.DataFrame): Input data: feature dataframe. 

    Returns:
        np.ndarray: Predicted values vector. 
    """
    return model.predict(X)

