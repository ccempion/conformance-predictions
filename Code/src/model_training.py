"""
Module for training and applying predictive models (HL-08, HL-09).
Implements scikit-learn-style objects for preprocessing and prediction.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from typing import Dict, Any, Tuple, Optional


def create_model_pipeline(ml_params: Optional[Dict[str, Any]] = None) -> Pipeline:
    """
    Creates a scikit-learn Pipeline with preprocessing and a Decision Tree classifier (LL-08.1).
    
    Args:
        ml_params: Dictionary of hyperparameters for the DecisionTreeClassifier.
                   If None, default parameters are used.
    
    Returns:
        A scikit-learn Pipeline object.
    """
    if ml_params is None:
        ml_params = {}
    
    # 1. Preprocessing: Standardize features (LL-08.1)
    # This ensures that scale-sensitive components (if added later) or 
    # even the decision tree splits (though less sensitive) are handled consistently.
    scaler = StandardScaler()
    
    # 2. Classifier: Decision Tree (HL-08, HL-09)
    # The Decision Tree is the core requirement for the current sprint.
    clf = DecisionTreeClassifier(**ml_params)
    
    # Bundle into a Pipeline for clean integration and to prevent data leakage 
    # during later cross-validation or deployment.
    pipeline = Pipeline([
        ('scaler', scaler),
        ('classifier', clf)
    ])
    
    return pipeline


def train_model(X_train: pd.DataFrame, y_train: pd.Series, ml_params: Optional[Dict[str, Any]] = None) -> Pipeline:
    """
    Fits a predictive model to the training data (LL-08.2).
    
    Args:
        X_train: Feature matrix (prefixes + optional conformance diagnostics).
        y_train: Target vector (deviation risk or outcome).
        ml_params: Optional hyperparameters for the model.
        
    Returns:
        The fitted Pipeline object.
    """
    pipeline = create_model_pipeline(ml_params)
    pipeline.fit(X_train, y_train)
    return pipeline


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
        y_prob = model.predict_proba(X)[:, 1]
    else:
        # Fallback for models without predict_proba (not expected for DecisionTree)
        y_prob = y_pred.astype(float)
        
    return y_pred, y_prob


def get_feature_importance(model: Pipeline, feature_names: list) -> pd.Series:
    """
    Extracts feature importance scores from the fitted Decision Tree.
    Useful for evaluation and comparing baseline vs conformance-aware features.
    
    Args:
        model: A fitted scikit-learn Pipeline.
        feature_names: List of names corresponding to the columns in X.
        
    Returns:
        A Series mapping feature names to importance scores.
    """
    importances = model.named_steps['classifier'].feature_importances_
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)
