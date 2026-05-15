"""
Module for training and applying predictive models (HL-08, HL-09).
Implements scikit-learn-style objects for preprocessing and prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from typing import Dict, Any, List, Tuple, Optional


def create_model_pipeline(ml_params: Optional[Dict[str, Any]] = None) -> Pipeline:
    """
    Creates a scikit-learn Pipeline with preprocessing and a Decision Tree classifier (LL-08.1).
    
    Args:
        ml_params: Dictionary of hyperparameters for the DecisionTreeClassifier.
                   If None, default parameters for full branching are used.
    
    Returns:
        A scikit-learn Pipeline object.
    """
    # 1. Default configurations for "complete branching" (MVP rule of thumb)
    # Allows the tree to grow until all leaves are sufficiently pure.
    model_defaults = {
        'criterion': 'gini',
        'max_depth': 4,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42  # Ensuring reproducibility
    }
    
    if ml_params is not None:
        model_defaults.update(ml_params)
    
    # 2. Preprocessing: Standardize features (LL-08.1)
    # This ensures that scale-sensitive components are handled consistently.
    scaler = StandardScaler()
    
    # 3. Classifier: Decision Tree (HL-08, HL-09)
    clf = DecisionTreeClassifier(**model_defaults)
    
    # Bundle into a Pipeline for clean integration and to prevent data leakage.
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


def visualize_decision_tree(
    model: Pipeline,
    feature_names: List[str],
    class_names: List[str],
    output_path: Optional[str] = None,
) -> None:
    """
    Renders the fitted Decision Tree from the pipeline.
    If output_path is given, saves to that file; otherwise opens an interactive window.
    """
    clf = model.named_steps['classifier']
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        ax=ax
    )
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()