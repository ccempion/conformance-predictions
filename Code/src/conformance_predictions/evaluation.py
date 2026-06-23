"""
Evaluation module for the predictive monitoring pipeline.

This module is responsible for (list not guaranteed to be complete):
1. Infrastructure utilities (called during the execution of our CLI):
    - measure_runtime / measure_memory: wrappers around (relatively big) pipeline fragments
2. Reporting (called by pipeline fragments and CLI after predictions are collected):
    - compute_classification_metrics: standard sklearn metrics for one model run
    - compute_regression_metrics: standard metrics for one model run
    - evaluate_by_prefix_length: metrics of models masked by a prefix of given length 
"""

from __future__ import annotations

import time
import sys
from typing import Any, Callable, Dict, Set, Tuple, Optional, List

try:
    import resource
except ImportError:
    resource = None

import pm4py
import numpy as np
import pandas as pd
import psutil
import sklearn.metrics as metrics
from sklearn.pipeline import Pipeline
from conformance_predictions import pipeline
from conformance_predictions import config
from conformance_predictions import model_training

def measure_runtime(func: Callable, *args: Any, **kwargs: Any) -> Tuple[Any, float]:
    """Measure the runtime of an arbitrary function.

    Args:
        func (Callable): The Python function whose runtime is to be measured.

    Returns:
        Tuple[Any, float]: The return value of the passed function and the elapsed time (in seconds).
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start_time
    return result, elapsed


def measure_memory(func: Callable, *args: Any, **kwargs: Any) -> Tuple[Any, float]:
    """Measure the peak memory usage of an arbitrary function.

    Args:
        func (Callable): The Python function whose peak memory usage is to be measured. 

    Returns:
        Tuple[Any, float]: The return value of the passed function and the peak RSS in MB. 
    """
    if resource is None:
        # Fallback to psutil delta for non-Unix if resource is unavailable
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)
        result = func(*args, **kwargs)
        mem_after = process.memory_info().rss / (1024 * 1024)
        return result, max(0.0, mem_after - mem_before)

    # resource.ru_maxrss is monotonic: it tracks the maximum RSS used by the process so far.
    result = func(*args, **kwargs)
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # On Linux, ru_maxrss is in KB. On macOS, it's in Bytes.
    factor = 1.0 / 1024.0 if sys.platform != 'darwin' else 1.0 / (1024.0 * 1024.0)
    return result, peak_kb * factor

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
)-> Dict[str, float]:
    """Aggregation function for all available regression metrics.

    Args:
        y_true (np.ndarray): Target vector. 
        y_pred (np.ndarray): Predicted vector. 

    Returns:
        Dict[str, float]: A dictionary containing regression metrics: 
        Root Mean Squared Error ('RMSE'),
        Mean Squared Error ('MSE'),
        Mean Absolute Error ('MAE'),
    """
    mae = compute_mae(y_true, y_pred)
    mse = compute_mse(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
    }

def pick_duration_unit(values_seconds) -> Tuple[float, str]:
    """Pick ONE human-readable time unit for second-valued data, chosen from the
    ~95th-percentile magnitude so the whole series can be shown in a single unit.

    Used to render remaining-time errors (MAE/RMSE, both in seconds) as e.g. hours
    or days instead of unreadable raw seconds. The unit is selected once and applied
    uniformly by the caller (divide values by the returned divisor).

    Args:
        values_seconds: Array-like of durations in seconds (e.g. the test labels).

    Returns:
        Tuple[float, str]: (divisor, label), e.g. (3600.0, "hours"). Empty/all-zero
        input falls back to (1.0, "seconds").
    """
    values = np.abs(np.asarray(values_seconds, dtype=float))
    values = values[~np.isnan(values)]
    if values.size == 0:
        return (1.0, "seconds")

    magnitude = float(np.percentile(values, 95))
    if magnitude >= 86400:
        return (86400.0, "days")
    if magnitude >= 3600:
        return (3600.0, "hours")
    if magnitude >= 60:
        return (60.0, "minutes")
    return (1.0, "seconds")

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Aggregation of classification metrics.

    Args:
        y_true (np.ndarray): Target vector. 
        y_pred (np.ndarray): Predicted vector.
        y_pred_proba (Optional[np.ndarray], optional): Class probability. Defaults to None.

    Returns:
        Dict[str, float]: Dictionary of classification scores such as:
        Accuracy, Precision, Recall, F1, ROC/AUC, Brier score.
    """
    n_unique = len(np.unique(y_true))
    # Use proba dimensionality to decide binary vs multi-class.
    is_binary_proba = np.ndim(y_pred_proba) == 1 if y_pred_proba is not None else False

    roc_auc = 0.5
    if y_pred_proba is not None and n_unique > 1:
        if is_binary_proba:
            roc_auc = float(metrics.roc_auc_score(y_true, y_pred_proba))
        else:
            try:
                roc_auc = float(metrics.roc_auc_score(y_true, y_pred_proba, multi_class='ovr'))
            except ValueError:
                roc_auc = 0.0

    return {
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "precision": float(metrics.precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        "recall": float(metrics.recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        "f1": float(metrics.f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        "roc_auc": roc_auc,
        "brier_score": float(metrics.brier_score_loss(y_true, y_pred_proba)) if (is_binary_proba and n_unique == 2) else 0.0
    }

def compute_roc_curve(y_true: np.ndarray, y_score: np.ndarray, pos_label=None) -> Tuple[np.ndarray, np.ndarray]:
    """Invokes the sklearn.metrics function that calculates the ROC curve.

    Args:
        y_true (np.ndarray): Target vector. 
        y_score (np.ndarray): Predicted vector.
        pos_label (_type_, optional): Class probability. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Lists of false positive and true positive rates (FPR, TPR).
    """
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=pos_label)
    return (fpr, tpr)

def compute_pr_curve(y_true: np.ndarray, y_score: np.ndarray, pos_label=None) -> Tuple[np.ndarray, np.ndarray, float]:
    prec, recall, _ = metrics.precision_recall_curve(y_true, y_score, pos_label=pos_label)
    if pos_label is not None:
        avg_prec = float(metrics.average_precision_score(y_true, y_score, pos_label=pos_label))
    else:
        avg_prec = float(metrics.average_precision_score(y_true, y_score))
    return (prec, recall, avg_prec)

def prepare_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Tuple[np.ndarray, np.ndarray, float]]]:
    """Preparation of data for and aggregation of curves (ROC / PR) per class.

    Args:
        y_true (np.ndarray): Target vector.
        y_prob (np.ndarray): Class probability.
        class_names (List[str]): Class names. 

    Returns:
        Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Tuple[np.ndarray, np.ndarray, float]]]: Tuple of prepared data for ROC and PR.
    """
    roc_data = {}
    pr_data = {}
    n_classes = len(class_names)

    if n_classes == 2:
        # Binary case: use probabilities of the positive class (assumed to be index 1)
        y_score = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
        fpr, tpr = compute_roc_curve(y_true, y_score, pos_label=1)
        roc_data["Model"] = (fpr, tpr)
        
        prec, recall, avg_prec = compute_pr_curve(y_true, y_score, pos_label=1)
        pr_data["Model"] = (prec, recall, avg_prec)
    else:
        # Multi-class: One-vs-Rest for each class
        for i, class_name in enumerate(class_names):
            y_true_binary = (y_true == i).astype(int)
            y_score = y_prob[:, i]
            
            fpr, tpr = compute_roc_curve(y_true_binary, y_score, pos_label=1)
            roc_data[class_name] = (fpr, tpr)
            
            prec, recall, avg_prec = compute_pr_curve(y_true_binary, y_score, pos_label=1)
            pr_data[class_name] = (prec, recall, avg_prec)
            
    return roc_data, pr_data

def metrics_by_prefix_length(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    prefix_lengths: np.ndarray,
    is_regression: bool = False,
    min_group_size: int = 10,
) -> pd.DataFrame:
    """Slice one pooled model's predictions by prefix length and compute metrics per length.

    Args:
        y_true (np.ndarray): Target vector.
        y_pred (np.ndarray): Prediction vector.
        y_prob (Optional[np.ndarray]): Class probability.
        prefix_lengths (np.ndarray): Prefix length of each test instance.
        is_regression (bool, optional): Determins whether to compute regression or classification metrics. Defaults to False.
        min_group_size (int, optional): Groups with fewer samples than specified are dropped. Defaults to 10.

    Returns:
        pd.DataFrame: Dataframe of metrics by the prefix length. One row per prefix length - metric combination.
    """
    y_true, y_pred, prefix_lengths = np.asarray(y_true), np.asarray(y_pred), np.asarray(prefix_lengths)

    rows = []
    for k in sorted(np.unique(prefix_lengths)):
        mask = prefix_lengths == k
        if int(mask.sum()) < min_group_size:
            continue
        
        if is_regression:
            m = compute_regression_metrics(y_true[mask], y_pred[mask])
        else:
            m = compute_classification_metrics(y_true[mask], y_pred[mask], y_prob[mask] if y_prob is not None else None)
        
        rows.append({"prefix_length": int(k), "n": int(mask.sum()), **m})

    return pd.DataFrame(rows)


def evaluate_by_prefix_length(
    target: config.PredictionTarget,
    feature: config.PredictionFeature,
    model: Tuple[Any, Any, Any],
    log: pd.DataFrame,
    outcome_activities: Optional[List[str]] = None,
    prefix_range_start: Optional[int] = None,
    prefix_range_end: Optional[int] = None,
    algorithm: Optional[Any] = None,
    ml_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Trains models with trace-prefixes of specified length and aggregates the resulting F1 and Accuracy scores.

    Args:
        target (config.PredictionTarget): User specified target type.
        feature (config.PredictionFeature): User specified feature type (should include trace, otherwise pointless). 
        model (Tuple[Any, Any, Any]): Petri net.
        log (pd.DataFrame): Event log.
        outcome_activities (Optional[List[str]], optional): Optionally specified outcome activities. Required for prediction type 'process outcome'. Defaults to None.
        prefix_range_start (Optional[int], optional): Specified lower bound of prefix length range. Defaults to None. If not specified, the maximum of (1, lower quartile trace length) is used.
        prefix_range_end (Optional[int], optional): Specified upper bound of prefix length range. Defaults to None. If not specified, the median trace length is identified and used.
        algorithm (Optional[Any], optional): User specified algorithm (ClassificationAlgorithm or RegressionAlgorithm). Defaults to None.
        ml_params (Optional[Dict[str, Any]], optional): Parameters for the ML algorithm. Defaults to None.

    Returns:
        pd.DataFrame: A dataframe containing the F1 and accuracy score per prefix length in the provided range.
    """
    case_id_col = "case:concept:name"
    is_regression = (target == config.PredictionTarget.REMAINING_TIME)

    if algorithm is None:
        if is_regression:
            algorithm = config.RegressionAlgorithm.DECISION_TREE
        else:
            algorithm = config.ClassificationAlgorithm.DECISION_TREE
    
    if prefix_range_start is None or prefix_range_end is None:
        case_lengths = log.groupby(case_id_col).size()
        if prefix_range_start is None:
            prefix_range_start = max(1, int(case_lengths.quantile(0.25)))
        if prefix_range_end is None:
            prefix_range_end = int(case_lengths.median())

    prefix_range = range(prefix_range_start, prefix_range_end + 1) 
    results = []
    log_train, log_test= pm4py.split_train_test(log, train_percentage=0.75)

    if log_train.empty or log_test.empty:
        cols = ["prefix_length", "MSE", "RMSE", "MAE"] if is_regression else ["prefix_length", "accuracy", "f1"]
        return pd.DataFrame(columns=cols)

    for prefix_length in prefix_range:
        X_train, y_train, class_names = pipeline.extract_target_and_feature(
            target=target,
            feature=feature,
            model=model,
            log=log_train,
            outcome_activities=outcome_activities,
            prefix_length=prefix_length
        )
        X_test, y_test, _ = pipeline.extract_target_and_feature(
            target=target,
            feature=feature,
            model=model,
            log=log_test,
            outcome_activities=outcome_activities,
            prefix_length=prefix_length
        )
        
        # Train model specifically for this prefix length
        try:
            if is_regression:
                trained_model = model_training.train_regression_model(
                    X_train, y_train, algorithm=algorithm, ml_params=ml_params
                )
            else:
                if len(np.unique(y_train)) < 2:
                    # Fallback to create_classification_pipeline directly to avoid raising ValueError for single-class splits (e.g. in tests)
                    trained_model = model_training.create_classification_pipeline(algorithm=algorithm, ml_params=ml_params)
                    trained_model.fit(X_train, y_train)
                else:
                    trained_model = model_training.train_classification_model(
                        X_train, y_train, algorithm=algorithm, ml_params=ml_params
                    )
            
            result = pipeline.run_conformance_prediction_pipeline(
                X_test=X_test,
                y_test=y_test,
                X_train=X_train,
                y_train=y_train,
                ml_params=ml_params,
                preloaded_model=trained_model,
                is_regression=is_regression,
            )
            metrics = result.get("metrics", {})
            if is_regression:
                # Regression metrics are keyed MSE/RMSE/MAE; spreading them keeps the
                # by-length DataFrame usable (the old accuracy/f1 keys were always None here).
                results.append({"prefix_length": prefix_length, **metrics})
            else:
                results.append({
                    "prefix_length": prefix_length,
                    "accuracy": metrics.get("accuracy"),
                    "f1": metrics.get("f1")
                })
        except ValueError as e:
            # Handle cases where training fails (e.g. single-class split)
            print(f"Skipping prefix length {prefix_length} due to training error: {e}")
    
    return pd.DataFrame(results)

def compare_models(
    baseline_results: Dict[int, Dict[str, float]],
    conformance_results: Dict[int, Dict[str, float]],
) -> pd.DataFrame:
    ...

def compute_mse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.float64:
    """Computation of Mean Squared Error (MSE).

    Args:
        y_true (np.ndarray): Target vector. 
        y_pred (np.ndarray): Predicted vector.

    Returns:
        np.float64: MSE.
    """
    n = len(y_true)
    return (1/n)*(sum((y_true-y_pred)**2))

def compute_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.float64:
    """Computation of Root Mean Squared Error (RMSE).

    Args:
        y_true (np.ndarray): Target vector. 
        y_pred (np.ndarray): Predicted vector.

    Returns:
        np.float64: RMSE.
    """
    mse = compute_mse(y_true, y_pred)
    return np.sqrt(mse)

def compute_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.float64:
    """Computation of Mean Absolute Error (MAE).

    Args:
        y_true (np.ndarray): Target vector. 
        y_pred (np.ndarray): Predicted vector.

    Returns:
        np.float64: MAE.
    """
    n = len(y_true)
    return (1/n)*(sum(np.abs(y_true-y_pred)))


    """Extracts and ranks feature importances from a fitted pipeline.

    Supports models with 'feature_importances_' (e.g., Decision Trees, Random Forests)
    and linear models with 'coef_' (e.g., Logistic Regression).
    """
def feature_importance_comparison(X_train: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    """Extracts and ranks feature importances from a fitted pipeline.

    Supports models with 'feature_importances_' (e.g., Decision Trees, Random Forests)
    and linear models with 'coef_' (e.g., Logistic Regression).

    Args:
        X_train (pd.DataFrame): Training data: feature dataframe. 
        model (Pipeline): Scikit-learn pipeline object.

    Returns:
        pd.DataFrame: Dataframe containing the ten most important attributes for the predictor. Cols: {- (Feature name), importance, rank}
    """
    # The pipeline is assumed to be [("scaler", ...), ("classifier/regressor", estimator)]
    # Extract the final estimator
    estimator = model.steps[-1][1]

    if hasattr(estimator, 'feature_importances_'):
        importances_values = estimator.feature_importances_
    elif hasattr(estimator, 'coef_'):
        # For linear models, use absolute coefficients as a proxy for importance.
        # If multi-class, coef_ has shape (n_classes, n_features).
        if estimator.coef_.ndim > 1:
            importances_values = np.mean(np.abs(estimator.coef_), axis=0)
        else:
            importances_values = np.abs(estimator.coef_.flatten())
    else:
        return pd.DataFrame(columns=['importance', 'rank'])

    features = list(X_train.columns)
    # Ensure dimensions match (sometimes coef_ might be different if scaling/encoding changed it)
    if len(importances_values) != len(features):
        return pd.DataFrame(columns=['importance', 'rank'])

    importances = pd.DataFrame(index=features)
    importances['importance'] = importances_values
    importances['rank'] = importances['importance'].rank(ascending=False)
    return importances.sort_values('rank').head(10)