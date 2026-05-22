"""
Evaluation module for the conformance-aware predictive monitoring pipeline.

Responsibilities:
  1. Infrastructure utilities (called by pipeline.py during execution):
       - temporal_split: case-level train/val/test partitioning by time
       - measure_runtime / measure_memory: wrappers around conformance and feature steps

  2. Reporting (called by pipeline.py after predictions are collected):
       - compute_classification_metrics: standard sklearn metrics for one model run
       - evaluate_by_prefix_length: metrics sliced by prefix length
       - compare_models: aggregated table across multiple seeds (baseline vs conformance-aware)
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Set, Tuple, Optional, List

import pm4py
import numpy as np
import pandas as pd
import psutil
import sklearn.metrics as metrics
from src import pipeline
from src import config

# ---------------------------------------------------------------------------
# 1. Infrastructure utilities
# ---------------------------------------------------------------------------

def temporal_split(
    log: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    case_id_col: str = "case:concept:name",
    timestamp_col: str = "time:timestamp",
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Partition cases into train / val / test strictly by case end-time."""
    # Find the last timestamp for each case
    case_ends = log.groupby(case_id_col)[timestamp_col].max().sort_values()
    
    n_cases = len(case_ends)
    n_train = int(n_cases * train_ratio)
    n_val = int(n_cases * val_ratio)
    
    train_case_ids = set(case_ends.index[:n_train])
    val_case_ids = set(case_ends.index[n_train:n_train + n_val])
    test_case_ids = set(case_ends.index[n_train + n_val:])
    
    return train_case_ids, val_case_ids, test_case_ids


def measure_runtime(func: Callable, *args: Any, **kwargs: Any) -> Tuple[Any, float]:
    """Call *func* and return its result together with wall-clock seconds elapsed."""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start_time
    return result, elapsed


def measure_memory(func: Callable, *args: Any, **kwargs: Any) -> Tuple[Any, float]:
    """Call *func* and return its result together with peak RSS increase in MB."""
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)
    result = func(*args, **kwargs)
    mem_after = process.memory_info().rss / (1024 * 1024)
    return result, max(0.0, mem_after - mem_before)


# ---------------------------------------------------------------------------
# 2. Metrics
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
) -> Dict[str, float]:
    """Compute standard binary classification metrics."""
    n_unique = len(np.unique(y_true))
    # Use proba dimensionality (what the model produced) to decide binary vs multi-class,
    # not y_true's unique count — the two can differ after a temporal split.
    is_binary_proba = np.ndim(y_pred_proba) == 1

    if n_unique <= 1:
        roc_auc = 0.5
    elif is_binary_proba:
        roc_auc = float(metrics.roc_auc_score(y_true, y_pred_proba))
    else:
        try:
            roc_auc = float(metrics.roc_auc_score(y_true, y_pred_proba, multi_class='ovr'))
        except ValueError:
            roc_auc = 0.0  # train/test class mismatch — model never saw all test classes

    return {
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "precision": float(metrics.precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        "recall": float(metrics.recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        "f1": float(metrics.f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        "roc_auc": roc_auc,
        "brier_score": float(metrics.brier_score_loss(y_true, y_pred_proba)) if (is_binary_proba and n_unique == 2) else 0.0
    }




def evaluate_by_prefix_length(
    target: config.PredictionTarget,
    feature: config.PredictionFeature,
    model: Tuple[Any, Any, Any],
    log: pd.DataFrame,
    outcome_activities: Optional[List[str]] = None,
    prefix_range_start: Optional[int] = None,
    prefix_range_end: Optional[int] = None,
) -> pd.DataFrame:
    case_id_col = "case:concept:name"
    
    if prefix_range_start is None or prefix_range_end is None:
        case_lengths = log.groupby(case_id_col).size()
        if prefix_range_start is None:
            prefix_range_start = max(1, int(case_lengths.quantile(0.25)))
        if prefix_range_end is None:
            prefix_range_end = int(case_lengths.median())

    prefix_range = range(prefix_range_start, prefix_range_end + 1) 
    results = []
    log_train, log_test= pm4py.split_train_test(log, train_percentage=0.75)

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
        result = pipeline.run_conformance_prediction_pipeline(
            X_test=X_test,
            y_test=y_test,
            X_train=X_train,
            y_train=y_train,
            # ml_params=ml_params, --> for now use the default values
        )
        metrics = result.get("metrics", {})
        results.append({
            "prefix_length": prefix_length,
            "accuracy": metrics.get("accuracy"),
            "f1": metrics.get("f1")
        })
    
    return pd.DataFrame(results)

def compare_models(
    baseline_results: Dict[int, Dict[str, float]],
    conformance_results: Dict[int, Dict[str, float]],
) -> pd.DataFrame:
    """Aggregate metrics across seeds and produce a comparison table.

    Args:
        baseline_results: Mapping seed -> metrics dict (from compute_classification_metrics)
            for the log-only baseline model.
        conformance_results: Same structure for the conformance-aware model.

    Returns:
        DataFrame with one row per metric, columns:
            baseline_mean, baseline_std, conformance_mean, conformance_std.
    """
    ...

def compute_mse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.float64:
    n = len(y_true)
    return (1/n)*(sum((y_true-y_pred)**2))

def compute_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.float64:
    mse = compute_mse(y_true, y_pred)
    return np.sqrt(mse)

def compute_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.float64:
    n = len(y_true)
    return (1/n)*(sum(np.abs(y_true-y_pred)))
