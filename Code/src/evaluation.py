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
from typing import Any, Callable, Dict, Set, Tuple

import numpy as np
import pandas as pd
import psutil
import sklearn.metrics as metrics

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
    prefix_results: pd.DataFrame,
) -> pd.DataFrame:
    """Compute metrics grouped by prefix length for early-vs-late warning analysis.

    Args:
        prefix_results: DataFrame with columns
            [case_id, prefix_length, y_true, y_pred, y_pred_proba].
            prefix_length is a grouping key, not a model feature.

    Returns:
        DataFrame indexed by prefix_length; columns are the same metric names
        as compute_classification_metrics produces. Groups with fewer than
        10 samples are dropped to avoid unstable estimates.
    """
    ...


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