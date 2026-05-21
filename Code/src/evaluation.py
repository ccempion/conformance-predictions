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
       - plot_roc_curves / plot_pr_curves / plot_metrics_by_prefix /
         plot_feature_importance / plot_resource_usage: all visualizations
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pm4py as pm
import psutil
import matplotlib.pyplot as plt
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


# ---------------------------------------------------------------------------
# 3. CLI display
# ---------------------------------------------------------------------------

def display_confusion_matrix_text(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> None:
    """Print a formatted confusion matrix to the terminal."""
    cm = metrics.confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    col_w = max(max(len(n) for n in class_names), 6) + 2
    indent = " " * (col_w + 2)

    print("\nConfusion Matrix (rows = Actual, cols = Predicted):")
    print(indent + "".join(n.center(col_w) for n in class_names))
    for i, name in enumerate(class_names):
        row = name.ljust(col_w + 2) + "".join(
            str(cm[i, j]).center(col_w) for j in range(len(class_names))
        )
        print(row)


def display_per_class_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> None:
    """Print per-class correct / incorrect percentages to the terminal."""
    cm = metrics.confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    print("\nPer-Class Prediction Breakdown:")
    for i, name in enumerate(class_names):
        total = int(cm[i].sum())
        correct = int(cm[i, i])
        wrong = total - correct
        pct_c = correct / total * 100 if total > 0 else 0.0
        pct_w = wrong / total * 100 if total > 0 else 0.0
        print(f"  {name}: {pct_c:.1f}% correct ({correct}/{total}),  {pct_w:.1f}% incorrect ({wrong}/{total})")


# ---------------------------------------------------------------------------
# 4. Image output
# ---------------------------------------------------------------------------

def plot_confusion_matrix_image(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: str,
) -> None:
    """Save a simple confusion-matrix heatmap to *output_path*."""
    cm = metrics.confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    n = len(class_names)

    fig, ax = plt.subplots(figsize=(max(4, n * 1.2 + 1), max(4, n * 1.2)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_per_class_bar_chart(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: str,
) -> None:
    """Save a stacked bar chart (green = correct, red = incorrect) per class."""
    cm = metrics.confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    correct_pcts, wrong_pcts = [], []
    for i in range(len(class_names)):
        total = int(cm[i].sum())
        correct = int(cm[i, i])
        correct_pcts.append(correct / total * 100 if total > 0 else 0.0)
        wrong_pcts.append((total - correct) / total * 100 if total > 0 else 0.0)

    x = list(range(len(class_names)))
    fig, ax = plt.subplots(figsize=(max(5, len(class_names) * 1.4 + 1), 5))
    ax.bar(x, correct_pcts, label="Correct", color="green", alpha=0.85)
    ax.bar(x, wrong_pcts, bottom=correct_pcts, label="Incorrect", color="red", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Per-Class Prediction Accuracy")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def save_evaluation_visuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    base_dir: Optional[str] = None,
) -> str:
    """Create a timestamped folder and save confusion matrix + bar chart inside it.

    Returns the absolute path of the created folder.
    """
    import pathlib
    import datetime

    root = pathlib.Path(base_dir) if base_dir else pathlib.Path.home() / "Downloads"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = root / f"conformance_results_{timestamp}"
    folder.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix_image(y_true, y_pred, class_names, str(folder / "confusion_matrix.png"))
    plot_per_class_bar_chart(y_true, y_pred, class_names, str(folder / "per_class_accuracy.png"))

    return str(folder)


# ---------------------------------------------------------------------------
# 5. Visualizations (stubs)
# ---------------------------------------------------------------------------

def plot_roc_curves(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    output_path: Optional[str] = None,
) -> None:
    """Plot overlaid ROC curves for multiple models.

    Args:
        roc_data: Mapping model_name -> (fpr, tpr, auc_value).
            Compute with sklearn.metrics.roc_curve + roc_auc_score before calling.
        output_path: If given, save figure to this path; otherwise display interactively.
    """
    ...


def plot_pr_curves(
    pr_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    output_path: Optional[str] = None,
) -> None:
    """Plot overlaid Precision-Recall curves for multiple models.

    Args:
        pr_data: Mapping model_name -> (precision, recall, pr_auc_value).
            Compute with sklearn.metrics.precision_recall_curve + auc before calling.
        output_path: If given, save figure; otherwise display interactively.
    """
    ...


def plot_metrics_by_prefix(
    baseline_prefix_df: pd.DataFrame,
    conformance_prefix_df: pd.DataFrame,
    metric: str = "roc_auc",
    output_path: Optional[str] = None,
) -> None:
    """Plot a chosen metric as a function of prefix length for both models.

    Visually answers: does conformance information help more at early prefixes
    (where the log-only signal is weak)?

    Args:
        baseline_prefix_df: Output of evaluate_by_prefix_length for the baseline model.
        conformance_prefix_df: Output of evaluate_by_prefix_length for conformance-aware model.
        metric: Column name to plot (must exist in both DataFrames).
        output_path: If given, save figure; otherwise display interactively.
    """
    ...


def plot_feature_importance(
    importances: pd.Series,
    conformance_feature_names: List[str],
    top_n: int = 20,
    output_path: Optional[str] = None,
) -> None:
    """Bar chart of feature importances with conformance features highlighted.

    Args:
        importances: Series mapping feature_name -> importance score.
            Obtain from model.feature_importances_ or a permutation importance result.
        conformance_feature_names: List of feature names that are conformance-derived,
            as declared by feature_extraction.py. Used for color coding.
        top_n: How many top features to display.
        output_path: If given, save figure; otherwise display interactively.
    """
    ...


def plot_resource_usage(
    runtime_records: List[Dict[str, Any]],
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Plot runtime and memory usage as a function of log size, return summary table.

    Each record in runtime_records should have keys:
        step (str), log_size (int), elapsed_seconds (float), peak_memory_mb (float).
    Separate series are drawn for conformance checking and feature generation steps.

    Args:
        runtime_records: Collected by pipeline.py using measure_runtime / measure_memory.
        output_path: If given, save figure; otherwise display interactively.

    Returns:
        Summary DataFrame (step x [mean_seconds, mean_memory_mb]) for the report.
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