from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree
import sklearn.metrics as metrics
import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking

from typing import Dict, Any, List, Tuple, Optional

from src.config import ModelRepresentation


# ------------------------------------------
# Image display (and save)
# ------------------------------------------

def visualize_process_model(model: Tuple[PetriNet, Marking, Marking], path: Optional[str], model_representation: ModelRepresentation):
    net, im, fm = model
    if model_representation == ModelRepresentation.PROCESS_TREE:
        process_tree = pm4py.convert_to_process_tree(net, im, fm)
        if path:
            pm4py.save_vis_process_tree(process_tree, path)
        else:
            pm4py.view_process_tree(process_tree)

    elif model_representation == ModelRepresentation.BPMN:
        bpmn = pm4py.convert_to_bpmn(net, im, fm)
        if path:
            pm4py.save_vis_bpmn(bpmn, path)
        else:
            pm4py.view_bpmn(bpmn)

    elif model_representation == ModelRepresentation.PETRI_NET:
        if path:
            pm4py.save_vis_petri_net(net, im, fm, path)
        else:
            pm4py.view_petri_net(net, im, fm)


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


def plot_confusion_matrix_image(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: str,
) -> None:
    """Save a simple confusion-matrix heatmap to *output_path*."""
    cm = metrics.confusion_matrix(
        y_true, y_pred, labels=list(range(len(class_names))))
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
    cm = metrics.confusion_matrix(
        y_true, y_pred, labels=list(range(len(class_names))))

    correct_pcts, wrong_pcts = [], []
    for i in range(len(class_names)):
        total = int(cm[i].sum())
        correct = int(cm[i, i])
        correct_pcts.append(correct / total * 100 if total > 0 else 0.0)
        wrong_pcts.append((total - correct) / total *
                          100 if total > 0 else 0.0)

    x = list(range(len(class_names)))
    fig, ax = plt.subplots(figsize=(max(5, len(class_names) * 1.4 + 1), 5))
    ax.bar(x, correct_pcts, label="Correct", color="green", alpha=0.85)
    ax.bar(x, wrong_pcts, bottom=correct_pcts,
           label="Incorrect", color="red", alpha=0.85)

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

    root = pathlib.Path(
        base_dir) if base_dir else pathlib.Path.home() / "Downloads"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = root / f"conformance_results_{timestamp}"
    folder.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix_image(
        y_true, y_pred, class_names, str(folder / "confusion_matrix.png"))
    plot_per_class_bar_chart(y_true, y_pred, class_names, str(
        folder / "per_class_accuracy.png"))

    return str(folder)


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

# ------------------------------------------
# CLI output
# ------------------------------------------


def display_confusion_matrix_text(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> None:
    """Print a formatted confusion matrix to the terminal."""
    cm = metrics.confusion_matrix(
        y_true, y_pred, labels=list(range(len(class_names))))
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
    cm = metrics.confusion_matrix(
        y_true, y_pred, labels=list(range(len(class_names))))

    print("\nPer-Class Prediction Breakdown:")
    for i, name in enumerate(class_names):
        total = int(cm[i].sum())
        correct = int(cm[i, i])
        wrong = total - correct
        pct_c = correct / total * 100 if total > 0 else 0.0
        pct_w = wrong / total * 100 if total > 0 else 0.0
        print(
            f"  {name}: {pct_c:.1f}% correct ({correct}/{total}),  {pct_w:.1f}% incorrect ({wrong}/{total})")
