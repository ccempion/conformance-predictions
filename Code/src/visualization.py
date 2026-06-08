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
    # Support both the classification pipeline ('classifier' step) and the
    # remaining-time regression pipeline ('regressor' step).
    steps = model.named_steps
    estimator = steps.get('classifier') or steps.get('regressor')
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        estimator,
        feature_names=feature_names,
        class_names=class_names or None,  # regressors have no class names
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
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Optional[str] = None,
) -> None:
    """Plot overlaid ROC curves for multiple models.
    
    Args:
        roc_data: Mapping label -> (fpr, tpr)
        output_path: Path to save the figure. If None, show interactively.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for name, (fpr, tpr) in roc_data.items():
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(
            fpr=fpr, tpr=tpr, roc_auc=roc_auc, name=name
        )
        display.plot(ax=ax)
    
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Chance')
    ax.set_title("Receiver Operating Characteristic (ROC) Curves")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_pr_curves(
    pr_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    output_path: Optional[str] = None,
) -> None:
    """Plot overlaid Precision-Recall curves for multiple models.
    
    Args:
        pr_data: Mapping label -> (precision, recall, average_precision)
        output_path: Path to save the figure. If None, show interactively.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for name, (precision, recall, avg_prec) in pr_data.items():
        display = metrics.PrecisionRecallDisplay(
            precision=precision, recall=recall, 
            average_precision=avg_prec, name=name
        )
        display.plot(ax=ax)
    
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="lower left")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()



def plot_evaluate_by_prefix_length(
    results_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> None:
    """
    Plots the results of metrics_by_prefix_length.
    X-axis: prefix_length. Y-axis: Metrics (Accuracy/F1 for clf, MAE for reg).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if "accuracy" in results_df.columns:
        ax.plot(results_df["prefix_length"], results_df["accuracy"], marker='o', label="Accuracy", color="blue")
    if "f1" in results_df.columns:
        ax.plot(results_df["prefix_length"], results_df["f1"], marker='o', label="F1 Score", color="green")
    
    if "MAE" in results_df.columns:
        ax.plot(results_df["prefix_length"], results_df["MAE"], marker='s', label="MAE", color="red")
        ax.set_ylabel("Mean Absolute Error (seconds)")
    else:
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
    
    ax.set_xlabel("Prefix Length")
    ax.set_title("Model Performance by Prefix Length")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_processing_times(
    time_list: List[Tuple[str, float]],
    output_path: Optional[str] = None,
) -> None:
    """
    Creates a pie chart of processing times for different pipeline steps.
    If output_path is provided, saves to that path; otherwise displays interactively.
    """
    if not time_list:
        print("No timing data available to visualize.")
        return

    labels = [f"{name} ({duration:.2f}s)" for name, duration in time_list]
    sizes = [duration for _, duration in time_list]

    # Handle cases where total time is 0 to avoid ZeroDivisionError in pie chart
    if sum(sizes) == 0:
        print("Total processing time is zero, skipping pie chart.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Use a visually pleasing color palette
    colors = plt.cm.Paired(np.linspace(0, 1, len(time_list)))
    
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=None,
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors,
        pctdistance=0.85,
        explode=[0.05] * len(time_list)  # Slight separation for better visibility
    )

    # Add a legend
    ax.legend(
        wedges, 
        labels,
        title="Pipeline Steps",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )

    # Formatting
    plt.setp(autotexts, size=10, weight="bold")
    ax.set_title("Processing Time Distribution per Pipeline Step", pad=20)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Processing time pie chart saved to: {output_path}")
    else:
        plt.show()


def plot_runtime_log_size_relation(log_sizes: List[float], runtimes: List[float], output_path: str = None):
    """
    Plots a simple line graph of log size vs runtime.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(log_sizes, runtimes, marker="o", linestyle="-", label="Total Runtime")
    
    ax.set_xlabel("Log Size")
    ax.set_ylabel("Total Runtime")
    ax.set_title("Relation of Total Runtime and Log Sizes")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

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

