"""
In this module, all visualizations, that can be created for our conformance predictions project, are bundled. This includes a great variety of different visualizations.
The are mainly created using matplotlib or built-in visualizations from other libraries that we listed in our requirements engineering document.
"""

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

def visualize_process_model(model: Tuple[PetriNet, Marking, Marking], output_path: Optional[str], model_representation: ModelRepresentation):
    """Visualize a process model with PM4Py.

    Args:
        model (Tuple[PetriNet, Marking, Marking]): Petri Net as internal (PM4Py) representation.
        output_path (Optional[str]): Path where output of visualizations shall be saved to.
        model_representation (ModelRepresentation): Desired model representation (BPMN / Petri Net / Process Tree).
    """
    net, im, fm = model
    if model_representation == ModelRepresentation.PROCESS_TREE:
        process_tree = pm4py.convert_to_process_tree(net, im, fm)
        if output_path:
            pm4py.save_vis_process_tree(process_tree, output_path)
        else:
            pm4py.view_process_tree(process_tree)

    elif model_representation == ModelRepresentation.BPMN:
        bpmn = pm4py.convert_to_bpmn(net, im, fm)
        if output_path:
            pm4py.save_vis_bpmn(bpmn, output_path)
        else:
            pm4py.view_bpmn(bpmn)

    elif model_representation == ModelRepresentation.PETRI_NET:
        if output_path:
            pm4py.save_vis_petri_net(net, im, fm, output_path)
        else:
            pm4py.view_petri_net(net, im, fm)


def visualize_decision_tree(
    model: Pipeline,
    feature_names: List[str],
    class_names: List[str],
    output_path: Optional[str],
) -> None:
    """Renders the fitted Decision Tree from the pipeline.
    If output_path is given, saves to that file; otherwise opens an interactive window.

    Args:
        model (Pipeline): Sklearn pipeline object. 
        feature_names (List[str]): Names of features. 
        class_names (List[str]): Names of target classes.
        output_path (Optional[str], optional): Path where output of visualizations shall be saved to.
    """
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
    output_path: Optional[str],
) -> None:
    """Plots a atandard confusion matrix and saves it to the specified output path.

    Args:
        y_true (np.ndarray): Target vector.
        y_pred (np.ndarray): Predicted vector.
        class_names (List[str]): Class names of targets.
        output_path (Optional[str]): Path where output of visualizations shall be saved to.
    """
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
    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_per_class_bar_chart(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: Optional[str],
) -> None:
    """Save a stacked bar chart (green = correct, red = incorrect) per class.

    Args:
        y_true (np.ndarray): Target vector.
        y_pred (np.ndarray): Predicted vector.
        class_names (List[str]): Class names of targets.
        output_path (Optional[str]): Path where output of visualizations shall be saved to.
    """
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
    if output_path: 
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


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


def save_curve_visuals(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    pr_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    base_dir: Optional[str] = None,
) -> str:
    """Create a timestamped folder and save ROC and PR curves inside it.

    Returns the absolute path of the created folder.
    """
    import pathlib
    import datetime

    root = pathlib.Path(base_dir) if base_dir else pathlib.Path.home() / "Downloads"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = root / f"conformance_curves_{timestamp}"
    folder.mkdir(parents=True, exist_ok=True)

    plot_roc_curves(roc_data, output_path=str(folder / "roc_curves.png"))
    plot_pr_curves(pr_data, output_path=str(folder / "pr_curves.png"))

    return str(folder)


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Optional[str] = None,
    unit_divisor: float = 1.0,
    unit_label: str = "seconds",
) -> None:
    """Scatter of actual vs. predicted remaining time, with a y=x reference line.

    Points hugging the diagonal mean accurate predictions; systematic departure reveals
    over-/under-estimation. Values are divided by ``unit_divisor`` so the axes read in a
    human unit (see ``evaluation.pick_duration_unit``).

    Args:
        y_true (np.ndarray): True remaining time in seconds.
        y_pred (np.ndarray): Predicted remaining time in seconds.
        output_path (Optional[str], optional): Path to save to; if None, show interactively.
        unit_divisor (float, optional): Seconds-per-unit divisor. Defaults to 1.0.
        unit_label (str, optional): Display unit label. Defaults to "seconds".
    """
    y_true = np.asarray(y_true, dtype=float) / unit_divisor
    y_pred = np.asarray(y_pred, dtype=float) / unit_divisor

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.4, edgecolors='none', color="steelblue")

    lo = float(min(y_true.min(), y_pred.min())) if y_true.size else 0.0
    hi = float(max(y_true.max(), y_pred.max())) if y_true.size else 1.0
    ax.plot([lo, hi], [lo, hi], color='navy', linestyle='--', label='Perfect prediction (y=x)')

    ax.set_xlabel(f"Actual remaining time ({unit_label})")
    ax.set_ylabel(f"Predicted remaining time ({unit_label})")
    ax.set_title("Predicted vs. Actual Remaining Time")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_residual_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Optional[str] = None,
    unit_divisor: float = 1.0,
    unit_label: str = "seconds",
) -> None:
    """Histogram of residuals (predicted - actual), centred on zero.

    A residual cloud centred and symmetric around 0 indicates an unbiased model; a shifted
    centre reveals systematic over-/under-estimation. Residuals are divided by
    ``unit_divisor`` to read in a human unit.

    Args:
        y_true (np.ndarray): True remaining time in seconds.
        y_pred (np.ndarray): Predicted remaining time in seconds.
        output_path (Optional[str], optional): Path to save to; if None, show interactively.
        unit_divisor (float, optional): Seconds-per-unit divisor. Defaults to 1.0.
        unit_label (str, optional): Display unit label. Defaults to "seconds".
    """
    residuals = (np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)) / unit_divisor

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(residuals, bins=40, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(0.0, color='navy', linestyle='--', label='Zero error')

    ax.set_xlabel(f"Residual (predicted - actual) ({unit_label})")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def save_regression_visuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    base_dir: Optional[str] = None,
    unit_divisor: float = 1.0,
    unit_label: str = "seconds",
) -> str:
    """Create a timestamped folder and save predicted-vs-actual + residual plots inside it.

    Regression counterpart of ``save_evaluation_visuals``. Returns the absolute path of the
    created folder.
    """
    import pathlib
    import datetime

    root = pathlib.Path(base_dir) if base_dir else pathlib.Path.home() / "Downloads"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = root / f"regression_results_{timestamp}"
    folder.mkdir(parents=True, exist_ok=True)

    plot_predicted_vs_actual(
        y_true, y_pred, str(folder / "predicted_vs_actual.png"), unit_divisor, unit_label)
    plot_residual_distribution(
        y_true, y_pred, str(folder / "residuals.png"), unit_divisor, unit_label)

    return str(folder)


def plot_roc_curves(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Optional[str] = None,
) -> None:
    """Plot overlaid ROC curves for multiple models.

    Args:
        roc_data (Dict[str, Tuple[np.ndarray, np.ndarray]]): Mapping of label -> list of (false positive rate, true positive rate) tuples.
        output_path (Optional[str], optional): Path where output of visualizations shall be saved to. Defaults to None.
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
        pr_data (Dict[str, Tuple[np.ndarray, np.ndarray]]): Mapping of label -> (precision, recall, average_precision)
        output_path (Optional[str], optional): Path where output of visualizations shall be saved to. If None, show interactively.
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
    unit_divisor: float = 1.0,
    unit_label: str = "seconds",
) -> None:
    """
    Plots the results of metrics_by_prefix_length.
    X-axis: prefix_length. Y-axis: Metrics (Accuracy/F1 for clf, MAE/RMSE for reg).

    For regression, error columns (MAE/RMSE) are divided by ``unit_divisor`` so the
    y-axis reads in a human unit (see ``evaluation.pick_duration_unit``).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if "accuracy" in results_df.columns:
        ax.plot(results_df["prefix_length"], results_df["accuracy"], marker='o', label="Accuracy", color="blue")
    if "f1" in results_df.columns:
        ax.plot(results_df["prefix_length"], results_df["f1"], marker='o', label="F1 Score", color="green")

    if "MAE" in results_df.columns:
        ax.plot(results_df["prefix_length"], results_df["MAE"] / unit_divisor, marker='s', label="MAE", color="red")
        if "RMSE" in results_df.columns:
            ax.plot(results_df["prefix_length"], results_df["RMSE"] / unit_divisor, marker='^', label="RMSE", color="darkorange")
        ax.set_ylabel(f"Error ({unit_label})")
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
    """Creates a pie chart of processing times for different pipeline steps.
    If output_path is provided, saves to that path; otherwise displays interactively.


    Args:
        time_list (List[Tuple[str, float]]): List of keyed times: (function name, time).
        output_path (Optional[str], optional): Path where output of visualizations shall be saved to. Defaults to None.
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

    ax.legend(
        wedges, 
        labels,
        title="Pipeline Steps",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )

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
    """Plots a simple line graph of log size vs runtime.

    Args:
        log_sizes (List[float]): List of log sizes (in MB). 
        runtimes (List[float]): List of runtimes per log. 
        output_path (str, optional): Path where output of visualizations shall be saved to. Defaults to None.
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
    importances: pd.DataFrame,
    output_path: Optional[str] = None,
) -> None:
    """Renders a bar chart of the top predictive features.
    
    Color-codes features based on their origin:
    - Green: Conformance-derived (sync_move, log_move, model_move)
    - Skyblue: Control-flow (cf_act)
    - Lightgray: Case attributes (ca)

    Args:
        importances (pd.DataFrame): Dataframe of ranked feature importances.
        output_path (Optional[str], optional): Path to save the plot. Defaults to None.
    """
    if importances.empty:
        print("No feature importance data to plot.")
        return

    # Sort by importance ascending for the horizontal bar chart
    df = importances.sort_values('importance', ascending=True)

    # Assign colors based on feature prefix
    colors = []
    for feat in df.index:
        if any(x in feat for x in ['sync_move_', 'log_move_', 'model_move_']):
            colors.append('green')       # Conformance-derived
        elif feat.startswith('cf_act_'):
            colors.append('skyblue')     # Control-flow
        elif feat.startswith('ca_'):
            colors.append('lightgray')   # Case attributes
        else:
            colors.append('gray')        # Others (e.g. temporal features)

    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.5)))
    ax.barh(df.index, df['importance'], color=colors, alpha=0.8)
    
    ax.set_xlabel('Importance Score (normalized)')
    ax.set_title('Top Predictive Features')
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Create a custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=6, label='Conformance-derived'),
        Line2D([0], [0], color='skyblue', lw=6, label='Control-flow'),
        Line2D([0], [0], color='lightgray', lw=6, label='Case attributes')
    ]
    if 'gray' in colors:
         legend_elements.append(Line2D([0], [0], color='gray', lw=6, label='Other'))
         
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


# ------------------------------------------
# CLI output
# ------------------------------------------


def display_confusion_matrix_text(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> None:
    """Print a formatted confusion matrix to the terminal.

    Args:
        y_true (np.ndarray): Target vector. 
        y_pred (np.ndarray): Predicted vector. 
        class_names (List[str]): Target class names. 
    """
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
    """Print per-class correct / incorrect percentages to the terminal.

    Args:
        y_true (np.ndarray): Target vector. 
        y_pred (np.ndarray): Predicted vector.
        class_names (List[str]): Target class names. 
    """
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

