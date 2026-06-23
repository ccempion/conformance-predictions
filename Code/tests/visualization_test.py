import pytest
import numpy as np
import pandas as pd
import os
import io as python_io
import sys
from conformance_predictions import visualization
from conformance_predictions.evaluation import compute_roc_curve, compute_pr_curve

@pytest.fixture
def dummy_data():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    class_names = ["ClassA", "ClassB"]
    return y_true, y_pred, class_names

def test_plot_confusion_matrix_image(dummy_data, tmp_path):
    y_true, y_pred, class_names = dummy_data
    output_path = str(tmp_path / "cm.png")
    
    visualization.plot_confusion_matrix_image(y_true, y_pred, class_names, output_path)
    
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0

def test_plot_per_class_bar_chart(dummy_data, tmp_path):
    y_true, y_pred, class_names = dummy_data
    output_path = str(tmp_path / "bar.png")
    
    visualization.plot_per_class_bar_chart(y_true, y_pred, class_names, output_path)
    
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0

def test_save_evaluation_visuals(dummy_data, tmp_path):
    y_true, y_pred, class_names = dummy_data
    base_dir = str(tmp_path)
    
    folder_path = visualization.save_evaluation_visuals(y_true, y_pred, class_names, base_dir)
    
    assert os.path.isdir(folder_path)
    assert os.path.exists(os.path.join(folder_path, "confusion_matrix.png"))
    assert os.path.exists(os.path.join(folder_path, "per_class_accuracy.png"))

def test_plot_evaluate_by_prefix_length(tmp_path):
    df = pd.DataFrame({
        "prefix_length": [1, 2, 3],
        "accuracy": [0.7, 0.8, 0.9],
        "f1": [0.65, 0.75, 0.85]
    })
    output_path = str(tmp_path / "prefix.png")
    
    visualization.plot_evaluate_by_prefix_length(df, output_path)
    
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0

def test_plot_processing_times(tmp_path):
    time_list = [("Step1", 1.5), ("Step2", 2.5)]
    output_path = str(tmp_path / "times.png")
    
    visualization.plot_processing_times(time_list, output_path)
    
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0

def test_display_confusion_matrix_text(dummy_data):
    y_true, y_pred, class_names = dummy_data
    
    captured_output = python_io.StringIO()
    sys.stdout = captured_output
    try:
        visualization.display_confusion_matrix_text(y_true, y_pred, class_names)
    finally:
        sys.stdout = sys.__stdout__
    
    output = captured_output.getvalue()
    assert "Confusion Matrix" in output
    assert "ClassA" in output
    assert "ClassB" in output

def test_display_per_class_stats(dummy_data):
    y_true, y_pred, class_names = dummy_data
    
    captured_output = python_io.StringIO()
    sys.stdout = captured_output
    try:
        visualization.display_per_class_stats(y_true, y_pred, class_names)
    finally:
        sys.stdout = sys.__stdout__
    
    output = captured_output.getvalue()
    assert "Per-Class Prediction Breakdown" in output
    assert "ClassA" in output
    assert "ClassB" in output


def test_roc_pr_curves(tmp_path):
    y_true = np.array([0, 0, 1, 1])
    y_scores1 = np.array([0.1, 0.4, 0.35, 0.8])
    y_scores2 = np.array([0.2, 0.3, 0.6, 0.9])
    
    # Test evaluation functions
    fpr1, tpr1 = compute_roc_curve(y_true, y_scores1)
    fpr2, tpr2 = compute_roc_curve(y_true, y_scores2)
    
    prec1, recall1, ap1 = compute_pr_curve(y_true, y_scores1)
    prec2, recall2, ap2 = compute_pr_curve(y_true, y_scores2)
    
    assert len(fpr1) > 0
    assert len(tpr1) == len(fpr1)
    assert ap1 > 0
    
    # Test visualization functions
    roc_data = {
        "Model A": (fpr1, tpr1),
        "Model B": (fpr2, tpr2)
    }
    
    pr_data = {
        "Model A": (prec1, recall1, ap1),
        "Model B": (prec2, recall2, ap2)
    }
    
    roc_path = os.path.join(tmp_path, "roc_test.png")
    pr_path = os.path.join(tmp_path, "pr_test.png")
    
    visualization.plot_roc_curves(roc_data, output_path=roc_path)
    visualization.plot_pr_curves(pr_data, output_path=pr_path)
    
    assert os.path.exists(roc_path)
    assert os.path.exists(pr_path)
    print("ROC and PR curve tests passed and files created.")

