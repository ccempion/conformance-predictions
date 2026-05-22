import pytest
import numpy as np
import pandas as pd
from src.evaluation import compute_classification_metrics, temporal_split

def test_compute_classification_metrics():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.6, 0.8])
    
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    
    assert 'accuracy' in metrics
    assert 'f1' in metrics
    assert 'roc_auc' in metrics
    # 3 out of 4 labels are corretly predicted
    assert metrics['accuracy'] == 0.75

def test_temporal_split():
    # create sample log
    case_ids = [str(i) for i in range(10)]
    timestamps = pd.to_datetime([f'2023-01-{i+1:02d}' for i in range(10)])
    df = pd.DataFrame({
        'case:concept:name': case_ids,
        'time:timestamp': timestamps
    })
    
    train, val, test = temporal_split(df, train_ratio=0.6, val_ratio=0.2)
    
    assert len(train) == 6
    assert len(val) == 2
    assert len(test) == 2
    assert train.isdisjoint(val)
    assert val.isdisjoint(test)
    assert len(train | val | test) == 10
