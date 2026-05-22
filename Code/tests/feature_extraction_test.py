import pytest
import pandas as pd
import numpy as np
from src.feature_extraction import extract_label, preprocess_for_ml
from src.config import PredictionTarget

def test_extract_label_process_outcome():
    d = {'case:concept:name': ['1', '1', '2', '2'], 
         'concept:name': ['A', 'Approve', 'A', 'Reject'], 
         'time:timestamp': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:01:00', 
                                         '2023-01-01 00:02:00', '2023-01-01 00:03:00'])}
    df = pd.DataFrame(data=d)
    
    outcomes = extract_label(df, PredictionTarget.PROCESS_OUTCOME, outcome_activities=['Approve', 'Reject'])
    
    assert outcomes['1'] == 'Approve'
    assert outcomes['2'] == 'Reject'

def test_preprocess_for_ml():
    X_raw = pd.DataFrame({'feat1': [1, 2], 'feat2': [0.1, 0.2]}, index=['1', '2'])
    y_raw = pd.Series(['A', 'B'], index=['1', '2'])
    
    X, y, class_names = preprocess_for_ml(X_raw, y_raw)
    
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, np.ndarray)
    assert len(class_names) == 2
    assert 'A' in class_names
    assert 'B' in class_names
