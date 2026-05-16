import pytest
import pandas as pd
import numpy as np
from src.model_training import create_model_pipeline, train_model, predict

def test_create_model_pipeline():
    pipeline = create_model_pipeline()
    assert 'scaler' in pipeline.named_steps
    assert 'classifier' in pipeline.named_steps

def test_train_and_predict():
    X_train = pd.DataFrame({'feat1': [1, 2, 3, 4], 'feat2': [0.1, 0.2, 0.3, 0.4]})
    y_train = pd.Series([0, 0, 1, 1])
    
    model = train_model(X_train, y_train)
    
    X_test = pd.DataFrame({'feat1': [1.5, 3.5], 'feat2': [0.15, 0.35]})
    y_pred, y_prob = predict(model, X_test)
    
    assert len(y_pred) == 2
    assert len(y_prob) == 2
    assert y_pred[0] in [0, 1]
