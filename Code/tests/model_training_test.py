import pytest
import pandas as pd
import numpy as np
from src.model_training import (
    predict,
    create_regression_pipeline, train_regression_model, predict_regression,
    create_classification_pipeline, train_classification_model,
)
from src.config import RegressionAlgorithm, ClassificationAlgorithm


# --------------------------------------------------------------------------
# Regression (REMAINING_TIME target)
# --------------------------------------------------------------------------

def test_create_regression_pipeline():
    pipe = create_regression_pipeline()
    assert 'scaler' in pipe.named_steps
    assert 'regressor' in pipe.named_steps     # distinct from the classifier step name


@pytest.mark.parametrize("algorithm", list(RegressionAlgorithm))
def test_train_and_predict_regression(algorithm):
    # Continuous target (e.g. remaining seconds) roughly proportional to feat1.
    X_train = pd.DataFrame({'feat1': [1, 2, 3, 4, 5], 'feat2': [0.1, 0.2, 0.3, 0.4, 0.5]})
    y_train = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

    model = train_regression_model(X_train, y_train, algorithm)

    X_test = pd.DataFrame({'feat1': [1.5, 4.5], 'feat2': [0.15, 0.45]})
    y_pred = predict_regression(model, X_test)

    assert len(y_pred) == 2
    assert np.issubdtype(y_pred.dtype, np.floating)   # continuous output, not class ids


# --------------------------------------------------------------------------
# Classification menu (FUTURE_DEVIATIONS risk prediction)
# --------------------------------------------------------------------------

def test_create_classification_pipeline():
    pipe = create_classification_pipeline(ClassificationAlgorithm.LOGISTIC_REGRESSION)
    assert 'scaler' in pipe.named_steps
    assert 'classifier' in pipe.named_steps     # so predict()/visualize_decision_tree work


@pytest.mark.parametrize("algorithm", [
    ClassificationAlgorithm.DECISION_TREE,
    ClassificationAlgorithm.LOGISTIC_REGRESSION,
    ClassificationAlgorithm.GRADIENT_BOOSTING,
])
def test_train_and_predict_classification(algorithm):
    X_train = pd.DataFrame({'feat1': [1, 2, 3, 4, 5, 6], 'feat2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
    y_train = np.array([0, 0, 0, 1, 1, 1])

    model = train_classification_model(X_train, y_train, algorithm)

    X_test = pd.DataFrame({'feat1': [1.5, 5.5], 'feat2': [0.15, 0.55]})
    y_pred, y_prob = predict(model, X_test)

    assert len(y_pred) == 2 and len(y_prob) == 2
    assert set(np.unique(y_pred)) <= {0, 1}
    assert all(0.0 <= float(p) <= 1.0 for p in y_prob)   # risk probabilities in [0, 1]
