"""
Configuration module that holds static enum classes for better code interoperability between modules.
"""

from enum import Enum

class Discovery_alg(Enum):
    ALPHA = "ALPHA"
    INDUCTIVE = "INDUCTIVE"
    GENETIC = "GENETIC"

class ModelRepresentation(Enum):
    PROCESS_TREE = "PROCESS_TREE"
    BPMN = "BPMN"
    PETRI_NET = "PETRI_NET"

class PredictionTarget(Enum):
    PROCESS_OUTCOME = "PROCESS_OUTCOME"
    REMAINING_TIME = "REMAINING_TIME"
    FUTURE_DEVIATIONS = "FUTURE_DEVIATIONS"

class PredictionFeature(Enum):
    CONTROL_FLOW = "CONTROL_FLOW"
    DIAGNOSTICS = "DIAGNOSTICS"
    CASE_ATTRIBUTES = "CASE_ATTRIBUTES"
    HYBRID = "HYBRID"

class RegressionAlgorithm(Enum):
    DECISION_TREE = "DECISION_TREE"
    RANDOM_FOREST = "RANDOM_FOREST"
    GRADIENT_BOOSTING = "GRADIENT_BOOSTING"

class ClassificationAlgorithm(Enum):
    DECISION_TREE = "DECISION_TREE"
    LOGISTIC_REGRESSION = "LOGISTIC_REGRESSION"
    GRADIENT_BOOSTING = "GRADIENT_BOOSTING"