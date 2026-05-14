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