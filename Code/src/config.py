from enum import Enum

class Discovery_alg(Enum):
    ALPHA = "ALPHA",
    INDUCTIVE = "INDUCTIVE",
    GENETIC = "GENETIC",

class ModelRepresentation(Enum):
    PROCESS_TREE = "PROCESS_TREE",
    BPMN = "BPMN",
    PETRI_NET = "PETRI_NET",