"""
The input module handles the loading of event logs and process models, as well as model discovery from event logs.
"""

import pandas as pd
import pm4py
from typing import Tuple
from pm4py.objects.petri_net.obj import PetriNet, Marking

from src.config import Discovery_alg

def load_event_log(file_path: str) -> pd.DataFrame:
    """
    Parses event log data from XES or CSV files into a DataFrame (LL-05.1).
    
    Args:
        file_path: Path to the log file.
        
    Returns:
        A pandas DataFrame formatted for PM4Py.
    """
    if file_path.endswith('.xes'):
        return pm4py.read_xes(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        return pm4py.format_dataframe(df, case_id='case:concept:name', 
                                     activity_key='concept:name', 
                                     timestamp_key='time:timestamp')
    else:
        raise ValueError(f"Unsupported log format: {file_path}")

def load_process_model(file_path: str) -> Tuple[PetriNet, Marking, Marking]:
    """
    Loads a Petri net or BPMN representation from a file (LL-05.2).
    
    Args:
        file_path: Path to the model file (.pnml or .bpmn).
        
    Returns:
        A tuple containing (Petri net, initial marking, final marking).
    """
    if file_path.endswith('.pnml'):
        return pm4py.read_pnml(file_path)
    elif file_path.endswith('.bpmn'):
        # Internally converts BPMN to Petri net for conformance checking
        bpmn_graph = pm4py.read_bpmn(file_path)
        return pm4py.convert_to_petri_net(bpmn_graph)
    elif file_path.endswith('.ptml'):
        # Internally converts Process Tree to Petri net for conformance checking
        ptml_graph= pm4py.read_ptml(file_path)
        return pm4py.convert_to_petri_net(ptml_graph)
    else:
        raise ValueError(f"Unsupported model format: {file_path}")

def discover_process_model(log: pd.DataFrame,
                           discovery_alg: Discovery_alg,
                           **kwargs
                           ) -> Tuple[PetriNet, Marking, Marking]:
    """
    Discovers a Petri net from event log data using Inductive / Alpha / Genetic Miner (LL-05.3).
    
    Args:
        log: The event log DataFrame.
        noise_threshold: Threshold for the inductive miner.
        **kwargs: Algorithm-specific parameters (e.g. noise_threshold for Inductive Miner)
        
    Returns:
        A tuple containing (Petri net, initial marking, final marking).
    """
    if discovery_alg == Discovery_alg.INDUCTIVE:
        # params: noise_threshold
        return pm4py.discover_petri_net_inductive(log, **kwargs)
    
    elif discovery_alg == Discovery_alg.GENETIC:
        # params: population_size, elitism_rate, crossover_rate, mutation_rate, generations, elitism_min_sample
        return pm4py.discover_petri_net_genetic(log, **kwargs)
    
    elif discovery_alg == Discovery_alg.ALPHA:
        return pm4py.discover_petri_net_alpha(log, **kwargs)
    
    else:
        raise TypeError(f"Unsupported parameters or algorithm: {kwargs}, {discovery_alg}")
