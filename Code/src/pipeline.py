from typing import Tuple, Optional, Dict, Any
import pandas as pd
from pm4py.objects.petri_net.obj import PetriNet, Marking
from src import io, config

def prepare_log_and_model(
    log_path: str,
    model_path: Optional[str] = None,
    discovery_alg: Optional[config.Discovery_alg] = None,
    discovery_params: Optional[Dict[str, Any]] = None,
    vis_params: Optional[Dict[str, Any]] = None
) -> Tuple[Tuple[PetriNet, Marking, Marking], pd.DataFrame]:
    """
    Orchestrates the loading of data, model acquisition (load or discovery),
    and optional visualization.
    
    Returns:
        A tuple containing ((Net, Initial Marking, Final Marking), Event Log DataFrame).
    """
    
    # 1. Load the Event Log
    print(f"--- Loading event log: {log_path} ---")
    log = io.load_event_log(log_path)
    
    # Acquire the Normative Model
    model: Tuple[PetriNet, Marking, Marking]
    
    if model_path:
        print(f"--- Loading provided normative model: {model_path} ---")
        model = io.load_process_model(model_path)
    else:
        if discovery_alg is None:
            raise ValueError("Discovery algorithm must be provided if no model path is specified.")
        
        print(f"--- Discovering model using {discovery_alg.name} ---")
        # Unpack the discovery_params dict into the discover function
        params = discovery_params if discovery_params else {}
        model = io.discover_process_model(log, discovery_alg, **params)
    
    # Handle Initial Visualization
    if vis_params:
        print(f"--- Generating visualization ({vis_params['representation'].name}) ---")
        io.visualize_process_model(
            model=model,
            path=vis_params.get('output_path'),
            model_representation=vis_params['representation']
        )
        
    return model, log