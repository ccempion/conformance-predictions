from typing import Tuple, Optional, Dict, Any, List
import pandas as pd
import numpy as np
from pm4py.objects.petri_net.obj import PetriNet, Marking
from src import io, config, feature_extraction

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
    
    # Load the Event Log
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


def extract_target_and_feature(
    target: config.PredictionTarget,
    feature: config.PredictionFeature,
    model: Tuple[Any, Any, Any],
    log: pd.DataFrame,
    outcome_activities: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Orchestrates the extraction of labels (y) and features (X).
    Ensures that the features and targets are aligned by case/prefix.
    """
    # Determine if alignments are required
    needs_alignments = (
        feature in [config.PredictionFeature.DIAGNOSTICS, config.PredictionFeature.HYBRID] or
        target == config.PredictionTarget.FUTURE_DEVIATIONS
    )

    alignments = feature_extraction.compute_alignments(log, model) if needs_alignments else None
    
    # Extract Labels
    print(f"--- Extracting target: {target.name} ---")
    y_raw = feature_extraction.extract_label(
        log=log, 
        target_type=target, 
        outcome_activities=outcome_activities,
        alignments=alignments
    )

    # Extract Features
    print(f"--- Extracting features: {feature.name} ---")
    X_raw = feature_extraction.extract_features(
        log=log, 
        feature_type=feature, 
        model=model,
        alignments=alignments
    )

    # Preprocessing
    print("--- Preprocessing and encoding data ---")
    X, y = feature_extraction.preprocess_for_ml(X_raw, y_raw)
    
    return X, y