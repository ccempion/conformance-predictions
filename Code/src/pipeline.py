from typing import Tuple, Optional, Dict, Any, List
import pandas as pd
import numpy as np
from pm4py.objects.petri_net.obj import PetriNet, Marking
from src import io, config, feature_extraction, model_training, evaluation

def run_conformance_prediction_pipeline(
    log_path: str,
    model_path: Optional[str] = None,
    discovery_alg: Optional[config.Discovery_alg] = config.Discovery_alg.INDUCTIVE,
    discovery_params: Optional[Dict[str, Any]] = None,
    target: str = "1",
    features: str = "1",
    outcome_activities: Optional[List[str]] = None,
    ml_params: Optional[Dict[str, Any]] = None,
    preloaded_model: Optional[Tuple[PetriNet, Marking, Marking]] = None,
    preloaded_log: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Main orchestration function for the entire pipeline.
    """
    # 1. Map CLI choices to Enums
    target_map = {"1": config.PredictionTarget.FUTURE_DEVIATIONS, "2": config.PredictionTarget.PROCESS_OUTCOME}
    feature_map = {
        "1": config.PredictionFeature.CONTROL_FLOW, 
        "2": config.PredictionFeature.DIAGNOSTICS, 
        "3": config.PredictionFeature.CASE_ATTRIBUTES,
        "4": config.PredictionFeature.HYBRID
    }
    
    target_enum = target_map.get(target, config.PredictionTarget.FUTURE_DEVIATIONS)
    feature_enum = feature_map.get(features, config.PredictionFeature.CONTROL_FLOW)

    # 2. Prepare Data
    if preloaded_model is not None and preloaded_log is not None:
        model = preloaded_model
        log = preloaded_log
    else:
        model, log = prepare_log_and_model(
            log_path=log_path,
            model_path=model_path,
            discovery_alg=discovery_alg,
            discovery_params=discovery_params
        )

    # 3. Feature Extraction
    X, y = extract_target_and_feature(
        target=target_enum,
        feature=feature_enum,
        model=model,
        log=log,
        outcome_activities=outcome_activities
    )

    # 4. Temporal Split (HL-15)
    print("--- Splitting data temporally ---")
    # This is a simplification; evaluation.temporal_split returns case IDs
    # For now, we assume X and y are indexed or aligned such that we can split them
    # In a real implementation, you'd filter X/y based on the returned case IDs
    train_ids, val_ids, test_ids = evaluation.temporal_split(log)
    
    # Placeholder: filter X and y based on IDs (Richard's feature extraction should keep case IDs)
    # X_train, y_train = X.loc[train_ids], y.loc[train_ids]
    # X_test, y_test = X.loc[test_ids], y.loc[test_ids]
    X_train, X_test = X[:int(0.8*len(X))], X[int(0.8*len(X)):] # Simple split for now
    y_train, y_test = y[:int(0.8*len(y))], y[int(0.8*len(y)):]

    # 5. Model Training (HL-08, LL-08.2)
    print(f"--- Training model with parameters: {ml_params} ---")
    trained_model = model_training.train_model(X_train, y_train, ml_params)

    # 6. Evaluation (HL-16)
    print("--- Evaluating model ---")
    y_pred, y_prob = model_training.predict(trained_model, X_test)
    metrics = evaluation.compute_classification_metrics(y_test, y_pred, y_prob)
    
    print(f"Evaluation results: {metrics}")
    
    return {
        "model": trained_model,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test
    }

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