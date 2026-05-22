from typing import Tuple, Optional, Dict, Any, List
import pandas as pd
import numpy as np
from pm4py.objects.petri_net.obj import PetriNet, Marking
import pm4py
from src import io, config, feature_extraction, model_training, evaluation
from src import visualization as vis

def run_conformance_prediction_pipeline(
    X_test=np.ndarray,
    y_test=np.ndarray,
    X_train=np.ndarray,
    y_train=np.ndarray,
    ml_params: Optional[Dict[str, Any]] = None,
    preloaded_model=None,
    test_prefix_lengths: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Main orchestration function for the entire pipeline.
    """

    if preloaded_model is not None:
        print("--- Using pre-trained model ---")
        trained_model = preloaded_model
    else:
        print(f"--- Training model with parameters: {ml_params} ---")
        trained_model = model_training.train_model(X_train, y_train, ml_params)

    # 6. Evaluation (HL-16)
    print("--- Evaluating model ---")

    # Align test columns to training columns — fills any activity/move seen only in
    # one split with 0, preventing shape mismatches in the StandardScaler.
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    y_pred, y_prob = model_training.predict(trained_model, X_test)
    metrics = evaluation.compute_classification_metrics(y_test, y_pred, y_prob)

    print(f"Evaluation results: {metrics}")

    results = {
        "model": trained_model,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
    if test_prefix_lengths is not None:
        results["prefix_lengths"] = test_prefix_lengths
    return results

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
        vis.visualize_process_model(
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
    outcome_activities: Optional[List[str]] = None,
    prefix_length: Optional[int] = None,
    selected_case_attributes: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Orchestrates the extraction of labels (y) and features (X).
    Ensures that the features and targets are aligned by case/prefix.

    Returns:
        X: feature matrix.
        y: encoded label vector.
        class_names: ordered list of original class labels.
    """
    # Determine if alignments are required
    needs_alignments_for_feature = (
        feature in [config.PredictionFeature.DIAGNOSTICS, config.PredictionFeature.HYBRID]
    )
    needs_alignments_for_label = (
        target == config.PredictionTarget.FUTURE_DEVIATIONS
    )

    # Extract Labels from FULL log
    print(f"--- Extracting target: {target.name} ---")
    label_alignments = feature_extraction.compute_alignments(log, model) if needs_alignments_for_label else None
    y_raw = feature_extraction.extract_label(
        log=log,
        target_type=target,
        outcome_activities=outcome_activities,
        alignments=label_alignments
    )

    # Slicing
    if prefix_length is not None:
        print(f"--- Slicing log to prefix length: {prefix_length} ---")
        log_to_feature = pm4py.get_prefixes_from_log(log, length=prefix_length, case_id_key='case:concept:name')
    else:
        log_to_feature = log

    # Extract Features from SLICED log
    print(f"--- Extracting features: {feature.name} ---")
    feature_alignments = feature_extraction.compute_alignments(log_to_feature, model) if needs_alignments_for_feature else None
    X_raw = feature_extraction.extract_features(
        log=log_to_feature,
        feature_type=feature,
        model=model,
        alignments=feature_alignments,
        selected_case_attributes=selected_case_attributes
    )

    # Preprocessing
    print("--- Preprocessing and encoding data ---")
    X, y, class_names = feature_extraction.preprocess_for_ml(X_raw, y_raw)

    return X, y, class_names


def extract_target_and_feature_over_prefixes(
    target: config.PredictionTarget,
    feature: config.PredictionFeature,
    model: Tuple[Any, Any, Any],
    log: pd.DataFrame,
    outcome_activities: Optional[List[str]] = None,
    selected_case_attributes: Optional[List[str]] = None,
    min_length: int = 1,
    max_length: Optional[int] = None,
    step: int = 1,
    gate_outcome: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, List[str], np.ndarray]:
    """
    Prefix-level (pooled) variant of extract_target_and_feature (LL-07.1).

    Expands each case into varying-length prefixes (exactly-k => min==max==k;
    varying 1..k => min=1, max=k), pools them into one (X, y) dataset, and returns
    each row's prefix length as metadata for by-length analysis (HL-17).

    Labels are computed once on the FULL log per original case and broadcast to that
    case's prefixes; for PROCESS_OUTCOME, prefixes are gated to end before the first
    outcome activity so the label is never inside the features.

    Returns:
        X, y, class_names, prefix_lengths   (prefix_lengths row-aligned to X)
    """
    case_id_key = "case:concept:name"

    needs_alignments_for_feature = (
        feature in [config.PredictionFeature.DIAGNOSTICS, config.PredictionFeature.HYBRID]
    )
    needs_alignments_for_label = (
        target == config.PredictionTarget.FUTURE_DEVIATIONS
    )

    # 1. Labels on the FULL log, per original case.
    print(f"--- Extracting target (per case): {target.name} ---")
    label_alignments = feature_extraction.compute_alignments(log, model) if needs_alignments_for_label else None
    y_by_case = feature_extraction.extract_label(
        log=log,
        target_type=target,
        outcome_activities=outcome_activities,
        alignments=label_alignments,
    )

    # 2. Generate prefixes (with leakage gating for outcome prediction).
    gate = outcome_activities if (target == config.PredictionTarget.PROCESS_OUTCOME and gate_outcome) else None
    print(f"--- Generating prefixes (min={min_length}, max={max_length}, step={step}, gate={'on' if gate else 'off'}) ---")
    prefix_log, prefix_index = feature_extraction.generate_prefixes(
        log, min_length=min_length, max_length=max_length, step=step, gate_activities=gate
    )
    if prefix_log.empty:
        raise ValueError("No prefixes generated — check the prefix length range (and outcome gating).")

    # 3. Features on the prefix log (one alignment per prefix when needed).
    print(f"--- Extracting features over {len(prefix_index)} prefixes: {feature.name} ---")
    feature_alignments = feature_extraction.compute_alignments(prefix_log, model) if needs_alignments_for_feature else None
    X_raw = feature_extraction.extract_features(
        log=prefix_log,
        feature_type=feature,
        model=model,
        alignments=feature_alignments,
        selected_case_attributes=selected_case_attributes,
    )

    # 4. Broadcast per-case labels to prefixes; align everything on the prefix id index.
    idx = prefix_index.set_index("prefix_id")
    y_raw = idx["orig_case_id"].map(y_by_case).reindex(X_raw.index)
    prefix_lengths = idx["prefix_length"].reindex(X_raw.index).to_numpy()

    # 5. Encode (now index-aligned by construction).
    print("--- Preprocessing and encoding data ---")
    X, y, class_names = feature_extraction.preprocess_for_ml(X_raw, y_raw)

    return X, y, class_names, prefix_lengths