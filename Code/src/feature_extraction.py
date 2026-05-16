from typing import List, Dict, Optional, Any, Tuple

import pandas as pd
import numpy as np
import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking

from src.config import PredictionTarget, PredictionFeature
from src import config


def compute_alignments(log: pd.DataFrame, model: Tuple[PetriNet, Marking, Marking]) -> List[Dict[str, Any]]:
    """
    Computes optimal alignments between the log and the model (Petri Net).
    Returns the standard PM4Py alignment output (list of dictionaries).
    """
    (net, im, fm) = model
    return pm4py.conformance_diagnostics_alignments(log, net, im, fm)

def extract_label(
    log: pd.DataFrame, 
    target_type: PredictionTarget, 
    outcome_activities: Optional[List[str]] = None,
    alignments: Optional[List[Dict[str, Any]]] = None
) -> pd.Series:
    """
    Derives labels for each case.
    - PROCESS_OUTCOME: Checks if any activity in 'outcome_activities' exists in the trace.
    - FUTURE_DEVIATIONS: Uses 'alignments' to check for future move-on-log or move-on-model.
    - REMAINING_TIME: Calculates time delta from current event to trace-end.
    """
    case_id_key = 'case:concept:name' # PM4Py standard case ID column
    activity_key = 'concept:name'
    timestamp_key = 'time:timestamp'

    if target_type == config.PredictionTarget.PROCESS_OUTCOME:
        if not outcome_activities:
            raise ValueError("Process Outcome target requires a list of target activities.")
        
        outcome_set = set(outcome_activities)
        
        def determine_outcome(activities: pd.Series) -> str:
            # Find intersection between trace activities and target outcomes
            present_outcomes = set(activities).intersection(outcome_set)
            
            if len(present_outcomes) == 1:
                return present_outcomes.pop() # Exactly one outcome found
            else:
                return "uncertain_outcome" # 0 or >1 outcomes found
                
        # Group by case and apply the outcome logic
        return log.groupby(case_id_key)[activity_key].apply(determine_outcome)

    elif target_type == config.PredictionTarget.FUTURE_DEVIATIONS:
        if alignments is None:
            raise ValueError("Future Deviations target requires alignment results.")
        case_ids = log[case_id_key].unique()
        
        if len(case_ids) != len(alignments):
            raise ValueError("Number of alignments does not match number of unique cases in the log.")
            
        deviations = []
        for align_dict in alignments:
             # PM4Py alignments contain 'alignment', a list of tuples: (log_move, model_move)
             # A deviation is present if either the log move or model move is '>>' (skip)
            alignment_moves = align_dict.get('alignment', [])
            has_deviation = any(
                log_move == '>>' or model_move == '>>' 
                for log_move, model_move in alignment_moves
             )
             # You can also return a count by changing this to a sum() if you prefer regression
            deviations.append(int(has_deviation)) 
            
        return pd.Series(deviations, index=case_ids, name="future_deviations")
        #return None

    elif target_type == config.PredictionTarget.REMAINING_TIME:
        # Calculates the total duration of the case in seconds
        # (Assuming the log passed here represents the full trace or prefix to evaluate)
        # def calc_duration(timestamps: pd.Series) -> float:
        #     return (timestamps.max() - timestamps.min()).total_seconds()
            
        # return log.groupby(case_id_key)[timestamp_key].apply(calc_duration).rename("remaining_time_seconds")
        return None

    else:
        raise NotImplementedError(f"Target type {target_type} is not supported.")
        
def extract_features(    log: pd.DataFrame, 
    feature_type: PredictionFeature,
    model: Optional[Tuple] = None,
    alignments: Optional[List[Dict[str, Any]]] = None
) -> pd.DataFrame:
    """
    Extracts features. 
    - CONTROL_FLOW: Extracts the trace prefix (Bag-of-activities).
    - DIAGNOSTICS: Extracts the alignment-based trace prefix diagnostics (Bag-of-move-types).
    - CASE_ATTRIBUTES: Extracts static attributes about the case.
    - HYBRID: Combines the active feature sets.
    """
    case_id_key = 'case:concept:name'
    activity_key = 'concept:name'
    
    feature_matrices = []
    
    # CONTROL_FLOW: Bag-of-Activities
    if feature_type in [config.PredictionFeature.CONTROL_FLOW, config.PredictionFeature.HYBRID]:
        # pd.crosstab -> frequency matrix (rows=cases, cols=activities)
        cf_df = pd.crosstab(log[case_id_key], log[activity_key])
        
        # Prefix columns to avoid naming collisions with case attributes later
        cf_df.columns = [f"cf_act_{col}" for col in cf_df.columns]
        feature_matrices.append(cf_df)

    # DIAGNOSTICS: Bag-of-[activity x move_type]
    if feature_type in [config.PredictionFeature.DIAGNOSTICS, config.PredictionFeature.HYBRID]:
        if alignments is None:
            raise ValueError(f"{feature_type.name} requires alignment results.")
            
        case_ids = log[case_id_key].unique()
        if len(case_ids) != len(alignments):
            raise ValueError("Number of alignments does not match unique cases in log.")
            
        diag_records = []
        for case_id, align_dict in zip(case_ids, alignments):
            record = {'case_id': case_id}
            alignment_moves = align_dict.get('alignment', [])
            
            for log_move, model_move in alignment_moves:
                # Determine the specific move type and activity
                if log_move == model_move and log_move != '>>':
                    feat_name = f"sync_move_{log_move}"
                elif model_move == '>>' and log_move != None:
                    feat_name = f"log_move_{log_move}"
                elif log_move == '>>' and model_move != None:
                    feat_name = f"model_move_{model_move}"
                else:
                    # Invisible transitions
                    continue 
                
                record[feat_name] = record.get(feat_name, 0) + 1
                
            diag_records.append(record)
            
        # Convert to DataFrame
        diag_df = pd.DataFrame(diag_records).set_index('case_id').fillna(0).astype(int)
        feature_matrices.append(diag_df)

    # # CASE_ATTRIBUTES: Static trace-level data
    # if feature_type in [config.PredictionFeature.CASE_ATTRIBUTES, config.PredictionFeature.HYBRID]:
    #     # Find all columns that start with 'case:' (standard PM4Py notation)
    #     case_cols = [col for col in log.columns if col.startswith('case:') and col != case_id_key]
        
    #     if case_cols:
    #         # Since trace attributes are static, taking the first value per case is sufficient
    #         ca_df = log.groupby(case_id_key)[case_cols].first()
    #     else:
    #         # Fallback if the log has no case attributes
    #         ca_df = pd.DataFrame(index=log[case_id_key].unique())
            
    #     feature_matrices.append(ca_df)

    if not feature_matrices:
        raise ValueError(f"Unsupported feature type: {feature_type}")
        
    # Concatenate all active feature matrices side-by-side (to create the hybrid features), because they all share the case_id as the index, Pandas aligns them perfectly.
    final_df = pd.concat(feature_matrices, axis=1)
    
    return final_df

from sklearn.preprocessing import LabelEncoder

def preprocess_for_ml(X_raw: pd.DataFrame, y_raw: pd.Series) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Encodes and scales the raw data for ML training.

    Returns:
        X: feature DataFrame (columns preserved for downstream alignment).
        y: encoded label vector as numpy array.
        class_names: ordered list of original class labels (index i maps to encoded integer i).
    """

    # Feature Preprocessing (X)
    # Converts categorical Case Attributes (strings) into numeric One-Hot encoded columns.
    # Numerical features (Control Flow/Diagnostics) remain unchanged.
    X = pd.get_dummies(X_raw, drop_first=True)


    # Label Preprocessing (y): PROCESS_OUTCOME
    # Since labels are strings (e.g., 'Approve', 'uncertain_outcome'), we map them to integers (0, 1, 2...).
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = [str(c) for c in le.classes_]

    # Handling for Remaining Time (Regression)
    # pass

    # Handling for Future Deviations (Binary)
    # pass

    return X, y, class_names