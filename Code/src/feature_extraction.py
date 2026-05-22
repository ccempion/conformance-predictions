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


def generate_prefixes(
    log: pd.DataFrame,
    min_length: int = 1,
    max_length: Optional[int] = None,
    step: int = 1,
    gate_activities: Optional[List[str]] = None,
    case_id_key: str = "case:concept:name",
    timestamp_key: str = "time:timestamp",
    activity_key: str = "concept:name",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Expand each case into varying-length prefix pseudo-cases (LL-07.1).

    For each case and each k in range(min_length, max_length+1, step) with k <= case length,
    the first k events (time-ordered) become a pseudo-case whose id is f"{orig}@@k{k}".
    This lets all existing feature/alignment/label code (which keys off case_id_key) run
    unchanged — each prefix simply looks like another case.

    Leakage gating: if gate_activities is given (PROCESS_OUTCOME), only prefixes that end
    strictly before the first occurrence of any gate activity are emitted, so the outcome is
    never already inside the prefix.

    Returns:
        prefix_log:   event-level DataFrame, case_id_key replaced by the unique prefix id,
                      every other column (events, case attributes, timestamp) preserved.
        prefix_index: DataFrame [prefix_id, orig_case_id, prefix_length].
    """
    # Order events within each case; mergesort keeps original order for equal timestamps.
    g = log.sort_values([case_id_key, timestamp_key], kind="mergesort").copy()
    g["_ev"] = g.groupby(case_id_key).cumcount()          # 0-based event position
    case_len = g.groupby(case_id_key)["_ev"].max() + 1     # Series indexed by case id

    # Per-case cap on prefix length. With gating, cap at the position of the first outcome
    # event: a length-k prefix covers events 0..k-1, so k <= first_pos excludes the outcome.
    if gate_activities:
        is_gate = g[activity_key].isin(set(gate_activities))
        gate_pos = g.loc[is_gate].groupby(case_id_key)["_ev"].min()
        cap = case_len.copy()
        cap.loc[gate_pos.index] = gate_pos
    else:
        cap = case_len

    # Build the set of valid (case, k) pairs.
    specs = []
    for case_id, c_cap in cap.items():
        hi = int(c_cap) if max_length is None else min(int(max_length), int(c_cap))
        for k in range(min_length, hi + 1, step):
            specs.append((case_id, k))

    if not specs:
        return (
            pd.DataFrame(columns=list(log.columns)),
            pd.DataFrame(columns=["prefix_id", "orig_case_id", "prefix_length"]),
        )

    spec = pd.DataFrame(specs, columns=[case_id_key, "_k"])

    # Expand: each event joins every valid k of its case, keep events strictly inside the prefix.
    merged = g.merge(spec, on=case_id_key)
    merged = merged[merged["_ev"] < merged["_k"]].copy()

    merged["orig_case_id"] = merged[case_id_key]
    merged[case_id_key] = merged[case_id_key].astype(str) + "@@k" + merged["_k"].astype(str)

    prefix_index = (
        merged[[case_id_key, "orig_case_id", "_k"]]
        .drop_duplicates()
        .rename(columns={case_id_key: "prefix_id", "_k": "prefix_length"})
        .reset_index(drop=True)
    )
    # Keep prefix_length out of the event log so it can never leak into the feature matrix.
    prefix_log = merged.drop(columns=["_ev", "_k", "orig_case_id"]).reset_index(drop=True)
    return prefix_log, prefix_index


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
        
# Standard XES event-level / system columns that are never case attributes,
# even when they happen to be constant within a case (e.g. an all-"complete"
# lifecycle:transition). Used to filter the constant-per-case detection below.
_EVENT_LEVEL_COLS = {
    "concept:name", "time:timestamp", "start_timestamp",
    "lifecycle:transition", "lifecycle:state",
    "org:resource", "org:group", "org:role",
    "@@index", "@@case_index",
}


def list_case_attributes(log: pd.DataFrame) -> List[str]:
    """Return the case-level (trace) attribute column names present in the log.

    A column is treated as case-level if it is either explicitly prefixed
    'case:' (as PM4Py does for CSV logs) or — for XES logs, where PM4Py
    flattens trace attributes into unprefixed columns mixed with event
    attributes — its value is constant within every case. Standard event/
    system columns (concept:name, time:timestamp, lifecycle:transition, ...)
    are always excluded.
    """
    case_id_key = 'case:concept:name'
    cols = [c for c in log.columns if c != case_id_key]

    # nunique per case, computed in a single groupby pass.
    per_case_nunique = log.groupby(case_id_key).nunique(dropna=True)

    keep = set()
    for c in cols:
        if c.startswith('case:'):
            keep.add(c)                       # explicitly a case attribute
        elif c in _EVENT_LEVEL_COLS:
            continue                          # known event/system column
        elif c in per_case_nunique.columns and per_case_nunique[c].max() <= 1:
            keep.add(c)                       # constant within every case -> trace attribute

    return [c for c in cols if c in keep]     # preserve original column order


def summarize_case_attributes(log: pd.DataFrame) -> pd.DataFrame:
    """Small summary table for CLI display: one row per case-level attribute
    with columns [attribute, dtype, n_unique, example]. Helps the analyst
    choose meaningful attributes (HL-03/HL-04)."""
    case_id_key = 'case:concept:name'
    rows = []
    for col in list_case_attributes(log):
        per_case = log.groupby(case_id_key)[col].first()
        example = per_case.dropna().iloc[0] if per_case.notna().any() else ""
        rows.append({
            "attribute": col,
            "dtype": str(per_case.dtype),
            "n_unique": int(per_case.nunique(dropna=True)),
            "example": example,
        })
    return pd.DataFrame(rows, columns=["attribute", "dtype", "n_unique", "example"])


def extract_features(
    log: pd.DataFrame,
    feature_type: PredictionFeature,
    model: Optional[Tuple] = None,
    alignments: Optional[List[Dict[str, Any]]] = None,
    selected_case_attributes: Optional[List[str]] = None,
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

    # CASE_ATTRIBUTES: Static trace-level data
    if feature_type in [config.PredictionFeature.CASE_ATTRIBUTES, config.PredictionFeature.HYBRID]:
        # None => default to all available case attributes; [] => explicitly none.
        cols = selected_case_attributes if selected_case_attributes is not None else list_case_attributes(log)
        cols = [c for c in cols if c in log.columns and c != case_id_key]

        if cols:
            # Case attributes are static per case -> take the first observed value per case.
            ca_df = log.groupby(case_id_key)[cols].first()
            # Numeric attributes absent for some cases would feed NaN into StandardScaler;
            # fill with 0. Categorical NaN is handled by get_dummies (becomes all-zero).
            num_cols = ca_df.select_dtypes(include="number").columns
            ca_df[num_cols] = ca_df[num_cols].fillna(0)
            # Prefix mirrors cf_act_ to avoid naming collisions in the hybrid concat.
            ca_df.columns = [f"ca_{c}" for c in ca_df.columns]
        else:
            # Fallback if the log has no (selected) case attributes.
            ca_df = pd.DataFrame(index=log[case_id_key].unique())

        feature_matrices.append(ca_df)

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
    if X_raw.empty or len(X_raw.columns) == 0:
        X = X_raw.copy()
    else:
        X = pd.get_dummies(X_raw, drop_first=True)


    # Label Preprocessing (y): PROCESS_OUTCOME
    # Since labels are strings (e.g., 'Approve', 'uncertain_outcome'), we map them to integers (0, 1, 2...).
    if y_raw.empty:
        return X, np.array([]), []

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = [str(c) for c in le.classes_]

    # Handling for Remaining Time (Regression)
    # pass

    # Handling for Future Deviations (Binary)
    # pass

    return X, y, class_names
