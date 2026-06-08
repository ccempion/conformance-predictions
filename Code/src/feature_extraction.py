"""
This module is responsible for extracting features from input data such as 
1. Log (pandas dataframe): case attributes, control flow, conformance insights i.e. alignments
2. Model (petri net): conformance insights i.e. alignments

This data is also preprocesed for ML by applying scaling and ensuring that the extracted label- and feature-vectors
for the train and test sets are not missing any cells.
"""
from typing import List, Dict, Optional, Any, Tuple

import pandas as pd
import numpy as np
import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from sklearn.preprocessing import LabelEncoder

from src.config import PredictionTarget, PredictionFeature
from src import config


def compute_alignments(log: pd.DataFrame, model: Tuple[PetriNet, Marking, Marking]) -> List[Dict[str, Any]]:
    """Computes optimal alignments between the log and the model (Petri Net).

    Ordering matters: pm4py does NOT guarantee that the returned alignment list follows
    the case order of ``log[...].unique()`` (passing a DataFrame sorts cases by id
    internally). Every consumer here pairs alignments to cases positionally, so we convert
    to an EventLog once, key each alignment by its case id, and re-emit in appearance
    order — making the positional pairing correct regardless of pm4py's internal ordering.

    Args:
        log (pd.DataFrame): Event log.
        model (Tuple[PetriNet, Marking, Marking]): Petri net model (net, im, fm).

    Returns:
        List[Dict[str, Any]]: The standard PM4Py alignment output (list of dictionaries),
        ordered to match ``log[case_id].unique()`` (first-appearance order).
    """
    case_id_key = "case:concept:name"
    (net, im, fm) = model
    event_log = pm4py.convert_to_event_log(log)
    aligns = pm4py.conformance_diagnostics_alignments(event_log, net, im, fm)
    by_case = {trace.attributes["concept:name"]: a for trace, a in zip(event_log, aligns)}
    order = list(dict.fromkeys(log[case_id_key]))  # first-appearance order == unique()
    return [by_case[cid] for cid in order]


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

    Args:
        log (pd.DataFrame): The original event log.
        min_length (int, optional): Minimum prefix length. Defaults to 1.
        max_length (Optional[int], optional): Maximum prefix length. Defaults to None.
        step (int, optional): Step size between prefix lengths. Defaults to 1.
        gate_activities (Optional[List[str]], optional): Activities that gate prefix generation. Defaults to None.
        case_id_key (str, optional): Key for case ID. Defaults to "case:concept:name".
        timestamp_key (str, optional): Key for timestamp. Defaults to "time:timestamp".
        activity_key (str, optional): Key for activity name. Defaults to "concept:name".

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            prefix_log: event-level DataFrame, case_id_key replaced by the unique prefix id,
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


def extract_remaining_time(
    full_log: pd.DataFrame,
    prefix_log: pd.DataFrame,
    prefix_index: pd.DataFrame,
    case_id_key: str = "case:concept:name",
    timestamp_key: str = "time:timestamp",
) -> pd.Series:
    """Per-prefix remaining time in seconds — the regression label for REMAINING_TIME.

    Unlike the classification targets (one label per case, broadcast to all prefixes),
    remaining time differs for every prefix: it is the time from the END of the observed
    prefix to the completion of the original case.

        remaining = (last timestamp of the FULL original case)
                  - (last timestamp inside the prefix)

    The case-end is taken from full_log so the future the model must predict is never
    inside the features; "now" is the prefix's last event.

    Args:
        full_log (pd.DataFrame): Complete event log (original case ids).
        prefix_log (pd.DataFrame): Prefix event log from generate_prefixes; its case_id_key column
            already holds the unique prefix id.
        prefix_index (pd.DataFrame): [prefix_id, orig_case_id, prefix_length] from generate_prefixes.
        case_id_key (str, optional): Key for case ID. Defaults to "case:concept:name".
        timestamp_key (str, optional): Key for timestamp. Defaults to "time:timestamp".

    Returns:
        pd.Series: Series indexed by prefix_id, name "remaining_time_seconds" (float).
    """
    # Completion time of each ORIGINAL case.
    case_end = full_log.groupby(case_id_key)[timestamp_key].max()
    # "Now": last event timestamp inside each prefix (case_id_key == prefix id here).
    prefix_now = prefix_log.groupby(case_id_key)[timestamp_key].max()

    idx = prefix_index.set_index("prefix_id")
    end_for_prefix = idx["orig_case_id"].map(case_end)        # indexed by prefix_id
    remaining = (end_for_prefix - prefix_now).dt.total_seconds()
    return remaining.rename("remaining_time_seconds")


def extract_temporal_features(
    prefix_log: pd.DataFrame,
    case_id_key: str = "case:concept:name",
    timestamp_key: str = "time:timestamp",
) -> pd.DataFrame:
    """Per-prefix temporal predictors for remaining-time prediction.

    These are the standard, high-impact time signals for remaining-time models.

    Args:
        prefix_log (pd.DataFrame): Prefix event log.
        case_id_key (str, optional): Key for case ID. Defaults to "case:concept:name".
        timestamp_key (str, optional): Key for timestamp. Defaults to "time:timestamp".

    Returns:
        pd.DataFrame: DataFrame indexed by prefix id with columns:
            tf_elapsed_s: time from the prefix's first to last event (how long it has run)
            tf_since_prev_s: gap between the last two events of the prefix (0 if one event)
            tf_n_events: number of events observed so far
    """
    g = prefix_log.sort_values([case_id_key, timestamp_key], kind="mergesort").copy()
    grp_ts = g.groupby(case_id_key)[timestamp_key]

    elapsed = (grp_ts.max() - grp_ts.min()).dt.total_seconds()
    n_events = grp_ts.size().astype(float)
    # diff() yields NaT for the first event of each prefix; the LAST gap is the most
    # recent inter-event time. fillna(0) covers single-event prefixes.
    g["_gap_s"] = grp_ts.diff().dt.total_seconds()
    since_prev = g.groupby(case_id_key)["_gap_s"].last().fillna(0.0)

    return pd.DataFrame({
        "tf_elapsed_s": elapsed,
        "tf_since_prev_s": since_prev,
        "tf_n_events": n_events,
    })


def real_deviation_cost(alignment_moves: List[Tuple[Any, Any]]) -> int:
    """Count the REAL conformance deviations in one alignment.

    A pm4py alignment move is a (log_move, model_move) tuple:
        (a, a)        synchronous move                                         -> 0 (conforming)
        (a, '>>')     log move: event a not allowed by the model here         -> +1
        ('>>', a)     model move on a VISIBLE transition a (expected, missing) -> +1
        ('>>', None)  model move on an invisible/silent (tau) transition       -> 0 (routing)

    Silent/tau moves are normal routing in (e.g.) Inductive-Miner models and carry zero
    cost; counting them as deviations is exactly the bug this helper avoids. Mirrors the
    move classification used for DIAGNOSTICS features in extract_features.

    Args:
        alignment_moves (List[Tuple[Any, Any]]): List of alignment moves.

    Returns:
        int: Total cost of real deviations.
    """
    cost = 0
    for log_move, model_move in alignment_moves:
        if log_move == model_move and log_move != '>>':
            continue                                   # synchronous move
        elif model_move == '>>' and log_move is not None:
            cost += 1                                  # real log move (a, '>>')
        elif log_move == '>>' and model_move is not None:
            cost += 1                                  # real visible model move ('>>', a)
        # else: invisible/tau model move ('>>', None) -> not a deviation
    return cost


def extract_label(
    log: pd.DataFrame,
    target_type: PredictionTarget,
    outcome_activities: Optional[List[str]] = None,
    alignments: Optional[List[Dict[str, Any]]] = None,
    deviation_cost_threshold: int = 0,
) -> pd.Series:
    """Derives labels for each case.

    - PROCESS_OUTCOME: Checks if any activity in 'outcome_activities' exists in the trace.
    - FUTURE_DEVIATIONS: label = 1 if the case's real deviation cost exceeds
      deviation_cost_threshold (HL-08). Real cost ignores silent/tau routing moves.
    - REMAINING_TIME: per-prefix regression label (see extract_remaining_time).

    Args:
        log (pd.DataFrame): Event log.
        target_type (PredictionTarget): Type of prediction target.
        outcome_activities (Optional[List[str]], optional): Activities defining outcomes. Defaults to None.
        alignments (Optional[List[Dict[str, Any]]], optional): Alignment results. Defaults to None.
        deviation_cost_threshold (int, optional): Threshold for future deviations. Defaults to 0.

    Raises:
        ValueError: If required arguments for target types are missing.
        NotImplementedError: If target type is not supported.

    Returns:
        pd.Series: Series of labels indexed by case ID.
    """
    case_id_key = 'case:concept:name'
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
            
        # HL-08: label = 1 if the case's REAL deviation cost exceeds the threshold.
        # real_deviation_cost ignores silent/tau routing moves, so a conforming case that
        # merely skips an optional activity is correctly labeled 0. T defaults to 0
        # ("any real deviation"); a higher T tolerates minor deviations.
        deviations = [
            int(real_deviation_cost(a.get('alignment', [])) > deviation_cost_threshold)
            for a in alignments
        ]
        return pd.Series(deviations, index=case_ids, name="future_deviations")

    elif target_type == config.PredictionTarget.REMAINING_TIME:
        # Remaining time is a PER-PREFIX regression label (each prefix has a different
        # remaining time), not a per-case label, so it is produced by
        # extract_remaining_time() on the prefix path rather than here.
        raise ValueError(
            "REMAINING_TIME labels are computed per prefix via extract_remaining_time(); "
            "route this target through extract_target_and_feature_over_prefixes."
        )

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

    Args:
        log (pd.DataFrame): Event log.

    Returns:
        List[str]: List of case-level attribute column names.
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
    """Small summary table for CLI display of case-level attributes.

    Helps the analyst choose meaningful attributes (HL-03/HL-04).

    Args:
        log (pd.DataFrame): Event log.

    Returns:
        pd.DataFrame: Summary table with columns [attribute, dtype, n_unique, example].
    """
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
    """Extracts features based on the specified feature type.

    - CONTROL_FLOW: Extracts the trace prefix (Bag-of-activities).
    - DIAGNOSTICS: Extracts the alignment-based trace prefix diagnostics (Bag-of-move-types).
    - CASE_ATTRIBUTES: Extracts static attributes about the case.
    - HYBRID: Combines the active feature sets.

    Args:
        log (pd.DataFrame): Event log.
        feature_type (PredictionFeature): Type of features to extract.
        model (Optional[Tuple], optional): Process model. Defaults to None.
        alignments (Optional[List[Dict[str, Any]]], optional): Alignment results. Defaults to None.
        selected_case_attributes (Optional[List[str]], optional): Subset of case attributes. Defaults to None.

    Raises:
        ValueError: If required data for feature type is missing or if feature type is unsupported.

    Returns:
        pd.DataFrame: Feature matrix indexed by case ID.
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
        
    # Concatenate side-by-side; all matrices share the case_id index, so pandas aligns them.
    final_df = pd.concat(feature_matrices, axis=1)

    return final_df


def preprocess_for_ml(
    X_raw: pd.DataFrame,
    y_raw: pd.Series,
    is_regression: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Encodes and scales the raw data for ML training.

    Args:
        X_raw (pd.DataFrame): Raw feature matrix.
        y_raw (pd.Series): Raw label vector.
        is_regression (bool, optional): When True the target is continuous (e.g. remaining 
            time in seconds) — it is kept numeric and NOT label-encoded. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, np.ndarray, List[str]]: A tuple containing:
            - X: feature DataFrame (columns preserved for downstream alignment).
            - y: encoded label vector (classification) or float vector (regression).
            - class_names: ordered list of original class labels for classification, [] for regression.
    """

    # Feature Preprocessing (X)
    # Converts categorical Case Attributes (strings) into numeric One-Hot encoded columns.
    # Numerical features (Control Flow/Diagnostics/temporal) remain unchanged.
    if X_raw.empty or len(X_raw.columns) == 0:
        X = X_raw.copy()
    else:
        X = pd.get_dummies(X_raw, drop_first=True)

    # Empty / missing labels: nothing to encode.
    if y_raw is None or (hasattr(y_raw, "empty") and y_raw.empty):
        return X, np.array([]), []

    # Regression target (REMAINING_TIME): keep the continuous value, no encoding.
    if is_regression:
        y = np.asarray(y_raw, dtype=float)
        return X, y, []

    # Classification target (PROCESS_OUTCOME / FUTURE_DEVIATIONS):
    # map string/categorical labels to integers (0, 1, 2, ...).
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = [str(c) for c in le.classes_]

    return X, y, class_names
