import pytest
import pandas as pd
import numpy as np
from src import feature_extraction as fe
from src import pipeline
from src.feature_extraction import extract_label, preprocess_for_ml
from src.config import PredictionTarget, PredictionFeature

def test_extract_label_process_outcome():
    d = {'case:concept:name': ['1', '1', '2', '2'], 
         'concept:name': ['A', 'Approve', 'A', 'Reject'], 
         'time:timestamp': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:01:00', 
                                         '2023-01-01 00:02:00', '2023-01-01 00:03:00'])}
    df = pd.DataFrame(data=d)
    
    outcomes = extract_label(df, PredictionTarget.PROCESS_OUTCOME, outcome_activities=['Approve', 'Reject'])
    
    assert outcomes['1'] == 'Approve'
    assert outcomes['2'] == 'Reject'

def test_preprocess_for_ml():
    X_raw = pd.DataFrame({'feat1': [1, 2], 'feat2': [0.1, 0.2]}, index=['1', '2'])
    y_raw = pd.Series(['A', 'B'], index=['1', '2'])
    
    X, y, class_names = preprocess_for_ml(X_raw, y_raw)
    
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, np.ndarray)
    assert len(class_names) == 2
    assert 'A' in class_names
    assert 'B' in class_names


def _toy_log() -> pd.DataFrame:
    return pd.DataFrame({
        "case:concept:name": ["1", "1", "2", "2"],
        "concept:name":      ["a", "b", "a", "c"],
        "case:amount":       [100, 100, 50, 50],
        "case:channel":      ["web", "web", "phone", "phone"],
    })


def test_list_case_attributes():
    assert set(fe.list_case_attributes(_toy_log())) == {"case:amount", "case:channel"}


def test_summarize_case_attributes():
    summary = fe.summarize_case_attributes(_toy_log())
    assert list(summary.columns) == ["attribute", "dtype", "n_unique", "example"]
    amount_row = summary[summary["attribute"] == "case:amount"].iloc[0]
    assert amount_row["n_unique"] == 2  # 100 and 50


def test_extract_selected_case_attributes():
    X = fe.extract_features(
        _toy_log(),
        PredictionFeature.CASE_ATTRIBUTES,
        selected_case_attributes=["case:amount"],
    )
    assert list(X.index) == ["1", "2"]
    assert "ca_case:amount" in X.columns
    assert "ca_case:channel" not in X.columns      # not selected
    assert X.loc["1", "ca_case:amount"] == 100


def test_extract_all_case_attributes_by_default():
    # selected_case_attributes=None => include all available case attributes
    X = fe.extract_features(_toy_log(), PredictionFeature.CASE_ATTRIBUTES)
    assert {"ca_case:amount", "ca_case:channel"} <= set(X.columns)


def test_list_case_attributes_unprefixed_xes_style():
    # XES logs (this pm4py version) flatten trace attributes WITHOUT a 'case:' prefix.
    # They must be detected as constant-within-case; event columns must be excluded.
    log = pd.DataFrame({
        "case:concept:name":   ["1", "1", "2", "2"],
        "concept:name":        ["a", "b", "a", "c"],   # event activity (varies) -> excluded
        "lifecycle:transition":["complete"] * 4,        # constant but event/system -> excluded
        "Age":                 [40, 40, 65, 65],         # trace attribute (constant) -> kept
        "CRP":                 [1.0, 2.0, 3.0, 4.0],     # event measurement (varies) -> excluded
    })
    found = set(fe.list_case_attributes(log))
    assert found == {"Age"}


# --------------------------------------------------------------------------
# Prefix generation (LL-07.1, LL-11.3)
# --------------------------------------------------------------------------

def _prefix_toy() -> pd.DataFrame:
    # case 1: a,b,c,d (len 4)   case 2: a,b (len 2)
    ts = pd.to_datetime("2023-01-01") + pd.to_timedelta(range(6), unit="h")
    return pd.DataFrame({
        "case:concept:name": ["1", "1", "1", "1", "2", "2"],
        "concept:name":      ["a", "b", "c", "d", "a", "b"],
        "time:timestamp":    ts,
    })


def test_generate_prefixes_lengths_and_content():
    plog, pidx = fe.generate_prefixes(_prefix_toy(), min_length=1, max_length=3, step=1)
    # ids span the valid (case, k) pairs (case 2 capped at its length 2)
    assert set(pidx["prefix_id"]) == {"1@@k1", "1@@k2", "1@@k3", "2@@k1", "2@@k2"}
    # LL-11.3: a length-k prefix contains EXACTLY the first k events, in order
    p3 = plog[plog["case:concept:name"] == "1@@k3"].sort_values("time:timestamp")
    assert list(p3["concept:name"]) == ["a", "b", "c"]
    row = pidx[pidx.prefix_id == "1@@k3"].iloc[0]
    assert row.prefix_length == 3 and row.orig_case_id == "1"


def test_generate_prefixes_step_and_range():
    _, pidx = fe.generate_prefixes(_prefix_toy(), min_length=2, max_length=4, step=2)
    assert set(pidx[pidx.orig_case_id == "1"].prefix_length) == {2, 4}
    assert set(pidx[pidx.orig_case_id == "2"].prefix_length) == {2}   # case 2 has only 2 events


def test_generate_prefixes_gating():
    # case 1: a, b, Approve, d  -> first outcome at position 2 -> prefixes capped at length 2
    ts = pd.to_datetime("2023-01-01") + pd.to_timedelta(range(4), unit="h")
    log = pd.DataFrame({
        "case:concept:name": ["1", "1", "1", "1"],
        "concept:name":      ["a", "b", "Approve", "d"],
        "time:timestamp":    ts,
    })
    plog, pidx = fe.generate_prefixes(log, min_length=1, max_length=None, step=1,
                                      gate_activities=["Approve"])
    assert "Approve" not in set(plog["concept:name"])          # never leaks into a prefix
    assert pidx.prefix_length.max() == 2                        # stops before the outcome


def test_pooled_extraction_broadcasts_labels_without_leakage():
    # case 1 -> Approve, case 2 -> Reject; CONTROL_FLOW needs no alignments (model unused)
    ts = pd.to_datetime("2023-01-01") + pd.to_timedelta(range(6), unit="h")
    log = pd.DataFrame({
        "case:concept:name": ["1", "1", "1", "2", "2", "2"],
        "concept:name":      ["a", "b", "Approve", "a", "c", "Reject"],
        "time:timestamp":    ts,
    })
    X, y, class_names, plen = pipeline.extract_target_and_feature_over_prefixes(
        target=PredictionTarget.PROCESS_OUTCOME,
        feature=PredictionFeature.CONTROL_FLOW,
        model=None,
        log=log,
        outcome_activities=["Approve", "Reject"],
        min_length=1, max_length=None, step=1,
    )
    # gating caps each case at length 2 (outcome at position 2) -> lengths {1, 2} per case
    assert len(X) == 4
    assert set(plen) == {1, 2}
    # every prefix row's label equals its ORIGINAL case's outcome (correct broadcast)
    for pid, yi in zip(X.index, y):
        expected = "Approve" if pid.startswith("1@@") else "Reject"
        assert class_names[yi] == expected
    # outcome activities were gated out, so they never appear as features
    assert not any(c.endswith("Approve") or c.endswith("Reject") for c in X.columns)
