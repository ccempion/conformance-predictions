import pytest
import pandas as pd
import os
from conformance_predictions.io import load_event_log, load_process_model

def test_load_event_log_csv(tmp_path):
    # Create a dummy CSV log
    d = {'case:concept:name': ['1', '1', '2'], 
         'concept:name': ['A', 'B', 'A'], 
         'time:timestamp': ['2023-01-01 00:00:00', '2023-01-01 00:01:00', '2023-01-01 00:02:00']}
    df = pd.DataFrame(data=d)
    csv_path = tmp_path / "test_log.csv"
    df.to_csv(csv_path, index=False)
    
    # Ensure CSV to internal log representation conversion is performed correctly
    loaded_df = load_event_log(str(csv_path))
    assert isinstance(loaded_df, pd.DataFrame)
    assert len(loaded_df) == 3
    assert 'case:concept:name' in loaded_df.columns

def test_load_process_model_invalid_format():
    # If user supplies wrong model format, error is thrown
    with pytest.raises(ValueError, match="Unsupported model format"):
        load_process_model("invalid_file.txt")
