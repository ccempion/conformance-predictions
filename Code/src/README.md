# Conformance Predictions

Interactive CLI and pipeline for process mining-based predictive monitoring.

## What it does

- Loads an event log from XES or CSV.
- Loads or discovers a normative process model (PNML, BPMN, PTML).
- Extracts features from control-flow, diagnostics, case attributes, or hybrid views.
- Trains and evaluates classification or regression models for:
  - process outcome prediction
  - remaining time prediction
  - future deviation risk prediction

## Installation

```bash
python -m pip install -r requirements.txt
```

or install the package:

```bash
python -m pip install .
```

## Usage

Run the CLI entry point:

```bash
python -m conformance_predictions.main
```

If installed via `pyproject.toml`, the script name is:

```bash
conformance-predictions
```

## Code structure

- `conformance_predictions/config.py` — enums and configuration types.
- `conformance_predictions/io.py` — event log loading, process model loading, discovery.
- `conformance_predictions/feature_extraction.py` — prefix generation, temporal features, labels.
- `conformance_predictions/model_training.py` — training and prediction pipelines.
- `conformance_predictions/pipeline.py` — end-to-end orchestration.
- `conformance_predictions/evaluation.py` — metrics, runtime/memory measurement.
- `conformance_predictions/visualization.py` — process model and evaluation plot generation.

## Notes

- Uses `pm4py` for process discovery, conformance checking, and model visualization.
- Supports scikit-learn classifiers and regressors via standard pipelines.
