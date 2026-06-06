# Conformance Predictions

A Python tool for **predictive process monitoring**. Given an event log — and optionally a
normative process model — it discovers/loads the model, runs conformance checking against it,
and trains a machine-learning model to predict, from a partial trace, one of three targets:
whether the case will deviate further, which outcome it will reach, or how much time remains.
Built on [PM4Py](https://pm4py.fit.fraunhofer.de/) for the process-mining side and
[scikit-learn](https://scikit-learn.org/) for the prediction side, exposed through an
interactive command-line program.

## Requirements

- **Python 3.14.x** (latest version with all dependencies supported)
- **Git**
- **Graphviz** — the system binary, not just the `graphviz` Python package. Needed to render
  process-model and decision-tree images. Install separately (e.g. `brew install graphviz`,
  `apt install graphviz`).

## Installation

```bash
git clone https://git.rwth-aachen.de/schuppenerr/conformance-predictions.git
cd conformance-predictions/Code
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies are pinned in [`requirements.txt`](requirements.txt) (pm4py 2.7.22.2,
scikit-learn 1.8.0, pandas, numpy, matplotlib, graphviz, psutil, pytest, …).

## Quick start

```bash
cd Code
python main.py
```

The program is an interactive CLI — it walks you through each choice. A typical first run
(discover a model, predict the process outcome, train a Decision Tree) looks like:

```text
=== Conformance Predictions ===
Do you have a normative process model? (y/N): n
Enter the path to the event log (XES/CSV): Data/Sepsis_Cases.xes

Which discovery algorithm should be used?
1. Inductive Miner (recommended)
...
Select option (1-3): 1
Noise threshold (default 0.0): 0.2

What should be predicted?
1. Future conformance deviations
2. Process outcome
3. Remaining time
Select option (1-3): 2

What should the prediction be based upon?
1. Plain trace (control-flow)
...
Select option (1-4): 1
```

From there it asks for the train/test split, the ML algorithm and (optionally)
hyperparameters, then prints the evaluation. For the full prompt-by-prompt walkthrough see the
[User Guide](#documentation).

Sample event logs live in `Code/Data/` (e.g. `Sepsis_Cases.xes`, `pdc_2017_*.xes`). This
folder is not tracked in Git — supply your own logs there or point the prompt at any path.

## What it can do

**Prediction targets** (`src/config.py` → `PredictionTarget`)

| Target | Type | Notes |
|---|---|---|
| Future conformance deviations | Probabilistic binary | Risk score; a case is "deviant" when its real alignment cost exceeds a configurable threshold `T` |
| Process outcome | Multi-class | Predicts which user-defined outcome activity a case reaches |
| Remaining time | Regression (seconds) | Per-prefix; requires prefixing (see below) |

**Feature types** (`PredictionFeature`)

| Feature | Description |
|---|---|
| Control-flow | Bag-of-activities counts |
| Alignment diagnostics | Per-activity synchronous / log-only / model-only move counts |
| Case attributes | Selected static case-level attributes from the log |
| Hybrid | Combination of the above |

**Algorithms**

- Classification: Logistic Regression, Gradient Boosting (Decision Tree for process outcome)
- Regression: Decision Tree, Random Forest, Gradient Boosting Regressor
- Model discovery: Inductive, Alpha, Genetic miner (`Discovery_alg`)

**Prefix modes.** Predictions can run on whole cases (`none`), a single prefix length
(`exactly k`), or pooled varying-length prefixes (`varying 1..k`). **Remaining time** is undefined
on a complete case, so it always requires prefixing.

## Inputs and outputs

**Inputs**
- Event log: **XES** (`.xes`), or **CSV** (`.csv`) with the columns
  `case:concept:name`, `concept:name`, `time:timestamp` and optional case attributes.
- Normative model (optional): **Petri net** (`.pnml`), **BPMN** (`.bpmn`), or
  **process tree** (`.ptml`). BPMN and process trees are converted to Petri nets internally.

**Outputs**
- Evaluation metrics, a confusion matrix and per-class/by-prefix-length breakdowns in the CLI.
- Optional saved images: process model, decision tree, performance-by-prefix-length chart, and
  a runtime pie chart.
- Optional `report.json` with per-step runtimes and peak memory.

## Project layout

```
Code/
├── main.py            # Interactive CLI entry point; orchestrates the run
├── requirements.txt   # Pinned dependencies
├── src/
│   ├── config.py            # Enums: targets, features, algorithms, discovery
│   ├── io.py                # Load logs/models; model discovery
│   ├── feature_extraction.py# Alignments, labels, features, prefixes, preprocessing
│   ├── model_training.py    # Train/predict for classification & regression
│   ├── pipeline.py          # Per-case and over-prefixes extraction; eval orchestration
│   ├── evaluation.py        # Metrics, by-prefix-length, runtime/memory measurement
│   └── visualization.py     # Model, decision-tree, confusion-matrix, runtime plots
├── tests/             # pytest suite
└── Data/              # Event logs (local only, not tracked)
```

## Running tests

```bash
cd Conformance-predictions/Code
python -m pytest
```

## Documentation

A detailed **User Guide** with the complete prompt-by-prompt CLI walkthrough is maintained as a
LaTeX/PDF deliverable in the sprint folders — see the most recent
[`Sprint 3/UserGuide.pdf`](../Sprint%203/UserGuide.pdf).

## Team & context

University lab project at **RWTH Aachen University**, supervised by
**Dr. Alessandro Berti**.

- Richard Schuppener
- Cem Colak
