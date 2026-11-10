# Conformance Predictions

A university lab project at RWTH Aachen University, supervised by Dr. Alessandro Berti.

## What this is

We're building a Python-based tool that combines process mining and machine learning for predictive process monitoring. Given a partially observed process case and an event log, it predicts one of three targets: whether a conformance violation will occur later, which outcome the case will reach, or how much time remains. We use PM4Py for the process mining side and scikit-learn for the prediction part.

**To install and run the tool, see [`Code/README.md`](Code/README.md).**

## Team

- Richard Schuppener
- Cem Colak

## Tools

- **PM4Py** — process mining (conformance checking, event log handling)
- **Python / scikit-learn** — machine learning and data processing
- **Trello** — task tracking
- **GitLab (RWTH)** — version control and project management

## Repository layout

- **`Code/`** — the software: the interactive pipeline, source modules, and tests. See [`Code/README.md`](Code/README.md) to install and run it.
  ```text
  Code/
  ├── cli_test_runner.py
  ├── docs/
  │   ├── index.html
  │   ├── search.js
  │   ├── src/
  │   │   ├── config.html
  │   │   ├── evaluation.html
  │   │   ├── feature_extraction.html
  │   │   ├── io.html
  │   │   ├── model_training.html
  │   │   ├── pipeline.html
  │   │   └── visualization.html
  │   └── src.html
  ├── main.py
  ├── README.md
  ├── requirements.txt
  ├── src/
  │   ├── config.py
  │   ├── evaluation.py
  │   ├── feature_extraction.py
  │   ├── __init__.py
  │   ├── io.py
  │   ├── model_training.py
  │   ├── pipeline.py
  │   └── visualization.py
  └── tests/
      ├── evaluation_test.py
      ├── feature_extraction_test.py
      ├── __init__.py
      ├── io_test.py
      ├── model_training_test.py
      └── visualization_test.py
  ```
- **`Requirements Engineering/`**, **`Project Setup/`**, **`Sprint 1–3/`**, **`Assessment/`** — academic deliverables (LaTeX/PDF), including the per-sprint **User Guide**.
## Data

The event logs are not included in this repository. The experiments use the public **Sepsis Cases** event log (Mannhardt, 2016), available from the 4TU.ResearchData repository: https://doi.org/10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460. Download the `.xes` / `.xes.gz` file and point the tool to it (see [`Code/README.md`](Code/README.md)).

## Status

Working interactive pipeline. All three prediction targets (future deviations, process outcome, remaining time), the four feature types, and the discovery/ML algorithm options are implemented.

> This is a mirror of the original project repository on the RWTH Aachen GitLab, where development took place.

## License

The source code is released under the [MIT License](LICENSE). Academic reports and figures are © the authors.
