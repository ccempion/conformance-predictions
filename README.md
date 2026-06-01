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
- **GitLab** — version control and project management

## Repository layout

- **`Code/`** — the software: the interactive pipeline, source modules, and tests. See [`Code/README.md`](Code/README.md) to install and run it.
- **`Requirements Engineering/`**, **`Project Setup/`**, **`Sprint 1–3/`**, **`Assessment/`** — academic deliverables (LaTeX/PDF), including the per-sprint **User Guide**.
- **`Sources/`** — reference literature.

## Status

Working interactive pipeline. All three prediction targets (future deviations, process outcome, remaining time), the four feature types, and the discovery/ML algorithm options are implemented.
