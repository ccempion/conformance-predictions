# CLI option-coverage report

**8/10 scenarios passed** (prefix fixed = exactly 3; no manual hyperparameters; split 0.7)

| # | status | dur(s) | rc | missing artifacts | error tail |
|---|--------|--------|----|-------------------|------------|
| S1 | FAIL | 485.4 | 1 | s1_tree.png, s1_report.json | Traceback (most recent call last): ⏎   File "/Users/bojack/Documents/Projects/conformance-predictions/Code/main.py", line 610, in <module> ⏎     time_list, mem_peak = evaluation.measure_memory(run_interactive_cli) ⏎                           ~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^ ⏎   File "/Users/bojack/Documents/Projects/conformance-predictions/Code/src/evaluation.py", line 52, in measure_memory ⏎ |
| S2 | PASS | 6.1 | 0 | - | - |
| S3 | FAIL | 1163.2 | 0 | - | [EXTRACTION ERROR]: trying to apply alignments on a Petri net that is not a easy sound net!! ⏎ |
| S4 | PASS | 4.5 | 0 | - | - |
| S5 | PASS | 4.3 | 0 | - | - |
| S6 | PASS | 3.8 | 0 | - | - |
| S7 | PASS | 3.0 | 0 | - | - |
| S8 | PASS | 2.9 | 0 | - | - |
| S9 | PASS | 3.7 | 0 | - | - |
| S10 | PASS | 3.0 | 0 | - | - |

## Scenario descriptions

- **S1** — Inductive | FutureDeviations T=0 | control-flow | Decision Tree (+tree,+JSON)
- **S2** — Alpha | FutureDeviations T=5 | diagnostics | Logistic Regression (+pie)
- **S3** — Genetic | FutureDeviations T=0 | case-attrs(all) | Gradient Boosting
- **S4** — Inductive | model-viz Petri | ProcessOutcome(Release A) | control-flow | Decision Tree (+tree,+conf-chart)
- **S5** — Inductive | model-viz BPMN | ProcessOutcome(Release A,B) | hybrid(all) | Logistic Regression
- **S6** — Inductive | model-viz ProcessTree | ProcessOutcome(Release A) | diagnostics | Gradient Boosting
- **S7** — Inductive | RemainingTime | control-flow | Decision Tree Regressor (+tree)
- **S8** — Inductive | RemainingTime | case-attrs(all) | Random Forest Regressor
- **S9** — Inductive | RemainingTime | hybrid(all) | Gradient Boosting Regressor (+pie,+JSON)
- **S10** — PROVIDED model (pnml) | model-viz Petri | ProcessOutcome(Release A) | control-flow | Decision Tree (+tree)
