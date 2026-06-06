# CLI Option-Coverage Test Report

**Tool:** Conformance Predictions interactive CLI (`Code/main.py`)
**Event log:** `Code/Data/Sepsis_Cases.xes` (1050 cases, 15 214 events, 16 activities, 24 case-level attributes, median case length 13)
**Date:** 2026-06-05
**Constraints (per brief):** prefix fixed to **exactly k = 3** wherever offered · **no manual hyperparameter configuration** (defaults) · default train/test split 0.7 · display (`d`) visualization modes avoided (require a GUI)

---

## 0. Pre-test blocker: broken merge had to be reverted first

The CLI could not be run at all on arrival. The latest commit `081282a "Merge branch '75-additional-ml-algorithm'"` had been committed **with unresolved git conflict markers** in 5 files (`main.py`, `src/config.py`, `src/model_training.py`, `src/pipeline.py`, `tests/model_training_test.py`). `python main.py` raised `IndentationError`; all 5 pytest modules failed to collect.

**Resolution:** `git revert -m 1 081282a` (revert commit, no force-push — the broken merge was already on `origin/main`). This restored the merge's first parent `4dfa6bc`, which already contained the full Decision-Tree / Logistic-Regression / Gradient-Boosting classifier functionality, so no feature was lost. **After the revert, all 38 unit tests pass.**

---

## 1. Method

A non-interactive runner (`Code/cli_test_runner.py`) drives `main.py` by feeding scripted stdin answers — one full run per scenario — with `MPLBACKEND=Agg` so plotting never blocks. Ten scenarios were designed so that **every bounded menu value is exercised at least once**, while the unbounded knobs are held fixed (prefix = exactly 3, defaults for hyperparameters).

A scenario is **PASS** iff: process exit code 0 **and** stdout contains `"Pipeline completed successfully!"` **and** no error sentinel (`Traceback`, `[SETUP ERROR]`, `[EXTRACTION ERROR]`, `[EVALUATION ERROR]`) appears.

---

## 2. Results — 8 / 10 passed

| # | Discovery / model | Target | Feature | Algorithm | Extra options | Result | Time |
|---|-------------------|--------|---------|-----------|---------------|--------|------|
| S1 | Inductive | Future deviations, T=0 | control-flow | Decision Tree | tree save, JSON save | ❌ **FAIL** | 485 s |
| S2 | Alpha | Future deviations, T=5 | diagnostics | Logistic Regression | runtime pie save | ✅ PASS | 6 s |
| S3 | **Genetic** | Future deviations, T=0 | case-attrs (all) | Gradient Boosting | — | ❌ **FAIL** | 1163 s |
| S4 | Inductive | Process outcome (Release A) | control-flow | Decision Tree | model-viz **Petri**, tree save, conf-matrix chart | ✅ PASS | 5 s |
| S5 | Inductive | Process outcome (Release A, B) | hybrid (all) | Logistic Regression | model-viz **BPMN** | ✅ PASS | 4 s |
| S6 | Inductive | Process outcome (Release A) | diagnostics | Gradient Boosting | model-viz **Process Tree** | ✅ PASS | 4 s |
| S7 | Inductive | Remaining time | control-flow | Decision Tree Regressor | tree save | ✅ PASS | 3 s |
| S8 | Inductive | Remaining time | case-attrs (all) | Random Forest Regressor | — | ✅ PASS | 3 s |
| S9 | Inductive | Remaining time | hybrid (all) | Gradient Boosting Regressor | runtime pie save, JSON save | ✅ PASS | 4 s |
| S10 | **Provided model (.pnml)** | Process outcome (Release A) | control-flow | Decision Tree | model-viz Petri, tree save | ✅ PASS | 3 s |

### Coverage achieved (each bounded option ≥ 1×)

| Option group | Values exercised |
|---|---|
| Discovery algorithm | Inductive ✓ · Alpha ✓ · Genetic ✓ |
| Model source | discovery ✓ · provided `.pnml` ✓ |
| Model-viz representation (save) | Petri Net ✓ · BPMN ✓ · Process Tree ✓ |
| Prediction target | Future deviations ✓ · Process outcome ✓ · Remaining time ✓ |
| Feature type | control-flow ✓ · diagnostics ✓ · case-attributes ✓ · hybrid ✓ |
| Classification algorithm | Decision Tree ✓ · Logistic Regression ✓ · Gradient Boosting ✓ |
| Regression algorithm | Decision Tree ✓ · Random Forest ✓ · Gradient Boosting ✓ |
| Deviation threshold T | 0 ✓ · 5 ✓ |
| Outcome activities | single ✓ · multiple ✓ |
| Save paths | decision-tree image ✓ · confusion-matrix/accuracy chart ✓ · runtime pie ✓ · JSON report ✓ |

Saved artifacts verified present under `Code/.cli_test_out/` (model PNGs for all three representations, decision-tree PNGs for classifier and regressor, runtime pies, JSON reports, and S4's `conformance_results_…` folder).

---

## 3. Findings (the two failures are genuine, not test-harness artifacts)

### Finding A — Genetic miner is impractical on Sepsis (S3)
- **Runtime:** genetic discovery took ~19 minutes on the 1050-case log (vs. seconds for Inductive/Alpha).
- **Correctness:** the discovered net is *"not an easy sound net"*, so any alignment-dependent path (FUTURE_DEVIATIONS labels, DIAGNOSTICS / HYBRID features) fails with `[EXTRACTION ERROR]: trying to apply alignments on a Petri net that is not a easy sound net!!`.
- **Severity:** the error is *caught gracefully* (program exits cleanly), but the Genetic option is effectively unusable for downstream prediction on this log.
- **Recommendation:** gate/validate soundness of the discovered net before alignment, or warn the user that Genetic output may be unsound; consider excluding it from the alignment-based targets.

### Finding B — Future-deviations at short prefix crashes ungracefully (S1)
- **Cause:** with prefix length 3 and T = 0, every length-3 prefix is non-deviant (Sepsis cases open with conforming ER Registration / Triage / Sepsis-Triage steps), so the training labels are single-class `[0]`.
- **Symptom:** `train_classification_model` raises its descriptive `ValueError` ("Training labels contain only one class … For FUTURE_DEVIATIONS this usually means the deviation cost threshold T is too high …"), **but the classification path (`configure_and_train_classifier` in `main.py`) does not wrap the training call in try/except.** The whole program therefore crashes with a traceback (exit code 1) instead of printing the helpful message. The *extraction* step has graceful handling; the *training* step does not.
- **Severity:** poor UX — the helpful guidance is hidden inside a stack trace.
- **Recommendation:** wrap the classifier training call (and ideally the regressor training call) in try/except, mirroring the `[EXTRACTION ERROR]` handling, so the friendly message surfaces. Note also the message's hint ("T too high") is the opposite of this case (T = 0, all non-deviant) — the wording could mention the short-prefix / all-conforming case too.

---

## 4. Intentional coverage gaps (consequences of the bounding rules)

- **Prefix modes "none" and "varying" were not exercised** (prefix fixed = exactly 3). As a result, the *metrics-by-prefix-length* breakdown, the per-length confusion-matrix inspection loop, and the by-prefix-length chart (`main.py` ≈ lines 559–589) never trigger (they require ≥ 2 distinct lengths). For PROCESS_OUTCOME with pure CASE_ATTRIBUTES features the CLI does not offer prefixing at all, so S-style case-attribute classification runs on full traces by design.
- **Display (`d`) visualization paths** were not exercised (graphviz/Matplotlib `show()` need a GUI; would block a headless run). Save (`s`) and skip (`N`) were used instead.
- **Discovery and ML hyperparameters** were left at defaults ("no manual configuration"), so the manual-parameter prompts (noise threshold, genetic population/rates/generations, criterion/max_depth/C/n_estimators, and the decision-tree reconfigure-and-retrain loop) were intentionally not tuned.

---

## 5. Reproduction

```bash
cd Code
python -m pytest -q          # 38 passed (sanity)
python cli_test_runner.py    # runs all 10 scenarios; writes .cli_test_out/report.md + report.json
```

The runner holds prefix = exactly 3 and answers "n" to every manual-hyperparameter prompt. Per-scenario timeout is 1200 s (only the Genetic scenario approaches it).
