"""
Non-interactive CLI option-coverage runner for main.py.

Drives the interactive CLI (main.py) by feeding scripted stdin answers, one full
run per scenario. Every *bounded* menu value is exercised at least once across the
scenarios, while the unbounded knobs are held fixed per the test brief:
  * prefix mode = "exactly k" with k = 3 (wherever the CLI offers prefixing)
  * no manual hyperparameter configuration (always answer "n")
  * default train/test split (0.7)

Named cli_test_runner.py (not *_test.py) so pytest does NOT auto-collect it.
Run:  cd Code && python cli_test_runner.py
"""

import os
import sys
import json
import time
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(CODE_DIR, ".cli_test_out")
LOG_PATH = os.path.join(CODE_DIR, "Data", "Sepsis_Cases.xes")
MODEL_PNML = os.path.join(OUT_DIR, "sepsis_model.pnml")
PER_SCENARIO_TIMEOUT = 1200  # seconds; genetic miner is the slow one

ERROR_SENTINELS = ("Traceback", "[SETUP ERROR]", "[EXTRACTION ERROR]", "[EVALUATION ERROR]")
SUCCESS_SENTINEL = "Pipeline completed successfully!"


@dataclass
class Scenario:
    name: str
    desc: str
    has_model: bool = False
    model_path: Optional[str] = None
    discovery: Optional[int] = None          # 1 Inductive / 2 Alpha / 3 Genetic
    model_viz: Optional[Tuple[int, str]] = None  # (representation 1/2/3, save_path)
    target: int = 1                          # 1 FutureDeviations / 2 ProcessOutcome / 3 RemainingTime
    outcome_activities: List[str] = field(default_factory=list)
    deviation_T: int = 0
    feature: int = 1                         # 1 CF / 2 Diagnostics / 3 CaseAttrs / 4 Hybrid
    algo: int = 1                            # classifier or regressor slot 1/2/3
    tree_save: Optional[str] = None          # path, only used when a single Decision Tree is trained
    conf_chart_dir: Optional[str] = None     # process-outcome only
    pie_save: Optional[str] = None
    json_save: Optional[str] = None


def build_answers(s: Scenario) -> List[str]:
    """Construct the exact stdin answer sequence, mirroring main.py's prompt branching."""
    a: List[str] = []

    # --- data & model input ---
    a.append("y" if s.has_model else "n")            # "Do you have a normative process model?"
    a.append(LOG_PATH)                               # "Enter the path to the event log"
    if s.has_model:
        a.append(s.model_path)                       # "Enter the path to the normative model"
    else:
        a.append(str(s.discovery))                   # discovery "Select option (1-3)"
        if s.discovery == 1:                         # Inductive: noise threshold + fall-throughs
            a += ["", ""]
        elif s.discovery == 2:                       # Alpha: no parameters
            pass
        elif s.discovery == 3:                       # Genetic: 6 params, all default (blank)
            a += ["", "", "", "", "", ""]

    # --- optional model visualization (we only ever save, never display) ---
    if s.model_viz is None:
        a.append("n")                                # vis choice N
    else:
        rep, path = s.model_viz
        a.append("s")
        a.append(str(rep))                           # representation 1/2/3
        a.append(path)

    # --- prediction target ---
    a.append(str(s.target))
    if s.target == 2:                                # process outcome: outcome-activity loop
        a += list(s.outcome_activities)
        a.append("done")
    elif s.target == 1:                              # future deviations: threshold T
        a.append(str(s.deviation_T))

    # --- feature type ---
    a.append(str(s.feature))
    if s.feature in (3, 4):                          # case-attrs / hybrid -> attribute selection
        a.append("all")

    # --- prefix mode (fixed: exactly k=3, wherever offered) ---
    prefix_prompt = (s.target == 3) or (s.feature != 3 and s.target in (1, 2))
    if prefix_prompt:
        a.append("1")                                # "exactly k"
        a.append("3")                                # k = 3

    # --- train/test split ---
    a.append("0.7")

    # --- ML algorithm + (decline) manual hyperparameters ---
    a.append(str(s.algo))
    a.append("n")                                    # "configure the parameters yourself?"

    # --- decision-tree visualization loop (only when a single Decision Tree is trained) ---
    show_tree = (s.target == 3 and s.algo == 1) or (s.target in (1, 2) and s.algo == 1)
    if show_tree:
        if s.tree_save:
            a += ["s", s.tree_save]
        else:
            a.append("n")
        a.append("n")                                # "reconfigure hyperparameters and retrain?" -> break

    # --- post-evaluation prompts ---
    if s.target == 3:                                # remaining-time: _display_results emits three
        # save prompts in order — feature importance, per-prefix-length viz, regression plots.
        a += ["n", "n", "n"]
    elif s.target == 2:                              # confusion-matrix/accuracy chart (process outcome only)
        if s.conf_chart_dir:
            a += ["y", s.conf_chart_dir]
        else:
            a.append("n")
    if s.target != 3:
        a.append("n")                                # feature-importance / next post-eval prompt

    # --- __main__ post-run prompts ---
    if s.pie_save:
        a += ["y", "s", s.pie_save]                  # pie chart: save
    else:
        a.append("n")
    if s.json_save:
        a += ["y", s.json_save]                      # JSON report: save
    else:
        a.append("n")

    if s.target == 3:
        a.append("n")                                # "perform predictions on new data?" (restart) -> stop

    return a


def make_scenarios() -> List[Scenario]:
    o = OUT_DIR
    return [
        Scenario("S1", "Inductive | FutureDeviations T=0 | control-flow | Decision Tree (+tree,+JSON)",
                 discovery=1, target=1, deviation_T=0, feature=1, algo=1,
                 tree_save=os.path.join(o, "s1_tree.png"), json_save=os.path.join(o, "s1_report.json")),
        Scenario("S2", "Alpha | FutureDeviations T=5 | diagnostics | Logistic Regression (+pie)",
                 discovery=2, target=1, deviation_T=5, feature=2, algo=2,
                 pie_save=os.path.join(o, "s2_runtime.png")),
        Scenario("S3", "Genetic | FutureDeviations T=0 | case-attrs(all) | Gradient Boosting",
                 discovery=3, target=1, deviation_T=0, feature=3, algo=3),
        Scenario("S4", "Inductive | model-viz Petri | ProcessOutcome(Release A) | control-flow | Decision Tree (+tree,+conf-chart)",
                 discovery=1, model_viz=(1, os.path.join(o, "s4_model_petri.png")),
                 target=2, outcome_activities=["Release A"], feature=1, algo=1,
                 tree_save=os.path.join(o, "s4_tree.png"), conf_chart_dir=o),
        Scenario("S5", "Inductive | model-viz BPMN | ProcessOutcome(Release A,B) | hybrid(all) | Logistic Regression",
                 discovery=1, model_viz=(2, os.path.join(o, "s5_model_bpmn.png")),
                 target=2, outcome_activities=["Release A", "Release B"], feature=4, algo=2),
        Scenario("S6", "Inductive | model-viz ProcessTree | ProcessOutcome(Release A) | diagnostics | Gradient Boosting",
                 discovery=1, model_viz=(3, os.path.join(o, "s6_model_ptree.png")),
                 target=2, outcome_activities=["Release A"], feature=2, algo=3),
        Scenario("S7", "Inductive | RemainingTime | control-flow | Decision Tree Regressor (+tree)",
                 discovery=1, target=3, feature=1, algo=1,
                 tree_save=os.path.join(o, "s7_tree.png")),
        Scenario("S8", "Inductive | RemainingTime | case-attrs(all) | Random Forest Regressor",
                 discovery=1, target=3, feature=3, algo=2),
        Scenario("S9", "Inductive | RemainingTime | hybrid(all) | Gradient Boosting Regressor (+pie,+JSON)",
                 discovery=1, target=3, feature=4, algo=3,
                 pie_save=os.path.join(o, "s9_runtime.png"), json_save=os.path.join(o, "s9_report.json")),
        Scenario("S10", "PROVIDED model (pnml) | model-viz Petri | ProcessOutcome(Release A) | control-flow | Decision Tree (+tree)",
                 has_model=True, model_path=MODEL_PNML,
                 model_viz=(1, os.path.join(o, "s10_model_petri.png")),
                 target=2, outcome_activities=["Release A"], feature=1, algo=1,
                 tree_save=os.path.join(o, "s10_tree.png")),
    ]


def export_model_pnml() -> None:
    """Discover an inductive Petri net once and write it to pnml, for the has_model=y scenario."""
    import warnings
    warnings.filterwarnings("ignore")
    import pm4py
    print("[setup] discovering + exporting Petri net for the 'provided model' scenario ...")
    log = pm4py.read_xes(LOG_PATH)
    net, im, fm = pm4py.discover_petri_net_inductive(log)
    pm4py.write_pnml(net, im, fm, MODEL_PNML)
    print(f"[setup] wrote {MODEL_PNML}")


def run_scenario(s: Scenario) -> dict:
    answers = build_answers(s)
    stdin_data = "\n".join(answers) + "\n"
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONPATH": CODE_DIR}
    t0 = time.time()
    status, rc, err_tail = "PASS", None, ""
    try:
        proc = subprocess.run(
            [sys.executable, "main.py"],
            input=stdin_data, text=True, capture_output=True,
            cwd=CODE_DIR, env=env, timeout=PER_SCENARIO_TIMEOUT,
        )
        rc = proc.returncode
        out = proc.stdout + "\n" + proc.stderr
        if rc != 0:
            status = "FAIL"
        elif SUCCESS_SENTINEL not in proc.stdout:
            status = "FAIL"
        elif any(sig in out for sig in ERROR_SENTINELS):
            status = "FAIL"
        # capture a short error tail for the report
        for sig in ERROR_SENTINELS:
            i = out.find(sig)
            if i != -1:
                err_tail = out[i:i + 400].replace("\n", " ⏎ ").strip()
                break
        if status != "PASS" and not err_tail and proc.stderr.strip():
            err_tail = proc.stderr.strip().splitlines()[-1][:400]
    except subprocess.TimeoutExpired:
        status, err_tail = "TIMEOUT", f">{PER_SCENARIO_TIMEOUT}s"
    dur = time.time() - t0

    # which saved artifacts should exist?
    expected = [p for p in (
        (s.model_viz[1] if s.model_viz else None),
        s.tree_save, s.pie_save, s.json_save,
    ) if p]
    missing = [os.path.basename(p) for p in expected if not os.path.exists(p)]

    return {
        "name": s.name, "desc": s.desc, "status": status,
        "returncode": rc, "duration_s": round(dur, 1),
        "missing_artifacts": missing, "error_tail": err_tail,
        "n_answers": len(answers),
    }


def main() -> int:
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    export_model_pnml()

    scenarios = make_scenarios()
    results = []
    for s in scenarios:
        print(f"\n=== Running {s.name}: {s.desc} ===", flush=True)
        r = run_scenario(s)
        results.append(r)
        flag = {"PASS": "✅", "FAIL": "❌", "TIMEOUT": "⏱"}.get(r["status"], "?")
        extra = f" missing={r['missing_artifacts']}" if r["missing_artifacts"] else ""
        err = f"  | {r['error_tail']}" if r["error_tail"] and r["status"] != "PASS" else ""
        print(f"{flag} {s.name}  {r['status']}  ({r['duration_s']}s, rc={r['returncode']}){extra}{err}", flush=True)

    # write JSON + markdown report
    with open(os.path.join(OUT_DIR, "report.json"), "w") as f:
        json.dump(results, f, indent=2)

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    lines = ["# CLI option-coverage report", "",
             f"**{n_pass}/{len(results)} scenarios passed** "
             f"(prefix fixed = exactly 3; no manual hyperparameters; split 0.7)", "",
             "| # | status | dur(s) | rc | missing artifacts | error tail |",
             "|---|--------|--------|----|-------------------|------------|"]
    for r in results:
        lines.append(f"| {r['name']} | {r['status']} | {r['duration_s']} | {r['returncode']} | "
                     f"{', '.join(r['missing_artifacts']) or '-'} | {r['error_tail'] or '-'} |")
    lines += ["", "## Scenario descriptions", ""]
    for r in results:
        lines.append(f"- **{r['name']}** — {r['desc']}")
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n================ SUMMARY: {n_pass}/{len(results)} passed ================")
    print(f"Report: {os.path.join(OUT_DIR, 'report.md')}")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
