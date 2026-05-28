"""
Main entry point for the Conformance Predictions software.
Provides an interactive CLI to guide users through the process mining and 
machine learning pipeline as specified in the Requirements Engineering document.
"""

import sys
import json
from typing import Optional, Tuple, List, Any
from src import model_training, pipeline, config, evaluation, feature_extraction
from src import visualization as vis
import pm4py

def run_interactive_cli() -> List[Tuple[str, float]]:
    """
    Guides the user through configuration steps as an interactive command-line-interface program: (HL-03) & triggers the pipeline.
    """
    print("=== Conformance Predictions ===")
    time_list = []

    # Data and Model Input
    has_model = input("Do you have a normative process model? (y/N): ").lower().strip() == 'y'
    
    log_path = input("Enter the path to the event log (XES/CSV): ").strip()
    discovery_alg = None
    discovery_params = {}
    model_path: Optional[str] = None
    discovery_alg: Optional[config.Discovery_alg] = None
    discovery_params: dict = {}
    if has_model:
        model_path = input("Enter the path to the normative model (Petri net/BPMN): ").strip()
    else:
        print("Model discovery will be used.")
        # Prompt for discovery algorithm & noise thresholds
        print("\nWhich discovery algorithm should be used?")
        print("1. Inductive Miner (recommended)")
        print("2. Alpha Miner")
        print("3. Genetic Miner")
        choice = int(input("Select option (1-3): ").strip())
        # Map to config.Discovery_alg Enum
        discovery_alg = [
            config.Discovery_alg.INDUCTIVE,
            config.Discovery_alg.ALPHA,
            config.Discovery_alg.GENETIC
        ][choice - 1]

        if discovery_alg == config.Discovery_alg.INDUCTIVE:
            discovery_params['noise_threshold'] = float(input("Noise threshold (default 0.0): ") or 0.0)
            discovery_params['disable_fallthroughs'] = input("Disable fall-throughs? (y/N): ").lower() == 'y'

        elif discovery_alg == config.Discovery_alg.ALPHA:
            print("Alpha Miner has no configurable parameters.")

        elif discovery_alg == config.Discovery_alg.GENETIC:
            discovery_params['population_size'] = int(input("Population size (default 500): ") or 500)
            discovery_params['elitism_rate'] = float(input("Elitism rate (default 0.01): ") or 0.01)
            discovery_params['crossover_rate'] = float(input("Crossover rate (default 1.0): ") or 1.0)
            discovery_params['mutation_rate'] = float(input("Mutation rate (default 0.01): ") or 0.01)
            discovery_params['generations'] = int(input("Generations (default 100): ") or 100)
            discovery_params['elitism_min_sample'] = int(input("Elitism min sample (default 5): ") or 5)

    vis_params = None
    vis_prompt = f"\nDo you want a visualization of the {'discovered' if model_path is None else 'provided'} normative model saved (s) or directly displayed (d) or not (N)? (s/d/N): "
    vis_choice = input(vis_prompt).lower().strip()

    if vis_choice in ['s', 'd']:
        vis_params = {}
        # Ask for representation format 
        print("\nChoose model representation for visualization:")
        print("1. Petri Net")
        print("2. BPMN")
        print("3. Process Tree")
        rep_choice = int(input("Select option (1-3): ").strip())
        
        # Map to config.ModelRepresentation Enum
        try: 
            vis_params['representation'] = [
                config.ModelRepresentation.PETRI_NET,
                config.ModelRepresentation.BPMN,
                config.ModelRepresentation.PROCESS_TREE
            ][rep_choice - 1] 
        except (ValueError, IndexError): 
            print("Invalid choice. Defaulting to Petri Net.")
            vis_params['representation'] = config.ModelRepresentation.PETRI_NET

        if vis_choice == 's':
            vis_params['output_path'] = input("Enter the path to save the visualization (e.g., model.png): ").strip()
        else:
            vis_params['output_path'] = None  # This triggers the 'else' (view) in your io.py logic

    
    # Call Input + initial visualization pipeline -> read files, optionally disover model and visualize to user
    try:
        (model_tuple, log), prep_time = evaluation.measure_runtime(
            pipeline.prepare_log_and_model,
            log_path=log_path,
            model_path=model_path,
            discovery_alg=discovery_alg,
            discovery_params=discovery_params,
            vis_params=vis_params
        )
        time_list.append(("Read and Discovery", prep_time))
    
    except Exception as e:
        print(f"\n[SETUP ERROR]: {e}", file=sys.stderr)
        return time_list



    # Target and Feature Configuration
    print("\nWhat should be predicted?")
    print("1. Future conformance deviations")
    print("2. Process outcome")
    print("3. Remaining time")
    
    outcome_activities = []
    deviation_cost_threshold = 0
    try:
        target_choice = int(input("Select option (1-3): ").strip())
        target = [
            config.PredictionTarget.FUTURE_DEVIATIONS,
            config.PredictionTarget.PROCESS_OUTCOME,
            config.PredictionTarget.REMAINING_TIME
        ][target_choice - 1]

        if target == config.PredictionTarget.PROCESS_OUTCOME:
            print("\nDefine the target outcome:")
            print("Enter activity names that signify a outcome (one per line).")
            print("Type 'done' when finished.")
            while True:
                act = input("> ").strip()
                if act.lower() == 'done': break
                if act: outcome_activities.append(act)
        elif target == config.PredictionTarget.FUTURE_DEVIATIONS:
            # HL-08: a case is "deviant" if its real alignment cost (count of genuine
            # log-/visible-model-moves; silent routing excluded) exceeds T. T=0 => "any
            # real deviation"; raise T to tolerate minor deviations.
            raw = input("\nDeviation cost threshold T (default 0): ").strip()
            try:
                deviation_cost_threshold = int(raw) if raw else 0
            except ValueError:
                print("Invalid input. Using T=0.")
                deviation_cost_threshold = 0
    except (ValueError, IndexError):
        print("Invalid choice. Defaulting to Future Conformance Deviations.")
        target = config.PredictionTarget.FUTURE_DEVIATIONS

    is_future_deviations = (target == config.PredictionTarget.FUTURE_DEVIATIONS)

    print("\nWhat should the prediction be based upon?")
    print("1. Plain trace (control-flow)")
    print("2. Trace and alignment-based diagnostics")
    print("3. Case features")
    print("4. Mixture of all available information")
    
    try:
        feature_choice = int(input("Select option (1-4): ").strip())
        feature = [
            config.PredictionFeature.CONTROL_FLOW,
            config.PredictionFeature.DIAGNOSTICS,
            config.PredictionFeature.CASE_ATTRIBUTES,
            config.PredictionFeature.HYBRID
        ][feature_choice - 1]
    except (ValueError, IndexError):
        print("Invalid choice. Defaulting to Control-flow.")
        feature = config.PredictionFeature.CONTROL_FLOW

    # Case-attribute listing + selection (only when case features are involved)
    selected_case_attributes = None
    if feature in (config.PredictionFeature.CASE_ATTRIBUTES, config.PredictionFeature.HYBRID):
        available = feature_extraction.list_case_attributes(log)
        if not available:
            print("\nNo case-level attributes found in the log.")
        else:
            summary = feature_extraction.summarize_case_attributes(log)
            print("\nAvailable case-level attributes:")
            for i, r in enumerate(summary.itertuples(index=False), 1):
                print(f"  {i}. {r.attribute}  (type={r.dtype}, unique={r.n_unique}, e.g. {r.example})")
            raw = input("\nSelect attributes (comma-separated numbers, or 'all'): ").strip().lower()
            if raw in ("", "all"):
                selected_case_attributes = available
            else:
                try:
                    idx = [int(x) - 1 for x in raw.split(",") if x.strip()]
                    selected_case_attributes = [available[i] for i in idx if 0 <= i < len(available)]
                except ValueError:
                    print("Invalid input. Using all attributes.")
                    selected_case_attributes = available
            print(f"Selected: {selected_case_attributes}")

    # Prefix configuration. Default = no prefixing (one row per case, cheapest path:
    # at most one alignment per full case, never per prefix). Prefix work is opt-in only.
    prefix_mode = "none"
    prefix_min, prefix_max, prefix_step = None, None, 1

    is_remaining_time = (target == config.PredictionTarget.REMAINING_TIME)
    median_len = int(log.groupby("case:concept:name").size().median())

    if is_remaining_time:
        # Remaining time is undefined on a complete case (it is 0), so prefixing is
        # mandatory — "none" is not offered. Default to pooled varying 1..median.
        print("\nPrefix mode (required for remaining time)?")
        print(f"  [Enter] varying 1..median ({median_len})  (pooled - prefixes of length 1..k)")
        print("  1)      exactly k          (one prefix per case at length k)")
        choice = input("Select (Enter/1): ").strip()
        if choice == "1":
            try:
                k = int(input("  prefix length k: ").strip())
                prefix_mode, prefix_min, prefix_max, prefix_step = "exactly", k, k, 1
            except ValueError:
                print(f"Invalid input. Using varying 1..{median_len}.")
                prefix_mode, prefix_min, prefix_max = "varying", 1, median_len
        else:
            try:
                km = input(f"  max length k (blank = median {median_len}): ").strip()
                ks = input("  step (default 1): ").strip()
                prefix_max = int(km) if km else median_len
                prefix_step = int(ks) if ks else 1
            except ValueError:
                prefix_max, prefix_step = median_len, 1
            prefix_min, prefix_mode = 1, "varying"
            print(f"  using varying prefixes 1..{prefix_max}, step={prefix_step}")
    elif feature != config.PredictionFeature.CASE_ATTRIBUTES and target in (
        config.PredictionTarget.PROCESS_OUTCOME, config.PredictionTarget.FUTURE_DEVIATIONS,
    ):
        print("\nPrefix mode?")
        print("  [Enter] none      (default - one row per case, cheapest; no per-prefix alignments)")
        print("  1)      exactly k (one prefix per case at length k)")
        print("  2)      varying 1..k (pooled - prefixes of length 1..k)")
        choice = input("Select (Enter/1/2): ").strip()
        if choice == "1":
            try:
                k = int(input("  prefix length k: ").strip())
                prefix_mode, prefix_min, prefix_max, prefix_step = "exactly", k, k, 1
            except ValueError:
                print("Invalid input. Defaulting to no prefixing.")
        elif choice == "2":
            try:
                km = input("  max length k (blank = median case length): ").strip()
                ks = input("  step (default 1): ").strip()
                prefix_max = int(km) if km else median_len
                prefix_step = int(ks) if ks else 1
                prefix_min, prefix_mode = 1, "varying"
                print(f"  using varying prefixes 1..{prefix_max}, step={prefix_step}")
            except ValueError:
                print("Invalid input. Defaulting to no prefixing.")


    #ask for train test split-ratio
    try:
        split_value = float(input("\nEnter a split value for the train- / test-split (default=0.7): ").strip())
    except (ValueError):
        print("Invalid choice. Defaulting to 0.7")
        split_value = 0.7

    (log_train, log_test), split_time = evaluation.measure_runtime(pm4py.split_train_test, log, split_value)
    time_list.append(("Train/Test Split", split_time))

    # Feature extraction. Default (no prefix) uses the per-case path; the opt-in prefix modes
    # use the pooled path, which also returns each row's prefix length for by-length analysis.
    test_prefix_lengths = None
    try:
        if prefix_mode == "none":
            (X_train, y_train, class_names), extract_train_time = evaluation.measure_runtime(
                pipeline.extract_target_and_feature,
                target=target,
                feature=feature,
                model=model_tuple,
                log=log_train,
                outcome_activities=outcome_activities,
                selected_case_attributes=selected_case_attributes,
                deviation_cost_threshold=deviation_cost_threshold,
            )
            time_list.append(("Alignment & Feature Extraction (Train split)", extract_train_time))
            (X_test, y_test, _), extract_test_time = evaluation.measure_runtime(
                pipeline.extract_target_and_feature,
                target=target,
                feature=feature,
                model=model_tuple,
                log=log_test,
                outcome_activities=outcome_activities,
                selected_case_attributes=selected_case_attributes,
                deviation_cost_threshold=deviation_cost_threshold,
            )
            time_list.append(("Alignment & Feature Extraction (Test split)", extract_test_time))
        else:
            (X_train, y_train, class_names, _), extract_train_time = evaluation.measure_runtime(
                pipeline.extract_target_and_feature_over_prefixes,
                target=target,
                feature=feature,
                model=model_tuple,
                log=log_train,
                outcome_activities=outcome_activities,
                selected_case_attributes=selected_case_attributes,
                min_length=prefix_min,
                max_length=prefix_max,
                step=prefix_step,
                deviation_cost_threshold=deviation_cost_threshold,
            )
            time_list.append(("Alignment & Feature Extraction (Train split)", extract_train_time))
            (X_test, y_test, _, test_prefix_lengths), extract_test_time = evaluation.measure_runtime(
                pipeline.extract_target_and_feature_over_prefixes,
                target=target,
                feature=feature,
                model=model_tuple,
                log=log_test,
                outcome_activities=outcome_activities,
                selected_case_attributes=selected_case_attributes,
                min_length=prefix_min,
                max_length=prefix_max,
                step=prefix_step,
                deviation_cost_threshold=deviation_cost_threshold,
            )
            time_list.append(("Alignment & Feature Extraction (Test split)", extract_test_time))
    except Exception as e:
        print(f"\n[EXTRACTION ERROR]: {e}", file=sys.stderr)
        return time_list

    # ML Algorithm Configuration
    ml_params = {}
    regression_algorithm = None      # set only for REMAINING_TIME
    classification_algorithm = None  # set only for FUTURE_DEVIATIONS

    if is_remaining_time:
        # Regression menu — the "Algorithm 1/2/3" slots map to concrete regressors.
        print("\nWhich regression algorithm do you want to apply?")
        print("(1) Decision Tree Regressor")
        print("(2) Random Forest Regressor")
        print("(3) Gradient Boosting Regressor")
        try:
            ml_choice = int(input("Select option (1-3): ").strip())
            regression_algorithm = [
                config.RegressionAlgorithm.DECISION_TREE,
                config.RegressionAlgorithm.RANDOM_FOREST,
                config.RegressionAlgorithm.GRADIENT_BOOSTING,
            ][ml_choice - 1]
        except (ValueError, IndexError):
            print("Invalid choice. Defaulting to Decision Tree Regressor.")
            ml_choice = 1
            regression_algorithm = config.RegressionAlgorithm.DECISION_TREE
        print(f"{regression_algorithm.value} selected.")

        use_custom_ml = input("\nDo you want to configure the parameters yourself? (y/N): ").lower().strip() == 'y'
        if use_custom_ml and regression_algorithm == config.RegressionAlgorithm.DECISION_TREE:
            print("\nConfiguring Decision Tree Regressor Parameters:")
            max_depth = input("Max depth (int, default 4): ").strip()
            if max_depth:
                ml_params["max_depth"] = int(max_depth)
            min_samples_split = input("Min samples split (int, default 2): ").strip()
            if min_samples_split:
                ml_params["min_samples_split"] = int(min_samples_split)
            min_samples_leaf = input("Min samples leaf (int, default 1): ").strip()
            if min_samples_leaf:
                ml_params["min_samples_leaf"] = int(min_samples_leaf)
        elif use_custom_ml:
            print("\nConfiguring ensemble parameters:")
            n_estimators = input("Number of estimators (int, default 100): ").strip()
            if n_estimators:
                ml_params["n_estimators"] = int(n_estimators)
            max_depth = input("Max depth (int, blank = default): ").strip()
            if max_depth:
                ml_params["max_depth"] = int(max_depth)

        print("\nTraining model...")
        trained_model, train_time = evaluation.measure_runtime(
            model_training.train_regression_model,
            X_train,
            y_train,
            regression_algorithm,
            ml_params,
        )
        time_list.append(("Model Training", train_time))
    elif is_future_deviations:
        # FUTURE_DEVIATIONS is risk classification: probabilistic classifiers whose
        # predict_proba gives a calibrated-ish deviation risk score.
        print("\nWhich classification algorithm do you want to apply?")
        print("(1) Logistic Regression")
        print("(2) Gradient Boosting")
        try:
            ml_choice = int(input("Select option (1-2): ").strip())
            classification_algorithm = [
                config.ClassificationAlgorithm.LOGISTIC_REGRESSION,
                config.ClassificationAlgorithm.GRADIENT_BOOSTING,
            ][ml_choice - 1]
        except (ValueError, IndexError):
            print("Invalid choice. Defaulting to Logistic Regression.")
            ml_choice = 1
            classification_algorithm = config.ClassificationAlgorithm.LOGISTIC_REGRESSION
        print(f"{classification_algorithm.value} selected.")

        use_custom_ml = input("\nDo you want to configure the parameters yourself? (y/N): ").lower().strip() == 'y'
        if use_custom_ml and classification_algorithm == config.ClassificationAlgorithm.LOGISTIC_REGRESSION:
            print("\nConfiguring Logistic Regression Parameters:")
            c_val = input("Inverse regularization strength C (float, default 1.0): ").strip()
            if c_val:
                ml_params["C"] = float(c_val)
        elif use_custom_ml:  # Gradient Boosting
            print("\nConfiguring Gradient Boosting Parameters:")
            n_estimators = input("Number of estimators (int, default 100): ").strip()
            if n_estimators:
                ml_params["n_estimators"] = int(n_estimators)
            max_depth = input("Max depth (int, default 3): ").strip()
            if max_depth:
                ml_params["max_depth"] = int(max_depth)

        print("\nTraining model...")
        trained_model, train_time = evaluation.measure_runtime(
            model_training.train_classification_model,
            X_train,
            y_train,
            classification_algorithm,
            ml_params,
        )
        time_list.append(("Model Training", train_time))
    else:
        print("\nWhich ML algorithm do you want to apply on your data?")
        print("(1) Decision Tree")
        print("(2) Algorithm 2")
        print("(3) Algorithm 3")
        try:
            ml_choice = int(input("Select option (1-3): ").strip())
            if ml_choice == 1:
                print("Decision Tree selected.")
            elif ml_choice == 2:
                print("Algorithm 2 selected.")
            elif ml_choice == 3:
                print("Algorithm 3 selected.")
        except (ValueError, IndexError):
            print("Invalid choice. Defaulting to Decision Tree.")
            ml_choice = 1

        use_custom_ml = input("\nDo you want to configure the parameters yourself? (y/N): ").lower().strip() == 'y'
        if use_custom_ml and ml_choice == 1:  # Decision Tree
            print("\nConfiguring Decision Tree Parameters:")
            criterion = input("Criterion (gini/entropy, default=gini): ").strip().lower()
            if criterion in ["gini", "entropy"]:
                ml_params["criterion"] = criterion
            max_depth = input("Max depth (int, default 4): ").strip()
            if max_depth:
                ml_params["max_depth"] = int(max_depth)
            min_samples_split = input("Min samples split (int, default 20): ").strip()
            if min_samples_split:
                ml_params["min_samples_split"] = int(min_samples_split)
            min_samples_leaf = input("Min samples leaf (int, default 1): ").strip()
            if min_samples_leaf:
                ml_params["min_samples_leaf"] = int(min_samples_leaf)

        # Train initial model
        print("\nTraining model...")
        trained_model, train_time = evaluation.measure_runtime(
            model_training.train_model,
            X_train,
            y_train,
            ml_params
        )
        time_list.append(("Model Training", train_time))

    # Decision Tree visualisation + reconfigure loop — only for a single Decision Tree
    # (classification DT, or the Decision Tree Regressor for remaining time).
    is_process_outcome = (not is_remaining_time and not is_future_deviations)
    show_tree = (
        (is_remaining_time and regression_algorithm == config.RegressionAlgorithm.DECISION_TREE)
        or (is_process_outcome and ml_choice == 1)
    )
    if show_tree:
        while True:
            dt_vis_choice = input("\nDo you want the decision tree saved (s), displayed (d) or not shown (N)? (s/d/N): ").lower().strip()

            if dt_vis_choice in ['s', 'd']:
                dt_output_path = None
                if dt_vis_choice == 's':
                    dt_output_path = input("Enter the path to save the decision tree (e.g., tree.png): ").strip()
                vis.visualize_decision_tree(
                    model=trained_model,
                    feature_names=X_train.columns.tolist(),
                    class_names=class_names,  # [] for regression -> handled as None
                    output_path=dt_output_path
                )
                if dt_vis_choice == 's':
                    print(f"Decision tree saved to: {dt_output_path}")

            if input("\nDo you want to reconfigure the hyperparameters and retrain? (y/N): ").lower().strip() == 'y':
                ml_params = {}
                print("\nConfiguring Decision Tree Parameters:")
                # Criterion (gini/entropy) only applies to the classifier.
                if not is_remaining_time:
                    criterion = input("Criterion (gini/entropy, default gini): ").strip().lower()
                    if criterion in ["gini", "entropy"]:
                        ml_params["criterion"] = criterion
                max_depth = input("Max depth (int, default 4): ").strip()
                if max_depth:
                    ml_params["max_depth"] = int(max_depth)
                min_samples_split = input("Min samples split (int, default 2): ").strip()
                if min_samples_split:
                    ml_params["min_samples_split"] = int(min_samples_split)
                min_samples_leaf = input("Min samples leaf (int, default 1): ").strip()
                if min_samples_leaf:
                    ml_params["min_samples_leaf"] = int(min_samples_leaf)
                print("Retraining model...")
                if is_remaining_time:
                    trained_model, train_time = evaluation.measure_runtime(
                        model_training.train_regression_model,
                        X_train, y_train, regression_algorithm, ml_params,
                    )
                else:
                    trained_model, train_time = evaluation.measure_runtime(
                        model_training.train_model,
                        X_train, y_train, ml_params,
                    )
                # Update the last training time
                if time_list and time_list[-1][0] == "Model Training":
                    time_list[-1] = ("Model Training", train_time)
            else:
                break

    # Call evaluation pipeline (training already done above)
    print("\nStarting evaluation pipeline...")
    try:
        # This calls the orchestration logic in src/pipeline.py
        results, eval_time = evaluation.measure_runtime(
            pipeline.run_conformance_prediction_pipeline,
            X_test=X_test,
            y_test=y_test,
            X_train=X_train,
            y_train=y_train,
            ml_params=ml_params,
            preloaded_model=trained_model,
            test_prefix_lengths=test_prefix_lengths,
            is_regression=is_remaining_time,
        )
        time_list.append(("Evaluation", eval_time))
        print("\nPipeline completed successfully!")

        y_pred = results["y_pred"]
        y_test_arr = results["y_test"]

        if is_remaining_time:
            # Regression result reporting (metrics, by-prefix-length error table, and the
            # adaptive duration-unit display) is the evaluation module's responsibility
            # (a colleague's part). The pipeline already attaches whatever metrics that
            # module computed to results["metrics"]; echo them if present.
            metrics = results.get("metrics") or {}
            if metrics:
                print("\nRegression metrics:")
                for name, value in metrics.items():
                    print(f"  {name}: {value}")
            else:
                print("\nPrediction complete. Remaining-time reporting (metrics, "
                      "by-prefix-length errors, adaptive duration units) is handled by "
                      "the evaluation module — not yet available.")
        else:
            # Display confusion matrix and per-class stats in the CLI
            vis.display_confusion_matrix_text(y_test_arr, y_pred, class_names)
            vis.display_per_class_stats(y_test_arr, y_pred, class_names)

            # Prefix-length performance breakdown (HL-17) — only meaningful when lengths vary.
            if test_prefix_lengths is not None and len(set(test_prefix_lengths)) > 1:
                by_len = evaluation.metrics_by_prefix_length(
                    y_test_arr, y_pred, results["y_prob"], test_prefix_lengths
                )
                print("\nMetrics by prefix length:")
                print(by_len.to_string(index=False))

            # Offer to save both visualisations
            if input("\nSave confusion matrix and accuracy chart? (y/N): ").lower().strip() == 'y':
                path_input = input("Folder to save in (press Enter for Downloads): ").strip()
                base_dir = path_input if path_input else None
                saved_to = vis.save_evaluation_visuals(y_test_arr, y_pred, class_names, base_dir)
                print(f"Results saved to: {saved_to}")

        # 5. Prediction on new data
        if input("\nDo you want to perform predictions on new data? (y/N): ").lower().strip() == 'y':
            new_data_path = input("Enter the path to the file containing trace prefixes: ").strip()
            # logic to call prediction
            
    except Exception as e:
        print(f"\n[EVALUATION ERROR]: {e}", file=sys.stderr)
        return time_list

    return time_list
    
if __name__ == "__main__":
    time_list, mem_peak = evaluation.measure_memory(run_interactive_cli)
    print(f"The program consumed {mem_peak} MB memory in peak.")
    total_time = 0
    for (name, duration) in time_list:
        print("---------------------------------------")
        print(f"The program took {duration:.4f} seconds to execute the following step: \n{name}")
        total_time += duration
   
    print("---------------------------------------")
    print(f"In total, it took {total_time:.4f} seconds from start to finish (processing time only).")

    # Save as JSON (on user selection after last step of CLI)
    if input("\nDo you want to save the runtime and memory measurements as JSON? (y/N): ").lower().strip() == 'y':
        report_path = input("Enter filename (default: report.json): ").strip() or "report.json"
        report_data = {
            "runtimes": {name: duration for name, duration in time_list},
            "peak_memory_mb": mem_peak,
            "total_processing_time": total_time
        }
        try:
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=4)
            print(f"Report saved to: {report_path}")
        except Exception as e:
            print(f"Failed to save report: {e}", file=sys.stderr)
