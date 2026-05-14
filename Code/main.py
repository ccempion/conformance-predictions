"""
Main entry point for the Conformance Predictions software.
Provides an interactive CLI to guide users through the process mining and 
machine learning pipeline as specified in the Requirements Engineering document.
"""

import sys
from typing import Optional
from src import io, conformance, feature_extraction, model_training, evaluation, pipeline, config

def run_interactive_cli() -> None:
    """
    Guides the user through configuration steps as an interactive command-line-interface program: (HL-03) & triggers the pipeline.
    """
    print("=== Conformance Predictions ===")
    
    # Data and Model Input
    has_model = input("Do you have a normative process model? (y/n): ").lower().strip() == 'y'
    
    log_path = input("Enter the path to the event log (XES/CSV): ").strip()
    
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
            discovery_params['disable_fallthroughs'] = input("Disable fall-throughs? (y/n, default n): ").lower() == 'y'

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
        model_tuple, log_df = pipeline.prepare_log_and_model(
            log_path=log_path,
            model_path=model_path,
            discovery_alg=discovery_alg,
            discovery_params=discovery_params,
            vis_params=vis_params
        )
    
    except Exception as e:
        print(f"\n[SETUP ERROR]: {e}", file=sys.stderr)
        sys.exit(1)

    # Target and Feature Configuration
    print("\nWhat should be predicted?")
    print("1. Future conformance deviations")
    print("2. Process outcome")
    print("3. Remaining time")
    
    try:
        target_choice = int(input("Select option (1-3): ").strip())
        target = [
            config.PredictionTarget.FUTURE_DEVIATIONS, 
            config.PredictionTarget.PROCESS_OUTCOME, 
            config.PredictionTarget.REMAINING_TIME
        ][target_choice - 1]
        
        # for target == outcome
        outcome_activities = []
        if target == config.PredictionTarget.PROCESS_OUTCOME:
            print("\nDefine the target outcome:")
            print("Enter activity names that signify a outcome (one per line).")
            print("Type 'done' when finished.")
            while True:
                act = input("> ").lower().strip()
                if act == 'done': break
                if act: outcome_activities.append(act)
    except (ValueError, IndexError):
        print("Invalid choice. Defaulting to Future Conformance Deviations.")
        target = config.PredictionTarget.FUTURE_DEVIATIONS

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

    # ML Algorithm Configuration
    
    print("\nWhich ML algorithm do you want to apply on your data?")
    print("Decision Tree")
    print("Algorithm 2")
    print("Algorithm 3")
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
        ml_choice = int(input("Select option (1-3): ").strip())
        feature = [
            config.Algorithm.DECISION_TREE,
            config.Algorithm.ALGORITHM_2,
            config.Algorithm.ALGORITHM_3,
        ][ml_choice - 1]

    use_custom_ml = input("\nDo you want to configure the {ml_choice}? (y/n): ").lower().strip() == 'y'

    #here we put all the parameters together
    ml_params = {}
    if use_custom_ml== "y" and ml_choice == 1:  # Decision Tree
        print("\nConfiguring Decision Tree Parameters:")
        criterion = input("Criterion (gini/entropy, default gini): ").strip().lower()
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
    # Call model training pipeline 
    print("\nStarting execution pipeline...")
    try:
        # This calls the orchestration logic in src/pipeline.py
        results = pipeline.run_conformance_prediction_pipeline(
            log_path=log_path,
            model_path=model_path,
            discovery_alg=discovery_alg,
            discovery_params=discovery_params,
            target=str(target_choice),
            features=str(feature_choice),
            outcome_activities=outcome_activities,
            ml_params=ml_params,
            preloaded_model=model_tuple,
            preloaded_log=log_df
        )
        print("\nPipeline completed successfully!")
        
        # 5. Prediction on new data
        if input("\nDo you want to perform predictions on new data? (y/n): ").lower().strip() == 'y':
            new_data_path = input("Enter the path to the file containing trace prefixes: ").strip()
            # logic to call prediction
            
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    run_interactive_cli()
