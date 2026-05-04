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
    if has_model:
        model_path = input("Enter the path to the normative model (Petri net/BPMN): ").strip()
    else:
        print("Model discovery will be used.")
        # Prompt for discovery algorithm & noise thresholds
        # print("\nWhich discovery algorithm should be used?")
        # print("1. Inductive Miner (recommended)")
        # print("2. Alpha Miner")
        # print("3. Heuristic Miner")
        # discovery_alg_choice = input("Select option (1-3):").strip()
        

    # Prediction Configuration
    print("\nWhat should be predicted?")
    print("1. Future conformance deviations")
    print("2. Process outcome")
    target_choice = input("Select option (1-2): ").strip()
    
    print("\nWhat should the prediction be based upon?")
    print("1. Plain trace (control-flow)")
    print("2. Trace and alignment-based diagnostics")
    print("3. Case features")
    print("4. Mixture of all available information")
    feature_choice = input("Select option (1-4): ").strip()


    # 3. ML Algorithm Configuration
    use_custom_ml = input("\nDo you want to configure the ML algorithm? (y/n): ").lower().strip() == 'y'
    ml_params = {}
    if use_custom_ml:
        # Placeholder for ML configuration logic
        print("\nWhich ML algorithm should be used?")
        # print("1. Neural Network")
        # print("2. Decision Tree")
        # print("3. Regression")
        # ml_alg_choice = input("Select option (1-3): ").strip()


    # 4. Trigger Pipeline
    print("\nStarting execution pipeline...")
    try:
        # This calls the orchestration logic in src/pipeline.py
        results = pipeline.run_conformance_prediction_pipeline(
            log_path=log_path,
            model_path=model_path,
            target=target_choice,
            features=feature_choice,
            ml_params=ml_params
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
