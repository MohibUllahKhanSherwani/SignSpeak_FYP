"""
evaluate_models.py
==================
SignSpeak FYP Diagnostic Suite.
Generates objective performance metrics including precision, recall, 
F1-score, confusion analysis, and hardware latency statistics.
"""

import os
import time
import json
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from actions_config import load_actions, SEQUENCE_LENGTH
from train_combined import load_combined_data, DATA_SOURCES

# -- Hardcoded Model Paths --
MODEL_BASELINE    = "all_models/action_model_baseline_new.h5"
MODEL_AUGMENTED   = "all_models/action_model_augmented_new.h5"
ENCODER_BASELINE  = "all_models/label_encoder_baseline_new.pkl"
ENCODER_AUGMENTED = "all_models/label_encoder_augmented_new.pkl"

def run_diagnostics(model_path, encoder_path, mode_name):
    """
    Executes a clinical evaluation of a specific model instance.
    """
    print("-" * 70)
    print(f"EVALUATION REPORT: {mode_name}")
    print("-" * 70)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None

    # 1. Load Resources
    print(f"Status: Loading model {os.path.basename(model_path)}...")
    model = load_model(model_path)
    le = joblib.load(encoder_path)
    actions = le.classes_
    
    # 2. Load Data
    # Using load_combined_data from train_combined to ensure we test on full dataset
    X, y = load_combined_data(actions, DATA_SOURCES)
    
    # 3. Predict and Measure Latency
    print(f"Status: Running inference on {len(X)} test sequences...")
    start_time = time.time()
    predictions = model.predict(X, verbose=0)
    total_time = (time.time() - start_time) * 1000 # ms
    latency = total_time / len(X)
    
    y_pred = np.argmax(predictions, axis=1)
    y_true = le.transform(y)
    
    # 4. Global Metrics
    report = classification_report(y_true, y_pred, target_names=actions, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n[GLOBAL METRICS]")
    print(f"Overall Accuracy : {report['accuracy']*100:.2f}%")
    print(f"Average Latency  : {latency:.2f} ms / sequence")
    print(f"Total Test Time  : {total_time/1000:.2f} seconds")
    
    # 5. Class-wise Analysis
    f1_scores = [(action, report[action]['f1-score']) for action in actions]
    f1_scores.sort(key=lambda x: x[1])
    
    print(f"\n[CLASS PERFORMANCE: LOWEST]")
    for i in range(min(5, len(f1_scores))):
        action, score = f1_scores[i]
        print(f"  Rank {i+1}: {action:<20} F1: {score:.4f}")
        
    print(f"\n[CLASS PERFORMANCE: HIGHEST]")
    for i in range(1, min(6, len(f1_scores)+1)):
        action, score = f1_scores[-i]
        print(f"  Rank {i}: {action:<20} F1: {score:.4f}")

    # 6. Confusion Matrix Analysis
    print(f"\n[INTER-CLASS CONFUSION]")
    confusion_found = False
    for i in range(len(actions)):
        for j in range(len(actions)):
            if i != j and cm[i, j] > 0:
                print(f"  Class '{actions[i]}' misclassified as '{actions[j]}' ({cm[i,j]} instances)")
                confusion_found = True
    
    if not confusion_found:
        print("  No misclassifications detected in the current test set.")

    # 7. Save results for documentation
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"eval_{mode_name.lower().replace(' ', '_')}_{int(time.time())}.json"
    save_path = os.path.join(output_dir, filename)
    
    # JSON-friendly dict
    results = {
        "model": mode_name,
        "accuracy": report['accuracy'],
        "latency_ms": latency,
        "full_report": report
    }
    
    with open(save_path, 'w') as f:
        # Convert classification report to string to avoid serialization issues if nested
        json.dump(results, f, indent=4)
        
    print(f"\nSummary: Full telemetry saved to {save_path}")
    print("-" * 70)
    
    return {
        "accuracy": report['accuracy'],
        "latency": latency,
        "report": report,
        "mode": mode_name
    }

def print_comparison(b_res, a_res):
    """
    Prints a side-by-side comparison of Baseline vs Augmented metrics.
    """
    print("\n" + "="*70)
    print("FINAL COMPARISON: BASELINE VS AUGMENTED")
    print("="*70)
    print(f"{'Metric':<25} {'Baseline':<20} {'Augmented':<20} {'Delta'}")
    print("-" * 70)
    
    b_acc = b_res['accuracy'] * 100
    a_acc = a_res['accuracy'] * 100
    acc_delta = a_acc - b_acc
    
    print(f"{'Accuracy (%)':<25} {b_acc:<20.2f} {a_acc:<20.2f} {acc_delta:+.2f}%")
    print(f"{'Latency (ms)':<25} {b_res['latency']:<20.2f} {a_res['latency']:<20.2f} {a_res['latency'] - b_res['latency']:+.2f} ms")
    
    print("\n[VERDICT]")
    if acc_delta > 1.0:
        print(f"Augmentation improved accuracy by {acc_delta:.2f}%. Recommended for production.")
    elif acc_delta < -1.0:
        print(f"Baseline outperformed augmented by {abs(acc_delta):.2f}%. Check augmentation parameters.")
    else:
        print("Both models performed within 1% of each other.")
    print("="*70 + "\n")

def main():
    print("SignSpeak Framework Evaluation Suite")
    print("Version: 1.0.0 (Research Edition)")
    
    baseline_results = None
    augmented_results = None
    
    # Evaluate Baseline
    if os.path.isfile(MODEL_BASELINE):
        baseline_results = run_diagnostics(MODEL_BASELINE, ENCODER_BASELINE, "Baseline Model")
    
    # Evaluate Augmented
    if os.path.isfile(MODEL_AUGMENTED):
        augmented_results = run_diagnostics(MODEL_AUGMENTED, ENCODER_AUGMENTED, "Augmented Model")
        
    # Compare if both exist
    if baseline_results and augmented_results:
        print_comparison(baseline_results, augmented_results)
    elif not baseline_results and not augmented_results:
        print("Error: No models found for evaluation. Please train models first.")


if __name__ == "__main__":
    main()
