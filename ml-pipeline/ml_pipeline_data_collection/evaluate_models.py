import os
import time
import json
import numpy as np
import joblib
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from actions_config import load_actions, SEQUENCE_LENGTH
from train_combined import load_combined_data, DATA_SOURCES

MODEL_BASELINE    = "all_models/action_model_baseline_new.h5"
MODEL_AUGMENTED   = "all_models/action_model_augmented_new.h5"
ENCODER_BASELINE  = "all_models/label_encoder_baseline_new.pkl"
ENCODER_AUGMENTED = "all_models/label_encoder_augmented_new.pkl"

TEST_SIZE    = 0.2
RANDOM_STATE = 42


def run_diagnostics(model_path, encoder_path, mode_name):
    print("\n" + "=" * 70)
    print(f"  EVALUATION REPORT: {mode_name}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    if not os.path.exists(model_path):
        print(f"  Error: Model file not found at {model_path}")
        return None

    print(f"\nLoading model: {os.path.basename(model_path)}")
    model = load_model(model_path)
    le = joblib.load(encoder_path)
    actions = le.classes_
    
    print(f"Loading dataset from {len(DATA_SOURCES)} source(s)...")
    X_full, y_full = load_combined_data(actions, DATA_SOURCES)
    
    y_encoded = le.transform(y_full)
    
    _, X_test, _, y_test_encoded = train_test_split(
        X_full, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )
    
    y_test_labels = le.inverse_transform(y_test_encoded)
    
    total_samples = len(X_full)
    test_samples  = len(X_test)
    print(f"\nDataset: {total_samples} total → {test_samples} test sequences (held-out {TEST_SIZE*100:.0f}%)")
    print(f"Classes: {len(actions)} signs")
    print(f"Split:   random_state={RANDOM_STATE}, stratified=True")
    
    print(f"\nRunning inference on {test_samples} test sequences...")
    
    _ = model.predict(X_test[:1], verbose=0)
    
    start_time = time.time()
    predictions = model.predict(X_test, verbose=0)
    total_time_ms = (time.time() - start_time) * 1000
    avg_latency = total_time_ms / test_samples
    
    y_pred = np.argmax(predictions, axis=1)
    
    report = classification_report(
        y_test_encoded, y_pred,
        target_names=actions,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test_encoded, y_pred)
    
    accuracy  = report["accuracy"]
    macro_f1  = report["macro avg"]["f1-score"]
    weighted_f1 = report["weighted avg"]["f1-score"]
    
    print(f"\n  {'─' * 50}")
    print(f"RESULTS ON HELD-OUT TEST SET ({test_samples} sequences)")
    print(f"  {'─' * 50}")
    print(f"  Overall Accuracy  : {accuracy*100:.2f}%")
    print(f"  Macro F1-Score    : {macro_f1*100:.2f}%")
    print(f"  Weighted F1-Score : {weighted_f1*100:.2f}%")
    print(f"  Avg Latency       : {avg_latency:.2f} ms / sequence")
    print(f"  Total Inference   : {total_time_ms/1000:.2f} seconds")
    
    class_metrics = []
    for action in actions:
        m = report[action]
        class_metrics.append({
            "sign": action,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1-score"],
            "support": int(m["support"]),
        })
    
    class_metrics_sorted = sorted(class_metrics, key=lambda x: x["f1"])
    
    print(f"\n  {'Sign':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print(f"  {'─'*60}")
    for m in class_metrics_sorted:
        print(f"  {m['sign']:<20} {m['precision']*100:>9.1f}% {m['recall']*100:>9.1f}% {m['f1']*100:>9.1f}% {m['support']:>10}")
    print(f"  {'─'*60}")
    print(f"  {'MACRO AVG':<20} {report['macro avg']['precision']*100:>9.1f}% {report['macro avg']['recall']*100:>9.1f}% {macro_f1*100:>9.1f}% {int(report['macro avg']['support']):>10}")
    print(f"  {'WEIGHTED AVG':<20} {report['weighted avg']['precision']*100:>9.1f}% {report['weighted avg']['recall']*100:>9.1f}% {weighted_f1*100:>9.1f}% {int(report['weighted avg']['support']):>10}")

    print(f"\nINTER-CLASS CONFUSION (misclassifications only)")
    print(f"  {'─'*60}")
    
    confusions = []
    for i in range(len(actions)):
        for j in range(len(actions)):
            if i != j and cm[i, j] > 0:
                confusions.append((actions[i], actions[j], int(cm[i, j])))
    
    if confusions:
        confusions.sort(key=lambda x: x[2], reverse=True)
        for true_label, pred_label, count in confusions:
            print(f"  '{true_label}' ->'{pred_label}' ({count} instances)")
    else:
        print("No misclassifications detected - perfect classification!")

    max_confidences = np.max(predictions, axis=1)
    correct_mask = (y_pred == y_test_encoded)
    
    print(f"\nCONFIDENCE DISTRIBUTION")
    print(f"  {'─'*60}")
    print(f"  Overall   → Mean: {np.mean(max_confidences)*100:.1f}%, "
          f"Min: {np.min(max_confidences)*100:.1f}%, "
          f"Max: {np.max(max_confidences)*100:.1f}%")
    if correct_mask.sum() > 0:
        print(f"  Correct   → Mean: {np.mean(max_confidences[correct_mask])*100:.1f}%")
    if (~correct_mask).sum() > 0:
        print(f"  Incorrect → Mean: {np.mean(max_confidences[~correct_mask])*100:.1f}%")
    
    low_conf = np.sum(max_confidences < 0.7)
    if low_conf > 0:
        print(f"{low_conf} predictions ({low_conf/test_samples*100:.1f}%) had confidence < 70%")

    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_{mode_name.lower().replace(' ', '_')}_{timestamp}.json"
    save_path = os.path.join(output_dir, filename)
    
    results_json = {
        "model": mode_name,
        "model_file": os.path.basename(model_path),
        "evaluation_date": datetime.now().isoformat(),
        "dataset": {
            "total_samples": total_samples,
            "test_samples": test_samples,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
            "num_classes": len(actions),
            "classes": actions.tolist(),
        },
        "metrics": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "avg_latency_ms": avg_latency,
        },
        "per_class": class_metrics,
        "confusion_matrix": cm.tolist(),
        "confidence_stats": {
            "mean": float(np.mean(max_confidences)),
            "min": float(np.min(max_confidences)),
            "max": float(np.max(max_confidences)),
            "low_confidence_count": int(low_conf),
        },
        "classification_report": report,
    }
    
    with open(save_path, "w") as f:
        json.dump(results_json, f, indent=4)
        
    print(f"\nFull results saved to: {save_path}")
    print("=" * 70)
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "latency": avg_latency,
        "report": report,
        "mode": mode_name,
    }


def print_comparison(b_res, a_res):
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON: BASELINE vs AUGMENTED")
    print("=" * 70)
    print(f"  {'Metric':<25} {'Baseline':>12} {'Augmented':>12} {'Delta':>12}")
    print(f"  {'─'*61}")
    
    b_acc = b_res["accuracy"] * 100
    a_acc = a_res["accuracy"] * 100
    b_f1  = b_res["macro_f1"] * 100
    a_f1  = a_res["macro_f1"] * 100
    
    print(f"  {'Accuracy (%)':<25} {b_acc:>11.2f}% {a_acc:>11.2f}% {a_acc - b_acc:>+11.2f}%")
    print(f"  {'Macro F1 (%)':<25} {b_f1:>11.2f}% {a_f1:>11.2f}% {a_f1 - b_f1:>+11.2f}%")
    print(f"  {'Latency (ms)':<25} {b_res['latency']:>11.2f}  {a_res['latency']:>11.2f}  {a_res['latency'] - b_res['latency']:>+11.2f}")
    
    acc_delta = a_acc - b_acc
    
    print(f"\n  🏆 VERDICT")
    print(f"  {'─'*61}")
    if acc_delta > 1.0:
        print(f"  Augmentation improved accuracy by {acc_delta:.2f}%.")
        print(f"     Recommended model: AUGMENTED")
    elif acc_delta < -1.0:
        print(f"  Baseline outperformed augmented by {abs(acc_delta):.2f}%.")
        print(f"     Recommended model: BASELINE")
    else:
        print(f"  Both models performed within 1% of each other.")
        print(f"     Either model is suitable for deployment.")
    print("=" * 70 + "\n")


def main():
    print("\n" + "=" * 70)
    print("  SignSpeak – Model Evaluation Suite")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Method: Held-out test set ({TEST_SIZE*100:.0f}%, stratified, seed={RANDOM_STATE})")
    print("=" * 70)
    
    baseline_results = None
    augmented_results = None
    
    if os.path.isfile(MODEL_BASELINE):
        baseline_results = run_diagnostics(
            MODEL_BASELINE, ENCODER_BASELINE, "Baseline Model"
        )
    else:
        print(f"\n  Baseline model not found: {MODEL_BASELINE}")
    
    if os.path.isfile(MODEL_AUGMENTED):
        augmented_results = run_diagnostics(
            MODEL_AUGMENTED, ENCODER_AUGMENTED, "Augmented Model"
        )
    else:
        print(f"\nAugmented model not found: {MODEL_AUGMENTED}")
        
    if baseline_results and augmented_results:
        print_comparison(baseline_results, augmented_results)
    elif not baseline_results and not augmented_results:
        print("\nError: No models found. Please run train_combined.py first.")


if __name__ == "__main__":
    main()
