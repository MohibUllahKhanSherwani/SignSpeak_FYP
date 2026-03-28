import os
import csv
from datetime import datetime

DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_LOG = os.path.join(DIR, "training_history.csv")
COMPARISON_LOG = os.path.join(DIR, "comparison_history.csv")

def log_training_session(duration_seconds, num_words, training_acc, val_acc, 
                         epochs, batch_size, augmented, model_path):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    file_exists = os.path.isfile(TRAINING_LOG)
    
    with open(TRAINING_LOG, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Date", "Time", "Duration (s)", "Words (Classes)", 
                "Train Accuracy", "Val Accuracy", "Epochs", 
                "Batch Size", "Augmented", "Model Saved As"
            ])
        
        writer.writerow([
            date_str, time_str, round(duration_seconds, 2), num_words,
            f"{training_acc:.4f}", f"{val_acc:.4f}", epochs,
            batch_size, "Yes" if augmented else "No", os.path.basename(model_path)
        ])
    
    print(f"Training session logged to {os.path.basename(TRAINING_LOG)}")


def log_comparison_session(total_duration, baseline_duration, augmented_duration, 
                           num_words, baseline_acc, augmented_acc):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    improvement = augmented_acc - baseline_acc
    
    file_exists = os.path.isfile(COMPARISON_LOG)
    
    with open(COMPARISON_LOG, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Date", "Time", "Total Duration (s)", "Baseline Dur (s)", 
                "Augmented Dur (s)", "Words", "Baseline Acc", 
                "Augmented Acc", "Improvement"
            ])
        
        writer.writerow([
            date_str, time_str, round(total_duration, 2), 
            round(baseline_duration, 2), round(augmented_duration, 2),
            num_words, f"{baseline_acc:.4f}", f"{augmented_acc:.4f}", 
            f"{improvement:+.4f}"
        ])
    
    print(f"Comparison session logged to {os.path.basename(COMPARISON_LOG)}")
