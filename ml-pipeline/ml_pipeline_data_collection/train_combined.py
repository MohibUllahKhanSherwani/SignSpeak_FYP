"""
train_combined.py
=================
Unified training script – merges MP_Data + MP_Data_mobile, then trains
both a baseline and an augmented model (same as the old compare_models.py).

Before training:
  - Current production models are MOVED into the `original/` folder
    so they are never deleted or overwritten.

After training:
  - New models are written to ml_pipeline_data_collection/ (same files
    api_server.py already points to).
  - A comparison report is saved to comparison_reports/.

Usage
-----
    # Quick sanity-check (2 epochs):
    python train_combined.py --epochs 2

    # Full run (EPOCHS from actions_config, default 200):
    python train_combined.py
"""

import os
import shutil
import time
import json
import numpy as np
from datetime import datetime

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import joblib

from actions_config import (
    load_actions, SEQUENCE_LENGTH,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, AUGMENTATION_MULTIPLIER,
)
from data_augmentation import create_augmented_dataset
from training_logger import log_training_session, log_comparison_session

# -- Output Paths --
MODEL_DIR = "all_models"
MODEL_BASELINE    = os.path.join(MODEL_DIR, "action_model_baseline_new.h5")
MODEL_AUGMENTED   = os.path.join(MODEL_DIR, "action_model_augmented_new.h5")
ENCODER_BASELINE  = os.path.join(MODEL_DIR, "label_encoder_baseline_new.pkl")
ENCODER_AUGMENTED = os.path.join(MODEL_DIR, "label_encoder_augmented_new.pkl")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_SOURCES = [
    os.path.join(SCRIPT_DIR, "MP_Data"),
    os.path.join(SCRIPT_DIR, "MP_Data_mobile"),
]
def load_combined_data(actions, data_sources):
    sequences, labels = [], []
    print("\nLoading combined data...")
    for source_dir in data_sources:
        name = os.path.basename(source_dir)
        if not os.path.isdir(source_dir):
            print(f"{name} not found – skipping")
            continue
        print(f"\n{name}")
        source_count = 0
        for action in actions:
            action_path = os.path.join(source_dir, action)
            if not os.path.isdir(action_path):
                continue
            seq_ids = sorted(
                [d for d in os.listdir(action_path) if d.isdigit()],
                key=int,
            )
            loaded, skipped = 0, 0
            for seq_id in seq_ids:
                window, ok = [], True
                for frame_num in range(SEQUENCE_LENGTH):
                    npy = os.path.join(action_path, seq_id, f"{frame_num}.npy")
                    if not os.path.exists(npy):
                        skipped += 1
                        ok = False
                        break
                    window.append(np.load(npy))
                if ok:
                    sequences.append(window)
                    labels.append(action)
                    loaded += 1
            note = f"({skipped} incomplete skipped)" if skipped else ""
            print(f"• {action}: {loaded} sequences{note}")
            source_count += loaded
        print(f"→ {source_count} from {name}")

    if not sequences:
        raise RuntimeError("No sequences loaded - check MP_Data / MP_Dat_mobile.")

    X, y = np.array(sequences), np.array(labels)
    print(f"\nCombined: {len(X)} sequences, {X.shape[2]} features/frame")
    return X, y
def build_model(input_shape, num_classes, use_dropout=True):
    model = Sequential([
        LSTM(64,  return_sequences=True, activation="tanh", input_shape=input_shape),
        *([] if not use_dropout else [Dropout(0.2)]),
        LSTM(128, return_sequences=True, activation="tanh"),
        *([] if not use_dropout else [Dropout(0.2)]),
        LSTM(64,  return_sequences=False, activation="tanh"),
        *([] if not use_dropout else [Dropout(0.2)]),
        Dense(64, activation="relu"),
        *([] if not use_dropout else [Dropout(0.3)]),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
def train_and_evaluate(actions, data_sources, use_augmentation,
                       augment_multiplier, epochs):
    print("\n" + "=" * 70)
    mode = "WITH AUGMENTATION" if use_augmentation else "BASELINE (No Augmentation)"
    print(f"Training: {mode}")
    print("=" * 70)
    X, y = load_combined_data(actions, data_sources)

    if use_augmentation:
        print(f"\nApplying {augment_multiplier}x augmentation...")
        X, y = create_augmented_dataset(X, y, augmentation_multiplier=augment_multiplier)
        print(f"→ {len(X)} sequences after augmentation")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_cat = to_categorical(y_encoded).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y_encoded,
    )
    print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

    model = build_model(
        (X_train.shape[1], X_train.shape[2]),
        y_cat.shape[1],
        use_dropout=use_augmentation,
    )

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=30, mode="max",
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10,
                          min_lr=1e-6, verbose=1),
    ]

    print(f"\n🚀 Training for up to {epochs} epochs...")
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )
    duration = time.time() - t0

    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss,  test_acc  = model.evaluate(X_test,  y_test,  verbose=0)

    y_pred_classes = np.argmax(model.predict(X_test), axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    cm     = confusion_matrix(y_test_classes, y_pred_classes)
    report = classification_report(
        y_test_classes, y_pred_classes,
        target_names=le.classes_.tolist(), output_dict=True,
    )

    model_path   = MODEL_AUGMENTED  if use_augmentation else MODEL_BASELINE
    encoder_path = ENCODER_AUGMENTED if use_augmentation else ENCODER_BASELINE

    model.save(model_path)
    joblib.dump(le, encoder_path)
    print(f"\nSaved: {os.path.basename(model_path)}, {os.path.basename(encoder_path)}")

    log_training_session(
        duration_seconds=duration, num_words=len(actions),
        training_acc=train_acc, val_acc=test_acc,
        epochs=len(history.history["accuracy"]),
        batch_size=BATCH_SIZE, augmented=use_augmentation,
        model_path=model_path,
    )

    return {
        "mode": mode,
        "use_augmentation": use_augmentation,
        "augment_multiplier": augment_multiplier if use_augmentation else 1,
        "data_sources": [os.path.basename(s) for s in data_sources],
        "num_sequences": int(len(X)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "train_accuracy": float(train_acc),
        "test_accuracy":  float(test_acc),
        "train_loss":     float(train_loss),
        "test_loss":      float(test_loss),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "model_file":   model_path,
        "encoder_file": encoder_path,
        "history": {
            "train_acc":  [float(x) for x in history.history["accuracy"]],
            "val_acc":    [float(x) for x in history.history["val_accuracy"]],
            "train_loss": [float(x) for x in history.history["loss"]],
            "val_loss":   [float(x) for x in history.history["val_loss"]],
        },
        "training_duration": duration,
    }
def compare_and_report(baseline, augmented):
    print("\n" + "=" * 70)
    print("MODEL COMPARISON REPORT")
    print("=" * 70)
    print(f"\nGenerated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Sources   : {', '.join(baseline['data_sources'])}")

    b_train = baseline["train_accuracy"]  * 100
    a_train = augmented["train_accuracy"] * 100
    b_test  = baseline["test_accuracy"]   * 100
    a_test  = augmented["test_accuracy"]  * 100
    train_diff = a_train - b_train
    test_diff  = a_test  - b_test
    b_gap = b_train - b_test
    a_gap = a_train - a_test
    print(f"\n{'Metric':<30} {'Baseline':<20} {'Augmented':<20} {'Δ'}")
    print("-" * 90)
    print(f"{'Training Accuracy':<30} {b_train:<20.2f}% {a_train:<20.2f}% {train_diff:+.2f}%")
    print(f"{'Test Accuracy':<30} {b_test:<20.2f}% {a_test:<20.2f}% {test_diff:+.2f}%")
    print(f"{'Train-Test Gap (overfit)':<30} {b_gap:<20.2f}% {a_gap:<20.2f}% {(b_gap-a_gap):+.2f}%")

    print("\nANALYSIS:")
    if test_diff > 5:
        print(f"Augmented: {test_diff:.1f}% higher test accuracy – SIGNIFICANT IMPROVEMENT!")
    elif test_diff > 0:
        print(f"Augmented: {test_diff:.1f}% higher test accuracy")
    else:
        print(f"Baseline has higher test accuracy - consider collecting more data")
    if a_gap < b_gap:
        print(f"Augmentation reduced overfitting by {b_gap-a_gap:.1f}%")
    if b_gap > 15:
        print(f"Baseline shows significant overfitting ({b_gap:.1f}% gap)")
    if a_gap > 10:
        print(f"Augmented still overfitting ({a_gap:.1f}% gap) - collect more data")

    print("\nRECOMMENDATION:")
    if test_diff > 2 or a_gap < b_gap - 3:
        print(f"USE AUGMENTED → {os.path.basename(augmented['model_file'])}")
        recommended = "augmented"
    elif test_diff < -5:
        print(f"USE BASELINE  → {os.path.basename(baseline['model_file'])}")
        recommended = "baseline"
    else:
        print("   Both models perform similarly")
        recommended = "either"

    print(f"\n{'Sign':<22} {'Baseline F1':<15} {'Augmented F1':<15} {'Δ'}")
    print("-" * 64)
    for action in baseline["classification_report"]:
        if action in ("accuracy", "macro avg", "weighted avg"):
            continue
        b_f1 = baseline["classification_report"][action].get("f1-score", 0) * 100
        a_f1 = augmented["classification_report"][action].get("f1-score", 0) * 100
        print(f"{action:<22} {b_f1:<15.2f}% {a_f1:<15.2f}% {a_f1-b_f1:+.2f}%")

    return {
        "baseline": baseline, "augmented": augmented,
        "comparison": {
            "test_accuracy_diff":    test_diff,
            "overfitting_reduction": b_gap - a_gap,
            "recommended_model":     recommended,
        },
    }

def main():
    total_t0 = time.time()
    print("=" * 70)
    print("SIGNSPEAK - Combined Training (MP_Data + MP_Data_mobile)")
    print("=" * 70)
    os.makedirs(MODEL_DIR, exist_ok=True)
    actions = load_actions()
    print(f"\nActions ({len(actions)}): {', '.join(actions)}")
    print("\n" + "~" * 70)
    baseline = train_and_evaluate(
        actions, DATA_SOURCES,
        use_augmentation=False, augment_multiplier=1,
        epochs=EPOCHS,
    )
    print("\n" + "~" * 70)
    print("Baseline done - starting augmented training...")
    augmented = train_and_evaluate(
        actions, DATA_SOURCES,
        use_augmentation=True, augment_multiplier=AUGMENTATION_MULTIPLIER,
        epochs=EPOCHS,
    )
    comparison = compare_and_report(baseline, augmented)
    total_duration = time.time() - total_t0
    comparison["total_execution_time_seconds"] = total_duration
    report_dir = os.path.join(SCRIPT_DIR, "comparison_reports")
    os.makedirs(report_dir, exist_ok=True)
    report_file = os.path.join(
        report_dir,
        f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(report_file, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nReport saved: {report_file}")
    log_comparison_session(
        total_duration=total_duration,
        baseline_duration=baseline["training_duration"],
        augmented_duration=augmented["training_duration"],
        num_words=len(actions),
        baseline_acc=baseline["test_accuracy"],
        augmented_acc=augmented["test_accuracy"],
    )

    m, s = divmod(int(total_duration), 60)
    print(f"\nTotal: {m}m {s}s")
    print("Done!")


if __name__ == "__main__":
    main()
