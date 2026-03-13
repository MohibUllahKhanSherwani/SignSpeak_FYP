# realtime_inference_minimal.py
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import mediapipe as mp
from mediapipe_utils import mediapipe_detection, extract_keypoints, draw_landmarks
import argparse
import os

from actions_config import load_actions, SEQUENCE_LENGTH, PREDICTION_THRESHOLD

def main():
    parser = argparse.ArgumentParser(description='Minimal SignSpeak Inference')
    parser.add_argument('--augmented', action='store_true', help='Use augmented model')
    parser.add_argument('--baseline', action='store_true', help='Use baseline model')
    args = parser.parse_args()

    # Model Selection
    model_path = "all_models/action_model_augmented_new.h5" if args.augmented else "all_models/action_model_baseline_new.h5"
    encoder_path = "all_models/label_encoder_augmented_new.pkl" if args.augmented else "all_models/label_encoder_baseline_new.pkl"
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Running baseline instead.")
        model_path, encoder_path = "all_models/action_model_baseline_new.h5", "all_models/label_encoder_baseline_new.pkl"

    # Load resources
    print(f"Loading {model_path}...")
    actions = load_actions()
    model = load_model(model_path)
    le = joblib.load(encoder_path)

    # State variables
    sequence = []
    sentence = []
    predictions = []
    frame_count = 0
    
    # Simple display variables
    current_action = "Warming up..."
    current_confidence = 0.0
    
    cap = cv2.VideoCapture(0)
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.4) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_count += 1
            image, results = mediapipe_detection(frame, holistic)
            
            # Optional: Draw only hands for better speed
            draw_landmarks(image, results)

            # Process keypoints
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            if len(sequence) > SEQUENCE_LENGTH:
                sequence.pop(0)

            # OPTIMIZED: Predict every 3 frames to save CPU and reduce lag
            if len(sequence) == SEQUENCE_LENGTH and frame_count % 3 == 0:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                prediction_idx = np.argmax(res)
                action = le.inverse_transform([prediction_idx])[0]
                confidence = res[prediction_idx]
                
                current_action = action
                current_confidence = confidence

                # Update stability tracking
                predictions.append(prediction_idx)
                if len(predictions) > 10: predictions.pop(0)

                # Higher stability: > 8 matches required in the last 10 predictions
                if confidence > PREDICTION_THRESHOLD and predictions.count(prediction_idx) > 8:
                    # Don't add "nothing" to the sentence
                    if action != "nothing":
                        if not sentence or sentence[-1] != action:
                            sentence.append(action)
                
                # Increase history capacity to 20 words
                if len(sentence) > 20: sentence = sentence[-20:]

            # --- MINIMALIST UI WITH VERTICAL HISTORY ---
            h, w, _ = image.shape
            
            # 1. Prediction Header (Top)
            display_text = f"CURRENT: {current_action.upper().replace('_', ' ')} ({current_confidence:.0%})"
            header_color = (0, 255, 0) if current_confidence > PREDICTION_THRESHOLD else (0, 255, 255)
            if current_action == "nothing":
                display_text = "SILENT..."
                header_color = (150, 150, 150)

            cv2.rectangle(image, (0, 0), (w, 50), (20, 20, 20), -1)
            cv2.putText(image, display_text, (20, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, header_color, 2, cv2.LINE_AA)

            # 2. Vertical History Sidebar (Left)
            sidebar_width = 250
            # Overlay sidebar background (semi-transparent)
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 50), (sidebar_width, h), (40, 40, 40), -1)
            image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
            
            cv2.putText(image, "HISTORY LOG", (20, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.line(image, (20, 95), (sidebar_width-20, 95), (100, 100, 100), 1)

            # Draw vertical list of words
            for i, word in enumerate(sentence[-15:]): # Show last 15 words
                y_pos = 130 + (i * 35)
                # Highlight the most recent word
                word_color = (255, 255, 255) if i < len(sentence[-15:]) - 1 else (0, 255, 0)
                prefix = "> " if i < len(sentence[-15:]) - 1 else "NOW: "
                
                cv2.putText(image, f"{prefix}{word.upper()}", (25, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, word_color, 1, cv2.LINE_AA)

            # 3. Footer Helper
            cv2.putText(image, "[Q] Quit | [C] Clear History", (sidebar_width + 20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # --- DISPLAY ---
            # Create a named window that can be resized
            cv2.namedWindow('SignSpeak - Pro History Mode', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('SignSpeak - Pro History Mode', 1000, 750)
            
            cv2.imshow('SignSpeak - Pro History Mode', image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                sentence = []

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
