# realtime_inference_minimal.py
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import mediapipe as mp
from mediapipe_utils import mediapipe_detection, extract_keypoints, draw_landmarks
import os

from actions_config import load_actions, SEQUENCE_LENGTH, PREDICTION_THRESHOLD

def main():
    model_path = "all_models/action_model_baseline_new.h5"
    encoder_path = "all_models/label_encoder_baseline_new.pkl"
    print(f"Loading {model_path}...")
    actions = load_actions()
    model = load_model(model_path)
    le = joblib.load(encoder_path)

    sequence = []
    sentence = []
    predictions = []
    frame_count = 0
    
    current_action = "Warming up..."
    current_confidence = 0.0
    
    cap = cv2.VideoCapture(0)
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.4) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_count += 1
            image, results = mediapipe_detection(frame, holistic)
            
            draw_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            if len(sequence) > SEQUENCE_LENGTH:
                sequence.pop(0)
            # Predict every 3 frames
            if len(sequence) == SEQUENCE_LENGTH and frame_count % 3 == 0:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                prediction_idx = np.argmax(res)
                action = le.inverse_transform([prediction_idx])[0]
                confidence = res[prediction_idx]
                
                current_action = action
                current_confidence = confidence

                predictions.append(prediction_idx)
                if len(predictions) > 10: predictions.pop(0)

                if confidence > PREDICTION_THRESHOLD and predictions.count(prediction_idx) > 8:
                    if action != "nothing":
                        if not sentence or sentence[-1] != action:
                            sentence.append(action)
                
                if len(sentence) > 20: sentence = sentence[-20:]

            h, w, _ = image.shape
            
            display_text = f"CURRENT: {current_action.upper().replace('_', ' ')} ({current_confidence:.0%})"
            header_color = (0, 255, 0) if current_confidence > PREDICTION_THRESHOLD else (0, 255, 255)
            if current_action == "nothing":
                display_text = "SILENT..."
                header_color = (150, 150, 150)

            cv2.rectangle(image, (0, 0), (w, 50), (20, 20, 20), -1)
            cv2.putText(image, display_text, (20, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, header_color, 2, cv2.LINE_AA)

            sidebar_width = 250
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 50), (sidebar_width, h), (40, 40, 40), -1)
            image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
            
            cv2.putText(image, "HISTORY LOG", (20, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.line(image, (20, 95), (sidebar_width-20, 95), (100, 100, 100), 1)

            for i, word in enumerate(sentence[-15:]): # Show last 15 words
                y_pos = 130 + (i * 35)
                word_color = (255, 255, 255) if i < len(sentence[-15:]) - 1 else (0, 255, 0)
                prefix = "> " if i < len(sentence[-15:]) - 1 else "NOW: "
                
                cv2.putText(image, f"{prefix}{word.upper()}", (25, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, word_color, 1, cv2.LINE_AA)

            cv2.putText(image, "[Q] Quit | [C] Clear History", (sidebar_width + 20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

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
