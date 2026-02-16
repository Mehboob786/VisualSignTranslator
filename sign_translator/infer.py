from __future__ import annotations

import joblib
import cv2
import mediapipe as mp
import numpy as np

from .features import hand_landmarks_to_feature_vector


def run_inference(model_path: str, camera_index: int = 0, threshold: float = 0.6) -> None:
    model = joblib.load(model_path)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            prediction_text = "No hand"
            if result.multi_hand_landmarks:
                hand_lm = result.multi_hand_landmarks[0]
                vector = hand_landmarks_to_feature_vector(hand_lm)
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

                if vector is not None:
                    probs = model.predict_proba(np.array([vector]))[0]
                    best_idx = int(np.argmax(probs))
                    best_prob = float(probs[best_idx])
                    best_label = model.classes_[best_idx]

                    if best_prob >= threshold:
                        prediction_text = f"{best_label} ({best_prob:.2f})"
                    else:
                        prediction_text = f"Uncertain ({best_prob:.2f})"

            cv2.putText(
                frame,
                f"Prediction: {prediction_text}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "q: quit",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imshow("VisualSignTranslator", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
