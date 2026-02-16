from __future__ import annotations

import csv
import os
from typing import List

import cv2
import mediapipe as mp

from .features import hand_landmarks_to_feature_vector


def _ensure_csv(csv_path: str, feature_count: int) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", *[f"f{i}" for i in range(feature_count)]])


def _append_rows(csv_path: str, rows: List[List[float]], label: str) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow([label, *row])


def collect_samples(label: str, sample_target: int, output_csv: str, camera_index: int = 0) -> None:
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    captured_rows: List[List[float]] = []

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

            vector = None
            if result.multi_hand_landmarks:
                hand_lm = result.multi_hand_landmarks[0]
                vector = hand_landmarks_to_feature_vector(hand_lm)
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            cv2.putText(
                frame,
                f"Label: {label} | Captured: {len(captured_rows)}/{sample_target}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "SPACE: capture, q: quit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Collect Sign Samples", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if vector is not None:
                    captured_rows.append(vector)
            elif key == ord("q"):
                break

            if len(captured_rows) >= sample_target:
                break

    cap.release()
    cv2.destroyAllWindows()

    if not captured_rows:
        print("No samples captured.")
        return

    _ensure_csv(output_csv, feature_count=len(captured_rows[0]))
    _append_rows(output_csv, captured_rows, label=label)
    print(f"Saved {len(captured_rows)} samples for label '{label}' to {output_csv}")
