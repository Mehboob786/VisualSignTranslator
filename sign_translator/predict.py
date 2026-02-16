from __future__ import annotations

from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .features import hand_landmarks_to_feature_vector


def predict_from_bgr_frame(
    model,
    frame_bgr: np.ndarray,
    threshold: float = 0.6,
) -> Tuple[str, float, np.ndarray]:
    """
    Predict a gesture label from a BGR frame.

    Returns:
    - label text (`No hand`, `Uncertain`, or class label)
    - confidence score in [0, 1]
    - annotated BGR frame
    """
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    annotated = frame_bgr.copy()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    ) as hands:
        result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return "No hand", 0.0, annotated

    hand_lm = result.multi_hand_landmarks[0]
    mp_draw.draw_landmarks(annotated, hand_lm, mp_hands.HAND_CONNECTIONS)
    vector = hand_landmarks_to_feature_vector(hand_lm)
    if vector is None:
        return "Uncertain", 0.0, annotated

    probs = model.predict_proba(np.array([vector]))[0]
    best_idx = int(np.argmax(probs))
    best_prob = float(probs[best_idx])
    best_label = str(model.classes_[best_idx])

    if best_prob < threshold:
        return "Uncertain", best_prob, annotated
    return best_label, best_prob, annotated
