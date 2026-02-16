from __future__ import annotations

from collections import deque
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np


def _finger_states(hand_landmarks, handedness: str) -> List[bool]:
    lm = hand_landmarks.landmark

    # Index, middle, ring, pinky: tip above pip means finger up.
    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up = lm[16].y < lm[14].y
    pinky_up = lm[20].y < lm[18].y

    # Thumb requires handedness-aware x comparison.
    if handedness.lower() == "right":
        thumb_up = lm[4].x > lm[3].x
    else:
        thumb_up = lm[4].x < lm[3].x

    return [thumb_up, index_up, middle_up, ring_up, pinky_up]


def _classify_gesture(states: List[bool], hand_landmarks) -> Tuple[str, float]:
    thumb, index, middle, ring, pinky = states

    if not any(states):
        return "YES", 0.85  # Fist-like

    if index and middle and not ring and not pinky:
        return "NO", 0.82  # Two-finger style

    if thumb and index and middle and ring and pinky:
        return "HELLO", 0.80  # Open palm

    if index and not middle and not ring and not pinky:
        return "POINT", 0.78

    # Thumbs up gesture: thumb above thumb MCP and others folded.
    lm = hand_landmarks.landmark
    if thumb and not index and not middle and not ring and not pinky and lm[4].y < lm[2].y:
        return "GOOD", 0.80

    return "UNKNOWN", 0.40


def predict_pretrained_from_bgr_frame(frame_bgr: np.ndarray) -> Tuple[str, float, np.ndarray]:
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
    handedness = "Right"
    if result.multi_handedness:
        handedness = result.multi_handedness[0].classification[0].label

    states = _finger_states(hand_lm, handedness)
    label, conf = _classify_gesture(states, hand_lm)
    return label, conf, annotated


def run_pretrained_inference(
    camera_index: int = 0,
    window_size: int = 12,
    min_stable_frames: int = 8,
    repeat_cooldown_frames: int = 18,
) -> None:
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    history = deque(maxlen=max(3, window_size))
    sentence_tokens: List[str] = []
    stable_label = "No hand"
    cooldown = 0

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

            label, conf = "No hand", 0.0
            if result.multi_hand_landmarks:
                hand_lm = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
                handedness = "Right"
                if result.multi_handedness:
                    handedness = result.multi_handedness[0].classification[0].label
                states = _finger_states(hand_lm, handedness)
                label, conf = _classify_gesture(states, hand_lm)

            history.append(label)
            if cooldown > 0:
                cooldown -= 1

            if history:
                candidate = max(set(history), key=list(history).count)
                count = list(history).count(candidate)
                if count >= max(2, min_stable_frames) and candidate not in {"UNKNOWN", "No hand"}:
                    stable_label = candidate
                    if cooldown == 0:
                        if not sentence_tokens or sentence_tokens[-1] != candidate.lower():
                            sentence_tokens.append(candidate.lower())
                        cooldown = max(1, repeat_cooldown_frames)

            sentence_text = " ".join(sentence_tokens) if sentence_tokens else "(start signing...)"

            cv2.putText(
                frame,
                f"Prediction: {label} ({conf:.2f}) | Stable: {stable_label}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Translation: {sentence_text[:70]}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "q: quit | c: clear sentence",
                (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.imshow("VisualSignTranslator (Pretrained)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                sentence_tokens.clear()
            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
