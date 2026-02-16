from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import joblib
import mediapipe as mp
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import hand_landmarks_to_feature_vector


@dataclass
class SequenceModelBundle:
    model: Pipeline
    target_len: int
    feature_dim: int


def _resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    if seq.shape[0] == target_len:
        return seq
    if seq.shape[0] < 2:
        return np.repeat(seq, target_len, axis=0)

    src_x = np.linspace(0.0, 1.0, seq.shape[0], dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    out = np.zeros((target_len, seq.shape[1]), dtype=np.float32)
    for i in range(seq.shape[1]):
        out[:, i] = np.interp(dst_x, src_x, seq[:, i])
    return out


def _flatten_for_model(seq: np.ndarray) -> np.ndarray:
    return seq.reshape(-1)


def collect_sequence_samples(
    label: str,
    clips: int,
    frames_per_clip: int,
    output_dir: str = "data/sequences",
    camera_index: int = 0,
) -> None:
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    label_dir = Path(output_dir) / label
    label_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    recorded = 0
    recording = False
    current_clip: List[List[float]] = []
    last_vec = np.zeros((63,), dtype=np.float32)

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

            if result.multi_hand_landmarks:
                hand_lm = result.multi_hand_landmarks[0]
                vec = hand_landmarks_to_feature_vector(hand_lm)
                if vec is not None:
                    last_vec = np.array(vec, dtype=np.float32)
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            if recording:
                current_clip.append(last_vec.tolist())
                if len(current_clip) >= frames_per_clip:
                    arr = np.array(current_clip, dtype=np.float32)
                    stamp = int(time.time() * 1000)
                    out = label_dir / f"{stamp}_{recorded:03d}.npz"
                    np.savez_compressed(out, sequence=arr, label=label)
                    recorded += 1
                    recording = False
                    current_clip = []

            cv2.putText(
                frame,
                f"Label: {label} | Clips: {recorded}/{clips}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "r: record clip | q: quit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            if recording:
                cv2.putText(
                    frame,
                    f"REC {len(current_clip)}/{frames_per_clip}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            cv2.imshow("Sequence Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("r") and not recording:
                recording = True
                current_clip = []
            elif key == ord("q"):
                break

            if recorded >= clips:
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {recorded} clips to {label_dir}")


def _load_sequence_dataset(data_dir: str, target_len: int) -> Tuple[np.ndarray, np.ndarray, int]:
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"Sequence dataset dir not found: {data_dir}")

    xs: List[np.ndarray] = []
    ys: List[str] = []
    feature_dim = 63

    for label_dir in sorted(base.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for npz_file in sorted(label_dir.glob("*.npz")):
            data = np.load(npz_file, allow_pickle=False)
            seq = data["sequence"]
            if seq.ndim != 2:
                continue
            feature_dim = int(seq.shape[1])
            seq = _resample_sequence(seq.astype(np.float32), target_len=target_len)
            xs.append(_flatten_for_model(seq))
            ys.append(label)

    if not xs:
        raise ValueError("No sequence samples found.")

    return np.array(xs, dtype=np.float32), np.array(ys), feature_dim


def train_sequence_model(
    data_dir: str = "data/sequences",
    model_out: str = "models/sequence_model.joblib",
    target_len: int = 40,
) -> None:
    x, y, feature_dim = _load_sequence_dataset(data_dir=data_dir, target_len=target_len)
    if len(set(y.tolist())) < 2:
        raise ValueError("Need at least two labels for sequence training.")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=3.0, gamma="scale", probability=True)),
        ]
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    print("Sequence validation report:")
    print(classification_report(y_test, preds))

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    bundle = SequenceModelBundle(model=model, target_len=target_len, feature_dim=feature_dim)
    joblib.dump(bundle, model_out)
    print(f"Saved sequence model to {model_out}")


def infer_sequence_live(
    model_path: str = "models/sequence_model.joblib",
    camera_index: int = 0,
    stride: int = 5,
) -> None:
    bundle: SequenceModelBundle = joblib.load(model_path)
    model = bundle.model
    target_len = bundle.target_len
    feature_dim = bundle.feature_dim

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    window: List[np.ndarray] = []
    frame_count = 0
    last_vec = np.zeros((feature_dim,), dtype=np.float32)
    last_pred = "No sign"
    last_conf = 0.0

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

            if result.multi_hand_landmarks:
                hand_lm = result.multi_hand_landmarks[0]
                vec = hand_landmarks_to_feature_vector(hand_lm)
                if vec is not None:
                    last_vec = np.array(vec, dtype=np.float32)
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            window.append(last_vec.copy())
            if len(window) > target_len:
                window.pop(0)

            frame_count += 1
            if len(window) >= target_len and frame_count % max(1, stride) == 0:
                seq = np.array(window[-target_len:], dtype=np.float32)
                flat = _flatten_for_model(seq)[None, :]
                probs = model.predict_proba(flat)[0]
                idx = int(np.argmax(probs))
                last_conf = float(probs[idx])
                last_pred = str(model.classes_[idx])

            cv2.putText(
                frame,
                f"Sequence Prediction: {last_pred} ({last_conf:.2f})",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
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
            cv2.imshow("Sequence Live Inference", frame)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
