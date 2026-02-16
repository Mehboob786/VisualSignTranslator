from __future__ import annotations

from typing import List, Optional

import numpy as np


def hand_landmarks_to_feature_vector(landmarks) -> Optional[List[float]]:
    """Convert MediaPipe hand landmarks into normalized flat features."""
    if landmarks is None:
        return None

    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)
    if points.shape != (21, 3):
        return None

    origin = points[0]
    centered = points - origin
    scale = np.max(np.linalg.norm(centered[:, :2], axis=1))
    if scale <= 1e-6:
        return None

    normalized = centered / scale
    return normalized.flatten().tolist()
