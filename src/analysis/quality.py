from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
import yaml


@dataclasses.dataclass
class QualityThresholds:
    """Configuration values for quality heuristics."""

    blur_min: float = 120.0
    exposure_min: float = 60.0
    exposure_max: float = 190.0
    roll_max_deg: float = 10.0

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "QualityThresholds":
        return cls(**{k: float(v) for k, v in data.items()})


def load_thresholds(path: Optional[Path]) -> QualityThresholds:
    if path is None or not path.exists():
        return QualityThresholds()
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return QualityThresholds.from_dict(raw)


def compute_blur_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_exposure_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def compute_roll_degrees(landmarks: np.ndarray) -> Optional[float]:
    if landmarks is None or len(landmarks) < 46:
        return None
    left_eye_outer = landmarks[36]
    right_eye_outer = landmarks[45]
    dx = right_eye_outer[0] - left_eye_outer[0]
    dy = right_eye_outer[1] - left_eye_outer[1]
    if dx == 0:
        return 90.0 if dy > 0 else -90.0
    return float(math.degrees(math.atan2(dy, dx)))


def assess_quality(
    image: np.ndarray,
    thresholds: QualityThresholds,
    landmarks: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    blur = compute_blur_score(image)
    exposure = compute_exposure_score(image)
    roll = compute_roll_degrees(landmarks) if landmarks is not None else None

    warnings: List[str] = []
    if blur < thresholds.blur_min:
        warnings.append(f"blur_score {blur:.1f} < min {thresholds.blur_min}")
    if exposure < thresholds.exposure_min or exposure > thresholds.exposure_max:
        warnings.append(
            f"exposure_mean {exposure:.1f} outside [{thresholds.exposure_min}, {thresholds.exposure_max}]"
        )
    if roll is not None and abs(roll) > thresholds.roll_max_deg:
        warnings.append(f"roll {roll:.1f}deg exceeds ±{thresholds.roll_max_deg}")

    return {
        "blur_score": blur,
        "exposure_mean": exposure,
        "roll_degrees": roll,
        "status": "warn" if warnings else "ok",
        "warnings": warnings,
    }


def summarize_warnings(quality_result: Dict[str, object]) -> str:
    warnings = quality_result.get("warnings") or []
    if not warnings:
        return "ok"
    return "; ".join(str(w) for w in warnings)
