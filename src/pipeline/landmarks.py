from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np
from face_alignment import FaceAlignment, LandmarksType

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class LandmarkResult:
    """Stores landmarks and metadata for a detected face."""

    points: np.ndarray  # shape (68, 2)
    bbox: Optional[Sequence[float]] = None
    score: Optional[float] = None

    def to_json(self) -> dict:
        data = {"points": self.points.astype(float).tolist()}
        if self.bbox is not None:
            data["bbox"] = [float(v) for v in self.bbox]
        if self.score is not None:
            data["score"] = float(self.score)
        return data


class FaceAlignmentWrapper:
    """Wraps face-alignment library for batched landmark extraction."""

    def __init__(self, device: Optional[str] = None) -> None:
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:  # pragma: no cover
                device = "cpu"
        LOGGER.info("Initializing face-alignment on %s", device)
        self._fa = FaceAlignment(LandmarksType.TWO_D, device=device)

    def predict(self, image: np.ndarray, bboxes: Optional[Iterable[Sequence[float]]] = None) -> List[np.ndarray]:
        if bboxes is not None:
            preds = self._fa.get_landmarks_from_image(image, detected_faces=list(bboxes))
        else:
            preds = self._fa.get_landmarks(image)
        return preds or []


def overlay_landmarks(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Draw 68-point landmarks on a copy of the image."""
    out = image.copy()
    for x, y in landmarks:
        cv2.circle(out, (int(round(x)), int(round(y))), 2, (0, 255, 255), -1)
    return out


def crop_face(image: np.ndarray, bbox: Sequence[float], padding: float = 0.2) -> np.ndarray:
    """Crop face region with optional padding percentage."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    pad_x = width * padding
    pad_y = height * padding
    x1p = max(int(x1 - pad_x), 0)
    y1p = max(int(y1 - pad_y), 0)
    x2p = min(int(x2 + pad_x), w)
    y2p = min(int(y2 + pad_y), h)
    return image[y1p:y2p, x1p:x2p]

