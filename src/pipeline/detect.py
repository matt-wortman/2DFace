from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np
from insightface.app import FaceAnalysis

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class Detection:
    """Structured representation of a single face detection."""

    bbox: Sequence[float]  # [x1, y1, x2, y2]
    score: float
    landmarks: Optional[np.ndarray] = None  # shape (5, 2)

    def to_json(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "bbox": [float(v) for v in self.bbox],
            "score": float(self.score),
        }
        if self.landmarks is not None:
            data["landmarks"] = self.landmarks.astype(float).tolist()
        return data


class InsightFaceDetector:
    """Thin wrapper around InsightFace RetinaFace detector."""

    def __init__(
        self,
        det_size: Sequence[int] = (640, 640),
        threshold: float = 0.3,
        ctx_id: Optional[int] = None,
        providers: Optional[Sequence[str]] = None,
        model_root: Optional[Path] = None,
    ) -> None:
        if model_root:
            os.environ.setdefault("INSIGHTFACE_HOME", str(model_root))

        # Default to GPU if available; InsightFace expects -1 for CPU.
        if ctx_id is None:
            try:
                import torch

                ctx_id = 0 if torch.cuda.is_available() else -1
            except Exception:  # pragma: no cover
                ctx_id = -1

        self._threshold = threshold

        self._app = FaceAnalysis(
            name="buffalo_l",
            root=str(model_root) if model_root else None,
            providers=list(providers) if providers else ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=ctx_id, det_size=tuple(det_size))
        LOGGER.debug("InsightFace detector ready (ctx_id=%s, det_size=%s)", ctx_id, det_size)

    def detect_array(self, image: np.ndarray) -> List[Detection]:
        faces = self._app.get(image)
        detections: List[Detection] = []
        for face in faces:
            score = float(getattr(face, "det_score", 0.0))
            if score < self._threshold:
                continue
            bbox = np.array(face.bbox, dtype=float)
            landmarks = np.array(face.kps, dtype=float) if getattr(face, "kps", None) is not None else None
            detections.append(Detection(bbox=bbox, score=score, landmarks=landmarks))
        LOGGER.debug("Detected %d faces (threshold=%s)", len(detections), self._threshold)
        return detections


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def annotate_detections(image: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{det.score:.2f}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        if det.landmarks is not None:
            for (lx, ly) in det.landmarks:
                cv2.circle(annotated, (int(round(lx)), int(round(ly))), 2, (0, 128, 255), -1)
    return annotated

