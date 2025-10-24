from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional

import numpy as np

from src.analysis.wrinkles import analyze_crows_feet, analyze_nasolabial_folds

Landmarks = np.ndarray  # shape (68, 2)


@dataclasses.dataclass
class Measurement:
    name: str
    value: float
    units: str
    description: str
    valid: bool = True
    notes: Optional[str] = None
    segments: Optional[List[List[float]]] = None
    polylines: Optional[List[List[float]]] = None
    label_point: Optional[List[float]] = None

    def to_json(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "name": self.name,
            "value": float(self.value),
            "units": self.units,
            "description": self.description,
            "valid": self.valid,
        }
        if self.notes:
            data["notes"] = self.notes
        if self.segments:
            data["segments"] = [list(map(float, seg)) for seg in self.segments]
        if self.polylines:
            data["polylines"] = [list(map(float, poly)) for poly in self.polylines]
        if self.label_point:
            data["label_point"] = list(map(float, self.label_point))
        return data


def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def reference_scale(landmarks: Landmarks) -> float:
    left_eye = landmarks[36]
    right_eye = landmarks[45]
    scale = _distance(left_eye, right_eye)
    return scale if scale > 1e-6 else 1.0


def _segment(p1: np.ndarray, p2: np.ndarray) -> List[float]:
    return [float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])]


def _midpoint(p1: np.ndarray, p2: np.ndarray) -> List[float]:
    mid = (p1 + p2) / 2.0
    return [float(mid[0]), float(mid[1])]


def symmetry_score(landmarks: Landmarks) -> Measurement:
    left = landmarks[0:17]
    right = landmarks[16:33][::-1]
    mid_x = landmarks[27][0]
    reflected_right = right.copy()
    reflected_right[:, 0] = 2 * mid_x - reflected_right[:, 0]
    rms = float(np.sqrt(np.mean(np.sum((left - reflected_right) ** 2, axis=1))))
    scale = reference_scale(landmarks)
    score = max(0.0, 100.0 - (rms / scale) * 100.0)
    return Measurement(
        name="facial_symmetry",
        value=score,
        units="percent",
        description="Symmetry score (100 = perfect bilateral symmetry)",
        segments=[_segment(np.array([mid_x, landmarks[0][1]]), np.array([mid_x, landmarks[8][1]]))],
        label_point=[float(mid_x), float(landmarks[8][1]) - 40.0],
    )


def facial_thirds(landmarks: Landmarks) -> List[Measurement]:
    brow = landmarks[19:25].mean(axis=0)
    subnasale = landmarks[33]
    menton = landmarks[8]
    trichion = landmarks[27] + (landmarks[27] - menton) * 0.33

    upper = _distance(trichion, brow)
    middle = _distance(brow, subnasale)
    lower = _distance(subnasale, menton)
    scale = reference_scale(landmarks)

    return [
        Measurement(
            "facial_third_upper",
            upper / scale,
            "ratio",
            "Upper third normalized length",
            segments=[_segment(np.array([landmarks[0][0], brow[1]]), np.array([landmarks[16][0], brow[1]]))],
            label_point=[float(landmarks[27][0]), float(brow[1])],
        ),
        Measurement(
            "facial_third_middle",
            middle / scale,
            "ratio",
            "Midface normalized length",
            segments=[_segment(np.array([landmarks[0][0], subnasale[1]]), np.array([landmarks[16][0], subnasale[1]]))],
            label_point=[float(landmarks[27][0]), float(subnasale[1])],
        ),
        Measurement(
            "facial_third_lower",
            lower / scale,
            "ratio",
            "Lower face normalized length",
            segments=[_segment(np.array([landmarks[0][0], menton[1]]), np.array([landmarks[16][0], menton[1]]))],
            label_point=[float(landmarks[27][0]), float(menton[1])],
        ),
    ]


def nose_width_ratio(landmarks: Landmarks) -> Measurement:
    alar_left = landmarks[31]
    alar_right = landmarks[35]
    width = _distance(alar_left, alar_right)
    scale = reference_scale(landmarks)
    return Measurement(
        name="nose_width_ratio",
        value=width / scale,
        units="ratio",
        description="Nasal width relative to interpupillary distance",
        segments=[_segment(alar_left, alar_right)],
        label_point=_midpoint(alar_left, alar_right),
    )


def mouth_width_measurements(landmarks: Landmarks) -> tuple[float, float]:
    left = landmarks[48]
    right = landmarks[54]
    width = _distance(left, right)
    scale = reference_scale(landmarks)
    return width, width / scale


def mouth_width_ratio(landmarks: Landmarks) -> Measurement:
    _, ratio = mouth_width_measurements(landmarks)
    return Measurement(
        name="mouth_width_ratio",
        value=ratio,
        units="ratio",
        description="Mouth width relative to interpupillary distance",
        segments=[_segment(landmarks[48], landmarks[54])],
        label_point=_midpoint(landmarks[48], landmarks[54]),
    )


def upper_lip_thickness_ratio(landmarks: Landmarks) -> Measurement:
    pairs = [(50, 61), (51, 62), (52, 63)]
    distances = [abs(landmarks[a][1] - landmarks[b][1]) for a, b in pairs]
    thickness = float(np.mean(distances))
    width, _ = mouth_width_measurements(landmarks)
    normalized = thickness / width if width > 1e-6 else 0.0
    return Measurement(
        name="upper_lip_thickness_ratio",
        value=normalized,
        units="ratio",
        description="Upper lip thickness relative to mouth width",
        segments=[_segment(landmarks[a], landmarks[b]) for a, b in pairs],
        label_point=_midpoint(landmarks[51], landmarks[62]),
    )


def nasolabial_fold_ratio(landmarks: Landmarks) -> Measurement:
    left = _distance(landmarks[31], landmarks[48])
    right = _distance(landmarks[35], landmarks[54])
    scale = reference_scale(landmarks)
    ratio = ((left + right) / 2.0) / scale
    return Measurement(
        name="nasolabial_fold_ratio",
        value=ratio,
        units="ratio",
        description="Average distance from alar base to mouth corner",
        segments=[_segment(landmarks[31], landmarks[48]), _segment(landmarks[35], landmarks[54])],
        label_point=_midpoint(landmarks[33], landmarks[48]),
    )


def chin_midline_deviation(landmarks: Landmarks) -> Measurement:
    mid_x = landmarks[27][0]
    chin = landmarks[8]
    deviation = abs(chin[0] - mid_x)
    scale = reference_scale(landmarks)
    return Measurement(
        name="chin_midline_deviation",
        value=deviation / scale,
        units="ratio",
        description="Horizontal chin deviation relative to interpupillary distance",
        segments=[
            _segment(np.array([mid_x, chin[1]]), np.array([mid_x, landmarks[27][1]])),
            _segment(np.array([chin[0], chin[1]]), np.array([mid_x, chin[1]])),
        ],
        label_point=[float(chin[0]), float(chin[1]) - 20.0],
    )


def jaw_width_ratio(landmarks: Landmarks) -> Measurement:
    left = landmarks[3]
    right = landmarks[13]
    width = _distance(left, right)
    scale = reference_scale(landmarks)
    return Measurement(
        name="jaw_width_ratio",
        value=width / scale,
        units="ratio",
        description="Mandibular width relative to interpupillary distance",
        segments=[_segment(landmarks[3], landmarks[13])],
        label_point=_midpoint(landmarks[3], landmarks[13]),
    )


def compute_all_metrics(landmarks: Landmarks, image: Optional[np.ndarray] = None) -> List[Measurement]:
    metrics: List[Measurement] = []
    if landmarks is None:
        return metrics

    scale = reference_scale(landmarks)

    metrics.append(symmetry_score(landmarks))
    metrics.extend(facial_thirds(landmarks))
    metrics.append(nose_width_ratio(landmarks))
    metrics.append(mouth_width_ratio(landmarks))
    metrics.append(upper_lip_thickness_ratio(landmarks))
    metrics.append(nasolabial_fold_ratio(landmarks))
    metrics.append(chin_midline_deviation(landmarks))
    metrics.append(jaw_width_ratio(landmarks))

    if image is not None:
        for metric in analyze_crows_feet(image, landmarks, scale):
            metrics.append(
                Measurement(
                    name=metric.name,
                    value=metric.value,
                    units=metric.units,
                    description=metric.description,
                    segments=metric.segments,
                    polylines=metric.polylines,
                    label_point=metric.label_point,
                )
            )
        for metric in analyze_nasolabial_folds(image, landmarks, scale):
            metrics.append(
                Measurement(
                    name=metric.name,
                    value=metric.value,
                    units=metric.units,
                    description=metric.description,
                    segments=metric.segments,
                    polylines=metric.polylines,
                    label_point=metric.label_point,
                )
            )

    return metrics
