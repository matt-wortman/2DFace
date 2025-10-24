from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize


@dataclass
class WrinkleMetric:
    name: str
    value: float
    units: str
    description: str
    segments: List[List[float]]
    polylines: List[List[float]]
    label_point: List[float]


def _extract_rotated_roi(
    image: np.ndarray, center: Tuple[float, float], size: Tuple[float, float], angle: float
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    width, height = size
    rotation_matrix = cv2.getRotationMatrix2D(center, math.degrees(angle), 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    x = int(center[0] - width / 2)
    y = int(center[1] - height / 2)
    x = max(x, 0)
    y = max(y, 0)
    width = int(min(width, image.shape[1] - x))
    height = int(min(height, image.shape[0] - y))
    roi = rotated[y : y + height, x : x + width]

    inverse = cv2.invertAffineTransform(rotation_matrix)
    return roi, inverse, (x, y)


def _enhance_texture(gray_roi: np.ndarray) -> np.ndarray:
    gray = gray_roi.astype(np.float32) / 255.0
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray_eq = clahe.apply((gray * 255).astype(np.uint8)).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray_eq, (0, 0), sigmaX=1.0)
    high_freq = cv2.subtract(gray_eq, blur)

    orientations = np.linspace(0, np.pi, 12, endpoint=False)
    responses = []
    for theta in orientations:
        kernel = cv2.getGaborKernel((9, 9), sigma=1.5, theta=theta, lambd=4.0, gamma=0.5, psi=0)
        responses.append(cv2.filter2D(high_freq, cv2.CV_32F, kernel))
    response = np.max(responses, axis=0)
    response = cv2.normalize(response, None, 0, 1, cv2.NORM_MINMAX)
    return response


def _segment_ridges(response: np.ndarray) -> Tuple[np.ndarray, float]:
    thresh_val = np.percentile(response, 80)
    _, mask = cv2.threshold(response, thresh_val, 1.0, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool), 0.0
    skeleton = skeletonize(mask > 0)
    intensity = float(response[skeleton].mean()) if skeleton.any() else 0.0
    return skeleton, intensity


def _roi_segments(x: int, y: int, w: int, h: int, inverse: np.ndarray) -> List[List[float]]:
    corners_rot = np.array(
        [
            [x, y, 1],
            [x + w, y, 1],
            [x + w, y + h, 1],
            [x, y + h, 1],
        ],
        dtype=np.float32,
    )
    corners = corners_rot @ inverse.T
    segments: List[List[float]] = []
    for i in range(len(corners)):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % len(corners)]
        segments.append([float(x1), float(y1), float(x2), float(y2)])
    return segments


def _polyline_from_points(points: np.ndarray) -> List[float]:
    if len(points) < 2:
        return []
    centered = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = centered @ vt[0]
    order = np.argsort(axis)
    ordered = points[order]
    return ordered.flatten().tolist()


def _analyze_wrinkle_region(
    image: np.ndarray,
    center: Tuple[float, float],
    size: Tuple[float, float],
    angle: float,
    scale: float,
    label_offset: Tuple[float, float],
    name_prefix: str,
    description: str,
) -> Tuple[WrinkleMetric, WrinkleMetric]:
    roi, inverse, (x_off, y_off) = _extract_rotated_roi(image, center, size, angle)
    segments = _roi_segments(x_off, y_off, roi.shape[1], roi.shape[0], inverse)
    label_point = [center[0] + label_offset[0], center[1] + label_offset[1]]

    if roi.size == 0:
        zero_length = WrinkleMetric(
            name=f"{name_prefix}_length_ratio",
            value=0.0,
            units="ratio",
            description=description,
            segments=segments,
            polylines=[],
            label_point=label_point,
        )
        zero_intensity = WrinkleMetric(
            name=f"{name_prefix}_intensity",
            value=0.0,
            units="score",
            description=f"Average texture intensity score for {name_prefix.replace('_', ' ')} region",
            segments=[],
            polylines=[],
            label_point=label_point,
        )
        return zero_length, zero_intensity

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    response = _enhance_texture(gray)
    skeleton, intensity = _segment_ridges(response)

    polyline: List[float] = []
    if skeleton.any():
        ys, xs = np.where(skeleton)
        coords_rot = np.column_stack([x_off + xs, y_off + ys, np.ones_like(xs)])
        coords_orig = coords_rot @ inverse.T
        polyline = _polyline_from_points(coords_orig[:, :2])

    length_pixels = float(np.count_nonzero(skeleton))
    length_ratio = length_pixels / scale if scale > 1e-6 else 0.0

    length_metric = WrinkleMetric(
        name=f"{name_prefix}_length_ratio",
        value=length_ratio,
        units="ratio",
        description=description,
        segments=segments,
        polylines=[polyline] if polyline else [],
        label_point=label_point,
    )
    intensity_metric = WrinkleMetric(
        name=f"{name_prefix}_intensity",
        value=float(intensity),
        units="score",
        description=f"Average texture intensity score for {name_prefix.replace('_', ' ')} region",
        segments=[],
        polylines=[polyline] if polyline else [],
        label_point=label_point,
    )
    return length_metric, intensity_metric


def analyze_crows_feet(image: np.ndarray, landmarks: np.ndarray, scale: float) -> List[WrinkleMetric]:
    metrics: List[WrinkleMetric] = []
    if landmarks is None or scale <= 0:
        return metrics

    left_eye_outer = landmarks[36]
    left_eye_inner = landmarks[39]
    right_eye_inner = landmarks[42]
    right_eye_outer = landmarks[45]

    eye_vector_left = left_eye_inner - left_eye_outer
    eye_vector_right = right_eye_outer - right_eye_inner

    angle_left = math.atan2(eye_vector_left[1], eye_vector_left[0])
    angle_right = math.atan2(eye_vector_right[1], eye_vector_right[0])

    roi_width = scale * 0.7
    roi_height = scale * 0.5

    left_center = left_eye_outer + np.array([roi_width * 0.4, 0])
    right_center = right_eye_outer + np.array([-roi_width * 0.4, 0])

    left_length, left_intensity = _analyze_wrinkle_region(
        image,
        tuple(left_center),
        (roi_width, roi_height),
        angle_left,
        scale,
        (20, -10),
        "crows_feet",
        "Skeletonized crow's feet length relative to interpupillary distance",
    )
    right_length, right_intensity = _analyze_wrinkle_region(
        image,
        tuple(right_center),
        (roi_width, roi_height),
        angle_right,
        scale,
        (20, 10),
        "crows_feet",
        "Skeletonized crow's feet length relative to interpupillary distance",
    )

    avg_length = (left_length.value + right_length.value) / 2.0
    avg_intensity = (left_intensity.value + right_intensity.value) / 2.0

    length_metric = WrinkleMetric(
        name="crows_feet_length_ratio",
        value=avg_length,
        units="ratio",
        description="Average skeletonized crow's feet length relative to interpupillary distance",
        segments=left_length.segments + right_length.segments,
        polylines=left_length.polylines + right_length.polylines,
        label_point=left_length.label_point,
    )
    intensity_metric = WrinkleMetric(
        name="crows_feet_intensity",
        value=avg_intensity,
        units="score",
        description="Average texture intensity score for crow's feet region",
        segments=[],
        polylines=left_intensity.polylines + right_intensity.polylines,
        label_point=right_intensity.label_point,
    )
    metrics.extend([length_metric, intensity_metric])
    return metrics


def analyze_nasolabial_folds(image: np.ndarray, landmarks: np.ndarray, scale: float) -> List[WrinkleMetric]:
    metrics: List[WrinkleMetric] = []
    if landmarks is None or scale <= 0:
        return metrics

    left_ala = landmarks[31]
    right_ala = landmarks[35]
    left_mouth = landmarks[48]
    right_mouth = landmarks[54]

    vector_left = left_mouth - left_ala
    vector_right = right_mouth - right_ala

    angle_left = math.atan2(vector_left[1], vector_left[0])
    angle_right = math.atan2(vector_right[1], vector_right[0])

    roi_width = scale * 0.6
    roi_height = scale * 0.8

    left_center = (left_ala + left_mouth) / 2
    right_center = (right_ala + right_mouth) / 2

    left_length, left_intensity = _analyze_wrinkle_region(
        image,
        tuple(left_center),
        (roi_width, roi_height),
        angle_left,
        scale,
        (20, -20),
        "nasolabial_wrinkle",
        "Skeletonized nasolabial fold length relative to interpupillary distance",
    )
    right_length, right_intensity = _analyze_wrinkle_region(
        image,
        tuple(right_center),
        (roi_width, roi_height),
        angle_right,
        scale,
        (20, 20),
        "nasolabial_wrinkle",
        "Skeletonized nasolabial fold length relative to interpupillary distance",
    )

    avg_length = (left_length.value + right_length.value) / 2.0
    avg_intensity = (left_intensity.value + right_intensity.value) / 2.0

    length_metric = WrinkleMetric(
        name="nasolabial_wrinkle_length_ratio",
        value=avg_length,
        units="ratio",
        description="Average skeletonized nasolabial fold length relative to interpupillary distance",
        segments=left_length.segments + right_length.segments,
        polylines=left_length.polylines + right_length.polylines,
        label_point=left_length.label_point,
    )
    intensity_metric = WrinkleMetric(
        name="nasolabial_wrinkle_intensity",
        value=avg_intensity,
        units="score",
        description="Average texture intensity score for nasolabial region",
        segments=[],
        polylines=left_intensity.polylines + right_intensity.polylines,
        label_point=right_intensity.label_point,
    )
    metrics.extend([length_metric, intensity_metric])
    return metrics
