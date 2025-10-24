from __future__ import annotations

import numpy as np
import pytest

from src.analysis.measurements import (
    Measurement,
    chin_midline_deviation,
    compute_all_metrics,
    mouth_width_ratio,
    nasolabial_fold_ratio,
    nose_width_ratio,
    upper_lip_thickness_ratio,
)


def _base_landmarks():
    landmarks = np.zeros((68, 2), dtype=float)
    landmarks[36] = [0.0, 0.0]
    landmarks[45] = [60.0, 0.0]
    landmarks[31] = [-10.0, 0.0]
    landmarks[35] = [10.0, 0.0]
    landmarks[48] = [-15.0, -20.0]
    landmarks[54] = [15.0, -20.0]
    landmarks[27] = [0.0, 20.0]
    landmarks[8] = [4.0, -60.0]
    landmarks[3] = [-40.0, -40.0]
    landmarks[13] = [40.0, -40.0]
    landmarks[19:25] = [[-10 + i * 4, 10.0] for i in range(6)]
    landmarks[33] = [0.0, -5.0]
    landmarks[39] = [25.0, -2.0]
    landmarks[42] = [35.0, -2.0]
    landmarks[50] = [-12.0, -18.0]
    landmarks[61] = [-12.0, -22.0]
    landmarks[51] = [0.0, -15.0]
    landmarks[62] = [0.0, -19.0]
    landmarks[52] = [12.0, -18.0]
    landmarks[63] = [12.0, -22.0]
    landmarks[17] = [-30.0, 5.0]
    landmarks[26] = [90.0, 5.0]
    return landmarks


def test_nose_width_ratio():
    landmarks = _base_landmarks()
    nose = nose_width_ratio(landmarks)
    assert nose.value == pytest.approx(20.0 / 60.0)


def test_mouth_width_ratio():
    landmarks = _base_landmarks()
    mouth = mouth_width_ratio(landmarks)
    assert mouth.value == pytest.approx(30.0 / 60.0)


def test_upper_lip_thickness_ratio():
    landmarks = _base_landmarks()
    metric = upper_lip_thickness_ratio(landmarks)
    assert metric.value == pytest.approx(4.0 / 30.0, abs=1e-3)


def test_nasolabial_fold_ratio():
    landmarks = _base_landmarks()
    metric = nasolabial_fold_ratio(landmarks)
    expected = ((np.hypot(5.0, 20.0) + np.hypot(5.0, 20.0)) / 2) / 60.0
    assert metric.value == pytest.approx(expected)


def test_chin_midline_deviation():
    landmarks = _base_landmarks()
    chin = chin_midline_deviation(landmarks)
    assert chin.value == pytest.approx(abs(4.0 - 0.0) / 60.0)


def test_compute_all_metrics_includes_wrinkle_metrics():
    landmarks = _base_landmarks()
    metrics = compute_all_metrics(landmarks, image=np.zeros((256, 256, 3), dtype=np.uint8))
    names = {m.name for m in metrics}
    assert "crows_feet_length_ratio" in names
    assert "nasolabial_wrinkle_length_ratio" in names
