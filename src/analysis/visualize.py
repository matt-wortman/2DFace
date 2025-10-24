from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from src.analysis.measurements import Measurement

_PALETTE = [
    (230, 57, 70),
    (29, 53, 87),
    (69, 123, 157),
    (131, 197, 190),
    (168, 218, 220),
    (233, 196, 106),
    (244, 162, 97),
    (38, 70, 83),
]


def render_measurement_overlay(
    image: np.ndarray,
    landmarks: np.ndarray,
    measurements: Iterable[Measurement],
) -> np.ndarray:
    overlay = image.copy()
    height, width = overlay.shape[:2]
    margin_width = 280

    for point in landmarks:
        cv2.circle(overlay, (int(point[0]), int(point[1])), 1, (190, 190, 190), -1)

    label_entries = []

    for idx, measurement in enumerate(measurements):
        color = _PALETTE[idx % len(_PALETTE)]
        segments = measurement.segments or []
        for seg in segments:
            if len(seg) != 4:
                continue
            x1, y1, x2, y2 = map(int, seg)
            cv2.line(overlay, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
            cv2.circle(overlay, (x1, y1), 2, color, -1)
            cv2.circle(overlay, (x2, y2), 2, color, -1)

        polylines = measurement.polylines or []
        for poly in polylines:
            if len(poly) < 4:
                continue
            pts = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [pts.astype(np.int32)], False, color, 1, cv2.LINE_AA)

        if measurement.label_point:
            lx, ly = map(int, measurement.label_point)
            label_entries.append((ly, idx, (lx, ly, measurement, color)))

    canvas = np.zeros((height, width + margin_width, 3), dtype=np.uint8)
    canvas[:, :width] = overlay
    canvas[:, width:] = (16, 16, 16)

    if label_entries:
        label_entries.sort(key=lambda x: x[0])
        spacing = max(height / (len(label_entries) + 1), 26)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        line_anchor_x = width + 10
        text_start_x = width + 24

        for order, (_, _, (lx, ly, measurement, color)) in enumerate(label_entries):
            target_y = int(spacing * (order + 1))
            target_y = min(max(target_y, 18), height - 12)

            anchor = (int(lx), int(ly))
            leader_mid = (line_anchor_x, target_y)
            cv2.circle(canvas, anchor, 3, color, -1, cv2.LINE_AA)
            cv2.line(canvas, anchor, leader_mid, color, 1, cv2.LINE_AA)
            cv2.line(canvas, leader_mid, (width + margin_width - 12, target_y), color, 1, cv2.LINE_AA)

            text = f"{measurement.name}: {measurement.value:.2f}"
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_width, text_height = text_size
            box_tl = (text_start_x - 6, target_y - text_height - 6)
            box_br = (text_start_x + text_width + 6, target_y + 8)
            cv2.rectangle(canvas, box_tl, box_br, (0, 0, 0), -1)
            cv2.putText(
                canvas,
                text,
                (text_start_x, target_y),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

    return canvas
