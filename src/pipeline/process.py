from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from src.analysis.measurements import Measurement, compute_all_metrics
from src.analysis.quality import QualityThresholds, assess_quality
from src.analysis.rules import Finding, Threshold, evaluate_findings
from src.analysis.visualize import render_measurement_overlay
from src.pipeline.detect import Detection, InsightFaceDetector, annotate_detections, load_image
from src.pipeline.landmarks import FaceAlignmentWrapper, LandmarkResult, overlay_landmarks


@dataclass
class PipelineResult:
    image_name: str
    detections: List[Detection]
    faces: List[LandmarkResult]
    face_overlays: List[Path]
    quality: Dict[str, object]
    metrics: List[Measurement]
    findings: List[Finding]
    landmark_overlay: Optional[Path]
    metrics_overlay: Optional[Path]

    def to_json(self) -> Dict[str, object]:
        return {
            "image": self.image_name,
            "faces": [face.to_json() for face in self.faces],
            "quality": self.quality,
            "metrics": [m.to_json() for m in self.metrics],
            "findings": [finding.to_json() for finding in self.findings],
            "landmark_overlay": str(self.landmark_overlay) if self.landmark_overlay else None,
            "metrics_overlay": str(self.metrics_overlay) if self.metrics_overlay else None,
            "face_overlays": [str(p) for p in self.face_overlays],
        }


def run_pipeline(
    image_path: Path,
    output_dir: Path,
    detector: InsightFaceDetector,
    landmark_model: FaceAlignmentWrapper,
    quality_thresholds: QualityThresholds,
    rule_thresholds: Dict[str, Threshold],
    save_overlays: bool = True,
) -> PipelineResult:
    image = load_image(image_path)
    detections = detector.detect_array(image)

    faces: List[LandmarkResult] = []
    face_overlay_paths: List[Path] = []
    landmark_overlay_path: Optional[Path] = None
    metrics_overlay_path: Optional[Path] = None

    if detections:
        bboxes = [det.bbox for det in detections]
        landmarks_list = landmark_model.predict(image, bboxes=bboxes) if bboxes else []
        for idx, (det, landmarks) in enumerate(zip(detections, landmarks_list)):
            faces.append(LandmarkResult(points=landmarks, bbox=det.bbox, score=det.score))
            if save_overlays:
                overlay = overlay_landmarks(image, landmarks)
                path = output_dir / f"{image_path.stem}_face{idx:02d}_landmarks.png"
                cv2.imwrite(str(path), overlay)
                face_overlay_paths.append(path)
    else:
        landmarks_list = []

    if save_overlays:
        annotated = annotate_detections(image, detections)
        landmark_overlay_path = output_dir / f"{image_path.stem}_detections.png"
        cv2.imwrite(str(landmark_overlay_path), annotated)

    primary_landmarks = faces[0].points if faces else None
    quality = assess_quality(image, quality_thresholds, primary_landmarks)

    metrics: List[Measurement] = (
        compute_all_metrics(primary_landmarks, image=image) if primary_landmarks is not None else []
    )
    findings = evaluate_findings(metrics, rule_thresholds) if metrics else []

    if save_overlays and primary_landmarks is not None and metrics:
        metric_overlay_img = render_measurement_overlay(image, primary_landmarks, metrics)
        metrics_overlay_path = output_dir / f"{image_path.stem}_metrics.png"
        cv2.imwrite(str(metrics_overlay_path), metric_overlay_img)

    return PipelineResult(
        image_name=image_path.name,
        detections=detections,
        faces=faces,
        face_overlays=face_overlay_paths,
        quality=quality,
        metrics=metrics,
        findings=findings,
        landmark_overlay=landmark_overlay_path,
        metrics_overlay=metrics_overlay_path,
    )
