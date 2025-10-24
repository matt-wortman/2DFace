from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from src.analysis.measurements import Measurement


@dataclasses.dataclass
class Threshold:
    warn_above: Optional[float] = None
    flag_above: Optional[float] = None
    warn_below: Optional[float] = None
    flag_below: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "Threshold":
        valid_keys = {"warn_above", "flag_above", "warn_below", "flag_below"}
        params = {k: float(v) for k, v in data.items() if k in valid_keys}
        return cls(**params)


@dataclasses.dataclass
class Finding:
    name: str
    severity: str  # normal, warn, flag
    metric_value: float
    threshold: float
    comparison: str
    message: str

    def to_json(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "severity": self.severity,
            "metric_value": float(self.metric_value),
            "threshold": float(self.threshold),
            "comparison": self.comparison,
            "message": self.message,
        }


_MESSAGES = {
    "facial_symmetry": "Facial symmetry below ideal range",
    "nasolabial_fold_ratio": "Prominent nasolabial folds",
    "chin_midline_deviation": "Chin deviates from facial midline",
    "jaw_width_ratio": "Jaw width exceeds proportional range",
    "crow_feet_ratio": "Crow's feet region appears pronounced",
    "facial_third_lower": "Lower face length exceeds expected proportion",
    "upper_lip_thickness_ratio": "Upper lip appears thin",
}


def load_rule_thresholds(path: Path) -> Dict[str, Threshold]:
    if not path.exists():
        raise FileNotFoundError(f"Rules config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    thresholds = {name: Threshold.from_dict(values or {}) for name, values in data.items()}
    return thresholds


def _evaluate_single(name: str, value: float, threshold: Threshold) -> Optional[Finding]:
    # Check "flag" conditions first for higher severity.
    if threshold.flag_above is not None and value > threshold.flag_above:
        return Finding(name, "flag", value, threshold.flag_above, ">", _MESSAGES.get(name, name))
    if threshold.flag_below is not None and value < threshold.flag_below:
        return Finding(name, "flag", value, threshold.flag_below, "<", _MESSAGES.get(name, name))

    if threshold.warn_above is not None and value > threshold.warn_above:
        return Finding(name, "warn", value, threshold.warn_above, ">", _MESSAGES.get(name, name))
    if threshold.warn_below is not None and value < threshold.warn_below:
        return Finding(name, "warn", value, threshold.warn_below, "<", _MESSAGES.get(name, name))

    return None


def evaluate_findings(metrics: Iterable[Measurement], thresholds: Dict[str, Threshold]) -> List[Finding]:
    metric_map = {m.name: m for m in metrics}
    findings: List[Finding] = []
    for name, threshold in thresholds.items():
        measurement = metric_map.get(name)
        if measurement is None:
            continue
        finding = _evaluate_single(name, measurement.value, threshold)
        if finding:
            findings.append(finding)
    return findings
