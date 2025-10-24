from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import click
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
from rich.console import Console
from rich.progress import track

from src.analysis.quality import load_thresholds
from src.analysis.rules import load_rule_thresholds
from src.pipeline.detect import InsightFaceDetector
from src.pipeline.landmarks import FaceAlignmentWrapper
from src.pipeline.process import PipelineResult, run_pipeline

console = Console()


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise click.ClickException(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def prepare_detector(config: Dict[str, Any], model_root: Path) -> InsightFaceDetector:
    det_cfg = config.get("detection", {})
    threshold = float(det_cfg.get("threshold", 0.3))
    det_size = det_cfg.get("det_size", [640, 640])
    return InsightFaceDetector(
        det_size=det_size,
        threshold=threshold,
        model_root=model_root,
    )


def render_report(result: PipelineResult, template_path: Path, output_path: Path) -> None:
    env = Environment(
        loader=FileSystemLoader(template_path.parent),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template(template_path.name)

    context = {
        "image_name": result.image_name,
        "quality": result.quality,
        "findings": [finding.to_json() for finding in result.findings],
        "metrics": [metric.to_json() for metric in result.metrics],
        "landmark_overlay": Path(result.landmark_overlay).name if result.landmark_overlay else None,
        "face_overlays": [path.name for path in result.face_overlays],
        "metrics_overlay": Path(result.metrics_overlay).name if result.metrics_overlay else None,
    }

    output_path.write_text(template.render(**context), encoding="utf-8")


@click.command()
@click.option("--input", "input_dir", type=click.Path(path_type=Path), required=True, help="Directory of input images.")
@click.option("--output", "output_dir", type=click.Path(path_type=Path), required=True, help="Directory for analysis outputs.")
@click.option(
    "--config",
    type=click.Path(path_type=Path),
    default=Path("configs/pipeline.yaml"),
    show_default=True,
    help="Pipeline configuration YAML.",
)
def main(input_dir: Path, output_dir: Path, config: Path) -> None:
    if not input_dir.exists():
        raise click.ClickException(f"Input directory not found: {input_dir}")

    cfg = load_config(config)

    model_root = Path(cfg.get("model_root", "models/insightface"))
    quality_cfg_path = Path(cfg.get("quality_config", "configs/quality.yaml"))
    rules_cfg_path = Path(cfg.get("rules_config", "configs/rules.yaml"))
    template_path = Path(cfg.get("report_template", "templates/report.html.j2"))

    output_dir.mkdir(parents=True, exist_ok=True)

    detector = prepare_detector(cfg, model_root)
    landmark_model = FaceAlignmentWrapper()
    thresholds = load_thresholds(quality_cfg_path)
    rule_thresholds = load_rule_thresholds(rules_cfg_path)

    image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not image_paths:
        console.print(f"[yellow]No images found in {input_dir}.")
        return

    summary: Dict[str, Any] = {"images": []}

    for image_path in track(image_paths, description="Analyzing"):
        result = run_pipeline(
            image_path=image_path,
            output_dir=output_dir,
            detector=detector,
            landmark_model=landmark_model,
            quality_thresholds=thresholds,
            rule_thresholds=rule_thresholds,
            save_overlays=bool(cfg.get("output_overlays", True)),
        )

        payload = result.to_json()
        if cfg.get("output_json", True):
            json_path = output_dir / f"{image_path.stem}_analysis.json"
            json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        report_path = output_dir / f"{image_path.stem}_report.html"
        render_report(result, template_path, report_path)

        summary_entry = {
            "image": result.image_name,
            "quality": result.quality,
            "findings": payload["findings"],
            "metrics": payload["metrics"],
            "report": report_path.name,
        }
        summary["images"].append(summary_entry)

        severity_summary = (
            ", ".join(f"{f['name']}({f['severity']})" for f in payload["findings"])
            if payload["findings"]
            else "none"
        )
        console.log(
            f"{image_path.name}: {len(result.faces)} faces | quality: {result.quality.get('status', 'n/a')} | findings: {severity_summary}"
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    console.print(f"[green]Analysis complete. Reports in {output_dir}")


if __name__ == "__main__":
    main()
