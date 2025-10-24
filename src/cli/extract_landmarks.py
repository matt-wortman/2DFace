from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.progress import track

from src.analysis.quality import load_thresholds
from src.analysis.rules import load_rule_thresholds
from src.pipeline.detect import InsightFaceDetector
from src.pipeline.landmarks import FaceAlignmentWrapper
from src.pipeline.process import PipelineResult, run_pipeline

console = Console()


@click.command()
@click.option("--input", "input_dir", type=click.Path(path_type=Path), required=True, help="Directory of input images.")
@click.option("--output", "output_dir", type=click.Path(path_type=Path), required=True, help="Directory to save outputs.")
@click.option(
    "--model-root",
    type=click.Path(path_type=Path),
    default=Path("models/insightface"),
    show_default=True,
    help="Directory for InsightFace cached models.",
)
@click.option(
    "--quality-config",
    type=click.Path(path_type=Path),
    default=Path("configs/quality.yaml"),
    show_default=True,
    help="YAML file with quality thresholds.",
)
@click.option(
    "--rules-config",
    type=click.Path(path_type=Path),
    default=Path("configs/rules.yaml"),
    show_default=True,
    help="YAML file with rule thresholds.",
)
def main(input_dir: Path, output_dir: Path, model_root: Path, quality_config: Path, rules_config: Path) -> None:
    if not input_dir.exists():
        raise click.ClickException(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    detector = InsightFaceDetector(model_root=model_root)
    landmark_model = FaceAlignmentWrapper()
    thresholds = load_thresholds(quality_config)
    rule_thresholds = load_rule_thresholds(rules_config)

    image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not image_paths:
        console.print(f"[yellow]No images found in {input_dir}.")
        return

    for image_path in track(image_paths, description="Processing"):
        result = run_pipeline(
            image_path=image_path,
            output_dir=output_dir,
            detector=detector,
            landmark_model=landmark_model,
            quality_thresholds=thresholds,
            rule_thresholds=rule_thresholds,
            save_overlays=True,
        )

        payload = result.to_json()
        json_path = output_dir / f"{image_path.stem}_landmarks.json"
        json_path.write_text(json.dumps(payload, indent=2))

        severity_summary = (
            ", ".join(f"{f['name']}({f['severity']})" for f in payload["findings"])
            if payload["findings"]
            else "none"
        )
        console.log(
            f"{image_path.name}: {len(result.faces)} faces landmarked | quality: {result.quality.get('status', 'unknown')} | findings: {severity_summary}"
        )

    console.print(f"[green]Processing complete. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
