from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import click
import cv2
from rich.console import Console
from rich.progress import track

from src.pipeline.detect import InsightFaceDetector, annotate_detections, load_image

console = Console()


def iter_image_paths(input_dir: Path, patterns: Sequence[str]) -> Sequence[Path]:
    image_paths = []
    for pattern in patterns:
        image_paths.extend(sorted(input_dir.glob(pattern)))
    return image_paths


@click.command()
@click.option("--input", "input_dir", type=click.Path(path_type=Path), required=True, help="Directory containing images.")
@click.option(
    "--output",
    "output_dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory to store detection overlays and metadata.",
)
@click.option("--threshold", type=float, default=0.3, show_default=True, help="Detection confidence threshold.")
@click.option("--det-size", type=(int, int), default=(640, 640), show_default=True, help="RetinaFace detection window.")
@click.option(
    "--model-root",
    type=click.Path(path_type=Path),
    default=Path("models/insightface"),
    show_default=True,
    help="Directory for caching InsightFace models.",
)
def main(input_dir: Path, output_dir: Path, threshold: float, det_size: Sequence[int], model_root: Path) -> None:
    """Run InsightFace RetinaFace detections across a directory of images."""
    if not input_dir.exists():
        raise click.ClickException(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    detector = InsightFaceDetector(det_size=det_size, threshold=threshold, model_root=model_root)
    image_paths = iter_image_paths(input_dir, ("*.jpg", "*.jpeg", "*.png"))

    if not image_paths:
        console.print(f"[yellow]No images found in {input_dir} (jpg/jpeg/png).[/yellow]")
        return

    console.print(f"[bold cyan]Processing {len(image_paths)} images from {input_dir}[/bold cyan]")

    for image_path in track(image_paths, description="Detecting faces"):
        image = load_image(image_path)
        detections = detector.detect_array(image)

        result_json = {
            "image": image_path.name,
            "detections": [det.to_json() for det in detections],
        }

        json_path = output_dir / f"{image_path.stem}_detections.json"
        json_path.write_text(json.dumps(result_json, indent=2))

        overlay = annotate_detections(image, detections)
        overlay_path = output_dir / f"{image_path.stem}_detections.png"
        cv2.imwrite(str(overlay_path), overlay)

        console.log(f"{image_path.name}: {len(detections)} faces detected -> {overlay_path.name}")

    console.print(f"[green]Completed detection run. Outputs in {output_dir}[/green]")


if __name__ == "__main__":
    main()
