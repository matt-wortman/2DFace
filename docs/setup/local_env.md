# Local Environment Notes

## GPU & Driver Snapshot (2025-10-23)
- GPU: NVIDIA GeForce RTX 4090
- Driver Version: 576.28
- CUDA Version: 12.9 (per `nvidia-smi`)
- NVIDIA-SMI Version: 575.51.03

## Next Actions
- Install/configure Python environment (`faceapp-local`).
- Record exact package versions once environment is bootstrapped.

## Python Environment (`faceapp-local`)
- Python: 3.10.19 (conda env `faceapp-local`)
- Installed key packages (2025-10-23): torch 2.5.1+cu121, torchvision 0.20.1+cu121, opencv-python-headless 4.12.0.88, face-alignment 1.4.1, insightface 0.7.3, onnxruntime-gpu 1.23.2, scipy 1.15.3, matplotlib 3.10.7, click 8.3.0, rich 14.2.0.
- Requirements export: `requirements-local.txt` (auto-generated after install).

## Outstanding Items
- Populate `data/sanity/` with test images (pending).
- Begin Stage 1 detection module once sample assets are ready.

## Stage 1 Progress (2025-10-23)
- InsightFace `buffalo_l` models cached under `models/insightface`.
- Detection CLI: `python -m src.cli.detect_faces --input data/sanity --output outputs/detections` (threshold 0.3, det_size 640x640).
- 12 sanity images processed successfully; each yielded 1 detection with overlays stored alongside JSON metadata.

## Stage 2 Progress (2025-10-23)
- Added `src/pipeline/landmarks.py` (face-alignment wrapper) and `src/cli/extract_landmarks.py`.
- Command: `python -m src.cli.extract_landmarks --input data/sanity --output outputs/landmarks`.
- Downloads: face-alignment FAN (s3fd + 2D model) cached in `~/.cache/torch/hub/`.
- All 12 sanity images processed with 68-point overlays + JSON landmarks.
- Outputs stored under `outputs/landmarks/` (one PNG + JSON per image).

## Stage 3 Progress (2025-10-23)
- Added quality heuristics (`src/analysis/quality.py`) with Laplacian blur, mean exposure, and roll angle checks.
- Quality thresholds configurable via `configs/quality.yaml`.
- Landmark CLI embeds quality block into JSON and logs warnings (e.g., roll angle exceedances) without blocking execution.

## Stage 4 Progress (2025-10-23)
- Added measurement library (`src/analysis/measurements.py`) computing symmetry, facial thirds, nasolabial, crow's feet, lip thickness, etc.
- Implemented unit tests (`tests/test_measurements.py`) – all passing via `pytest -q`.
- Updated extraction CLI to embed metrics and evaluate rule-based findings using `configs/rules.yaml` and new `src/analysis/rules.py` engine.
- JSON outputs now include `metrics` array and `findings` list with severity levels, logged in console during runs.
- Measurement overlays generated at `outputs/landmarks/<image>_metrics.png` to visualize segments used for each metric.

## Stage 6 Progress (2025-10-23)
- Refactored pipeline execution into `src/pipeline/process.py` for reuse.
- Added `src/cli/analyze.py` orchestrating detection→landmarks→quality→metrics→findings with HTML reporting via `templates/report.html.j2`.
- Created `configs/pipeline.yaml` to centralize model paths and output preferences.
- Reports generated alongside overlays (`*_report.html`) plus master `summary.json` (example run: `python -m src.cli.analyze --input data/nano --output outputs/nano_reports`).
- Added wrinkle analysis (`src/analysis/wrinkles.py`) using Gabor-based ridge detection for crow's feet and nasolabial folds; new metrics (`*_length_ratio`, `*_intensity`) now flow into rules and reports.
