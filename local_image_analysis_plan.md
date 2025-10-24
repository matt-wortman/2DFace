# Local Image Analysis Bring-Up Plan

**Objective:** Implement a dependable face detection, landmarking, measurement, and reporting pipeline on the local workstation before layering additional product features.

---

## Stage 0 – Baseline Environment (Day 0–1)
1. **GPU & Driver Verification**
   - Run `nvidia-smi` to confirm GPU visibility.
   - Capture CUDA toolkit and driver versions; record in `docs/setup/local_env.md`.
2. **Python Environment Bootstrap**
   - Install Miniconda if missing.
   - Create environment `faceapp-local` with Python ≥3.10:  
     `conda create -n faceapp-local python=3.10 -y`
   - Activate env and install minimal dependencies:  
     `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`  
     `pip install opencv-python-headless face-alignment insightface onnxruntime-gpu numpy scipy matplotlib click rich`
   - Export `requirements-local.txt`.
3. **Dataset Folder Prep**
   - Create `data/sanity/` with 5–10 test images (frontal + profile if available).  
   - Add README noting provenance and consent.

**Exit Criteria:** conda env active, GPU accessible, sample images ready.

---

## Stage 1 – Face Detection Module (Day 1–2)
1. **RetinaFace Integration**
   - Download InsightFace RetinaFace ResNet50 ONNX (store at `models/detectors/retinaface_r50.onnx`).
   - Implement `src/pipeline/detect.py` exposing `detect_faces(image_path, config)` returning bounding boxes + confidences.
2. **CLI Wrapper**
   - Create `scripts/detect_faces.py` to batch through input folder, save annotated PNGs (with boxes + scores) to `outputs/detections/`.
3. **Validation**
   - Run on sanity dataset; review overlays for false positives/negatives.
   - Log inference time and GPU usage (target <100ms/image).

**Exit Criteria:** reliable detections across sample set; CLI produces annotated outputs and JSON metadata.

---

## Stage 2 – Landmark Extraction (Day 2–3)
1. **face-alignment Wrapper**
   - Implement `src/pipeline/landmarks.py` using 68-point FAN model; accept optional bounding boxes for faster inference.
2. **Visualization**
   - Add overlay generation to `outputs/landmarks/` with landmarks drawn on face crop.
3. **Quality Assessment**
   - Inspect each sample; note failure cases, adjust padding or detection thresholds as needed.
   - Record baseline latency per image.

**Exit Criteria:** consistent 68-point landmarks on all test photos with overlays saved.

---

## Stage 3 – Quality Gating Utilities (Day 3)
1. **Image Quality Metrics**
   - Implement blur detection (variance of Laplacian) and exposure score (histogram-based).
   - Add head tilt heuristic using selected landmarks (eye-to-eye line).
2. **Integration**
   - Store results in metadata JSON; warn (not block) when thresholds breached.
   - Document threshold defaults in `configs/quality.yaml`.

**Exit Criteria:** pipeline flags low-quality inputs and records quality metadata.

---

## Stage 4 – Measurement Library (Day 3–4)
1. **Measurement Catalog**
   - Implement module `src/analysis/measurements.py` with initial metrics: facial symmetry score, facial thirds, nose width ratio, chin deviation, jaw width, mouth width, lip height, nasolabial approximation.
   - Normalize distances with interpupillary distance; track assumptions.
2. **Unit Testing**
   - Add fixtures with synthetic landmarks to verify formulas (store under `tests/fixtures/`).
3. **Metric Output**
   - Produce structured JSON (`outputs/metrics/<case_id>.json`) listing metric values, units, and validity flags.

**Exit Criteria:** measurement functions tested, metrics generated for sample set.

---

## Stage 5 – Rule-Based Findings (Day 4)
1. **Rule Engine**
   - Create `src/analysis/rules.py` reading thresholds from `configs/rules.yaml`.
   - Map metric deviations to findings with confidence levels.
2. **Summary Generator**
   - Output per-case summary JSON and plaintext snippet.
   - Log triggered rules for debug traceability.

**Exit Criteria:** rules produce interpretable findings for each case; configs easily tweakable.

---

## Stage 6 – End-to-End CLI & Reports (Day 5)
1. **Pipeline Orchestration**
   - Implement `src/cli/analyze.py` composing detection → landmarks → quality → measurements → rules.
   - CLI command: `python -m src.cli.analyze --input data/sanity --output outputs/sanity --config configs/pipeline.yaml`.
2. **HTML Report**
   - Generate simple Jinja2-based report summarizing metrics, findings, quality warnings, and embed overlay images.
3. **Logging**
   - Use `rich` or `logging` to stream progress and timings; dump structured logs per run.

**Exit Criteria:** single command analyzes dataset, saves overlays, metrics, reports without errors.

---

## Stage 7 – Reliability & Performance Checks (Day 5–6)
1. **Repeatability Test**
   - Run pipeline twice on same dataset; verify identical outputs (hash or compare JSON).
2. **Performance Metrics**
   - Record per-stage latency, GPU memory usage; document in `docs/setup/local_env.md`.
3. **Failure Handling**
   - Confirm graceful behavior when images lack faces or quality thresholds fail.

**Exit Criteria:** pipeline stable, performance documented, failure cases handled with clear messaging.

---

## Stage 8 – Documentation & Next Steps (Day 6)
1. **Developer Notes**
   - Update `docs/setup/local_env.md` with exact steps, gotchas, and troubleshooting tips.
2. **Known Issues Log**
   - Start `docs/KNOWN_ISSUES.md` capturing misdetections or measurement inaccuracies.
3. **Docker Decision Point**
   - Evaluate need for containerization; if desired, draft Dockerfile mirroring working env (postpone build until sign-off).

**Exit Criteria:** documentation up to date, outstanding issues tracked, go/no-go decision recorded for next phase.

---

## Work Tracking & Tooling
- Use a simple checklist in `docs/progress/local_pipeline_checklist.md` to mark completion.
- Log commands and results in `logs/local_pipeline_dev.log` for reproducibility.
- Defer Docker, UI, ML classifier, and advanced security until this plan’s exit criteria are met.

