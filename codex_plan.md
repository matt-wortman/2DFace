# Solution 1 Implementation Plan – Deep Landmark & Feature Analysis Pipeline

## 1. Goals & Success Metrics
- Deliver a high-accuracy facial analysis workflow that can operate locally on a single NVIDIA RTX 4090 workstation (Windows 11 or Ubuntu 22.04) while remaining ready for hosted/web deployment.
- Ensure architecture can power a patient-facing web experience where users capture images via webcam or upload photos; offline deployment remains supported for clinics prioritizing on-prem privacy.
- Input: 1–3 high-resolution 2D photos per patient (frontal mandatory, 45° and/or profile optional); output within 5 seconds per image.
- Quantitative targets: ≥99% face-detection recall on internal validation images, ≤2.5 px normalized landmark error (NME) for front view, reproducible metric variance ≤5% across repeat runs, ≥0.85 AUROC for each ML attribute.
- Operational targets: deterministic report generation (JSON + clinician-friendly summary + annotated overlays), editable thresholds, complete offline install bundle, versioned models & configs.

## 2. System Architecture & Data Flow
- Modular pipeline: `ingestion → preprocessing → detection → landmarking → measurement → rule/ML inference → aggregation → reporting → feedback logging`.
- Data stores: (a) encrypted raw image vault; (b) processed artifacts (`.npy`, `.json`) per case; (c) model registry with semantic versioning; (d) SQLite feedback DB.
- Services: core pipeline exposed as a Python package + CLI; optional local FastAPI server or PyQt GUI; plan for a secure web front end (Next.js or similar) that interacts with the API for webcam capture/upload once regulatory approvals are cleared.
- Configuration: centralized YAML (`config/pipeline.yaml`) capturing model paths, thresholds, measurement definitions, UI toggles.
- Logging: structured logging (JSON) with unique case IDs, model versions, timings; persisted locally for audits.

## 3. Data & View Management
- Define capture protocol metadata schema: `case_id`, `capture_timestamp`, `view` (`front`, `profile_left`, `profile_right`, `oblique_left/right`), camera body/lens, distance notes.
- Implement ingestion layer that validates filename conventions and required metadata YAML/JSON per case; reject or quarantine non-compliant inputs.
- Orientation & quality heuristics: blur detection (Laplacian), exposure check (histogram), pose estimation (head tilt) to warn surgeons before processing.
- Support live webcam capture in the future web UI by exposing ingestion endpoints that accept base64-encoded frames and enforce the same quality gates.
- Multi-view orchestrator: front view drives symmetric measurements; profile-only metrics (nasolabial angle, chin projection) require profile images—flag “insufficient view” when missing.
- Normalized coordinate frame strategy: align images using inter-pupillary line, store transformation matrices for reproducibility.

## 4. Environment & DevOps
- Workstation prerequisites: NVIDIA driver ≥ 555.xx, CUDA 12.2 toolkit, cuDNN 9.x.
- Provision isolated conda environment (`conda env create -f env.yml`) with pinned versions (PyTorch w/ CUDA, torchvision, onnxruntime-gpu, opencv-python-headless, face-alignment, mediapipe, albumentations, pandas, scipy, scikit-image, PyQt6, FastAPI, uvicorn, SQLAlchemy, pytest, mypy, black).
- Dependency policy: default to the most recent stable releases for all libraries; only pin older versions when compatibility or reproducibility requires it, documenting each exception in `configs/pins.md`.
- Create internal PyPI mirror/cache (`devops/pypi-mirror/`) to enable completely offline installs; document refresh cadence.
- Repository scaffolding: `src/` for package, `models/` for weights, `configs/`, `tests/`, `ui/`, `notebooks/`, `docs/`.
- CI/CD: local Git hooks + GitHub Actions (if allowed) running lint, type-check, unit/integration suites; nightly job benchmarking inference latency and accuracy on a fixed validation set.

## 5. Face Detection & Preprocessing
- Acquire RetinaFace ResNet50 weights from InsightFace (Apache 2.0); store under `models/detectors/retinaface-r50.onnx` with SHA256 checksum file and license note.
- Implement detection module wrapping ONNXRuntime or PyTorch; support batching multi-view inputs; expose detection confidence threshold and NMS parameters via config.
- Preprocessing steps per image: convert to RGB, color constancy (Gray-World), resize keeping aspect ratio, pad to detector input size, run detection, crop with 15% margin, resize to landmark input size.
- Implement quality gating: skip/flag if detection confidence < threshold, if blur score < min, or face bounding box < config-defined min size.
- Persist intermediate artifacts (cropped faces, masks) for debugging; keep optional visualization flag.

## 6. Landmark Detection & Editing Tools
- Integrate `face-alignment` (FAN/HRNet) for 68-point 2D landmarks; download weights at install time, store locally with checksums.
- Implement wrapper supporting batched inference on GPU, fallback to CPU if CUDA unavailable; reuse detection boxes for speed.
- Optional dense mesh: integrate MediaPipe FaceMesh for secondary metrics (468 points) for surgeons wanting higher granularity.
- Profile handling: evaluate 2D/3D landmark model (e.g., `face-alignment` 3D model) for profile images; define fallback heuristics when 3D depth is unreliable.
- Build lightweight Qt-based landmark editor allowing manual drag-drop corrections, snap-to-line midpoints, and storing adjusted coordinates alongside original.
- Persist aligned landmarks in canonical coordinate space plus metadata (source model, manual edits flag).

## 7. Measurement & Anthropometric Library
- Define measurement catalog (e.g., symmetry scores, vertical thirds, nasal width ratios, chin projection proxies, lip fullness) each as a dataclass with inputs, dependencies, formula, units, normative range, citations.
- Normalize distances by inter-pupillary distance or other stable scaling factor to approximate millimeters; maintain scaling provenance in outputs.
- Implement measurement engine that runs only metrics whose required landmarks & views are present; mark results as `not_applicable` otherwise.
- Parameterize normative ranges by sex/ethnicity when provided; allow configuration overrides via `configs/norms.yaml`.
- Validate formulas via unit tests comparing to curated examples and cross-referencing literature values.

## 8. Rule-Based Issue Engine
- Define `issues.yaml` mapping measurements to cosmetic findings with severity buckets (normal, borderline, flagged) and templated narratives.
- Implement rule engine that consumes measurement outputs, applies thresholds, and aggregates multi-metric evidence with weighted scoring.
- Provide override hooks so surgeons can tune thresholds without redeploying (hot-reload or config reload).
- Emit explainability payload referencing specific measurements, values, and thresholds that triggered each issue.

## 9. Machine Learning Attribute Classifier
- Dataset provisioning: mirror CelebA locally, document download & checksum process, store in encrypted dataset volume; augment with in-house labeled cases as consent allows.
- Pre-processing: align faces using detected landmarks, generate standardized 224×224 crops, split into train/val/test ensuring no identity leakage.
- Modeling: build multi-label classifier (ResNet50 or EfficientNet-B0) with sigmoid outputs for attributes (wide nose, nasal asymmetry proxy, chin retrusion, lip thinness, brow ptosis, asymmetry).
- Training pipeline: implement PyTorch Lightning or custom training loop with mixed precision on 4090, early stopping, class-weighting; log metrics to local TensorBoard equivalent.
- Evaluation: compute AUROC, precision/recall, calibration curves; require human review of worst-performing samples.
- Model registry: package best checkpoint as TorchScript + ONNX, store under semantic version `models/classifiers/v1/`; document training data lineage, hyperparameters, and license notes.
- Bias review: run stratified analysis by gender/skin tone proxies (using CelebA attributes) to detect systematic errors; document mitigation steps.

## 10. Aggregation, Explainability & Confidence Scoring
- Design aggregator that combines rule engine outputs and classifier probabilities via configurable weighting; implement consistency checks (e.g., drop ML signal if measurement contradicts and ML confidence <0.6).
- Compute confidence scores per issue and overall case quality score; surface when recommendations are low confidence and require manual review.
- Generate human-readable explanations linking back to landmarks, metrics, and, if applicable, sample reference images.

## 11. Reporting & Visualization
- Produce structured JSON report (`reports/<case_id>.json`) capturing raw metrics, issues, confidences, and provenance (model versions, config hash).
- Generate clinician-facing PDF/HTML with summary table, severity color coding, and recommended next steps notes (template stored under `templates/report_template.html`).
- Create annotated imagery with landmarks, symmetry axes, measurement overlays, and highlight discrepancies; support multi-view layouts.
- Archive all outputs under case-specific folders with audit timestamps.

## 12. Surgeon Feedback & Active Learning Loop
- Extend UI to allow surgeons to: (a) toggle issue correctness; (b) adjust thresholds per case; (c) reposition landmarks; (d) add free-text comments.
- Feedback persistence: SQLite (`data/feedback.db`) with tables for `issues_feedback`, `landmark_corrections`, `notes`; enforce referential integrity to case IDs.
- Schedule retraining workflow (e.g., monthly) that ingests accumulated feedback, updates training datasets, retrains ML classifier, and recalibrates rule thresholds.
- Provide migration scripts to snapshot feedback DB and archive prior to each retraining cycle.

## 13. Quality Assurance & Testing
- Unit tests for each module (detection, preprocessing, measurement formulas, rule evaluation); aim for ≥85% coverage on core logic.
- Integration test suite using curated image set with ground-truth landmarks/measurements; include tolerance assertions.
- Visual regression tests: auto-generate overlays and compare against baselines using SSIM; flag drift.
- Performance tests benchmarking inference latency and GPU memory usage per stage; record in CI artifacts.
- Manual review protocol: quarterly clinician sign-off on randomly sampled reports.

## 14. Deployment, Packaging & Operations
- Build offline installer: bundle conda env, models, configs, and license docs into a signed archive; include checksum manifest.
- Provide CLI entry point (`faceapp analyze --input <path>`) and optional PyQt desktop app packaged via PyInstaller/MSIX.
- Implement runtime checks for GPU availability, disk space, and driver versions; fall back to CPU (with warning) when necessary.
- Logging & monitoring: rotate logs, expose simple dashboard (Streamlit or textual CLI) summarizing recent analyses, performance metrics, and feedback counts.
- Backup strategy: nightly local backup of reports and feedback DB to encrypted external drive or secure NAS.

## 15. Security, Privacy & License Compliance
- Enforce full-disk encryption and OS-level user access controls; store encryption keys securely (TPM on Windows, LUKS passphrase on Linux).
- Scrub EXIF metadata on ingestion; hash patient identifiers and avoid storing PII beyond case IDs.
- Maintain `docs/licenses.md` enumerating third-party assets (models, datasets) with usage terms; add automated check ensuring assets are present with corresponding license files.
- Document patient consent workflow and data retention policy aligned with clinical governance.

## 16. Project Management, Staffing & Timeline
- Roles: ML engineer (landmarks & classifiers), CV engineer (detection & measurements), Full-stack engineer (UI, packaging), Clinical SME (validation), DevOps (offline infra support).
- Sprint 0 (1 week): finalize requirements, set up repo, secure data storage, establish offline dependency mirror.
- Sprint 1 (2 weeks): implement detection + preprocessing + quality gates; deliver CLI prototype with cropped face outputs.
- Sprint 2 (2 weeks): integrate landmark models, build manual editor, persist aligned landmarks; complete measurement engine MVP.
- Sprint 3 (2 weeks): finalize anthropometric catalog, implement rule engine & reporting templates; clinician review of measurement outputs.
- Sprint 4 (3 weeks): stand up ML classifier training pipeline, run initial training, integrate aggregator, conduct bias review.
- Sprint 5 (2 weeks): deliver UI/feedback loop, add retraining automation, finalize packaging & installer, execute end-to-end acceptance tests with surgeons.
- Post-launch: monthly maintenance cadence (feedback review, retraining, dependency updates, security patching).

## 17. Risk Register & Mitigations
- **Limited labeled data**: prioritize rule-based engine, leverage transfer learning, invest in clinician labeling sessions.
- **Landmark failures on atypical faces**: provide manual editing, maintain fallback detectors, log failure cases for future tuning.
- **Regulatory scrutiny**: keep detailed audit logs, include disclaimers and human-in-the-loop requirement.
- **Dependency drift**: pin versions, run quarterly regression suite before upgrades.
- **Hardware downtime**: document recovery procedure, maintain spare GPU-enabled workstation or cloud burst plan (with secure tunnel) as contingency.
