# Phase 2 Plan – Wrinkle Analysis Alignment & Expansion

**Date:** October 24, 2025  
**Maintainers:** FaceApp Computer Vision Team  
**Purpose:** Working plan for the next implementation sprint on wrinkle analysis. This consolidates the October 24 reviews (Codex + Cline) with prior roadmap commitments. Wrinkle detection is one pillar inside the broader mandate: identify facial attributes that may warrant cosmetic surgical correction (symmetry, proportions, contours, texture).

---

## 1. Current Snapshot
- **Baseline in place:** CLAHE-driven preprocessing, single-scale (λ=4.0) Gabor ridge detection for crow’s feet & nasolabial folds, skeletonization, length/intensity metrics, interpupillary normalization, rule-based severity tagging. (`src/analysis/wrinkles.py`, `src/analysis/measurements.py`, `configs/rules.yaml`)
- **Review highlights:** Missing multi-scale filtering, limited geometric cleanup, no segment count or depth proxy metrics, severity is threshold-only.
- **Outstanding Stage work (`docs/PROGRESS.md`):** Stage 7 benchmarks & repeatability logs, Stage 8 documentation polish, surgeon-threshold calibration.
- **Parallel initiatives:** Learning-based wrinkle exploration, multi-view orchestration, 3D groundwork, surgeon feedback tooling (`codex_plan.md`, `local_image_analysis_plan.md`).

---

## 2. Priority Backlog
Time assumptions: ~80% focus for one engineer. Update inline status/owners as work progresses.

### 2.1 Immediate (Target: 1–2 days)
1. **Multi-scale Gabor bank**  
   - Extend `_enhance_texture` to iterate over fine/medium/deep kernels; combine responses via per-pixel max.  
   - Benchmark on crow’s feet + nasolabial sample set.
2. **Geometric cleanup pass**  
   - Add dilation/thinning plus length/aspect-ratio component filtering and optional short-gap bridging.  
   - Provide debug overlays for kept vs. rejected segments.
3. **Wrinkle configuration surface**  
   - Introduce `configs/wrinkle.yaml` with defaults (see Section 3).  
   - Ensure CLI loads config automatically with backward-compatible fallbacks.

### 2.2 Near-Term (Target: follow-on 1–2 days)
4. **Wrinkle segment counts & labeling**  
   - Emit per-region segment count, segment metadata (centroid, length) into metrics JSON.
5. **Cross-line intensity profiling**  
   - Sample perpendicular intensity drop per segment to approximate depth proxy.  
   - Store in `WrinkleMetric` for downstream scoring.
6. **Composite severity index v1**  
   - Combine length ratio, average intensity, segment count via configurable weights; output both scalar score (0–1) and grade (`normal/warn/flag`).  
   - Document calibration assumptions pending surgeon feedback.

### 2.3 Validation & Harden (Target: 1 day)
7. **Benchmark + regression pack**  
   - Add unit tests for multi-scale config parsing, geometric filtering, metric serialization.  
   - Capture before/after overlays (≥3 cases) in `outputs/regression/`.  
   - Record per-ROI latency; target <25 ms/ROI on RTX 4090.
8. **Rules refresh & docs**  
   - Update rule thresholds to leverage severity index while retaining compatibility with raw metrics.  
   - Draft wrinkle tuning guide (`docs/guides/wrinkle_playbook.md`) with parameter explanations & troubleshooting.
9. **Stage 7/8 carry-over**  
   - Log repeatability/performance results post-change.  
   - Update `docs/KNOWN_ISSUES.md` and launch surgeon feedback capture template (see Section 7).

---

## 3. Key Technical Defaults (Phase 2 Targets)
| Parameter | Value / Target | Notes |
|-----------|----------------|-------|
| Gabor scales | λ ∈ {3.0, 4.0, 6.0}, σ ∈ {1.2, 1.5, 2.0}, kernel sizes {7, 9, 11} | Keep 12 orientations, γ=0.5, ψ=0 |
| Morphology | 3×3 kernel, 1 iteration open + optional closing | Keep ROI-constrained |
| Component filters | Min length 10 px, min aspect ratio 2.0, max gap bridge 5 px | Tune as necessary |
| Severity weights | Default length/intensity/count = 0.4/0.3/0.3 | Adjustable via config |
| Severity scale | Score 0–1 → grade (`normal/warn/flag`) | Map thresholds in config |
| Performance budget | <25 ms per ROI (wrinkle stage) | Record baseline before/after |
| Config file | `configs/wrinkle.yaml` | Mirror `quality.yaml`/`rules.yaml` style |

Document all defaults in the config file with inline comments.

**Example `configs/wrinkle.yaml`:**
```yaml
gabor:
  orientations: 12
  gamma: 0.5
  psi: 0.0
  scales:
    - lambda: 3.0
      sigma: 1.2
      kernel_size: 7
    - lambda: 4.0
      sigma: 1.5
      kernel_size: 9
    - lambda: 6.0
      sigma: 2.0
      kernel_size: 11

morphology:
  open_kernel: 3
  bridge_gap_px: 5
  min_length_px: 10
  min_aspect_ratio: 2.0

severity:
  weights:
    length_ratio: 0.4
    intensity: 0.3
    segment_count: 0.3
  thresholds:
    warn: 0.45
    flag: 0.65
  notes: "Weights/thresholds are provisional; tune with surgeon feedback."
```

**Severity score formula (default):**
\[
\text{score} = w_L \cdot \hat{L} + w_I \cdot \hat{I} + w_C \cdot \hat{C}
\]
where each term is min-max normalized to [0, 1] per region, weights come from config, and grades map as `normal` < warn ≤ `warn` < flag ≤ `flag`.

---

## 4. Execution Notes & Integration
- **Pipeline order:** Maintain current ROI extraction flow; avoid full-frame passes unless profiling demands.  
- **Config loading:** `src/cli/analyze.py` should load `configs/wrinkle.yaml` (fallback to defaults when absent) and pass settings to wrinkle analyzers.  
- **Pipeline wiring:** `src/pipeline/process.py` passes config into `analyze_crows_feet` / `analyze_nasolabial_folds`, collects new metrics (segment_count, depth_proxy, severity_score, severity_grade) for JSON/HTML outputs.  
- **Report template:** Update `templates/report.html.j2` to display severity grade (color-coded), segment count, and depth proxy per region.  
- **Visualization toggles:** Expose CLI flag or config switch for overlaying rejected segments during debugging.  
- **Branching:** Use `feature/wrinkle-phase2`; commit per backlog item for traceability.

---

## 5. Open Questions
1. **Severity scale alignment:** Do we align with WSRS/IGAIS immediately or keep an internal 0–1 scale until calibration data arrives?  
2. **Minimum length sanity check:** Does 10 px hold across our current resolution range? Validate against sample set.  
3. **Additional regions:** Should forehead/glabellar detection join Phase 2 or wait for Phase 3?  
4. **Annotation workflow ownership:** Who gathers surgeon labels and where do we store them?  

Document decisions inline here when resolved.

---

## 6. Validation & Exit Criteria
We consider Phase 2 complete when all items below pass:
1. **Functional improvements:** Multi-scale + geometric filtering visibly improve traces and reduce obvious false positives on sanity dataset. Target ≥30% reduction in pore/makeup false positives versus current baseline with <5% loss of true wrinkles.  
2. **Metric coverage:** Metrics JSON reports length, intensity, segment count, depth proxy, severity score/grade per region; serialization tests pass.  
3. **Performance:** Wrinkle stage stays within 25 ms/ROI (record baseline and post-change).  
4. **Rules integration:** Composite severity drives rule outputs without breaking existing downstream consumers (regress CLI JSON schema/tests).  
5. **Documentation:** Config, playbook, changelog, and surgeon feedback template updates completed (Section 7).  
6. **Benchmarks logged:** Stage 7 repeatability/performance notes updated; Stage 8 documentation tasks closed.  
7. **Sign-off:** Surgeon-feedback loop ready (template + instructions) even if data collection is pending.

---

## 7. Documentation & Deliverables
- `configs/wrinkle.yaml` with inline comments & sensible defaults.  
- `docs/guides/wrinkle_playbook.md` covering configuration parameters, tuning guidance, common pitfalls.  
- `docs/SURGEON_FEEDBACK_TEMPLATE.md` (lightweight form: case ID, region, severity rating, missed wrinkles, comments, follow-up).  
- README update noting severity scoring capability + link to playbook.  
- `docs/api/README.md` additions for new metrics fields.  
- `CHANGELOG.md` entry (target version bump if warranted) summarizing Phase 2 changes.  
- `docs/KNOWN_ISSUES.md` updates for 2D wrinkle limitations (lighting, heavy makeup, curvature edge cases).

---

## 8. Testing Strategy
- **Unit tests:**  
  - Validate Gabor config parsing and default fallbacks.  
  - Verify `_filter_geometric_constraints` removes sub-threshold components and preserves valid ridges using synthetic masks.  
  - Confirm severity scoring math with deterministic normalized inputs and weight permutations.  
- **Integration/regression tests:**  
  - Golden-output comparisons for ≥3 representative images capturing metrics JSON, wrinkle overlays, and severity grades (store under `tests/data/wrinkle_regression/`).  
  - CLI smoke test (`python -m src.cli.analyze data/sanity/*.jpg`) to ensure orchestration still succeeds with new metrics.  
- **Performance checks:**  
  - Scripted benchmark recording per-stage timings before/after multi-scale rollout (log to `outputs/benchmarks/wrinkle_phase2.json`).  
  - Fail CI if ROI latency exceeds 25 ms by >10% margin.

---

## 9. Risks & Mitigations
- **Performance regression:** Multi-scale filters may exceed ROI budget → profile early; consider pruning scales or leveraging OpenCV optimizations if >25 ms/ROI.  
- **Threshold drift:** Default length/aspect thresholds may not generalize → bake config overrides, keep debug overlays to inspect segments quickly.  
- **Severity calibration gap:** Without surgeon data, severity weights could mis-rank cases → mark outputs as “preliminary” and capture clinician feedback via template.  
- **Scope creep:** Forehead/other regions tempting to add mid-sprint → defer to backlog unless needed for validation.  
- **Documentation debt:** Multiple deliverables risk slipping → tie completion to exit criteria (Section 6).

---

## 10. Future Directions (Maintained)
- **Learning-based wrinkle detection & aggregated reporting** – Resume research once Phase 2 stabilizes.  
- **Multi-view expansion** – Integrate oblique/profile metrics per long-term plan.  
- **3D readiness** – Begin stereophotogrammetry requirements gathering.  
- **Surgeon feedback tooling** – Build lightweight portal or script for annotations tied to case IDs.  
- **Developer experience** – Evaluate Sphinx/MkDocs publishing and internal packaging once pipeline solidifies.  
- **UI/report polish** – Enhance HTML reports / consider desktop or web viewer.  
- **Performance automation** – Finish Stage 7 benchmarking scripts and GPU monitoring.

---

## 11. References
- Technical Summary “2D vs 3D Wrinkle Detection Approaches”.  
- Reviews: `codex-review.md`, `cline-review.md`.  
- Planning artifacts: `Phase2.md`, `codex_plan.md`, `local_image_analysis_plan.md`.  
- Baseline implementation: `src/analysis/wrinkles.py`, `src/analysis/measurements.py`, `src/pipeline/process.py`, `templates/report.html.j2`.

Update this document whenever backlog status, technical defaults, or decisions change.
