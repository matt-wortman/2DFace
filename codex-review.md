# Code Review – 2025-10-24

## Findings
- **Major** – `src/analysis/wrinkles.py:49` only applies a single 9×9 Gabor kernel across orientations. The 2D plan expects multi-scale Gabor/Hessian filtering so both fine lines and deeper folds are captured; without additional scales the detector can miss wrinkle classes the spec calls out.
- **Major** – `src/analysis/wrinkles.py:59` performs a single morphological open before skeletonizing. The reference approach requires a dilation/erosion cleanup plus geometric filtering (length, aspect ratio) to suppress pores, makeup lines, and other false positives; those constraints are absent.
- **Major** – `src/analysis/wrinkles.py:149` and `src/analysis/measurements.py:221` surface only average length ratios and mean intensity per region. The document specifies counting wrinkle segments and sampling cross-line intensity drops to support severity grading, so key metrics are missing.
- **Minor** – `src/analysis/rules.py:47` together with `configs/rules.yaml:1` still rely on static thresholds for “severity.” The spec envisions a weighted wrinkle severity index calibrated with surgeon feedback; even a placeholder composite score is not present.

## Recommendations
1. Expand `_enhance_texture` to fuse responses from multiple kernel sizes (and optionally Hessian-based ridges) before thresholding.
2. Introduce morphological dilation/thinning and component-level filters (length, aspect ratio, region masks) prior to skeletonization.
3. Track wrinkle counts and cross-line contrast metrics, expose them via `WrinkleMetric`, and feed a composite score into the CLI/findings pipeline.
