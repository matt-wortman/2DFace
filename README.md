# 2DFace - Offline AI Solutions for Cosmetic Facial Analysis

A GPU-accelerated, privacy-first facial analysis pipeline designed for plastic surgeons to perform detailed cosmetic assessments using 2D images. Built for offline deployment on NVIDIA RTX 4090 workstations.

## Overview

2DFace provides an end-to-end solution for analyzing facial features, proportions, and cosmetic considerations from 2D photographs. The system combines state-of-the-art face detection (RetinaFace), precise landmark extraction (68-point FAN), and clinically-relevant measurements to assist surgical planning. Wrinkle analysis is an important module, but the primary mandate is to surface any facial attributes that could inform cosmetic surgical correction (symmetry, proportions, contour, volume cues, texture) so clinicians get a holistic view rather than a wrinkle-only readout.

**Key Features:**
- 🔒 **100% Offline** - No cloud dependencies, all processing runs locally
- ⚡ **GPU-Accelerated** - Optimized for NVIDIA RTX 4090 (CUDA 12.x)
- 📏 **Comprehensive Analysis** - Symmetry, proportions, angles, ratios, and wrinkle detection
- 🎯 **Quality Gates** - Automatic image quality assessment (blur, exposure, head pose)
- 📊 **Clinical Reports** - HTML reports with measurements, findings, and visualizations
- 🔧 **Configurable** - YAML-based thresholds and rules for customization

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [Development](#development)
- [Citation](#citation)
- [License](#license)

## Installation

### Prerequisites

- **Hardware:** NVIDIA GPU (tested on RTX 4090)
- **OS:** Linux (tested on WSL2 Ubuntu), macOS, or Windows
- **CUDA:** 12.x or later
- **Python:** 3.10 or later
- **conda:** Miniconda or Anaconda

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/matt-wortman/2DFace.git
cd 2DFace
```

2. **Create conda environment:**
```bash
conda create -n 2dface python=3.10
conda activate 2dface
```

3. **Install dependencies:**
```bash
pip install -r requirements-local.txt
```

4. **Verify GPU setup:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Model Downloads

On first run, the system will automatically download required models (~300MB):
- RetinaFace (buffalo_l variant from InsightFace)
- FAN landmark detector (from face-alignment)

Models are cached in `./models/` directory.

## Quick Start

### Basic Analysis

Analyze a single image:

```bash
python -m src.cli.analyze data/sanity/sample_face.jpg
```

This will:
1. Detect faces in the image
2. Extract 68 facial landmarks
3. Perform quality checks (blur, exposure, pose)
4. Calculate measurements (symmetry, proportions, ratios)
5. Apply rule-based analysis for cosmetic findings
6. Generate an HTML report in `outputs/`

### View Results

The analysis produces:
- **JSON metrics:** `outputs/[image_name]_analysis.json`
- **Annotated image:** `outputs/[image_name]_annotated.png`
- **HTML report:** `outputs/[image_name]_report.html`

Open the HTML report in a browser to view the comprehensive analysis.

### Batch Processing

Process multiple images in a directory:

```bash
python -m src.cli.analyze data/sanity/*.jpg --output-dir outputs/batch_results
```

### Individual Pipeline Steps

Run components separately for debugging or custom workflows:

**1. Face Detection:**
```bash
python -m src.cli.detect_faces data/sanity/sample_face.jpg
```

**2. Landmark Extraction:**
```bash
python -m src.cli.extract_landmarks data/sanity/sample_face.jpg
```

## Project Structure

```
2DFace/
├── src/
│   ├── pipeline/          # Core processing pipeline
│   │   ├── detect.py      # Face detection (RetinaFace)
│   │   ├── landmarks.py   # Landmark extraction (FAN 68-point)
│   │   └── process.py     # End-to-end orchestration
│   │
│   ├── analysis/          # Analysis modules
│   │   ├── quality.py     # Image quality assessment
│   │   ├── measurements.py # Facial metrics (symmetry, proportions, ratios)
│   │   ├── rules.py       # Rule-based finding engine
│   │   ├── wrinkles.py    # Wrinkle detection (Gabor-based)
│   │   └── visualize.py   # Visualization utilities
│   │
│   └── cli/               # Command-line interfaces
│       ├── detect_faces.py
│       ├── extract_landmarks.py
│       └── analyze.py
│
├── configs/               # Configuration files
│   ├── pipeline.yaml      # Pipeline settings
│   ├── quality.yaml       # Quality gate thresholds
│   └── rules.yaml         # Rule engine thresholds
│
├── data/                  # Test datasets
│   ├── sanity/           # Validation images
│   └── nano/             # Additional test data
│
├── models/                # Cached ML models
├── outputs/               # Analysis results
├── templates/             # HTML report templates
├── tests/                 # Unit tests
└── docs/                  # Documentation
    ├── setup/            # Environment setup guides
    ├── api/              # API reference
    └── guides/           # Usage guides
```

## Usage

### Command-Line Interface

#### Full Analysis Pipeline

```bash
python -m src.cli.analyze <input_image> [OPTIONS]

Options:
  --output-dir PATH       Output directory (default: outputs/)
  --config PATH           Pipeline config (default: configs/pipeline.yaml)
  --quality-config PATH   Quality config (default: configs/quality.yaml)
  --rules-config PATH     Rules config (default: configs/rules.yaml)
  --visualize            Generate annotated images (default: True)
  --report               Generate HTML report (default: True)
```

#### Face Detection Only

```bash
python -m src.cli.detect_faces <input_image> [OPTIONS]

Options:
  --output-dir PATH      Output directory
  --det-size INT         Detection size (default: 640)
  --visualize           Draw bounding boxes on image
```

#### Landmark Extraction Only

```bash
python -m src.cli.extract_landmarks <input_image> [OPTIONS]

Options:
  --output-dir PATH      Output directory
  --visualize           Draw landmarks on image
  --measurements        Calculate facial measurements
```

### Python API

```python
from src.pipeline.process import FacialAnalysisPipeline

# Initialize pipeline
pipeline = FacialAnalysisPipeline(
    config_path='configs/pipeline.yaml',
    quality_config_path='configs/quality.yaml',
    rules_config_path='configs/rules.yaml'
)

# Analyze an image
results = pipeline.process_image('path/to/image.jpg')

# Access results
print(f"Quality score: {results['quality']['overall_score']}")
print(f"Findings: {results['findings']}")
print(f"Measurements: {results['measurements']}")
```

## Configuration

The system uses YAML configuration files for customization:

### Pipeline Configuration (`configs/pipeline.yaml`)

```yaml
detection:
  model_name: "buffalo_l"
  det_size: 640

landmarks:
  device: "cuda"
  flip_input: false

quality:
  config: "configs/quality.yaml"

rules:
  config: "configs/rules.yaml"
```

### Quality Thresholds (`configs/quality.yaml`)

```yaml
blur_threshold: 100.0      # Laplacian variance threshold
exposure_threshold: 0.3    # Acceptable under/over exposure ratio
max_roll_degrees: 15       # Maximum head roll angle
```

### Rule Engine (`configs/rules.yaml`)

```yaml
symmetry:
  deviation_threshold: 5.0  # mm threshold for asymmetry detection

proportions:
  nose_width_ratio:
    min: 0.25
    max: 0.35

angles:
  nasolabial_angle:
    min: 90
    max: 110
```

See [Configuration Guide](docs/guides/configuration.md) for detailed options.

## Documentation

- **[Installation Guide](docs/setup/local_env.md)** - Detailed environment setup
- **[CLI Usage Guide](docs/guides/cli_usage.md)** - Command-line examples
- **[API Reference](docs/api/README.md)** - Module and function documentation
- **[Architecture](docs/architecture.md)** - System design and data flow
- **[Development Guide](docs/development.md)** - Contributing and testing
- **[Progress Tracker](docs/PROGRESS.md)** - Implementation status

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_measurements.py

# Run with coverage
pytest --cov=src tests/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for:
- Code style guidelines
- Development workflow
- Pull request process
- Testing requirements

## Technology Stack

- **Deep Learning:** PyTorch 2.5.1, TorchVision 0.20.1
- **Face Detection:** InsightFace 0.7.3 (RetinaFace)
- **Landmark Detection:** face-alignment 1.4.1 (FAN/HRNet)
- **Computer Vision:** OpenCV 4.12.0, scikit-image 0.25.2
- **Scientific Computing:** NumPy 2.1.2, SciPy 1.15.3
- **CLI Framework:** Click 8.3.0, Rich 14.2.0
- **Acceleration:** ONNX Runtime GPU 1.23.2, Numba 0.62.1

## Citation

If you use 2DFace in your research or clinical practice, please cite:

```bibtex
@software{2dface2025,
  title = {2DFace: Offline AI Solutions for Cosmetic Facial Analysis},
  author = {Wortman, Matt},
  year = {2025},
  url = {https://github.com/matt-wortman/2DFace},
  note = {GPU-accelerated facial analysis pipeline for cosmetic surgery planning}
}
```

## License

This project includes components under different licenses:

- **InsightFace (RetinaFace):** MIT License
- **face-alignment:** BSD-3-Clause License
- **Project Code:** [Specify your license here]

See [LICENSE](LICENSE) for full details.

## Acknowledgments

- **InsightFace** - Face detection models
- **face-alignment** - Landmark detection library
- **PyTorch** - Deep learning framework

## Disclaimer

⚠️ **Medical Device Notice:** This software is intended for research and educational purposes. It is NOT approved as a medical device. Clinical decisions should be made by qualified medical professionals using approved diagnostic tools.

## Contact & Support

- **Issues:** [GitHub Issues](https://github.com/matt-wortman/2DFace/issues)
- **Documentation:** [Full Documentation](docs/)
- **Email:** [Contact information]

---

**Current Status:** ✅ Stage 6 Complete (End-to-end orchestration with HTML reporting)

**Next Steps:**
- Stage 7: Repeatability benchmarks and failure handling
- Stage 8: Documentation polish and deployment guide
- Future: Multi-view aggregation and learning-based enhancements
