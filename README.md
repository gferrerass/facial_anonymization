# Facial Anonymization

A ComfyUI-based facial anonymization pipeline for image generation and evaluation with automatic parameter tuning.

## Metrics

- InsightFace distance: embedding-space distance between detected faces (identity removal).
- CLIP similarity: semantic similarity between original and anonymized images (semantic preservation).
- LPIPS distance: perceptual distance between original and anonymized images (perceptual quality preservation).

## Project structure

```text
facial_anonymization/
|- main.py
|- generation.py
|- evaluation.py
|- shared_utils.py
|- setup.py
|- requirements.txt
|- README.md
|- input/
|- output/
|- models/
|- ComfyUI/
`- venv/
```

## Main scripts

- `setup.py`: creates `venv`, installs dependencies, and configures ComfyUI with required custom nodes.
- `main.py`: complete workflow (generation + iterative evaluation).
- `generation.py`: generation-only pipeline.
- `evaluation.py`: evaluation-only workflow for a single original/anonymized pair.
- `shared_utils.py`: shared helpers and CLI argument definitions.

## Requirements

- Python 3.10+
- Git
- NVIDIA GPU recommended for better performance
- Enough disk space for models

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd facial_anonymization
```

### 2. Run setup

```bash
python setup.py
```

`setup.py` automatically:

- Creates `venv/`.
- Installs Python dependencies in the virtual environment.
- Downloads and configures `ComfyUI/`.
- Installs required custom nodes and their dependencies.

Runtime scripts relaunch using the project's `venv` Python interpreter when needed.

## Required models

Download and place the following model files:

- `models/text_encoders/qwen_3_4b.safetensors`
- `models/unet/z_image_turbo_bf16.safetensors`
- `models/vae/ae.safetensors`
- `models/ultralytics/bbox/face_yolov8m.pt`
- `models/controlnet/Z-Image-Turbo-Fun-Controlnet-Union.safetensors`

## Usage

### Full workflow (recommended)

```bash
python main.py
```

This mode:

- Generates anonymized output.
- Evaluates metrics.
- Adjusts `strength` and `denoise` on each iteration until thresholds are met or `--max-iterations` is reached.

Examples:

```bash
python main.py --input input --output output
python main.py --strength 0.7 --denoise 0.6 --max-images 10
python main.py --insightface-threshold 0.65 --clip-threshold 0.75 --lpips-threshold 0.3
```

### Generation only

```bash
python generation.py
python generation.py --input input --output output --strength 0.7 --denoise 0.6 --max-images 5
```

### Evaluation only

```bash
python evaluation.py input/original.jpg output/original_anonymized_20260305_120000.png
python evaluation.py input/original.jpg output/original_anonymized_20260305_120000.png --no_crop
```

## CLI parameters

### Shared by `main.py` and `generation.py`

- `--input`: input directory path (default: `input`)
- `--output`: output directory path (default: `output`)
- `--max-images`: maximum number of images to process
- `--strength`: ControlNet strength (default: `0.7`)
- `--denoise`: sampler denoise strength (default: `0.6`)

### `main.py` only

- `--insightface-threshold` (default: `0.65`)
- `--clip-threshold` (default: `0.75`)
- `--lpips-threshold` (default: `0.3`)
- `--max-iterations` (default: `3`)

### `evaluation.py` only

- `original`: path to the original image
- `anonymized`: path to the anonymized image
- `--no_crop`: skips face detection and cropping (use pre-cropped face images)