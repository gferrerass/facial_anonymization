# Facial anonymization

Python application for facial anonymization through image generation, using ComfyUI as the processing engine with advanced generation models.

## Requirements

- **Python 3.10+** installed on your system
- **CUDA 11.8+** or **CUDA 12.x** (for GPU acceleration)
- **8GB+ VRAM** (recommended for smooth generation)
- **16GB+ RAM** (minimum)
- **50GB+ free disk space** (for models)
- **Git** installed

### Windows Notes

The setup script automatically handles all dependencies on Windows:
- Installs required packages in the project virtual environment
- No manual compilation or build tools needed for core functionality

## Custom Nodes (Auto-installed)

The setup script automatically installs these ComfyUI custom nodes:

1. **ComfyUI-Florence2** - Image captioning and visual understanding
2. **ComfyUI-Impact-Pack** - Face detection and segmentation tools
3. **ComfyUI-Impact-Subpack** - Ultralytics YOLO integration
4. **ComfyUI-KJNodes** - Advanced mask operations
5. **ComfyUI-Inpaint-CropAndStitch** - Intelligent inpainting workflow
6. **comfyui_controlnet_aux** - ControlNet preprocessors (Canny edge detection, etc.)

All dependencies for these nodes (including OpenCV, Ultralytics, etc.) are automatically installed during setup.

### Evaluation Metrics

The project includes advanced evaluation metrics for anonymized face quality:
- **CLIP Similarity** - Semantic similarity using OpenAI CLIP
- **LPIPS Distance** - Perceptual similarity using Learned Perceptual Image Patch Similarity
- **InsightFace Similarity** - Face embedding cosine similarity (optional, requires `insightface` package)

## Project Structure

```
facial_anonymization/
├── main.py               # Generation + Evaluation workflow
├── evaluate.py           # Evaluation only (for existing images)
├── generate.py           # Generation only script
├── setup.py              # Setup script
├── run.py                # Run script
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── ComfyUI/              # ComfyUI installation (auto-downloaded after setup)
├── models/
│   ├── text_encoders/    # CLIP/Text encoder models
│   ├── unet/             # Diffusion model
│   ├── vae/              # VAE encoder-decoder models
│   ├── controlnet/       # ControlNet models (edge guidance)
│   └── ultralytics/
│       └── bbox/         # YOLO face detector
├── input/                # Input directory
├── output/               # Generated output images
└── venv/                 # Virtual environment (created after setup)
```

## Required Models

Before running, download and place these models in their respective folders:

### 1. Qwen 3.4B (Text Encoder)
- **Size:** ~7GB
- **Location:** `models/text_encoders/`
- **Filename:** `qwen_3_4b.safetensors`
- **Purpose:** Text prompt processing

### 2. Z-Image Turbo (Diffusion model)
- **Size:** ~2.5GB
- **Location:** `models/unet/`
- **Filename:** `z_image_turbo_bf16.safetensors`
- **Purpose:** Image generation

### 3. AE VAE (Autoencoder)
- **Size:** ~200MB
- **Location:** `models/vae/`
- **Filename:** `ae.safetensors`
- **Purpose:** Latent encoding/decoding

### 4. YOLOv8 Face Detector
- **Location:** `models/ultralytics/bbox/`
- **Filename:** `face_yolov8m.pt`
- **Purpose:** Face detection

### 5. Z-Image-Turbo ControlNet Union
- **Size:** ~350MB
- **Location:** `models/controlnet/`
- **Filename:** `Z-Image-Turbo-Fun-Controlnet-Union.safetensors`
- **Purpose:** Edge-based guidance for improved face inpainting (uses Canny edge detection)

## Installation

### Step 1: Clone and Setup the Environment

```bash
# Clone the repository
git clone <repository-url>
cd facial_anonymization

# Run setup (automatically creates virtual environment and installs all dependencies)
python setup.py
```

The `setup.py` script handles:
- Creating a virtual environment in `venv/`
- Installing all dependencies inside `venv` (no global Python pollution)
- Setting up ComfyUI
- Cloning and validating required custom nodes
- Installing dependencies from each custom node `requirements.txt`

### Step 2: Download Models

Download the models listed above and place them in their respective directories:
- `models/text_encoders/qwen_3_4b.safetensors`
- `models/unet/z_image_turbo_bf16.safetensors`
- `models/vae/ae.safetensors`
- `models/ultralytics/bbox/face_yolov8m.pt`
- `models/controlnet/Z-Image-Turbo-Fun-Controlnet-Union.safetensors`

## Usage

### Basic Usage

```bash
# Simple execution with default settings
python run.py
```

**Note:** No need to manually activate the virtual environment. The `run.py` script automatically activates it and runs the application.

### Advanced Usage with Parameters

You can customize the anonymization process using command-line parameters:

```bash
# General syntax
python run.py [--strength VALUE] [--denoise VALUE] [--input DIR] [--output DIR] [--max-images N]
```

#### Available Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--strength` | float | 0.7 | ControlNet strength (0.0-1.0). Higher values = stronger edge guidance for better structure preservation |
| `--denoise` | float | 0.6 | Denoising strength (0.0-1.0). Higher values = more changes to the face |
| `--input` | string | `input` | Input directory path (absolute or relative) |
| `--output` | string | `output` | Output directory path (absolute or relative) |
| `--max-images` | int | all | Maximum number of images to process |

#### Usage Examples

```bash
# Use default settings (strength=0.7, denoise=0.6)
python run.py

# Stronger facial anonymization
python run.py --strength 0.6 --denoise 0.75

# Complete custom configuration
python run.py --strength 0.6 --denoise 0.7 --input ./photos --output ./anonymized --max-images 10


# Show help and all available options
python run.py --help
```

## Evaluation

The project includes two scripts for evaluating anonymized images:

### Using `main.py` (Generation + Evaluation)

The `main.py` script combines generation and evaluation in a single workflow. It loads all models once at startup and processes images with evaluation metrics.

```bash
# Run generation with automatic evaluation
python main.py

# With custom parameters
python main.py --input photos --output results --strength 0.7 --denoise 0.6
```

**Output:** Generates anonymized images and displays metrics for each:
- CLIP Similarity
- LPIPS Distance
- InsightFace Similarity (if available)

### Using `evaluate.py` (Evaluation Only)

Use this script to evaluate already generated images without running the generation process again.

```bash
# Evaluate a single pair of images
python evaluate.py input/original.jpg output/original_anonymized_0001.png

# Save face crops and open them
python evaluate.py input/photo.jpg output/photo_anonymized_0001.png --show

# Specify custom output directory for crops
python evaluate.py input/photo.jpg output/photo_anonymized_0001.png --output-dir evaluation_results
```

**Output:** Displays similarity metrics and saves cropped faces to a timestamped directory.

### Installing InsightFace (Optional)

For face embedding similarity metrics, install InsightFace:

```bash
# Activate virtual environment first
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install InsightFace
pip install insightface onnxruntime-gpu
```

If InsightFace is not installed, the scripts will skip this metric and display "N/A".