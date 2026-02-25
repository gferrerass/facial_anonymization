# Facial anonymization

Python application for facial anonymization through image generation, using ComfyUI as the processing engine with advanced generation models.

## Requirements

- **Python 3.10+** installed on your system
- **CUDA 11.8+** or **CUDA 12.x** (for GPU acceleration)
- **8GB+ VRAM** (recommended for smooth generation)
- **16GB+ RAM** (minimum)
- **50GB+ free disk space** (for models)
- **Git** installed

## Custom Nodes (Auto-installed)

The setup script automatically installs these ComfyUI custom nodes:

1. **ComfyUI-Florence2** - Image captioning and visual understanding
2. **ComfyUI-Impact-Pack** - Face detection and segmentation tools
3. **ComfyUI-Impact-Subpack** - Ultralytics YOLO integration
4. **ComfyUI-KJNodes** - Advanced mask operations
5. **ComfyUI-Inpaint-CropAndStitch** - Intelligent inpainting workflow

All dependencies for these nodes (including OpenCV, Ultralytics, etc.) are automatically installed during setup.

## Project Structure

```
facial_anonymization/
├── main.py               # Main execution script
├── setup.py              # Setup script
├── run.py                # Run script
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── ComfyUI/              # ComfyUI installation (auto-downloaded after setup)
├── models/
│   ├── text_encoders/    # CLIP/Text encoder models
│   ├── unet/             # Diffusion model
│   └── vae/              # VAE encoder-decoder models
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
- Installing all Python dependencies
- Setting up ComfyUI

### Step 2: Download Models

Download the three required models listed above and place them in their respective directories:
- `models/text_encoders/qwen_3_4b.safetensors`
- `models/unet/z_image_turbo_bf16.safetensors`
- `models/vae/ae.safetensors`

## Usage

### Running the Application

```bash
# Simple execution (automatically uses the virtual environment created by setup.py)
python run.py
```

**Note:** No need to manually activate the virtual environment. The `run.py` script automatically activates it and runs the application.

The Python application:
1. Initializes the ComfyUI engine
2. Loads all required models
3. Processes prompts and generates images
4. Saves outputs to the `output/` directory