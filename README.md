# Facial Anonymisation - ComfyUI Workflow

Automated script for running a generative workflow with ComfyUI using advanced image generation models.

## Requirements

- **Python 3.10+** installed on your system
- **CUDA 11.8+** or **CUDA 12.x** (for GPU acceleration)
- **8GB+ VRAM** (recommended for smooth generation)
- **16GB+ RAM** (minimum)
- **50GB+ free disk space** (for models)
- **Git** installed (optional, for ComfyUI download)

## Project Structure

```
facial_anonymisation/
├── main.py               # Main execution script
├── setup.py              # Setup script
├── run.py                # Run script
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── ComfyUI/              # ComfyUI installation (auto-downloaded)
├── models/
│   ├── text_encoders/    # CLIP/Text encoder models
│   ├── unet/             # Diffusion/UNET models
│   └── vae/              # VAE encoder-decoder models
├── input/                # Input directory (optional)
├── output/               # Generated output images
└── venv/                 # Virtual environment
```

## Required Models

Before running, download and place these models in their respective folders:

### 1. Qwen 3.4B (Text Encoder)
- **Size:** ~7GB
- **Location:** `models/text_encoders/`
- **Filename:** `qwen_3_4b.safetensors`
- **Purpose:** Text prompt processing

### 2. Z-Image Turbo (UNET)
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
cd facial_anonymisation

# Run setup (creates virtual environment and installs dependencies)
python setup.py
```

### Step 2: Download Models

Download the three required models listed above and place them in their respective directories:
- `models/text_encoders/qwen_3_4b.safetensors`
- `models/unet/z_image_turbo_bf16.safetensors`
- `models/vae/ae.safetensors`

### Step 3: Verify Installation

```bash
# Activate virtual environment (if needed)
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

## Usage

### Running the Workflow

```bash
# Simple execution
python run.py

# Or directly
python main.py
```

The script will:
1. Initialize ComfyUI
2. Load all required models
3. Process prompts and generate images
4. Save outputs to the `output/` directory

### Example Output

```
============================================================
   STARTING BATCH IMAGE GENERATION (3 images)
============================================================

▶ Loading models...
✓ Models loaded in 15.23 seconds

============================================================
   GENERATING IMAGE 1/3
============================================================
Prompt: Cinematic portrait of a futuristic cyberpunk woman...

✓ Image 1 generated in 8.45 seconds

============================================================
   BATCH GENERATION COMPLETED
============================================================
✓ Total images: 3
✓ Total time: 25.34 seconds (0.42 minutes)
✓ Average time per image: 8.45 seconds
============================================================
```

## Configuration

Edit `main.py` to customize:

- **Prompts:** Modify the `prompts` list to generate different images
- **Image Size:** Change `width` and `height` parameters (default: 1024x1024)
- **Sampling Steps:** Adjust `steps` parameter (default: 9)
- **Output Directory:** Change with `folder_paths.set_output_directory()`

## File Descriptions

| File | Description |
|------|-------------|
| `main.py` | Main script that executes the generation workflow |
| `setup.py` | Sets up virtual environment and installs dependencies |
| `run.py` | Convenient script to run main.py |
| `requirements.txt` | List of Python dependencies |
| `florence_generation.py` | Florence model utilities (optional) |

## Troubleshooting

### Model Not Found Error
```
FileNotFoundError: Model in folder 'xxx' with filename 'yyy' not found.
```
**Solution:** Ensure all three models are downloaded and placed in the correct directories.

### CUDA/GPU Issues
```
RuntimeError: CUDA out of memory
```
**Solution:** 
- Close other GPU-intensive applications
- Reduce batch size
- Ensure you have CUDA 11.8+ or higher installed
- Check GPU drivers are up to date

### Virtual Environment Issues
```bash
# Recreate virtual environment
python -m venv venv
python setup.py
```

## Performance Notes

- First run will be slower (models loading)
- GPU acceleration requires CUDA-compatible NVIDIA graphics card
- Typical generation time per image: 8-15 seconds
- Memory usage peaks during model loading (~12-14GB)

## System Requirements Checklist

- [ ] Python 3.10+
- [ ] CUDA 11.8+ or 12.x
- [ ] 8GB+ VRAM
- [ ] 16GB+ System RAM
- [ ] 50GB+ Free Storage
- [ ] All three model files downloaded
- [ ] Models placed in correct directories

## Support

For issues or questions, check:
- ComfyUI documentation: https://github.com/comfyanonymous/ComfyUI
- Model sources for download links
- GPU driver compatibility

## License

[Add your license here]

## Notes

- Models are large files; ensure stable internet connection during download
- First initialization downloads ComfyUI (~500MB)
- Keep models in their designated folders for proper operation
- Output images are saved automatically in the `output/` directory
