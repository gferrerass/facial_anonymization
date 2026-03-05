"""
Shared utilities for facial anonymization scripts.
Contains common functions used by main.py, generate.py, and evaluate.py.
"""

import argparse
import contextlib
import io
import logging
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import cv2


# Configure UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass  # Ignore if reconfigure is not available


# ============================================================================
# WARNING SUPPRESSION
# ============================================================================

# Suppress all warnings globally
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# ============================================================================
# VENV MANAGEMENT
# ============================================================================

def ensure_running_in_venv() -> None:
    """Relaunch the script with the project's virtualenv Python if needed."""
    if os.environ.get("FACIAL_ANON_RELAUNCHED") == "1":
        return

    in_venv = hasattr(sys, "real_prefix") or (getattr(sys, "base_prefix", sys.prefix) != sys.prefix)
    if in_venv:
        return

    # Attempting to relaunch with the project's venv
    project_root = Path(__file__).resolve().parent
    venv_dir = project_root / "venv"
    if platform.system() == "Windows":
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:
        venv_python = venv_dir / "bin" / "python"

    if not venv_python.exists():
        print(f"✗ Virtual environment not found at: {venv_python}")
        sys.exit(1)

    env = os.environ.copy()
    env["FACIAL_ANON_RELAUNCHED"] = "1"
    # Use sys.argv[0] to get the actual script being run, not this util file
    cmd = [str(venv_python)] + sys.argv
    result = subprocess.run(cmd, cwd=project_root, env=env, check=False)
    raise SystemExit(result.returncode)


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def suppress_verbose_logging() -> None:
    """Suppress transformers and torch logging warnings for cleaner output."""
    logging.basicConfig(level=logging.CRITICAL, force=True)
    
    for name in logging.root.manager.loggerDict:
        if 'transformers' in name or 'torch' in name or 'controlnet_aux' in name:
            logging.getLogger(name).setLevel(logging.CRITICAL)
            logging.getLogger(name).propagate = False
    
    os.environ['COMFYUI_CONTROLNET_AUX_VERBOSE'] = '0'


suppress_verbose_logging()


# ============================================================================
# IMAGE UTILITIES
# ============================================================================

def load_image_cv2(path: Path, label: str) -> cv2.typing.MatLike:
    """Load image using OpenCV."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load {label} image: {path}")
    return image


# ============================================================================
# COMFYUI UTILITIES
# ============================================================================

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping."""
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """Recursively find a folder/file in parent directories."""
    if path is None:
        path = os.getcwd()
    
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        # print(f"{name} found: {path_name}")  # Silenced
        return path_name
    
    parent_directory = os.path.dirname(path)
    if parent_directory == path:
        return None
    
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """Add ComfyUI directory to Python path."""
    for existing_path in sys.path:
        if os.path.basename(existing_path) == "ComfyUI" and os.path.isdir(existing_path):
            return
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        # print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """Load extra model paths configuration."""
    try:
        from comfy.cli_args import load_extra_path_config
    except ImportError:
        try:
            from utils.extra_config import load_extra_path_config
        except ImportError:
            return

    extra_model_paths = find_path("extra_model_paths.yaml")
    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)


def configure_local_paths(output_dir_override=None) -> None:
    """Configure ComfyUI to use local models and output directories."""
    # Initialize ComfyUI paths if not already done
    initialize_comfyui_paths()
    
    import folder_paths
    
    facial_anonymization_dir = Path(__file__).parent
    models_dir = facial_anonymization_dir / "models"
    
    if output_dir_override:
        output_dir = Path(output_dir_override)
        if not output_dir.is_absolute():
            output_dir = facial_anonymization_dir / output_dir
    else:
        output_dir = facial_anonymization_dir / "output"
    
    models_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    for subdir in ["text_encoders", "unet", "vae"]:
        (models_dir / subdir).mkdir(exist_ok=True)
        folder_paths.add_model_folder_path(subdir, str(models_dir / subdir))

    model_patches_dir = models_dir / "controlnet"
    model_patches_dir.mkdir(parents=True, exist_ok=True)
    folder_paths.add_model_folder_path("model_patches", str(model_patches_dir))
    
    ultralytics_bbox_dir = models_dir / "ultralytics" / "bbox"
    ultralytics_segm_dir = models_dir / "ultralytics" / "segm"
    ultralytics_bbox_dir.mkdir(parents=True, exist_ok=True)
    ultralytics_segm_dir.mkdir(parents=True, exist_ok=True)
    
    folder_paths.add_model_folder_path("ultralytics_bbox", str(ultralytics_bbox_dir))
    folder_paths.add_model_folder_path("ultralytics_segm", str(ultralytics_segm_dir))
    
    folder_paths.set_output_directory(str(output_dir))


# Track if ComfyUI paths have been initialized
_comfyui_initialized = False


def initialize_comfyui_paths() -> None:
    """Initialize ComfyUI paths (alias for add_comfyui_directory_to_sys_path + add_extra_model_paths)."""
    global _comfyui_initialized
    if _comfyui_initialized:
        return
    add_comfyui_directory_to_sys_path()
    add_extra_model_paths()
    _comfyui_initialized = True


def import_custom_nodes() -> None:
    """Initialize ComfyUI custom nodes."""
    # Initialize ComfyUI paths if not already done
    initialize_comfyui_paths()
    
    import asyncio
    
    try:
        import execution
        from nodes import init_extra_nodes, NODE_CLASS_MAPPINGS
        import server
    except ImportError:
        print("Note: ComfyUI frontend not found, but not required for batch processing")
        return

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)
    
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    
    try:
        loop.run_until_complete(init_extra_nodes())
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
    
    if "UltralyticsDetectorProvider" not in NODE_CLASS_MAPPINGS:
        print("Impact-Subpack not auto-loaded, loading manually...")
        try:
            impact_subpack_path = Path(__file__).parent / "ComfyUI" / "custom_nodes" / "ComfyUI-Impact-Subpack"
            sys.path.insert(0, str(impact_subpack_path))
            from modules import subpack_nodes
            NODE_CLASS_MAPPINGS.update(subpack_nodes.NODE_CLASS_MAPPINGS)
            print(f"Manually loaded Impact-Subpack nodes")
        except Exception as e:
            print(f"Failed to manually load Impact-Subpack: {e}")
        finally:
            if str(impact_subpack_path) in sys.path:
                sys.path.remove(str(impact_subpack_path))


# ============================================================================
# FILE AND ARGUMENT UTILITIES
# ============================================================================

def get_input_images(input_dir_override=None, max_images=None):
    """Collect valid image files from input folder."""
    images = []
    
    if input_dir_override:
        input_folder = Path(input_dir_override)
        if not input_folder.is_absolute():
            input_folder = Path(__file__).parent / input_dir_override
    else:
        input_folder = Path(__file__).parent / "input"
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
    if not input_folder.exists():
        print(f"Input folder does not exist: {input_folder}")
        return images
    
    for image_file in input_folder.iterdir():
        if image_file.is_file() and image_file.suffix.lower() in valid_extensions:
            images.append(str(image_file))
    
    images = sorted(images)
    
    if max_images and max_images > 0:
        images = images[:max_images]
    
    return images


def build_argument_parser(description: str, epilog: str = "") -> "argparse.ArgumentParser":
    """Build a standard argument parser for facial anonymization scripts."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default="input",
        help="Input directory path (absolute or relative). Default: input"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory path (absolute or relative). Default: output"
    )
    
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process. Default: all images"
    )
    
    parser.add_argument(
        "--strength",
        type=float,
        default=0.7,
        help="ControlNet strength (0.0-1.0). Higher values = stronger edge guidance. Default: 0.7"
    )
    
    parser.add_argument(
        "--denoise",
        type=float,
        default=0.6,
        help="Denoising strength (0.0-1.0). Higher values = more changes. Default: 0.6"
    )
    
    parser.add_argument(
        "--insightface-threshold",
        type=float,
        default=0.65,
        help="InsightFace similarity threshold (0.0-1.0). Higher values = more anonymization required. Default: 0.65"
    )
    
    parser.add_argument(
        "--clip-threshold",
        type=float,
        default=0.75,
        help="CLIP similarity threshold (0.0-1.0). Higher values = must preserve more context. Default: 0.75"
    )
    
    parser.add_argument(
        "--lpips-threshold",
        type=float,
        default=0.3,
        help="LPIPS perceptual similarity threshold (0.0-1.0). Lower values = more similarity required. Default: 0.3"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum number of generation iterations per image (1-10). Default: 3"
    )
    
    return parser


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

def log_evaluation_result(image_name: str, insightface_score: float, clip_score: float, lpips_score: float) -> None:
    """
    Log evaluation results to logs.txt file.
    Creates file with header on first write, appends subsequent entries.
    
    Args:
        image_name: Name of the generated image file
        insightface_score: InsightFace similarity score
        clip_score: CLIP similarity score  
        lpips_score: LPIPS perceptual distance score
    """
    log_file = Path(__file__).parent / "logs.txt"
    
    # Check if file exists to determine if we need to write header
    file_exists = log_file.exists()
    
    with open(log_file, "a", encoding="utf-8") as f:
        # Write header if file is new
        if not file_exists:
            f.write("Name Insightface_Score CLIP_Score LPIPS_Score\n")
        
        # Write evaluation result
        f.write(f"{image_name} {insightface_score:.6f} {clip_score:.6f} {lpips_score:.6f}\n")
