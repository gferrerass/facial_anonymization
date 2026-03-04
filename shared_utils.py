"""
Shared utilities for facial anonymization scripts.
Contains common functions used by main.py, generate.py, and evaluate.py.
"""

import io
import logging
import os
import platform
import subprocess
import sys
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
# VENV MANAGEMENT
# ============================================================================

def ensure_running_in_venv() -> None:
    """Relaunch the script with the project's virtualenv Python if needed."""
    if os.environ.get("FACIAL_ANON_RELAUNCHED") == "1":
        return

    in_venv = hasattr(sys, "real_prefix") or (getattr(sys, "base_prefix", sys.prefix) != sys.prefix)
    if in_venv:
        return

    print("Attempting to relaunch with the project's venv...")

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
        print(f"{name} found: {path_name}")
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
        print(f"'{comfyui_path}' added to sys.path")


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
    
    print(f"✓ Models directory: {models_dir}")


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
            print(f"✓ Manually loaded Impact-Subpack nodes")
        except Exception as e:
            print(f"Failed to manually load Impact-Subpack: {e}")
        finally:
            if str(impact_subpack_path) in sys.path:
                sys.path.remove(str(impact_subpack_path))
