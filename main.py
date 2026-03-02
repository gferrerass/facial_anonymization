#!/usr/bin/env python3
"""
Facial Anonymization with Evaluation
Combines generation and evaluation in a single workflow.
Loads all models once at startup.
"""

import argparse
import io
import logging
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import cv2
import lpips
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

try:
    from ultralytics import YOLO
except Exception as exc:
    YOLO = None
    ULTRALYTICS_IMPORT_ERROR = exc
else:
    ULTRALYTICS_IMPORT_ERROR = None


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
    cmd = [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]]
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


# Initialize ComfyUI paths
_initialized = False
if not _initialized:
    add_comfyui_directory_to_sys_path()
    add_extra_model_paths()
    _initialized = True


def configure_local_paths(output_dir_override=None) -> None:
    """Configure ComfyUI to use local models and output directories."""
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


def import_custom_nodes() -> None:
    """Initialize ComfyUI custom nodes."""
    import asyncio
    import execution
    from nodes import init_extra_nodes, NODE_CLASS_MAPPINGS
    import server

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
        print("⚠ Impact-Subpack not auto-loaded, loading manually...")
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


# ============================================================================
# MODEL LOADING (Combined from both scripts)
# ============================================================================

def load_comfyui_models():
    """Load all ComfyUI models for generation (from generate.py)."""
    from nodes import NODE_CLASS_MAPPINGS
    
    print("\n▶ Loading ComfyUI generation models...")
    
    if "UltralyticsDetectorProvider" not in NODE_CLASS_MAPPINGS:
        print("\nERROR: UltralyticsDetectorProvider not found in NODE_CLASS_MAPPINGS")
        raise KeyError("UltralyticsDetectorProvider node not available")
    
    models_load_start = time.time()
    
    # Load YOLO face detector
    ultralyticsdetectorprovider = NODE_CLASS_MAPPINGS["UltralyticsDetectorProvider"]()
    face_detector = ultralyticsdetectorprovider.doit(model_name="bbox/face_yolov8m.pt")
    
    # Load CLIP model
    cliploader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
    cliploader_39 = cliploader.load_clip(
        clip_name="qwen_3_4b.safetensors", 
        type="lumina2", 
        device="default"
    )
    
    # Load Florence2 model for captioning
    florence2_loader = NODE_CLASS_MAPPINGS["DownloadAndLoadFlorence2Model"]()
    florence2_model = florence2_loader.loadmodel(
        model="MiaoshouAI/Florence-2-base-PromptGen-v1.5",
        precision="fp16",
        attention="sdpa",
        convert_to_safetensors=False,
    )
    
    # Load VAE decoder
    vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
    vae = vaeloader.load_vae(vae_name="ae.safetensors")
    
    # Load UNET model
    unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
    unet = unetloader.load_unet(
        unet_name="z_image_turbo_bf16.safetensors", 
        weight_dtype="default"
    )

    # Load model patch (ControlNet)
    modelpatchloader = NODE_CLASS_MAPPINGS["ModelPatchLoader"]()
    model_patch = modelpatchloader.load_model_patch(
        name="Z-Image-Turbo-Fun-Controlnet-Union.safetensors"
    )
    
    models_load_time = time.time() - models_load_start
    print(f"✓ ComfyUI models loaded in {models_load_time:.2f} seconds")
    
    return {
        "face_detector": face_detector,
        "clip": cliploader_39,
        "florence2_model": florence2_model,
        "vae": vae,
        "unet": unet,
        "model_patch": model_patch,
        "load_time": models_load_time,
    }


def load_evaluation_models():
    """Load all models for evaluation (from evaluate.py)."""
    print("▶ Loading evaluation models...")
    
    if YOLO is None:
        raise RuntimeError(
            f"Ultralytics is not available. Import error: {ULTRALYTICS_IMPORT_ERROR}"
        )
    
    eval_start = time.time()
    
    # Load YOLO face detector
    model_path = Path(__file__).parent / "models" / "ultralytics" / "bbox" / "face_yolov8m.pt"
    if model_path.exists():
        print(f"Loading YOLO model: {model_path}")
        yolo_model = YOLO(str(model_path))
    else:
        print(f"⚠ YOLO model not found at {model_path}, using default")
        yolo_model = YOLO("yolov8n-face.pt")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading CLIP model (device: {device})...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai', device=device
    )
    
    print(f"Loading LPIPS model (device: {device})...")
    lpips_model = lpips.LPIPS(net='alex').to(device)
    
    eval_load_time = time.time() - eval_start
    print(f"✓ Evaluation models loaded in {eval_load_time:.2f} seconds")
    
    return {
        "yolo": yolo_model,
        "lpips": lpips_model,
        "clip_model": clip_model,
        "clip_preprocess": clip_preprocess,
        "device": device,
        "load_time": eval_load_time,
    }


# ============================================================================
# IMAGE GENERATION (from generate.py)
# ============================================================================

def process_and_generate_image(idx, total, image_path, comfyui_models, controlnet_strength=0.7, denoise_strength=0.6):
    """Generate anonymized image using ComfyUI workflow."""
    from nodes import NODE_CLASS_MAPPINGS
    
    print(f"\n{'='*60}")
    print(f"   GENERATING IMAGE {idx}/{total}")
    print(f"{'='*60}")
    print(f"Input image: {image_path}")
    
    original_filename = Path(image_path).stem
    gen_start_time = time.time()
    
    # Load input image
    loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
    loaded_image = loadimage.load_image(image=image_path)
    
    # Detect faces
    bboxdetectorcombined_v2 = NODE_CLASS_MAPPINGS["BboxDetectorCombined_v2"]()
    face_detection = bboxdetectorcombined_v2.doit(
        threshold=0.4,
        dilation=0,
        bbox_detector=get_value_at_index(comfyui_models["face_detector"], 0),
        image=get_value_at_index(loaded_image, 0),
    )
    print("✓ Face detection completed")
    
    # Create blurred mask
    growmaskwithblur = NODE_CLASS_MAPPINGS["GrowMaskWithBlur"]()
    blurred_mask = growmaskwithblur.expand_mask(
        expand=0,
        incremental_expandrate=0,
        tapered_corners=True,
        flip_input=False,
        blur_radius=16,
        lerp_alpha=1,
        decay_factor=1,
        fill_holes=False,
        mask=get_value_at_index(face_detection, 0),
    )
    print("✓ Blurred mask created")
    
    # Prepare inpainting crop
    inpaintcropimproved = NODE_CLASS_MAPPINGS["InpaintCropImproved"]()
    inpaint_crop = inpaintcropimproved.inpaint_crop(
        downscale_algorithm="bilinear",
        upscale_algorithm="bicubic",
        preresize=True,
        preresize_mode="ensure minimum resolution",
        preresize_min_width=1032,
        preresize_min_height=1032,
        preresize_max_width=16384,
        preresize_max_height=16384,
        mask_fill_holes=True,
        mask_expand_pixels=0,
        mask_invert=False,
        mask_blend_pixels=32,
        mask_hipass_filter=0.1,
        extend_for_outpainting=False,
        extend_up_factor=1,
        extend_down_factor=1,
        extend_left_factor=1,
        extend_right_factor=1,
        context_from_mask_extend_factor=1.2,
        output_resize_to_target_size=True,
        output_target_width=1024,
        output_target_height=1024,
        output_padding="32",
        device_mode="gpu (much faster)",
        image=get_value_at_index(loaded_image, 0),
        mask=get_value_at_index(blurred_mask, 0),
    )
    print("✓ Inpainting crop prepared")
    
    # Generate caption
    florence2run = NODE_CLASS_MAPPINGS["Florence2Run"]()
    
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    
    try:
        caption_result = florence2run.encode(
            text_input="",
            task="more_detailed_caption",
            fill_mask=True,
            keep_model_loaded=False,
            max_new_tokens=1024,
            num_beams=3,
            do_sample=True,
            output_mask_select="",
            seed=random.randint(1, 2**64),
            image=get_value_at_index(loaded_image, 0),
            florence2_model=get_value_at_index(comfyui_models["florence2_model"], 0),
        )
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
    
    caption = get_value_at_index(caption_result, 2)
    print(f"✓ Generated caption: {caption}")
    
    # Encode text conditioning
    cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
    positive_conditioning = cliptextencode.encode(
        text=caption,
        clip=get_value_at_index(comfyui_models["clip"], 0),
    )
    negative_conditioning = cliptextencode.encode(
        text="",
        clip=get_value_at_index(comfyui_models["clip"], 0),
    )
    
    # Setup inpainting conditioning
    inpaintmodelconditioning = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]()
    inpaint_conditioning = inpaintmodelconditioning.encode(
        noise_mask=True,
        positive=get_value_at_index(positive_conditioning, 0),
        negative=get_value_at_index(negative_conditioning, 0),
        vae=get_value_at_index(comfyui_models["vae"], 0),
        pixels=get_value_at_index(inpaint_crop, 1),
        mask=get_value_at_index(inpaint_crop, 2),
    )
    print("✓ Inpainting conditioning prepared")
    
    # Apply Canny edge detection
    aio_preprocessor = NODE_CLASS_MAPPINGS["AIO_Preprocessor"]()
    canny_edges = aio_preprocessor.execute(
        preprocessor="CannyEdgePreprocessor",
        resolution=512,
        image=get_value_at_index(inpaint_crop, 1),
    )
    print("✓ Canny edge detection applied")
    
    # Apply model patches
    modelsamplingauraflow = NODE_CLASS_MAPPINGS["ModelSamplingAuraFlow"]()
    model_patched = modelsamplingauraflow.patch_aura(
        shift=6,
        model=get_value_at_index(comfyui_models["unet"], 0)
    )
    
    differentialdiffusion = NODE_CLASS_MAPPINGS["DifferentialDiffusion"]()
    model_differential = differentialdiffusion.EXECUTE_NORMALIZED(
        strength=1,
        model=get_value_at_index(model_patched, 0)
    )
    print("✓ Model patches applied")
    
    # Apply ControlNet
    qwenimagediffsynthcontrolnet = NODE_CLASS_MAPPINGS["QwenImageDiffsynthControlnet"]()
    model_with_controlnet = qwenimagediffsynthcontrolnet.diffsynth_controlnet(
        strength=controlnet_strength,
        model=get_value_at_index(model_differential, 0),
        model_patch=get_value_at_index(comfyui_models["model_patch"], 0),
        vae=get_value_at_index(comfyui_models["vae"], 0),
        image=get_value_at_index(canny_edges, 0),
    )
    print(f"✓ ControlNet applied")
    
    # Sample
    ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
    samples = ksampler.sample(
        seed=random.randint(1, 2**64),
        steps=9,
        cfg=1,
        sampler_name="euler",
        scheduler="normal",
        denoise=denoise_strength,
        model=get_value_at_index(model_with_controlnet, 0),
        positive=get_value_at_index(inpaint_conditioning, 0),
        negative=get_value_at_index(inpaint_conditioning, 1),
        latent_image=get_value_at_index(inpaint_conditioning, 2),
    )
    print(f"✓ Inpainting completed")
    
    # Decode
    vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
    decoded = vaedecode.decode(
        samples=get_value_at_index(samples, 0),
        vae=get_value_at_index(comfyui_models["vae"], 0),
    )
    
    # Stitch result back
    inpaintstitchimproved = NODE_CLASS_MAPPINGS["InpaintStitchImproved"]()
    final_image = inpaintstitchimproved.inpaint_stitch(
        stitcher=get_value_at_index(inpaint_crop, 0),
        inpainted_image=get_value_at_index(decoded, 0),
    )
    print("✓ Image stitched")
    
    # Save result
    saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()
    saveimage.save_images(
        filename_prefix=f"{original_filename}_anonymized",
        images=get_value_at_index(final_image, 0)
    )
    
    gen_time = time.time() - gen_start_time
    print(f"✓ Image {idx} generated in {gen_time:.2f} seconds")
    
    # Find and return path to generated image
    import folder_paths
    output_dir = Path(folder_paths.get_output_directory())
    
    # Find the most recent file that matches the pattern
    pattern = f"{original_filename}_anonymized"
    matching_files = sorted(
        output_dir.glob(f"{pattern}*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    if not matching_files:
        raise FileNotFoundError(f"Generated image not found in {output_dir} matching pattern '{pattern}*.png'")
    
    return matching_files[0]


# ============================================================================
# IMAGE EVALUATION (from evaluate.py)
# ============================================================================

def load_image_cv2(path: Path, label: str) -> cv2.typing.MatLike:
    """Load image using OpenCV."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load {label} image: {path}")
    return image


def detect_largest_face_bbox(yolo_model: Any, image_bgr: cv2.typing.MatLike) -> tuple[int, int, int, int]:
    """Detect largest face using YOLO model."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = yolo_model.predict(source=image_rgb, verbose=False)

    if not results:
        raise RuntimeError("No detection results returned by YOLO.")

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        raise RuntimeError("No face detected in image.")

    xyxy = boxes.xyxy.cpu().numpy()
    max_area = -1.0
    best_box = None
    h, w = image_bgr.shape[:2]

    for row in xyxy:
        x1, y1, x2, y2 = row.tolist()
        x1_i = max(0, min(w - 1, int(round(x1))))
        y1_i = max(0, min(h - 1, int(round(y1))))
        x2_i = max(1, min(w, int(round(x2))))
        y2_i = max(1, min(h, int(round(y2))))
        width = max(0, x2_i - x1_i)
        height = max(0, y2_i - y1_i)
        area = float(width * height)
        if area > max_area:
            max_area = area
            best_box = (x1_i, y1_i, x2_i, y2_i)

    if best_box is None or max_area <= 0:
        raise RuntimeError("Could not determine a valid face bounding box.")

    return best_box


def scale_bbox(
    bbox: tuple[int, int, int, int],
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    src_h, src_w = src_size
    dst_h, dst_w = dst_size

    x1, y1, x2, y2 = bbox
    scale_x = dst_w / src_w
    scale_y = dst_h / src_h

    sx1 = int(round(x1 * scale_x))
    sy1 = int(round(y1 * scale_y))
    sx2 = int(round(x2 * scale_x))
    sy2 = int(round(y2 * scale_y))

    sx1 = max(0, min(dst_w - 1, sx1))
    sy1 = max(0, min(dst_h - 1, sy1))
    sx2 = max(sx1 + 1, min(dst_w, sx2))
    sy2 = max(sy1 + 1, min(dst_h, sy2))
    return sx1, sy1, sx2, sy2


def crop_by_bbox(image_bgr: cv2.typing.MatLike, bbox: tuple[int, int, int, int]) -> cv2.typing.MatLike:
    x1, y1, x2, y2 = bbox
    return image_bgr[y1:y2, x1:x2]


def calculate_clip_similarity(image1_bgr: cv2.typing.MatLike, image2_bgr: cv2.typing.MatLike, 
                            clip_model: Any, clip_preprocess: Any, device: str) -> float:
    """Calculate CLIP cosine similarity between two BGR images."""
    image1_rgb = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2_bgr, cv2.COLOR_BGR2RGB)
    
    image1_pil = Image.fromarray(image1_rgb)
    image2_pil = Image.fromarray(image2_rgb)
    
    image1_tensor = clip_preprocess(image1_pil).unsqueeze(0).to(device)
    image2_tensor = clip_preprocess(image2_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image1_features = clip_model.encode_image(image1_tensor)
        image2_features = clip_model.encode_image(image2_tensor)
    
    similarity = torch.cosine_similarity(image1_features, image2_features)
    return similarity.item()


def calculate_lpips_similarity(image1_bgr: cv2.typing.MatLike, image2_bgr: cv2.typing.MatLike, 
                              lpips_model: Any, device: str) -> float:
    """Calculate LPIPS distance between two BGR images."""
    image1_rgb = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2_bgr, cv2.COLOR_BGR2RGB)
    
    image1_pil = Image.fromarray(image1_rgb)
    image2_pil = Image.fromarray(image2_rgb)
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image1_tensor = transform(image1_pil).unsqueeze(0).to(device)
    image2_tensor = transform(image2_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        lpips_distance = lpips_model(image1_tensor, image2_tensor)
    
    return lpips_distance.item()


def evaluate_images(original_path: Path, generated_path: Path, eval_models: dict) -> dict:
    """Evaluate anonymized image quality."""
    try:
        original_bgr = load_image_cv2(original_path, "original")
        generated_bgr = load_image_cv2(generated_path, "generated")

        # Detect largest face
        bbox_original = detect_largest_face_bbox(eval_models["yolo"], original_bgr)
        bbox_generated = scale_bbox(
            bbox_original,
            src_size=original_bgr.shape[:2],
            dst_size=generated_bgr.shape[:2],
        )

        original_crop = crop_by_bbox(original_bgr, bbox_original)
        generated_crop = crop_by_bbox(generated_bgr, bbox_generated)

        # Calculate similarities
        clip_score = calculate_clip_similarity(
            original_crop, generated_crop, 
            eval_models["clip_model"], eval_models["clip_preprocess"], eval_models["device"]
        )
        
        lpips_distance = calculate_lpips_similarity(
            original_crop, generated_crop, eval_models["lpips"], eval_models["device"]
        )
        lpips_similarity = 1.0 - lpips_distance

        return {
            "clip_score": clip_score,
            "lpips_distance": lpips_distance,
            "lpips_similarity": lpips_similarity,
            "success": True,
        }
    except Exception as e:
        print(f"✗ Evaluation error: {e}")
        return {
            "clip_score": None,
            "lpips_distance": None,
            "lpips_similarity": None,
            "success": False,
            "error": str(e),
        }


# ============================================================================
# UTILITIES
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
        print(f"⚠ Input folder does not exist: {input_folder}")
        return images
    
    for image_file in input_folder.iterdir():
        if image_file.is_file() and image_file.suffix.lower() in valid_extensions:
            images.append(str(image_file))
    
    images = sorted(images)
    
    if max_images and max_images > 0:
        images = images[:max_images]
    
    return images


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Facial Anonymization with Evaluation - Batch processing with quality metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --input custom_input --output custom_output
  python main.py --strength 0.8 --denoise 0.7 --max-images 5
  python main.py --input ./photos --output ./results --max-images 10
        """
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
    
    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function."""
    # Check venv
    ensure_running_in_venv()
    
    # Parse arguments
    args = parse_arguments()
    
    print("\n" + "="*60)
    print("   FACIAL ANONYMIZATION WITH EVALUATION")
    print("="*60)
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    if args.max_images:
        print(f"Max images: {args.max_images}")
    print(f"ControlNet strength: {args.strength}")
    print(f"Denoise strength: {args.denoise}")
    print("="*60)
    
    # Setup
    configure_local_paths(output_dir_override=args.output)
    import_custom_nodes()
    
    # Get input images
    images = get_input_images(input_dir_override=args.input, max_images=args.max_images)
    if not images:
        print(f"No images found in input folder: {args.input}")
        return
    
    print(f"\n✓ Found {len(images)} image(s) to process\n")
    
    total_start_time = time.time()
    results = []
    
    with torch.inference_mode():
        # Load all models once at the beginning
        print("\n" + "="*60)
        print("   LOADING ALL MODELS")
        print("="*60)
        
        comfyui_models = load_comfyui_models()
        eval_models = load_evaluation_models()
        
        print("="*60)
        
        # Process each image
        for idx, image_path in enumerate(images, start=1):
            try:
                # Generate anonymized image
                generated_path = process_and_generate_image(
                    idx, len(images), image_path, comfyui_models,
                    controlnet_strength=args.strength,
                    denoise_strength=args.denoise
                )
                
                # Evaluate the generated image
                print(f"\n▶ Evaluating image {idx}/{len(images)}...")
                eval_result = evaluate_images(Path(image_path), generated_path, eval_models)
                
                if eval_result["success"]:
                    print(f"\n{'='*60}")
                    print(f"   EVALUATION RESULTS - IMAGE {idx}")
                    print(f"{'='*60}")
                    print(f"CLIP Similarity:  {eval_result['clip_score']:.4f}")
                    print(f"LPIPS Distance:   {eval_result['lpips_distance']:.4f}")
                    print(f"LPIPS Similarity: {eval_result['lpips_similarity']:.4f}")
                    print(f"{'='*60}")
                    
                    results.append({
                        "image": Path(image_path).name,
                        "generated": generated_path.name,
                        **eval_result
                    })
                else:
                    print(f"✗ Evaluation failed: {eval_result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"\n✗ Error processing image {idx}: {e}")
                results.append({
                    "image": Path(image_path).name,
                    "success": False,
                    "error": str(e)
                })
    
    # Summary
    total_time = time.time() - total_start_time
    comfyui_load_time = comfyui_models["load_time"]
    eval_load_time = eval_models["load_time"]
    total_load_time = comfyui_load_time + eval_load_time
    processing_time = total_time - total_load_time
    avg_time = processing_time / len(images) if images else 0
    
    print("\n" + "="*60)
    print("   BATCH PROCESSING COMPLETED")
    print("="*60)
    print(f"✓ Total images: {len(images)}")
    print(f"✓ Successful: {sum(1 for r in results if r.get('success', False))}")
    print(f"✓ Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"✓ Models load time: {total_load_time:.2f}s")
    print(f"  - ComfyUI models: {comfyui_load_time:.2f}s")
    print(f"  - Evaluation models: {eval_load_time:.2f}s")
    print(f"✓ Processing time: {processing_time:.2f}s")
    print(f"✓ Average time per image: {avg_time:.2f}s")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

