import os
import random
import sys
import time
import logging
import warnings
import io
import argparse
from typing import Sequence, Mapping, Any, Union
import torch
from pathlib import Path


# ============================================================================
# CONFIGURATION & INITIALIZATION
# ============================================================================

def suppress_verbose_logging() -> None:
    """Suppress transformers and torch logging warnings for cleaner output."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore")
    
    logging.basicConfig(level=logging.CRITICAL, force=True)
    
    for name in logging.root.manager.loggerDict:
        if 'transformers' in name or 'torch' in name or 'controlnet_aux' in name:
            logging.getLogger(name).setLevel(logging.CRITICAL)
            logging.getLogger(name).propagate = False
    
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass
    
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.handlers = [NullHandler()]
    transformers_logger.propagate = False
    
    # Suppress ComfyUI custom nodes verbose output
    os.environ['COMFYUI_CONTROLNET_AUX_VERBOSE'] = '0'


# Suppress logging on import
suppress_verbose_logging()


# ============================================================================
# UTILITY FUNCTIONS
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
        from main import load_extra_path_config
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


# ============================================================================
# COMFYUI SETUP
# ============================================================================

def configure_local_paths(output_dir_override=None) -> None:
    """Configure ComfyUI to use local models and output directories."""
    import folder_paths
    
    facial_anonymization_dir = Path(__file__).parent
    models_dir = facial_anonymization_dir / "models"
    
    # Use override if provided, otherwise default
    if output_dir_override:
        output_dir = Path(output_dir_override)
        if not output_dir.is_absolute():
            output_dir = facial_anonymization_dir / output_dir
    else:
        output_dir = facial_anonymization_dir / "output"
    
    models_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # Configure standard model directories
    for subdir in ["text_encoders", "unet", "vae"]:
        (models_dir / subdir).mkdir(exist_ok=True)
        folder_paths.add_model_folder_path(subdir, str(models_dir / subdir))

    # Configure model patches directory (ControlNet patches)
    model_patches_dir = models_dir / "controlnet"
    model_patches_dir.mkdir(parents=True, exist_ok=True)
    folder_paths.add_model_folder_path("model_patches", str(model_patches_dir))
    
    # Configure ultralytics directories (required by Impact-Subpack)
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
    
    # Suppress verbose output from custom nodes during initialization
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    
    try:
        loop.run_until_complete(init_extra_nodes())
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
    
    # Manually load ComfyUI-Impact-Subpack if not loaded
    if "UltralyticsDetectorProvider" not in NODE_CLASS_MAPPINGS:
        print("⚠ Impact-Subpack not auto-loaded, loading manually...")
        try:
            impact_subpack_path = Path(__file__).parent / "ComfyUI" / "custom_nodes" / "ComfyUI-Impact-Subpack"
            sys.path.insert(0, str(impact_subpack_path))
            from modules import subpack_nodes
            NODE_CLASS_MAPPINGS.update(subpack_nodes.NODE_CLASS_MAPPINGS)
            print(f"✓ Manually loaded Impact-Subpack nodes: {list(subpack_nodes.NODE_CLASS_MAPPINGS.keys())}")
        except Exception as e:
            print(f"❌ Failed to manually load Impact-Subpack: {e}")
        finally:
            if str(impact_subpack_path) in sys.path:
                sys.path.remove(str(impact_subpack_path))


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models():
    """Load all required models once at startup."""
    from nodes import NODE_CLASS_MAPPINGS
    
    print("\n▶ Loading models...")
    
    # Debug: Check if UltralyticsDetectorProvider is available
    if "UltralyticsDetectorProvider" not in NODE_CLASS_MAPPINGS:
        print("\n❌ ERROR: UltralyticsDetectorProvider not found in NODE_CLASS_MAPPINGS")
        print("\n📋 Available nodes containing 'Ultralytics' or 'Detector':")
        for key in sorted(NODE_CLASS_MAPPINGS.keys()):
            if 'ultralytics' in key.lower() or 'detector' in key.lower():
                print(f"  - {key}")
        print(f"\n📋 Total nodes loaded: {len(NODE_CLASS_MAPPINGS)}")
        print("\n💡 Checking if ComfyUI-Impact-Subpack is installed...")
        impact_subpack_path = Path(__file__).parent / "ComfyUI" / "custom_nodes" / "ComfyUI-Impact-Subpack"
        if impact_subpack_path.exists():
            print(f"✓ ComfyUI-Impact-Subpack found at: {impact_subpack_path}")
            print("\n💡 This might be due to missing dependencies. Try running:")
            print("   pip install -r ComfyUI/custom_nodes/ComfyUI-Impact-Subpack/requirements.txt")
        else:
            print(f"❌ ComfyUI-Impact-Subpack NOT found at: {impact_subpack_path}")
        raise KeyError("UltralyticsDetectorProvider node not available")
    
    models_load_start = time.time()
    
    # Load YOLO face detector
    ultralyticsdetectorprovider = NODE_CLASS_MAPPINGS["UltralyticsDetectorProvider"]()
    face_detector = ultralyticsdetectorprovider.doit(
        model_name="bbox/face_yolov8m.pt"
    )
    
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
    print(f"✓ Models loaded in {models_load_time:.2f} seconds\n")
    
    return {
        "face_detector": face_detector,
        "clip": cliploader_39,
        "florence2_model": florence2_model,
        "vae": vae,
        "unet": unet,
        "model_patch": model_patch,
        "load_time": models_load_time,
    }


# ============================================================================
# GENERATION WORKFLOW
# ============================================================================

def process_and_generate_image(idx, total, image_path, models, controlnet_strength=0.7, denoise_strength=0.6):
    """Complete workflow: load image → detect faces → inpaint faces with blurred mask."""
    from nodes import NODE_CLASS_MAPPINGS
    
    print(f"\n{'='*60}")
    print(f"   GENERATING IMAGE {idx}/{total}")
    print(f"{'='*60}")
    print(f"Input image: {image_path}")
    
    # Extract original filename (without extension) for output naming
    original_filename = Path(image_path).stem
    
    gen_start_time = time.time()
    
    # --- STEP 1: Load input image ---
    loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
    loaded_image = loadimage.load_image(image=image_path)
    
    # --- STEP 2: Detect faces with YOLO ---
    bboxdetectorcombined_v2 = NODE_CLASS_MAPPINGS["BboxDetectorCombined_v2"]()
    face_detection = bboxdetectorcombined_v2.doit(
        threshold=0.4,
        dilation=0,
        bbox_detector=get_value_at_index(models["face_detector"], 0),
        image=get_value_at_index(loaded_image, 0),
    )
    print("✓ Face detection completed")
    
    # --- STEP 3: Create blurred mask ---
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
    
    # --- STEP 4: Prepare inpainting crop ---
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
    
    # --- STEP 5: Generate caption from image using Florence2 ---
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
            florence2_model=get_value_at_index(models["florence2_model"], 0),
        )
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
    
    caption = get_value_at_index(caption_result, 2)
    print(f"✓ Generated caption: {caption}")
    
    # --- STEP 6: Encode text conditioning ---
    cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
    positive_conditioning = cliptextencode.encode(
        text=caption,
        clip=get_value_at_index(models["clip"], 0),
    )
    negative_conditioning = cliptextencode.encode(
        text="",
        clip=get_value_at_index(models["clip"], 0),
    )
    
    # --- STEP 7: Setup inpainting conditioning ---
    inpaintmodelconditioning = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]()
    inpaint_conditioning = inpaintmodelconditioning.encode(
        noise_mask=True,
        positive=get_value_at_index(positive_conditioning, 0),
        negative=get_value_at_index(negative_conditioning, 0),
        vae=get_value_at_index(models["vae"], 0),
        pixels=get_value_at_index(inpaint_crop, 1),
        mask=get_value_at_index(inpaint_crop, 2),
    )
    print("✓ Inpainting conditioning prepared")
    
    # --- STEP 8: Apply Canny edge detection (ControlNet preprocessing) ---
    aio_preprocessor = NODE_CLASS_MAPPINGS["AIO_Preprocessor"]()
    canny_edges = aio_preprocessor.execute(
        preprocessor="CannyEdgePreprocessor",
        resolution=512,
        image=get_value_at_index(inpaint_crop, 1),
    )
    print("✓ Canny edge detection applied")
    
    # --- STEP 9: Apply model patches ---
    modelsamplingauraflow = NODE_CLASS_MAPPINGS["ModelSamplingAuraFlow"]()
    model_patched = modelsamplingauraflow.patch_aura(
        shift=6,
        model=get_value_at_index(models["unet"], 0)
    )
    
    differentialdiffusion = NODE_CLASS_MAPPINGS["DifferentialDiffusion"]()
    model_differential = differentialdiffusion.EXECUTE_NORMALIZED(
        strength=1,
        model=get_value_at_index(model_patched, 0)
    )
    print("✓ Model patches applied")
    
    # --- STEP 10: Apply QwenImageDiffsynth ControlNet ---
    qwenimagediffsynthcontrolnet = NODE_CLASS_MAPPINGS["QwenImageDiffsynthControlnet"]()
    model_with_controlnet = qwenimagediffsynthcontrolnet.diffsynth_controlnet(
        strength=controlnet_strength,
        model=get_value_at_index(model_differential, 0),
        model_patch=get_value_at_index(models["model_patch"], 0),
        vae=get_value_at_index(models["vae"], 0),
        image=get_value_at_index(canny_edges, 0),
    )
    print(f"✓ ControlNet applied")
    
    # --- STEP 11: Sample (inpaint) ---
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
    
    # --- STEP 12: Decode ---
    vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
    decoded = vaedecode.decode(
        samples=get_value_at_index(samples, 0),
        vae=get_value_at_index(models["vae"], 0),
    )
    
    # --- STEP 13: Stitch result back ---
    inpaintstitchimproved = NODE_CLASS_MAPPINGS["InpaintStitchImproved"]()
    final_image = inpaintstitchimproved.inpaint_stitch(
        stitcher=get_value_at_index(inpaint_crop, 0),
        inpainted_image=get_value_at_index(decoded, 0),
    )
    print("✓ Image stitched")
    
    # --- STEP 14: Save result ---
    saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()
    saveimage.save_images(
        filename_prefix=f"{original_filename}_anonymized",
        images=get_value_at_index(final_image, 0)
    )
    
    gen_time = time.time() - gen_start_time
    print(f"✓ Image {idx} processed in {gen_time:.2f} seconds")


# ============================================================================
# MAIN
# ============================================================================

def get_input_images(input_dir_override=None, max_images=None):
    """Collect valid image files from input folder."""
    images = []
    
    # Use override if provided, otherwise default
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
    
    # Limit number of images if specified
    if max_images and max_images > 0:
        images = images[:max_images]
    
    return images


def main():
    """Main execution function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Facial Anonymization using ComfyUI")
    parser.add_argument("--strength", type=float, default=0.7, help="ControlNet strength (default: 0.7)")
    parser.add_argument("--denoise", type=float, default=0.6, help="Denoise strength (default: 0.6)")
    parser.add_argument("--input", type=str, default="input", help="Input directory (default: input)")
    parser.add_argument("--output", type=str, default="output", help="Output directory (default: output)")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum number of images to process")
    args = parser.parse_args()
    
    # Setup
    configure_local_paths(output_dir_override=args.output)
    import_custom_nodes()
    
    # Get input images
    images = get_input_images(input_dir_override=args.input, max_images=args.max_images)
    if not images:
        print(f"No images found in input folder: {args.input}")
        return
    
    # Header
    print("\n" + "="*60)
    print(f"   BATCH FACE ANONYMIZATION ({len(images)} images)")
    print("="*60)
    total_start_time = time.time()
    
    with torch.inference_mode():
        # Load all models once
        models = load_models()
        
        # Process each image
        for idx, image_path in enumerate(images, start=1):
            process_and_generate_image(
                idx, len(images), image_path, models,
                controlnet_strength=args.strength,
                denoise_strength=args.denoise
            )
    
    # Summary
    total_time = time.time() - total_start_time
    avg_time = (total_time - models["load_time"]) / len(images) if images else 0
    
    print("\n" + "="*60)
    print("   BATCH ANONYMIZATION COMPLETED")
    print("="*60)
    print(f"✓ Total images: {len(images)}")
    print(f"✓ Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"✓ Models load time: {models['load_time']:.2f}s")
    print(f"✓ Average time per image: {avg_time:.2f}s")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
