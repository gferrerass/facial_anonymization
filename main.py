import os
import random
import sys
import time
import logging
import warnings
import io
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
        if 'transformers' in name or 'torch' in name:
            logging.getLogger(name).setLevel(logging.CRITICAL)
            logging.getLogger(name).propagate = False
    
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass
    
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.handlers = [NullHandler()]
    transformers_logger.propagate = False


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

def configure_local_paths() -> None:
    """Configure ComfyUI to use local models and output directories."""
    import folder_paths
    
    facial_anonymisation_dir = Path(__file__).parent
    models_dir = facial_anonymisation_dir / "models"
    output_dir = facial_anonymisation_dir / "output"
    
    models_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    for subdir in ["text_encoders", "unet", "vae"]:
        (models_dir / subdir).mkdir(exist_ok=True)
        folder_paths.add_model_folder_path(subdir, str(models_dir / subdir))
    
    folder_paths.set_output_directory(str(output_dir))
    
    print(f"✓ Models directory: {models_dir}")
    print(f"✓ Output directory: {output_dir}")


def import_custom_nodes() -> None:
    """Initialize ComfyUI custom nodes."""
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)
    loop.run_until_complete(init_extra_nodes())


from nodes import NODE_CLASS_MAPPINGS


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models():
    """Load all required models once at startup."""
    print("\n▶ Loading models...")
    models_load_start = time.time()
    
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
    
    models_load_time = time.time() - models_load_start
    print(f"✓ Models loaded in {models_load_time:.2f} seconds\n")
    
    return {
        "clip": cliploader_39,
        "florence2_model": florence2_model,
        "vae": vae,
        "unet": unet,
        "load_time": models_load_time,
    }


# ============================================================================
# GENERATION WORKFLOW
# ============================================================================

def process_and_generate_image(idx, total, image_path, models):
    """Complete workflow: load image → generate caption → generate new image."""
    print(f"\n{'='*60}")
    print(f"   GENERATING IMAGE {idx}/{total}")
    print(f"{'='*60}")
    print(f"Input image: {image_path}")
    
    gen_start_time = time.time()
    
    # --- STEP 1: Load input image ---
    loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
    loaded_image = loadimage.load_image(image=image_path)
    
    # --- STEP 2: Generate caption from image using Florence2 ---
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
    print(f"Generated caption: {caption}")
    
    # --- STEP 3: Generate new image from caption ---
    cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
    emptylatentimage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
    modelsamplingauraflow = NODE_CLASS_MAPPINGS["ModelSamplingAuraFlow"]()
    conditioningzeroout = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
    ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
    vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
    saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()
    
    # Encode caption for image generation
    positive_conditioning = cliptextencode.encode(
        text=caption,
        clip=get_value_at_index(models["clip"], 0),
    )
    
    # Create latent representation
    latent_image = emptylatentimage.generate(width=1024, height=1024, batch_size=1)
    model_patched = modelsamplingauraflow.patch_aura(
        shift=1, model=get_value_at_index(models["unet"], 0)
    )
    negative_conditioning = conditioningzeroout.zero_out(
        conditioning=get_value_at_index(positive_conditioning, 0)
    )
    
    # Sample and decode
    samples = ksampler.sample(
        seed=random.randint(1, 2**64),
        steps=9,
        cfg=1,
        sampler_name="res_multistep",
        scheduler="simple",
        denoise=1,
        model=get_value_at_index(model_patched, 0),
        positive=get_value_at_index(positive_conditioning, 0),
        negative=get_value_at_index(negative_conditioning, 0),
        latent_image=get_value_at_index(latent_image, 0),
    )
    
    decoded = vaedecode.decode(
        samples=get_value_at_index(samples, 0),
        vae=get_value_at_index(models["vae"], 0),
    )
    
    # Save generated image
    saveimage.save_images(
        filename_prefix=f"z-image_{idx}",
        images=get_value_at_index(decoded, 0)
    )
    
    gen_time = time.time() - gen_start_time
    print(f"✓ Image {idx} generated in {gen_time:.2f} seconds")


# ============================================================================
# MAIN
# ============================================================================

def get_input_images():
    """Collect valid image files from input folder."""
    images = []
    input_folder = Path(__file__).parent / "input"
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
    if not input_folder.exists():
        return images
    
    for image_file in input_folder.iterdir():
        if image_file.is_file() and image_file.suffix.lower() in valid_extensions:
            images.append(str(image_file))
    
    return sorted(images)


def main():
    """Main execution function."""
    # Setup
    configure_local_paths()
    import_custom_nodes()
    
    # Get input images
    images = get_input_images()
    if not images:
        print("No images found in input folder!")
        return
    
    # Header
    print("\n" + "="*60)
    print(f"   BATCH CAPTION-TO-IMAGE GENERATION ({len(images)} images)")
    print("="*60)
    total_start_time = time.time()
    
    with torch.inference_mode():
        # Load all models once
        models = load_models()
        
        # Process each image
        for idx, image_path in enumerate(images, start=1):
            process_and_generate_image(idx, len(images), image_path, models)
    
    # Summary
    total_time = time.time() - total_start_time
    avg_time = (total_time - models["load_time"]) / len(images) if images else 0
    
    print("\n" + "="*60)
    print("   BATCH GENERATION COMPLETED")
    print("="*60)
    print(f"✓ Total images: {len(images)}")
    print(f"✓ Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"✓ Models load time: {models['load_time']:.2f}s")
    print(f"✓ Average time per image: {avg_time:.2f}s")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
