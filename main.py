#!/usr/bin/env python3
"""
Facial Anonymization with Evaluation
Combines generation and evaluation in a single workflow.
Loads all models once at startup.
"""

import argparse
import io
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import torch

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

# Import shared utilities
from shared_utils import (
    ensure_running_in_venv,
    suppress_verbose_logging,
    get_value_at_index,
    configure_local_paths,
    import_custom_nodes,
)

# Import image utilities
from image_utils import (
    load_image_cv2,
    detect_largest_face_bbox,
    scale_bbox,
    crop_by_bbox,
)

# Import evaluation utilities
from evaluation import (
    load_evaluation_models,
    calculate_clip_similarity,
    calculate_lpips_similarity,
    calculate_insightface_similarity,
)


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





# ============================================================================
# IMAGE GENERATION
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
# IMAGE EVALUATION
# ============================================================================


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

        insightface_similarity = None
        if eval_models.get("insightface") is not None:
            insightface_similarity = calculate_insightface_similarity(
                original_crop,
                generated_crop,
                eval_models["insightface"],
            )

        return {
            "clip_score": clip_score,
            "lpips_distance": lpips_distance,
            "lpips_similarity": lpips_similarity,
            "insightface_similarity": insightface_similarity,
            "success": True,
        }
    except Exception as e:
        print(f"✗ Evaluation error: {e}")
        return {
            "clip_score": None,
            "lpips_distance": None,
            "lpips_similarity": None,
            "insightface_similarity": None,
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
                    if eval_result['insightface_similarity'] is not None:
                        print(f"InsightFace Sim.: {eval_result['insightface_similarity']:.4f}")
                    else:
                        print("InsightFace Sim.: N/A")
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

