#!/usr/bin/env python3
"""
Facial Anonymization with Evaluation
Combines generation and evaluation in a single workflow.
Loads all models once at startup.
"""

import time
from pathlib import Path
import torch

# Import shared utilities
from shared_utils import (
    ensure_running_in_venv,
    configure_local_paths,
    import_custom_nodes,
    load_image_cv2,
    get_input_images,
    build_argument_parser,
    log_evaluation_result,
)

# Import generation utilities
from generation import (
    load_comfyui_models,
    process_and_generate_image,
)

# Import evaluation utilities
from evaluation import (
    load_evaluation_models,
    evaluate,
    print_metrics,
)

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function."""
    # Parse arguments first (so --help works without loading anything)
    parser = build_argument_parser(
        description="Facial Anonymization with Evaluation",
        epilog="""
Examples:
  python main.py
  python main.py --input custom_input --output custom_output
  python main.py --strength 0.8 --denoise 0.7 --max-images 5 --steps 7
  python main.py --insightface-threshold 0.7 --clip-threshold 0.8 --max-iterations 5
        """
    )
    args = parser.parse_args()
    
    # Check venv
    ensure_running_in_venv()
    
    print("\n" + "="*60)
    print("   FACIAL ANONYMIZATION WITH EVALUATION")
    print("="*60)
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    if args.max_images:
        print(f"Max images: {args.max_images}")
    print(f"Initial ControlNet strength: {args.strength}")
    print(f"Initial Denoise strength: {args.denoise}")
    print(f"KSampler steps: {args.steps}")
    print(f"Max iterations per image: {args.max_iterations}")
    print(f"InsightFace threshold: {args.insightface_threshold}")
    print(f"CLIP threshold: {args.clip_threshold}")
    print(f"LPIPS threshold: {args.lpips_threshold}")
    print("="*60)
    
    # Setup
    configure_local_paths(output_dir_override=args.output)
    import_custom_nodes()
    
    # Get input images
    images = get_input_images(input_dir_override=args.input, max_images=args.max_images)
    if not images:
        print(f"No images found in input folder: {args.input}")
        return
    
    total_start_time = time.time()
    results = []
    
    # Track metrics for statistics
    generation_times = []  # Time for each generation call
    image_total_times = []  # Total time per image (all iterations)
    iterations_per_image = []  # Number of iterations per image
    
    with torch.inference_mode():
        # Load all models once at the beginning
        print("   LOADING ALL MODELS")
        print("="*60)
        
        comfyui_models = load_comfyui_models()
        eval_models = load_evaluation_models()
        
        # Process each image with iterative optimization
        for idx, image_path in enumerate(images, start=1):
            print("="*60)
            print(f"   PROCESSING IMAGE {idx}/{len(images)}")
            image_start_time = time.time()
            
            try:
                # Initialize parameters for this image
                current_strength = args.strength
                current_denoise = args.denoise
                iteration = 0
                
                # Load original image once
                original_image = load_image_cv2(Path(image_path), "original")
                
                # Iterative optimization loop
                while iteration < args.max_iterations:
                    iteration += 1
                    print("="*60)
                    print(f"   ITERATION {iteration}/{args.max_iterations}")
                    print(f"   Strength: {current_strength:.3f}")
                    print(f"   Denoise: {current_denoise:.3f}")
                    
                    # Generate anonymized image and track time
                    generation_start_time = time.time()
                    anonymized_path = process_and_generate_image(
                        idx, len(images), image_path, comfyui_models,
                        controlnet_strength=current_strength,
                        denoise_strength=current_denoise,
                        steps=args.steps
                    )
                    generation_time = time.time() - generation_start_time
                    generation_times.append(generation_time)
                    
                    # Load anonymized image
                    anonymized_image = load_image_cv2(anonymized_path, "anonymized")
                    
                    # Evaluate
                    insightface_score, clip_score, lpips_score = evaluate(
                        original_image, anonymized_image, eval_models
                    )
                    
                    # Print metrics
                    print_metrics(insightface_score, clip_score, lpips_score)
                    
                    # Store result
                    current_result = {
                        "image": Path(image_path).name,
                        "generated": anonymized_path.name,
                        "iteration": iteration,
                        "strength": current_strength,
                        "denoise": current_denoise,
                        "insightface_score": insightface_score,
                        "clip_score": clip_score,
                        "lpips_score": lpips_score,
                    }
                    
                    # Log evaluation results immediately after evaluation
                    log_evaluation_result(
                        current_result["generated"],
                        current_result["insightface_score"],
                        current_result["clip_score"],
                        current_result["lpips_score"]
                    )
                    
                    # Check if metrics are satisfactory
                    if insightface_score < args.insightface_threshold:
                        print(f"  InsightFace score ({insightface_score:.3f}) is lower than threshold ({args.insightface_threshold})")
                        print(f"  Adjusting: strength*0.9, denoise*1.1")
                        current_strength *= 0.9
                        current_denoise *= 1.1
                        current_denoise = min(current_denoise, 1.0)  # Cap at 1.0
                    elif clip_score < args.clip_threshold:
                        print(f"  CLIP score ({clip_score:.3f}) is lower than threshold ({args.clip_threshold})")
                        print(f"  Adjusting: strength*1.075, denoise*0.95")
                        current_strength *= 1.075
                        current_denoise *= 0.95
                        current_strength = min(current_strength, 1.0)  # Cap at 1.0
                    elif lpips_score > args.lpips_threshold:
                        print(f"  LPIPS score ({lpips_score:.3f}) is higher than threshold ({args.lpips_threshold})")
                        print(f"  Adjusting: strength*1.075, denoise*0.95")
                        current_strength *= 1.075
                        current_denoise *= 0.95
                        current_strength = min(current_strength, 1.0)  # Cap at 1.0
                    else:
                        print(f"  All metrics within acceptable range!")
                        break
                    
                if iteration == args.max_iterations:
                    print(f"  Reached maximum iterations without meeting all thresholds.")
                
                # Track iterations and total time for this image
                iterations_per_image.append(iteration)
                image_total_time = time.time() - image_start_time
                image_total_times.append(image_total_time)
                       
            except Exception as e:
                print(f"\nError processing image {idx}: {e}")
                results.append({
                    "image": Path(image_path).name,
                    "success": False,
                    "error": str(e)
                })
    
    # Calculate statistics
    avg_generation_time = sum(generation_times) / len(generation_times) if generation_times else 0
    avg_image_total_time = sum(image_total_times) / len(image_total_times) if image_total_times else 0
    avg_iterations = sum(iterations_per_image) / len(iterations_per_image) if iterations_per_image else 0
    
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
    print(f"Total images: {len(images)}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"Models load time: {total_load_time:.2f}s")
    print(f"  - ComfyUI models: {comfyui_load_time:.2f}s")
    print(f"  - Evaluation models: {eval_load_time:.2f}s")
    print(f"Processing time: {processing_time:.2f}s")
    print(f"Average time per generation: {avg_generation_time:.2f}s")
    print(f"Average time per final image: {avg_image_total_time:.2f}s")
    print(f"Average iterations per image: {avg_iterations:.2f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

