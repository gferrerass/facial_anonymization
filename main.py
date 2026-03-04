#!/usr/bin/env python3
"""
Facial Anonymization with Evaluation
Combines generation and evaluation in a single workflow.
Loads all models once at startup.
"""

import argparse
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

# Import shared utilities (UTF-8 config already applied)
from shared_utils import (
    ensure_running_in_venv,
    configure_local_paths,
    import_custom_nodes,
    load_image_cv2,
    get_input_images,
    build_argument_parser,
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
        description="Facial Anonymization with Evaluation - Batch processing with quality metrics",
        epilog="""
Examples:
  python main.py
  python main.py --input custom_input --output custom_output
  python main.py --strength 0.8 --denoise 0.7 --max-images 5
  python main.py --input ./photos --output ./results --max-images 10
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
    
    print(f"\nFound {len(images)} image(s) to process\n")
    
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
                anonymized_path = process_and_generate_image(
                    idx, len(images), image_path, comfyui_models,
                    controlnet_strength=args.strength,
                    denoise_strength=args.denoise
                )
                
                # Load images
                original_image = load_image_cv2(Path(image_path), "original")
                anonymized_image = load_image_cv2(anonymized_path, "anonymized")
                
                # Evaluate
                print(f"\nEvaluating image {idx}/{len(images)}...")
                insightface_score, clip_score, lpips_score = evaluate(
                    original_image, anonymized_image, eval_models
                )
                
                # Print metrics
                print_metrics(insightface_score, clip_score, lpips_score)
                
                results.append({
                    "image": Path(image_path).name,
                    "generated": anonymized_path.name,
                    "insightface_score": insightface_score,
                    "clip_score": clip_score,
                    "lpips_score": lpips_score,
                    "success": True,
                })
                    
            except Exception as e:
                print(f"\nError processing image {idx}: {e}")
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
    print(f"Total images: {len(images)}")
    print(f"Successful: {sum(1 for r in results if r.get('success', False))}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"Models load time: {total_load_time:.2f}s")
    print(f"  - ComfyUI models: {comfyui_load_time:.2f}s")
    print(f"  - Evaluation models: {eval_load_time:.2f}s")
    print(f"Processing time: {processing_time:.2f}s")
    print(f"Average time per image: {avg_time:.2f}s")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

