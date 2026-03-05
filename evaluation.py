"""
Evaluation utilities for facial anonymization.
Contains model loading and similarity metric functions.
"""

import argparse
import time
from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Import utilities from shared_utils
from shared_utils import (
    ensure_running_in_venv,
    load_image_cv2,
    suppress_stdout_stderr,
)


def load_evaluation_models():
    """Load all models for evaluation."""
    import lpips
    import open_clip
    
    try:
        from ultralytics import YOLO
    except ImportError:
        YOLO = None
    
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        FaceAnalysis = None
    
    print("Loading evaluation models...")
    
    if YOLO is None:
        raise RuntimeError("Ultralytics is not available")
    
    eval_start = time.time()
    
    # Load YOLO face detector
    model_path = Path(__file__).parent / "models" / "ultralytics" / "bbox" / "face_yolov8m.pt"
    if model_path.exists():
        print(f"Loading YOLO model: {model_path}")
        yolo_model = YOLO(str(model_path))
    else:
        print(f"YOLO model not found at {model_path}, using default")
        yolo_model = YOLO("yolov8n-face.pt")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading CLIP model (device: {device})...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai', device=device
    )
    
    print(f"Loading LPIPS model (device: {device})...")
    with suppress_stdout_stderr():
        lpips_model = lpips.LPIPS(net='alex').to(device)

    insightface_model = None
    if FaceAnalysis is not None:
        insightface_root = Path(__file__).parent / "models" / "insightface"
        insightface_root.mkdir(parents=True, exist_ok=True)

        provider_candidates = ["CPUExecutionProvider"]
        if device == "cuda":
            provider_candidates = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        for provider_name in provider_candidates:
            try:
                print(f"Loading InsightFace model (provider: {provider_name})...")
                with suppress_stdout_stderr():
                    insightface_model = FaceAnalysis(
                        name="buffalo_l",
                        root=str(insightface_root),
                        providers=[provider_name],
                    )
                    ctx_id = 0 if provider_name == "CUDAExecutionProvider" else -1
                    insightface_model.prepare(ctx_id=ctx_id, det_size=(640, 640))
                break
            except Exception as e:
                print(f"Could not load InsightFace with {provider_name}: {e}")
                insightface_model = None
    else:
        print("InsightFace not available; skipping InsightFace similarity metric")
    
    eval_load_time = time.time() - eval_start
    print(f"Evaluation models loaded in {eval_load_time:.2f} seconds")
    
    return {
        "yolo": yolo_model,
        "lpips": lpips_model,
        "clip_model": clip_model,
        "clip_preprocess": clip_preprocess,
        "insightface": insightface_model,
        "device": device,
        "load_time": eval_load_time,
    }


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


def _extract_insightface_embedding(insightface_model: Any, image_bgr: cv2.typing.MatLike):
    """Extract largest-face normalized embedding using InsightFace."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    faces = insightface_model.get(image_rgb)
    if not faces:
        det_model = getattr(insightface_model, "det_model", None)
        if det_model is not None and hasattr(det_model, "input_size"):
            for size in range(640, 256, -64):
                det_model.input_size = (size, size)
                faces = insightface_model.get(image_rgb)
                if faces:
                    break

    if not faces:
        return None

    best_face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )
    return getattr(best_face, "normed_embedding", None)


def calculate_insightface_similarity(
    image1_bgr: cv2.typing.MatLike,
    image2_bgr: cv2.typing.MatLike,
    insightface_model: Any,
) -> float:
    """Calculate cosine distance between two face embeddings using InsightFace (matches faceanalysis.py)."""
    emb1 = _extract_insightface_embedding(insightface_model, image1_bgr)
    emb2 = _extract_insightface_embedding(insightface_model, image2_bgr)

    if emb1 is None or emb2 is None:
        raise RuntimeError("InsightFace could not detect a face in one or both crops")

    # Use the same formula as in Faceanalysis (ComfyUI custom node)
    # https://github.com/cubiq/ComfyUI_FaceAnalysis/blob/main/faceanalysis.py
    dist = np.float64(1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    return float(dist)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate facial anonymization quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evaluation.py input/person.jpg output/person_anonymized_0001.png\n"
            "  python evaluation.py C:/data/orig.png C:/data/anon.png\n"
            "  python evaluation.py input/person.jpg output/person_anonymized_0001.png --no_crop"
        ),
    )
    parser.add_argument("original", type=str, help="Path to the original image")
    parser.add_argument("anonymized", type=str, help="Path to the anonymized image")
    parser.add_argument(
        "--no_crop",
        action="store_true",
        help="Skip face detection and cropping; assumes images are already cropped to face region"
    )
    return parser.parse_args()


def detect_largest_face_bbox(yolo_model: Any, image_bgr: cv2.typing.MatLike) -> Tuple[int, int, int, int]:
    """Detect the largest face in the image and return its bounding box."""
    results = yolo_model.predict(image_bgr, verbose=False)
    
    if not results or len(results[0].boxes) == 0:
        raise RuntimeError("No faces detected in the image")
    
    # Get the largest face
    boxes = results[0].boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    largest_idx = areas.argmax()
    x1, y1, x2, y2 = boxes[largest_idx]
    
    return int(x1), int(y1), int(x2), int(y2)


def crop_by_bbox(image_bgr: cv2.typing.MatLike, bbox: Tuple[int, int, int, int]) -> cv2.typing.MatLike:
    """Crop image by bounding box."""
    x1, y1, x2, y2 = bbox
    return image_bgr[y1:y2, x1:x2]


def scale_bbox(
    bbox: Tuple[int, int, int, int],
    src_size: Tuple[int, int],
    dst_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """Scale bounding box from source image size to destination image size."""
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


def evaluate(original_image: cv2.typing.MatLike, anonymized_image: cv2.typing.MatLike, models: dict, no_crop: bool = False) -> Tuple[float, float, float]:
    """
    Evaluate the anonymization quality by comparing original and anonymized images.
    
    Args:
        original_image: Original image (BGR format)
        anonymized_image: Anonymized image (BGR format)
        models: Dictionary containing loaded models and device info
        no_crop: If True, skip face detection and use full images (assumes pre-cropped faces)
    
    Returns:
        Tuple of (insightface_score, clip_score, lpips_score)
    """
    if no_crop:
        # Use full images (assumes they are already cropped to face region)
        original_crop = original_image
        anonymized_crop = anonymized_image
    else:
        # Detect largest face in original image
        bbox_original = detect_largest_face_bbox(models["yolo"], original_image)
        
        # Scale bbox to anonymized image dimensions
        bbox_anonymized = scale_bbox(
            bbox_original,
            src_size=original_image.shape[:2],
            dst_size=anonymized_image.shape[:2],
        )
        
        # Crop faces
        original_crop = crop_by_bbox(original_image, bbox_original)
        anonymized_crop = crop_by_bbox(anonymized_image, bbox_anonymized)
    
    # Calculate metrics
    clip_score = calculate_clip_similarity(
        original_crop,
        anonymized_crop,
        models["clip_model"],
        models["clip_preprocess"],
        models["device"]
    )
    
    lpips_distance = calculate_lpips_similarity(
        original_crop,
        anonymized_crop,
        models["lpips"],
        models["device"]
    )
    lpips_score = lpips_distance  # Return distance as score (lower is more similar)
    
    insightface_score = None
    if models.get("insightface") is not None:
        try:
            insightface_score = calculate_insightface_similarity(
                original_crop,
                anonymized_crop,
                models["insightface"]
            )
        except Exception as e:
            print(f"⚠ InsightFace calculation failed: {e}")
            insightface_score = None
    
    return insightface_score, clip_score, lpips_score


def print_metrics(insightface_score: float, clip_score: float, lpips_score: float) -> None:
    """
    Print evaluation metrics in a formatted way.
    
    Args:
        insightface_score: InsightFace distance (0-1, lower is more similar)
        clip_score: CLIP similarity (0-1, higher is more similar)
        lpips_score: LPIPS distance (lower is more similar)
    """
    print(f"\n{'='*60}")
    print(f"Similarity Metrics")
    print(f"{'='*60}")
    if insightface_score is not None:
        print(f"InsightFace Dist.: {insightface_score:.4f}")
    else:
        print(f"InsightFace Dist.: N/A")
    print(f"CLIP Similarity:  {clip_score:.4f}")
    print(f"LPIPS Distance:   {lpips_score:.4f}")
    print(f"{'='*60}")


def main() -> None:
    """Main execution function."""
    # Parse arguments first (so --help works without loading anything)
    args = parse_args()
    
    # Check venv after parsing args
    ensure_running_in_venv()
    
    print("\n" + "="*60)
    print("   FACIAL ANONYMIZATION EVALUATION")
    print("="*60)
    
    # Load models
    models = load_evaluation_models()
    
    # Load images
    original_path = Path(args.original).expanduser().resolve()
    anonymized_path = Path(args.anonymized).expanduser().resolve()
    
    original_image = load_image_cv2(original_path, "Original")
    print(f"Loaded Original image: {original_path} ({original_image.shape[1]}x{original_image.shape[0]})")
    
    anonymized_image = load_image_cv2(anonymized_path, "Anonymized")
    print(f"Loaded Anonymized image: {anonymized_path} ({anonymized_image.shape[1]}x{anonymized_image.shape[0]})")
    
    # Evaluate
    print("\nEvaluating images...")
    if args.no_crop:
        print("Using pre-cropped images (skipping face detection)")
    insightface_score, clip_score, lpips_score = evaluate(original_image, anonymized_image, models, no_crop=args.no_crop)
    
    # Print results
    print_metrics(insightface_score, clip_score, lpips_score)


if __name__ == "__main__":
    main()

