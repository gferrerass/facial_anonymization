"""
Evaluation utilities for facial anonymization.
Contains model loading and similarity metric functions.
"""

import time
from pathlib import Path
from typing import Any
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


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
    
    print("▶ Loading evaluation models...")
    
    if YOLO is None:
        raise RuntimeError("Ultralytics is not available")
    
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
                insightface_model = FaceAnalysis(
                    name="buffalo_l",
                    root=str(insightface_root),
                    providers=[provider_name],
                )
                ctx_id = 0 if provider_name == "CUDAExecutionProvider" else -1
                insightface_model.prepare(ctx_id=ctx_id, det_size=(640, 640))
                break
            except Exception as e:
                print(f"⚠ Could not load InsightFace with {provider_name}: {e}")
                insightface_model = None
    else:
        print("⚠ InsightFace not available; skipping InsightFace similarity metric")
    
    eval_load_time = time.time() - eval_start
    print(f"✓ Evaluation models loaded in {eval_load_time:.2f} seconds")
    
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
    """Calculate cosine similarity between two face embeddings using InsightFace."""
    emb1 = _extract_insightface_embedding(insightface_model, image1_bgr)
    emb2 = _extract_insightface_embedding(insightface_model, image2_bgr)

    if emb1 is None or emb2 is None:
        raise RuntimeError("InsightFace could not detect a face in one or both crops")

    emb1 = torch.from_numpy(emb1).float().unsqueeze(0)
    emb2 = torch.from_numpy(emb2).float().unsqueeze(0)
    similarity = F.cosine_similarity(emb1, emb2)
    return float(similarity.item())
