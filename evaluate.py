#!/usr/bin/env python3
"""
Evaluate script for face crop comparison.

Workflow:
1) Load original and anonymized images.
2) Detect the largest face in the original image.
3) Crop both images using the same face region (scaled if dimensions differ).
4) Save both crops to a temporary directory.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import torch
import lpips
import open_clip
from torchvision import transforms
from PIL import Image

# Suppress torchvision deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

try:
	from ultralytics import YOLO
except Exception as exc:
	YOLO = None
	ULTRALYTICS_IMPORT_ERROR = exc
else:
	ULTRALYTICS_IMPORT_ERROR = None


def ensure_running_in_venv() -> None:
	"""Relaunch the script with the project's virtualenv Python if needed."""
	if os.environ.get("FACIAL_ANON_EVAL_RELAUNCHED") == "1":
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
		return

	env = os.environ.copy()
	env["FACIAL_ANON_EVAL_RELAUNCHED"] = "1"
	cmd = [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]]
	result = subprocess.run(cmd, cwd=project_root, env=env, check=False)
	raise SystemExit(result.returncode)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Crop largest detected face from original and anonymized images.",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog=(
			"Examples:\n"
			"  python evaluate.py input/person.jpg output/person_anonymized_0001.png\n"
			"  python evaluate.py C:/data/orig.png C:/data/anon.png --show"
		),
	)
	parser.add_argument("original", type=str, help="Path to the original image")
	parser.add_argument("anonymized", type=str, help="Path to the anonymized image")
	parser.add_argument(
		"--model",
		type=str,
		default="models/ultralytics/bbox/face_yolov8m.pt",
		help="Path to YOLO face detection model",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default=str(Path(tempfile.gettempdir()) / "facial_anonymization_eval"),
		help="Directory where cropped faces are saved",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="Open generated crops with the default image viewer",
	)
	return parser.parse_args()


def load_image(path: Path, label: str) -> cv2.typing.MatLike:
	"""Load image using OpenCV."""
	image = cv2.imread(str(path), cv2.IMREAD_COLOR)
	if image is None:
		raise FileNotFoundError(f"Could not load {label} image: {path}")
	return image


def detect_largest_face_bbox(model: Any, image_bgr: cv2.typing.MatLike) -> tuple[int, int, int, int]:
	"""Detect largest face using YOLO model."""
	image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
	results = model.predict(source=image_rgb, verbose=False)

	if not results:
		raise RuntimeError("No detection results returned by YOLO.")

	boxes = results[0].boxes
	if boxes is None or len(boxes) == 0:
		raise RuntimeError("No face detected in original image.")

	xyxy = boxes.xyxy.cpu().numpy()
	max_area = -1.0
	best_box: tuple[int, int, int, int] | None = None
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


def save_crop(image_bgr: cv2.typing.MatLike, output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	ok = cv2.imwrite(str(output_path), image_bgr)
	if not ok:
		raise RuntimeError(f"Could not write image: {output_path}")


def open_image(path: Path) -> None:
	if platform.system() == "Windows":
		os.startfile(str(path))
		return

	if platform.system() == "Darwin":
		subprocess.run(["open", str(path)], check=False)
		return

	subprocess.run(["xdg-open", str(path)], check=False)


def load_models(model_path: Path) -> tuple[Any, Any, Any, Any]:
	"""Load YOLO, CLIP and LPIPS models at the beginning."""
	print(f"Loading YOLO model: {model_path}")
	yolo_model = YOLO(str(model_path))
	
	device = "cuda" if torch.cuda.is_available() else "cpu"
	
	print(f"Loading CLIP model (device: {device})...")
	clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', device=device)
	
	print(f"Loading LPIPS model (device: {device})...")
	lpips_model = lpips.LPIPS(net='alex').to(device)
	
	return yolo_model, lpips_model, clip_model, clip_preprocess


def calculate_clip_similarity(image1_bgr: cv2.typing.MatLike, image2_bgr: cv2.typing.MatLike, clip_model: Any, clip_preprocess: Any) -> float:
	"""Calculate CLIP cosine similarity between two BGR images."""
	device = "cuda" if torch.cuda.is_available() else "cpu"
	
	# Convert BGR to RGB and then to PIL
	image1_rgb = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2RGB)
	image2_rgb = cv2.cvtColor(image2_bgr, cv2.COLOR_BGR2RGB)
	
	image1_pil = Image.fromarray(image1_rgb)
	image2_pil = Image.fromarray(image2_rgb)
	
	# Apply CLIP preprocessing
	image1_tensor = clip_preprocess(image1_pil).unsqueeze(0).to(device)
	image2_tensor = clip_preprocess(image2_pil).unsqueeze(0).to(device)
	
	# Get image features
	with torch.no_grad():
		image1_features = clip_model.encode_image(image1_tensor)
		image2_features = clip_model.encode_image(image2_tensor)
	
	# Calculate cosine similarity
	similarity = torch.cosine_similarity(image1_features, image2_features)
	return similarity.item()


def calculate_lpips_similarity(image1_bgr: cv2.typing.MatLike, image2_bgr: cv2.typing.MatLike, lpips_model: Any) -> float:
	"""Calculate LPIPS distance between two BGR images."""
	device = "cuda" if torch.cuda.is_available() else "cpu"
	
	# Convert BGR to RGB and then to PIL
	image1_rgb = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2RGB)
	image2_rgb = cv2.cvtColor(image2_bgr, cv2.COLOR_BGR2RGB)
	
	image1_pil = Image.fromarray(image1_rgb)
	image2_pil = Image.fromarray(image2_rgb)
	
	# Transform images for LPIPS (resize to 256x256 and normalize)
	transform = transforms.Compose([
		transforms.Resize((256, 256)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
	])
	
	image1_tensor = transform(image1_pil).unsqueeze(0).to(device)
	image2_tensor = transform(image2_pil).unsqueeze(0).to(device)
	
	# Calculate LPIPS distance
	with torch.no_grad():
		lpips_distance = lpips_model(image1_tensor, image2_tensor)
	
	return lpips_distance.item()


def main() -> None:
	ensure_running_in_venv()
	args = parse_args()

	if YOLO is None:
		raise RuntimeError(
			"Ultralytics is not available in the active environment. "
			f"Import error: {ULTRALYTICS_IMPORT_ERROR}"
		)

	original_path = Path(args.original).expanduser().resolve()
	anonymized_path = Path(args.anonymized).expanduser().resolve()
	model_path = Path(args.model).expanduser().resolve()

	if not original_path.exists():
		raise FileNotFoundError(f"Original image not found: {original_path}")
	if not anonymized_path.exists():
		raise FileNotFoundError(f"Anonymized image not found: {anonymized_path}")
	if not model_path.exists():
		raise FileNotFoundError(f"YOLO model not found: {model_path}")

	yolo_model, lpips_model, clip_model, clip_preprocess = load_models(model_path)

	original_bgr = load_image(original_path, "original")
	anonymized_bgr = load_image(anonymized_path, "anonymized")

	bbox_original = detect_largest_face_bbox(yolo_model, original_bgr)
	bbox_anonymized = scale_bbox(
		bbox_original,
		src_size=original_bgr.shape[:2],
		dst_size=anonymized_bgr.shape[:2],
	)

	original_crop = crop_by_bbox(original_bgr, bbox_original)
	anonymized_crop = crop_by_bbox(anonymized_bgr, bbox_anonymized)

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	output_dir = Path(args.output_dir).expanduser().resolve() / timestamp
	original_crop_path = output_dir / f"{original_path.stem}_face_crop.png"
	anonymized_crop_path = output_dir / f"{anonymized_path.stem}_face_crop.png"

	save_crop(original_crop, original_crop_path)
	save_crop(anonymized_crop, anonymized_crop_path)

	print(f"\n{'='*60}")
	print(f"Face crops generated successfully")
	print(f"Original crop: {original_crop_path}")
	print(f"Anonymized crop: {anonymized_crop_path}")
	print(f"{'='*60}")

	# Calculate similarities
	print(f"\nCalculating CLIP similarity...")
	clip_score = calculate_clip_similarity(original_crop, anonymized_crop, clip_model, clip_preprocess)
	
	print(f"Calculating LPIPS similarity...")
	lpips_distance = calculate_lpips_similarity(original_crop, anonymized_crop, lpips_model)
	lpips_similarity = 1.0 - lpips_distance  # Convert distance to similarity
	
	print(f"\n{'='*60}")
	print(f"Similarity Metrics")
	print(f"{'='*60}")
	print(f"CLIP Similarity:  {clip_score:.4f}")
	print(f"LPIPS Distance:   {lpips_distance:.4f}")
	print(f"LPIPS Similarity: {lpips_similarity:.4f}")
	print(f"{'='*60}")

	if args.show:
		print(f"\nOpening crops...")
		open_image(original_crop_path)
		open_image(anonymized_crop_path)


if __name__ == "__main__":
	main()