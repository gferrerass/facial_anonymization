"""
Image utilities for facial anonymization.
Contains image loading, face detection, and cropping functions.
"""

from pathlib import Path
from typing import Any
import cv2
from PIL import Image


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
    """Scale bounding box from source size to destination size."""
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
    """Crop image using bounding box coordinates."""
    x1, y1, x2, y2 = bbox
    return image_bgr[y1:y2, x1:x2]


def bgr_crop_to_pil_rgb(crop_bgr: cv2.typing.MatLike) -> Image.Image:
    """Convert BGR crop to PIL RGB image."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)


def save_crop(crop_bgr: cv2.typing.MatLike, output_path: Path) -> None:
    """Save BGR crop to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), crop_bgr)
