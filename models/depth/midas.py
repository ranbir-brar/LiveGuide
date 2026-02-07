"""
Depth Estimation Module using MiDaS

Provides relative depth estimation for detected objects.
Uses bottom-center of image as POV reference point (simulating a person's view).
"""

import threading
from io import BytesIO
from typing import Any

import numpy as np
import torch
from PIL import Image

# Use MiDaS small for speed
MODEL_TYPE = "MiDaS_small"

_model = None
_transform = None
_device = None
_load_lock = threading.Lock()


def _get_model():
    """Lazy-load MiDaS model (thread-safe, cached)."""
    global _model, _transform, _device
    
    if _model is None:
        with _load_lock:
            if _model is None:
                print(f"Loading MiDaS depth model...")
                _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # Load MiDaS small model
                _model = torch.hub.load("intel-isl/MiDaS", MODEL_TYPE, trust_repo=True)
                _model.to(_device)
                _model.eval()
                
                # Load transforms
                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
                _transform = midas_transforms.small_transform
                
                print(f"MiDaS loaded on {_device}")
    
    return _model, _transform, _device


def estimate_depth(img_bytes: bytes) -> np.ndarray:
    """
    Estimate depth map from image bytes.
    
    Args:
        img_bytes: Image as bytes.
    
    Returns:
        Depth map as 2D numpy array (higher values = further away).
    """
    model, transform, device = _get_model()
    
    # Load image
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img)
    
    # Transform for MiDaS
    input_batch = transform(img_np).to(device)
    
    with torch.no_grad():
        prediction = model(input_batch)
        
        # Resize to original image size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_np.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    
    # Normalize to 0-1 range (0 = close, 1 = far)
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    
    return depth_map


def get_object_depth(
    depth_map: np.ndarray,
    xyxy: list[float],
    reference: str = "bottom_center",
) -> dict[str, Any]:
    """
    Get depth info for an object bounding box.
    
    Args:
        depth_map: Depth map from estimate_depth().
        xyxy: Bounding box [x1, y1, x2, y2].
        reference: Reference point for distance ("bottom_center" = person's POV).
    
    Returns:
        Dict with depth metrics.
    """
    h, w = depth_map.shape
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    
    # Clamp to image bounds
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    
    if x1 >= x2 or y1 >= y2:
        return {"depth": 1.0, "distance_category": "far", "relative_distance": 1.0}
    
    # Get depth values for the object region
    obj_depth = depth_map[y1:y2, x1:x2]
    
    # Use median depth (more robust than mean)
    median_depth = float(np.median(obj_depth))
    
    # Reference point: bottom-center of image (where the person is "standing")
    ref_y, ref_x = h - 1, w // 2
    ref_depth = depth_map[ref_y, ref_x]
    
    # Relative distance from reference point
    # Objects at same depth as reference = 0, further = positive
    relative_distance = median_depth - ref_depth
    
    # Normalize to 0-1 scale based on typical depth range
    # Clamp to ensure valid range
    normalized_distance = max(0.0, min(1.0, (relative_distance + 0.5)))
    
    # Categorize distance
    if normalized_distance < 0.3:
        distance_category = "immediate"  # Very close, urgent
    elif normalized_distance < 0.5:
        distance_category = "near"
    elif normalized_distance < 0.7:
        distance_category = "medium"
    else:
        distance_category = "far"
    
    return {
        "depth": float(round(median_depth, 3)),
        "relative_distance": float(round(normalized_distance, 3)),
        "distance_category": distance_category,
    }


def enrich_detections_with_depth(
    detections: list[dict[str, Any]],
    depth_map: np.ndarray,
) -> list[dict[str, Any]]:
    """
    Add depth information to each detection.
    
    Args:
        detections: List of YOLO detections with 'xyxy' key.
        depth_map: Depth map from estimate_depth().
    
    Returns:
        Detections with added depth info.
    """
    enriched = []
    
    for det in detections:
        xyxy = det.get("xyxy", [0, 0, 0, 0])
        depth_info = get_object_depth(depth_map, xyxy)
        
        enriched_det = {**det, **depth_info}
        enriched.append(enriched_det)
    
    return enriched
