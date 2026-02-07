"""
Hazard Detection Module

Analyzes YOLO detection results to identify potential hazards
for users who are walking or driving.

Configuration is loaded from config/hazard_config.json
"""

import json
import re
from pathlib import Path
from typing import Any, Literal

# Load configuration from JSON file
CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "hazard_config.json"

_config = None


def _load_config() -> dict:
    """Load and cache the hazard configuration."""
    global _config
    if _config is None:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                _config = json.load(f)
        else:
            # Fallback defaults if config file doesn't exist
            _config = {
                "walking": {
                    "high": ["car", "truck", "bus", "motorcycle", "bicycle", "train"],
                    "medium": ["dog", "horse", "cow", "elephant", "bear", "zebra", "giraffe"],
                    "low": ["person", "skateboard", "sports ball", "frisbee"],
                },
                "driving": {
                    "high": ["person", "bicycle", "motorcycle", "dog", "cat", "horse", "cow", "elephant", "bear", "zebra", "deer"],
                    "medium": ["car", "truck", "bus", "traffic light", "stop sign"],
                    "low": ["bird", "sports ball", "frisbee"],
                },
                "scoring": {
                    "category_weights": {"high": 1.0, "medium": 0.5, "low": 0.2},
                    "distance_weights": {"immediate": 1.0, "near": 0.8, "medium": 0.5, "far": 0.2},
                    "thresholds": {"critical": 0.8, "high": 0.5, "medium": 0.3},
                },
                "filters": {"confidence_threshold": 0.5, "min_box_height": 40},
            }
    return _config


def reload_config():
    """Force reload configuration from file."""
    global _config
    _config = None
    return _load_config()


def get_hazard_classes(context: str) -> dict[str, set]:
    """Get hazard classes for a given context."""
    config = _load_config()
    context_config = config.get(context, config.get("walking", {}))
    return {
        "high": set(context_config.get("high", [])),
        "medium": set(context_config.get("medium", [])),
        "low": set(context_config.get("low", [])),
    }


def _match_keyword(label: str, keyword: str) -> bool:
    """Match keyword as a word/phrase boundary inside a class label."""
    return re.search(rf"\b{re.escape(keyword)}\b", label) is not None


def _classify_by_keywords(class_name: str, context: str, config: dict[str, Any]) -> str | None:
    """Fallback classification for larger vocab models (e.g., OpenImages 600 classes)."""
    auto_cfg = config.get("auto_classification", {})
    if not auto_cfg.get("enabled", False):
        return None

    ctx = auto_cfg.get(context, auto_cfg.get("walking", {}))
    for category in ("high", "medium", "low"):
        key = f"{category}_keywords"
        for kw in ctx.get(key, []):
            kw_norm = str(kw).strip().lower()
            if kw_norm and _match_keyword(class_name, kw_norm):
                return category
    return None


def classify_hazard(
    detections: list[dict[str, Any]],
    context: Literal["walking", "driving"] = "walking",
    confidence_threshold: float | None = None,
    min_box_height: int | None = None,
) -> dict[str, Any]:
    """
    Analyze detections and return hazard classification.
    
    Args:
        detections: List of YOLO detection dicts with 'class_name', 'confidence', 'xyxy'.
        context: Either 'walking' or 'driving' to determine hazard relevance.
        confidence_threshold: Min confidence (uses config default if None).
        min_box_height: Min box height in pixels (uses config default if None).
    
    Returns:
        Dict with hazard_score, hazard_level, and details.
    """
    config = _load_config()
    filters = config.get("filters", {})
    scoring = config.get("scoring", {})
    
    # Use config defaults if not specified
    if confidence_threshold is None:
        confidence_threshold = filters.get("confidence_threshold", 0.5)
    if min_box_height is None:
        min_box_height = filters.get("min_box_height", 40)
    
    hazard_map = get_hazard_classes(context)
    category_weights = scoring.get("category_weights", {"high": 1.0, "medium": 0.5, "low": 0.2})
    distance_weights = scoring.get("distance_weights", {"immediate": 1.0, "near": 0.8, "medium": 0.5, "far": 0.2})
    thresholds = scoring.get("thresholds", {"critical": 0.8, "high": 0.5, "medium": 0.3})
    
    high_hazards = []
    medium_hazards = []
    low_hazards = []
    ignored_distant = 0
    auto_classified = 0
    
    for det in detections:
        class_name = det.get("class_name", "").lower()
        confidence = det.get("confidence", 0)
        
        if confidence < confidence_threshold:
            continue
        
        xyxy = det.get("xyxy", [0, 0, 0, 0])
        box_height = xyxy[3] - xyxy[1]
        
        # Skip objects that are too small (far away)
        if box_height < min_box_height:
            ignored_distant += 1
            continue
        
        hazard_info = {
            "object": class_name,
            "confidence": round(confidence, 2),
            "box_height": int(box_height),
        }
        
        # Add depth info if available
        if "relative_distance" in det:
            hazard_info["relative_distance"] = det["relative_distance"]
            hazard_info["distance_category"] = det.get("distance_category", "unknown")
        
        if class_name in hazard_map.get("high", set()):
            high_hazards.append(hazard_info)
        elif class_name in hazard_map.get("medium", set()):
            medium_hazards.append(hazard_info)
        elif class_name in hazard_map.get("low", set()):
            low_hazards.append(hazard_info)
        else:
            auto_category = _classify_by_keywords(class_name, context, config)
            if auto_category == "high":
                high_hazards.append(hazard_info)
                auto_classified += 1
            elif auto_category == "medium":
                medium_hazards.append(hazard_info)
                auto_classified += 1
            elif auto_category == "low":
                low_hazards.append(hazard_info)
                auto_classified += 1
    
    # Calculate hazard score (0-1)
    max_score = 0.0
    cumulative_score = 0.0
    
    for h in high_hazards:
        if "relative_distance" in h:
            distance_weight = 1.0 - h["relative_distance"]
        else:
            dist_cat = h.get("distance_category", "medium")
            distance_weight = distance_weights.get(dist_cat, 0.5)
        
        score = category_weights.get("high", 1.0) * h["confidence"] * distance_weight
        max_score = max(max_score, score)
        cumulative_score += score * 0.3
    
    for h in medium_hazards:
        if "relative_distance" in h:
            distance_weight = 1.0 - h["relative_distance"]
        else:
            dist_cat = h.get("distance_category", "medium")
            distance_weight = distance_weights.get(dist_cat, 0.5)
        
        score = category_weights.get("medium", 0.5) * h["confidence"] * distance_weight
        max_score = max(max_score, score)
        cumulative_score += score * 0.2
    
    for h in low_hazards:
        if "relative_distance" in h:
            distance_weight = 1.0 - h["relative_distance"]
        else:
            dist_cat = h.get("distance_category", "medium")
            distance_weight = distance_weights.get(dist_cat, 0.5)
        
        score = category_weights.get("low", 0.2) * h["confidence"] * distance_weight
        max_score = max(max_score, score)
        cumulative_score += score * 0.1
    
    # Final score
    hazard_score = min(1.0, max_score + cumulative_score * 0.5)
    hazard_score = round(hazard_score, 2)
    
    # Determine hazard level from score using config thresholds
    if hazard_score >= thresholds.get("critical", 0.8):
        hazard_level = "critical"
    elif hazard_score >= thresholds.get("high", 0.5):
        hazard_level = "high"
    elif hazard_score >= thresholds.get("medium", 0.3):
        hazard_level = "medium"
    elif hazard_score > 0:
        hazard_level = "low"
    else:
        hazard_level = "clear"
    
    return {
        "hazard_score": hazard_score,
        "hazard_level": hazard_level,
        "context": context,
        "high_hazards": high_hazards,
        "medium_hazards": medium_hazards,
        "low_hazards": low_hazards,
        "total_hazards": len(high_hazards) + len(medium_hazards) + len(low_hazards),
        "ignored_distant": ignored_distant,
        "auto_classified": auto_classified,
    }


def format_hazard_alert(hazard_result: dict[str, Any]) -> str:
    """Format hazard result into a spoken alert string."""
    level = hazard_result["hazard_level"]
    
    if level == "clear":
        return "Path is clear."
    
    alerts = []
    
    for h in hazard_result["high_hazards"]:
        alerts.append(f"{h['object']} ({h['confidence']:.0%})")
    
    for h in hazard_result["medium_hazards"]:
        alerts.append(f"{h['object']} ({h['confidence']:.0%})")

    if not alerts:
        for h in hazard_result.get("low_hazards", []):
            alerts.append(f"{h['object']} ({h['confidence']:.0%})")
            if len(alerts) >= 3:
                break
    
    if level == "critical":
        prefix = "WARNING: "
    elif level == "high":
        prefix = "Caution: "
    else:
        prefix = ""
    
    return prefix + ", ".join(alerts[:3]) + "."
