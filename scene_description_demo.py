#**********************************************************
# Hazard Detection Demo                                   #
# YOLO detection with hazard classification               #
#**********************************************************

import base64
import json
import random
import time
from collections import deque
from io import BytesIO
from pathlib import Path

from PIL import Image

from models.llm import describe_scene_from_frame
from models.yolo import detect_image_bytes, classify_hazard, format_hazard_alert

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Standardized output size
OUTPUT_SIZE = 256
RUNTIME_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "runtime_config.json"
_LLM_CALL_TIMES = deque()


def load_runtime_config() -> dict:
    """Load runtime configuration for YOLO/LLM scheduling."""
    if RUNTIME_CONFIG_PATH.exists():
        with open(RUNTIME_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "pipeline": {"yolo_target_fps": 6.0},
        "llm": {
            "max_new_tokens": 80,
            "calls_per_second_constant": 0.8,
            "max_calls_per_second": 1.5,
            "min_interval_ms": 300,
            "probability_scale": 1.0,
            "base_probability_bias": 0.1,
            "rng_seed": 42,
        },
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def llm_send_probability(yolo_score: float, cfg: dict) -> float:
    """
    Bias outside scale:
    P = bias + (1 - bias) * scale01((yolo_score * constant) * probability_scale)
    """
    llm_cfg = cfg.get("llm", {})
    constant = float(llm_cfg.get("calls_per_second_constant", 0.8))
    bias = float(llm_cfg.get("base_probability_bias", 0.1))
    scale = float(llm_cfg.get("probability_scale", 1.0))
    bias = _clamp01(bias)
    scaled_part = _clamp01(float(yolo_score) * constant * scale)
    return bias + (1.0 - bias) * scaled_part


def _llm_rate_allowed(cfg: dict) -> bool:
    llm_cfg = cfg.get("llm", {})
    max_calls_per_sec = float(llm_cfg.get("max_calls_per_second", 1.5))
    min_interval_sec = float(llm_cfg.get("min_interval_ms", 300)) / 1000.0
    per_call_interval = 1.0 / max_calls_per_sec if max_calls_per_sec > 0 else 1.0
    now = time.monotonic()

    while _LLM_CALL_TIMES and (now - _LLM_CALL_TIMES[0]) > 1.0:
        _LLM_CALL_TIMES.popleft()

    if _LLM_CALL_TIMES and (now - _LLM_CALL_TIMES[-1]) < max(min_interval_sec, per_call_interval):
        return False

    _LLM_CALL_TIMES.append(now)
    return True


def should_send_to_llm(yolo_score: float, cfg: dict) -> tuple[bool, float, bool, bool]:
    probability = llm_send_probability(yolo_score, cfg)
    sampled = random.random() < probability
    rate_allowed = _llm_rate_allowed(cfg) if sampled else False
    return sampled and rate_allowed, probability, sampled, rate_allowed


def preprocess_image(img_bytes: bytes, target_size: int = 256) -> bytes:
    """Resize image to target_size x target_size (stretch to fit)."""
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    
    # Resize to exact dimensions (stretch/squash, no cropping)
    img_resized = img.resize((target_size, target_size), Image.Resampling.BILINEAR)
    
    # Convert back to bytes
    buf = BytesIO()
    img_resized.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def standardize_output_image(img_bytes: bytes, size: int = OUTPUT_SIZE) -> bytes:
    """Resize annotated image to standardized size."""
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img_resized = img.resize((size, size), Image.Resampling.BILINEAR)
    
    buf = BytesIO()
    img_resized.save(buf, format="PNG")
    return buf.getvalue()


def process_image(img_path: Path, context: str = "walking", use_depth: bool = True) -> dict:
    """Process a single image and return results."""
    img_bytes = img_path.read_bytes()
    
    total_start = time.perf_counter()
    
    # Preprocess image (resize for speed)
    preprocess_start = time.perf_counter()
    processed_bytes = preprocess_image(img_bytes, target_size=256)
    preprocess_time = time.perf_counter() - preprocess_start
    
    # Run YOLO detection
    yolo_start = time.perf_counter()
    res = detect_image_bytes(processed_bytes, return_image=True, imgsz=256, use_depth=use_depth)
    yolo_time = time.perf_counter() - yolo_start
    
    detections = res["detections"]
    depth_time = float(res.get("depth_ms", 0.0)) / 1000.0
    
    # Classify hazards (filters out distant objects)
    hazard_start = time.perf_counter()
    hazard_result = classify_hazard(detections, context=context)
    hazard_time = time.perf_counter() - hazard_start

    runtime_cfg = load_runtime_config()
    llm_cfg = runtime_cfg.get("llm", {})
    yolo_score = float(hazard_result.get("hazard_score", 0.0))
    send_to_llm, llm_prob, llm_sampled, llm_rate_allowed = should_send_to_llm(yolo_score, runtime_cfg)
    llm_desc = None
    llm_error = None

    if send_to_llm:
        try:
            llm_desc = describe_scene_from_frame(
                processed_bytes,
                detections,
                max_new_tokens=int(llm_cfg.get("max_new_tokens", 80)),
            )
        except Exception as e:  # noqa: BLE001
            llm_error = str(e)
    
    total_time = time.perf_counter() - total_start
    
    return {
        "detections": detections,
        "image_base64": res.get("image_base64"),
        "hazard": hazard_result,
        "llm": {
            "sent": send_to_llm,
            "probability": round(llm_prob, 4),
            "sampled": llm_sampled,
            "rate_allowed": llm_rate_allowed,
            "description": llm_desc,
            "error": llm_error,
        },
        "latency": {
            "preprocess_ms": round(preprocess_time * 1000, 1),
            "yolo_ms": round(yolo_time * 1000, 1),
            "depth_ms": round(depth_time * 1000, 1),
            "hazard_ms": round(hazard_time * 1000, 1),
            "total_ms": round(total_time * 1000, 1),
        },
    }


def main() -> int:
    here = Path(__file__).resolve().parent
    test_images_dir = here / "data" / "test_images"
    output_dir = here / "annotated-images"
    output_dir.mkdir(exist_ok=True)
    
    # Context: "walking" or "driving"
    context = "walking"
    runtime_cfg = load_runtime_config()
    random.seed(int(runtime_cfg.get("llm", {}).get("rng_seed", 42)))
    
    # Find all images
    image_files = [
        f for f in test_images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    
    if not image_files:
        print(f"No images found in {test_images_dir}")
        return 1
    
    print(f"Found {len(image_files)} image(s) in {test_images_dir.name}/")
    print(f"Context: {context}")
    print("=" * 60)
    
    all_results = []
    
    for img_path in sorted(image_files):
        print(f"\n>> {img_path.name}")
        
        result = process_image(img_path, context=context)
        hazard = result["hazard"]
        latency = result["latency"]
        
        # Display hazard info
        print(f"   Detected: {len(result['detections'])} objects, "
              f"Filtered: {hazard['ignored_distant']} distant")
        
        # List detected objects
        if result['detections']:
            objects = [f"{d['class_name']}" for d in result['detections']]
            print(f"   Objects: {', '.join(objects)}")
        
        print(f"   Hazard Score: {hazard['hazard_score']:.2f} ({hazard['hazard_level'].upper()})")
        print(f"   Alert: {format_hazard_alert(hazard)}")
        llm = result["llm"]
        print(
            f"   LLM Gate: sent={llm['sent']} prob={llm['probability']:.2f} "
            f"sampled={llm['sampled']} rate_ok={llm['rate_allowed']}"
        )
        if llm.get("description"):
            print(f"   LLM: {llm['description']}")
        elif llm.get("error"):
            print(f"   LLM Error: {llm['error']}")
        print(f"   Latency: {latency['total_ms']}ms "
              f"(YOLO: {latency['yolo_ms']}ms, Depth: {latency['depth_ms']}ms)")
        
        # Save annotated image (standardized to 256x256)
        if result.get("image_base64"):
            out_png = output_dir / f"{img_path.stem}_annotated.png"
            png_bytes = base64.b64decode(result["image_base64"])
            standardized = standardize_output_image(png_bytes)
            out_png.write_bytes(standardized)
        
        # Add to results
        result["source_image"] = img_path.name
        all_results.append(result)
    
    # Save combined results
    results_json = output_dir / "results.json"
    # Remove base64 images from JSON to keep it small
    for r in all_results:
        r.pop("image_base64", None)
    results_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    
    print("\n" + "=" * 60)
    print(f"Processed {len(image_files)} image(s)")
    print(f"Results saved to {output_dir}/")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
