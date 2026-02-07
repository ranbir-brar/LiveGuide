#**********************************************************
# Hazard Detection Demo                                   #
# YOLO detection with hazard classification               #
#**********************************************************

import base64
import json
import time
from io import BytesIO
from pathlib import Path

from PIL import Image

from models.yolo import detect_image_bytes, classify_hazard, format_hazard_alert

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Standardized output size
OUTPUT_SIZE = 256


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
    res = detect_image_bytes(processed_bytes, return_image=True, imgsz=256)
    yolo_time = time.perf_counter() - yolo_start
    
    detections = res["detections"]
    depth_time = 0.0
    
    # Depth estimation (optional)
    if use_depth and detections:
        from models.depth import estimate_depth, enrich_detections_with_depth
        
        depth_start = time.perf_counter()
        depth_map = estimate_depth(processed_bytes)
        detections = enrich_detections_with_depth(detections, depth_map)
        depth_time = time.perf_counter() - depth_start
    
    # Classify hazards (filters out distant objects)
    hazard_start = time.perf_counter()
    hazard_result = classify_hazard(detections, context=context)
    hazard_time = time.perf_counter() - hazard_start
    
    total_time = time.perf_counter() - total_start
    
    return {
        "detections": detections,
        "image_base64": res.get("image_base64"),
        "hazard": hazard_result,
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
    test_images_dir = here / "models" / "yolo" / "test-images"
    output_dir = here / "annotated-images"
    output_dir.mkdir(exist_ok=True)
    
    # Context: "walking" or "driving"
    context = "walking"
    
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
