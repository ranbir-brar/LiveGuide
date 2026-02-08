#**********************************************************
# Video Assist Image Demo                                 #
# Sequence pipeline via main interface                    #
#**********************************************************

import base64
import json
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main import run_frame_sequence

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OUTPUT_SIZE = 256


def standardize_output_image(img_bytes: bytes, size: int = OUTPUT_SIZE) -> bytes:
    """Resize annotated image to standardized size."""
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img_resized = img.resize((size, size), Image.Resampling.BILINEAR)

    buf = BytesIO()
    img_resized.save(buf, format="PNG")
    return buf.getvalue()


def main() -> int:
    here = Path(__file__).resolve().parent
    test_images_dir = here / "test_images"
    output_dir = here.parent / "annotated-images"
    output_dir.mkdir(exist_ok=True)

    image_files = sorted(
        [
            f
            for f in test_images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )

    if not image_files:
        print(f"No images found in {test_images_dir}")
        return 1

    frames = [p.read_bytes() for p in image_files]
    all_results = run_frame_sequence(
        frames,
        context="general",
        use_depth=True,
        preprocess_size=256,
        imgsz=256,
        return_annotated=True,
        wait_for_llm=True,
    )

    print(f"Found {len(image_files)} image(s) in {test_images_dir.name}/")
    print("=" * 60)

    for img_path, result in zip(image_files, all_results):
        latency = result["latency"]
        llm = result["llm"]

        print(f"\n>> {img_path.name}")
        print(f"   Detected: {len(result['detections'])} objects")

        if result["detections"]:
            objects = [f"{d['class_name']}" for d in result["detections"]]
            print(f"   Objects: {', '.join(objects)}")

        print(
            f"   LLM Gate: sent={llm['sent']} prob={llm['probability']:.2f} "
            f"sampled={llm['sampled']} rate_ok={llm['rate_allowed']} "
            f"provider={result['runtime'].get('llm_provider')}"
        )
        if llm.get("description"):
            print(f"   LLM: {llm['description']}")
        elif llm.get("error"):
            print(f"   LLM Error: {llm['error']}")

        print(
            f"   Latency: {latency['total_ms']}ms "
            f"(YOLO: {latency['yolo_ms']}ms, Depth: {latency['depth_ms']}ms, LLM: {latency['llm_ms']}ms)"
        )

        if result.get("image_base64"):
            out_png = output_dir / f"{img_path.stem}_annotated.png"
            png_bytes = base64.b64decode(result["image_base64"])
            standardized = standardize_output_image(png_bytes)
            out_png.write_bytes(standardized)

        result["source_image"] = img_path.name

    results_json = output_dir / "results.json"
    for r in all_results:
        r.pop("image_base64", None)
    results_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Processed {len(image_files)} image(s)")
    print(f"Results saved to {output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
