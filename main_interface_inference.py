#**********************************************************
# Main Interface API Example                              #
# Calls single-image pipeline: YOLO -> hazard -> LLM gate #
#**********************************************************

import base64
import json
from pathlib import Path

from main_interface import run_image_pipeline


def main() -> int:
    here = Path(__file__).resolve().parent
    test_dir = here / "data" / "test_images"
    image_candidates = sorted(
        p for p in test_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not image_candidates:
        print(f"No test images found in {test_dir}")
        return 1

    img_path = image_candidates[0]
    img_bytes = img_path.read_bytes()
    result = run_image_pipeline(img_bytes, context="walking", use_depth=True, return_annotated=True)

    out_png = here / "annotated.png"
    out_json = here / "result.json"
    if result.get("image_base64"):
        out_png.write_bytes(base64.b64decode(result["image_base64"]))
        print(f"saved {out_png.name}")

    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out_json.name}")
    print("source_image =", img_path.name)
    print("num_detections =", len(result["detections"]))
    print("hazard =", result["hazard"]["hazard_level"], result["hazard"]["hazard_score"])
    print("llm_gate =", result["llm"]["sent"], "prob=", result["llm"]["probability"])
    if result["llm"].get("description"):
        print("llm_desc =", result["llm"]["description"])
    elif result["llm"].get("error"):
        print("llm_error =", result["llm"]["error"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
