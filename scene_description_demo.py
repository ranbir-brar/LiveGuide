#**********************************************************
# Scene Description Demo                                  #
# Combines YOLO detection with Qwen2.5 scene description  #
#**********************************************************

import base64
import json
from pathlib import Path

from models.yolo import detect_image_bytes
from models.llm import describe_scene_from_detections


def main() -> int:
    here = Path(__file__).resolve().parent
    img_path = here / "models/yolo/test.jpg"
    img_bytes = img_path.read_bytes()
    
    # Run YOLO detection
    print("Running YOLO detection...")
    res = detect_image_bytes(img_bytes, return_image=True)
    
    print(f"Detected {len(res['detections'])} objects")
    
    # Generate scene description with Qwen2.5
    print("\nGenerating scene description...")
    description = describe_scene_from_detections(res["detections"])
    
    print("\n" + "="*50)
    print("SCENE DESCRIPTION:")
    print("="*50)
    print(description)
    print("="*50)
    
    # Save outputs
    out_png = here / "annotated.png"
    out_json = here / "result.json"
    
    png_bytes = base64.b64decode(res["image_base64"])
    out_png.write_bytes(png_bytes)
    
    # Add description to result
    res["scene_description"] = description
    out_json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print(f"\nSaved {out_png.name}")
    print(f"Saved {out_json.name}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
