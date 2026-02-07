#**********************************************************
# YOlOv11n API Example #
# Can be deleted after testing #
#**********************************************************

import base64
import json
from pathlib import Path
from models.yolo import detect_image_bytes

def main() -> int:
    here = Path(__file__).resolve().parent
    img_path = here / "models/yolo/test.jpg"
    img_bytes = img_path.read_bytes()
    
    #**********************************************************
    # Main function to detect the image #
    res = detect_image_bytes(img_bytes, return_image=True)
    #**********************************************************

    out_png = here / "annotated.png"
    out_json = here / "result.json"

    png_bytes = base64.b64decode(res["image_base64"])
    out_png.write_bytes(png_bytes)
    out_json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved {out_png.name}")
    print(f"saved {out_json.name}")
    print("num_detections =", len(res["detections"]))
    print("detections =", res["detections"][:5])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
