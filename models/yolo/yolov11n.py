import base64
import os
import threading
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from ultralytics import YOLO

MODULE_DIR = Path(__file__).resolve().parent  # .../models/yolo
DEFAULT_WEIGHTS = MODULE_DIR / "weights" / "yolov8n.pt"
DEFAULT_WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"

_MODEL_CACHE: dict[str, YOLO] = {}
_MODEL_CACHE_LOCK = threading.Lock()
_DOWNLOAD_LOCK = threading.Lock()


class APIError(Exception):
    pass


def _download_file(url: str, dst: Path, *, timeout: float = 60.0) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "YOLOv11n-local/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            if getattr(resp, "status", 200) != 200:
                raise APIError(f"Download failed (HTTP {getattr(resp, 'status', 'unknown')}).")
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        if tmp.stat().st_size < 1024 * 1024:
            raise APIError("Downloaded file looks too small; aborting.")
        tmp.replace(dst)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _resolve_weights_path(weights_path: str | None) -> Path:
    raw = (weights_path or "").strip()
    if not raw:
        p = DEFAULT_WEIGHTS
    else:
        rel = Path(raw)
        if rel.is_absolute():
            raise APIError("Please provide a relative weights path (relative to `models/yolo/`), e.g. `weights/yolo11n.pt`.")
        if any(part == ".." for part in rel.parts):
            raise APIError("Invalid weights path.")
        p = (MODULE_DIR / rel).resolve()

    if not p.exists():
        # Auto-download only for the default weights into `models/yolo/weights/`.
        # Set AUTO_DOWNLOAD=0 to disable.
        auto = os.getenv("AUTO_DOWNLOAD", "1").strip().lower() not in {"0", "false", "no", "off"}
        if p == DEFAULT_WEIGHTS and auto:
            with _DOWNLOAD_LOCK:
                if not p.exists():
                    try:
                        _download_file(DEFAULT_WEIGHTS_URL, p)
                    except Exception as e:  # noqa: BLE001
                        raise APIError(f"Failed to auto-download default weights: {e}") from e
        if not p.exists():
            raise APIError(
                "Weights file not found. Please put `yolo11n.pt` under `models/yolo/weights/`, "
                "or provide `weights_path` relative to `models/yolo/`."
            )
    return p


def _get_model(weights_path: str | None) -> YOLO:
    p = _resolve_weights_path(weights_path)
    key = str(p)
    with _MODEL_CACHE_LOCK:
        m = _MODEL_CACHE.get(key)
        if m is None:
            m = YOLO(key)
            _MODEL_CACHE[key] = m
    return m


def detect_pil(
    image: Image.Image,
    *,
    weights_path: str | None = None,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    device: str = "auto",
    return_image: bool = False,
) -> dict[str, Any]:
    if image is None:
        raise APIError("Image is required.")

    pil = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(np.array(image)).convert("RGB")
    model = _get_model(weights_path)

    results = model.predict(
        source=pil,
        conf=float(conf),
        iou=float(iou),
        imgsz=int(imgsz),
        device=None if device == "auto" else device,
        verbose=False,
    )

    r0 = results[0]
    boxes = getattr(r0, "boxes", None)
    detections: list[dict[str, Any]] = []

    if boxes is not None and getattr(boxes, "xyxy", None) is not None:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else np.zeros((xyxy.shape[0],))
        clss = (
            boxes.cls.cpu().numpy().astype(int)
            if getattr(boxes, "cls", None) is not None
            else np.zeros((xyxy.shape[0],), dtype=int)
        )
        names = r0.names if hasattr(r0, "names") else getattr(model, "names", {})

        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss, strict=False):
            detections.append(
                {
                    "class_id": int(k),
                    "class_name": str(names.get(int(k), int(k))),
                    "confidence": float(c),
                    "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                }
            )

    detections.sort(key=lambda d: d["confidence"], reverse=True)

    out: dict[str, Any] = {"detections": detections}
    if return_image:
        plotted_bgr = r0.plot()  # np.ndarray, BGR
        plotted_rgb = plotted_bgr[:, :, ::-1]
        out_img = Image.fromarray(plotted_rgb)
        buf = BytesIO()
        out_img.save(buf, format="PNG")
        out["image_base64"] = base64.b64encode(buf.getvalue()).decode("utf-8")
    return out


def detect_image_bytes(
    image_bytes: bytes,
    *,
    weights_path: str | None = None,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    device: str = "auto",
    return_image: bool = False,
) -> dict[str, Any]:
    if not image_bytes:
        raise APIError("Image bytes are empty.")
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise APIError(f"Failed to decode image: {e}") from e
    return detect_pil(
        img,
        weights_path=weights_path,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        return_image=return_image,
    )
