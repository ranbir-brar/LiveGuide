import json
import random
import time
from collections import deque
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from models.llm import describe_scene_from_frame
from models.yolo import classify_hazard, detect_image_bytes, format_hazard_alert

RUNTIME_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "runtime_config.json"
_LLM_CALL_TIMES: deque[float] = deque()
_RNG: random.Random | None = None


def load_runtime_config() -> dict[str, Any]:
    if RUNTIME_CONFIG_PATH.exists():
        return json.loads(RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
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


def llm_send_probability(yolo_score: float, cfg: dict[str, Any]) -> float:
    """
    Bias outside scale:
    P = bias + (1 - bias) * scale01((yolo_score * constant) * probability_scale)
    """
    llm_cfg = cfg.get("llm", {})
    constant = float(llm_cfg.get("calls_per_second_constant", 0.8))
    score_scale = float(llm_cfg.get("probability_scale", 1.0))
    bias = _clamp01(float(llm_cfg.get("base_probability_bias", 0.1)))
    scaled_part = _clamp01(float(yolo_score) * constant * score_scale)
    return bias + (1.0 - bias) * scaled_part


def _llm_rate_allowed(cfg: dict[str, Any]) -> bool:
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


def should_send_to_llm(yolo_score: float, cfg: dict[str, Any]) -> tuple[bool, float, bool, bool]:
    global _RNG
    if _RNG is None:
        seed = int(cfg.get("llm", {}).get("rng_seed", 42))
        _RNG = random.Random(seed)
    probability = llm_send_probability(yolo_score, cfg)
    sampled = _RNG.random() < probability
    rate_allowed = _llm_rate_allowed(cfg) if sampled else False
    return sampled and rate_allowed, probability, sampled, rate_allowed


def preprocess_image(img_bytes: bytes, target_size: int = 256) -> bytes:
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    resized = img.resize((target_size, target_size), Image.Resampling.BILINEAR)
    buf = BytesIO()
    resized.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def run_image_pipeline(
    image_bytes: bytes,
    *,
    context: str = "walking",
    use_depth: bool = True,
    preprocess_size: int = 256,
    imgsz: int = 256,
    return_annotated: bool = True,
) -> dict[str, Any]:
    cfg = load_runtime_config()
    llm_cfg = cfg.get("llm", {})

    total_start = time.perf_counter()

    preprocess_start = time.perf_counter()
    processed_bytes = preprocess_image(image_bytes, target_size=preprocess_size)
    preprocess_ms = (time.perf_counter() - preprocess_start) * 1000

    yolo_start = time.perf_counter()
    yolo_result = detect_image_bytes(
        processed_bytes,
        return_image=return_annotated,
        imgsz=imgsz,
        use_depth=use_depth,
    )
    yolo_ms = (time.perf_counter() - yolo_start) * 1000
    detections = yolo_result["detections"]

    hazard_start = time.perf_counter()
    hazard = classify_hazard(detections, context=context)
    hazard_ms = (time.perf_counter() - hazard_start) * 1000
    hazard_alert = format_hazard_alert(hazard)

    yolo_score = float(hazard.get("hazard_score", 0.0))
    send_to_llm, llm_prob, llm_sampled, llm_rate_allowed = should_send_to_llm(yolo_score, cfg)
    llm_desc = None
    llm_error = None
    llm_ms = 0.0
    if send_to_llm:
        try:
            llm_start = time.perf_counter()
            llm_desc = describe_scene_from_frame(
                processed_bytes,
                detections,
                max_new_tokens=int(llm_cfg.get("max_new_tokens", 80)),
            )
            llm_ms = (time.perf_counter() - llm_start) * 1000
        except Exception as e:  # noqa: BLE001
            llm_error = str(e)

    total_ms = (time.perf_counter() - total_start) * 1000
    return {
        "detections": detections,
        "image_base64": yolo_result.get("image_base64"),
        "hazard": hazard,
        "hazard_alert": hazard_alert,
        "llm": {
            "sent": send_to_llm,
            "probability": round(llm_prob, 4),
            "sampled": llm_sampled,
            "rate_allowed": llm_rate_allowed,
            "description": llm_desc,
            "error": llm_error,
        },
        "latency": {
            "preprocess_ms": round(preprocess_ms, 1),
            "yolo_ms": round(yolo_ms, 1),
            "depth_ms": round(float(yolo_result.get("depth_ms", 0.0)), 1),
            "hazard_ms": round(hazard_ms, 1),
            "llm_ms": round(llm_ms, 1),
            "total_ms": round(total_ms, 1),
        },
    }
