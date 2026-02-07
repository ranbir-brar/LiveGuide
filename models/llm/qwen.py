"""Qwen3-VL scene description module with local model download."""

import json
import threading
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from transformers import AutoProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration
except Exception:  # noqa: BLE001
    Qwen3VLForConditionalGeneration = None

try:
    from huggingface_hub import snapshot_download
except Exception:  # noqa: BLE001
    snapshot_download = None

_MODULE_DIR = Path(__file__).resolve().parent
_RUNTIME_CONFIG_YAML = _MODULE_DIR.parent.parent / "config" / "runtime_config.yaml"
_RUNTIME_CONFIG_JSON = _MODULE_DIR.parent.parent / "config" / "runtime_config.json"

_model = None
_processor = None
_load_lock = threading.Lock()


def _load_json_with_comments(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    out: list[str] = []
    i = 0
    in_string = False
    escape = False
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue

        if ch == "/" and nxt == "/":
            i += 2
            while i < len(text) and text[i] != "\n":
                i += 1
            continue

        if ch == "/" and nxt == "*":
            i += 2
            while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue

        out.append(ch)
        i += 1

    return json.loads("".join(out))


def _load_runtime_config() -> dict[str, Any]:
    if _RUNTIME_CONFIG_YAML.exists():
        data = yaml.safe_load(_RUNTIME_CONFIG_YAML.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    if _RUNTIME_CONFIG_JSON.exists():
        return _load_json_with_comments(_RUNTIME_CONFIG_JSON)
    return {
        "llm": {
            "model_id": "Qwen/Qwen3-VL-2B-Instruct",
            "local_model_dir": "models/llm/weights/Qwen3-VL-2B-Instruct",
            "max_new_tokens": 80,
            "temperature": 0.2,
            "top_p": 0.9,
        }
    }


def _resolve_model_paths() -> tuple[str, Path]:
    cfg = _load_runtime_config().get("llm", {})
    model_id = str(cfg.get("model_id", "Qwen/Qwen3-VL-2B-Instruct"))
    local_rel = str(cfg.get("local_model_dir", "models/llm/weights/Qwen3-VL-2B-Instruct"))
    local_dir = (_MODULE_DIR.parent.parent / local_rel).resolve()
    return model_id, local_dir


def _has_model_weights(local_dir: Path) -> bool:
    if not local_dir.exists():
        return False
    patterns = ("*.safetensors", "pytorch_model*.bin", "model*.safetensors", "*.index.json")
    for pat in patterns:
        if any(local_dir.glob(pat)):
            return True
    return False


def _ensure_local_model() -> Path:
    model_id, local_dir = _resolve_model_paths()
    if _has_model_weights(local_dir):
        return local_dir

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_id} to {local_dir} ...")
    if snapshot_download is not None:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
        )
    else:
        # Fallback download path through transformers cache if huggingface_hub is unavailable.
        AutoProcessor.from_pretrained(model_id)
        if Qwen3VLForConditionalGeneration is None:
            raise RuntimeError(
                "Qwen3-VL is not supported by current transformers version. "
                "Please upgrade: pip install -U git+https://github.com/huggingface/transformers"
            )
        Qwen3VLForConditionalGeneration.from_pretrained(model_id)
    return local_dir


def _get_model_and_processor():
    """Lazy-load model and processor (thread-safe, cached)."""
    global _model, _processor

    if _model is None:
        with _load_lock:
            if _model is None:
                if Qwen3VLForConditionalGeneration is None:
                    raise RuntimeError(
                        "Current transformers does not include Qwen3-VL architecture. "
                        "Please upgrade: pip install -U git+https://github.com/huggingface/transformers"
                    )
                local_dir = _ensure_local_model()
                print(f"Loading Qwen3-VL from {local_dir} ...")
                _processor = AutoProcessor.from_pretrained(local_dir, trust_remote_code=True)
                _model = Qwen3VLForConditionalGeneration.from_pretrained(
                    local_dir,
                    dtype="auto",
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                if not torch.cuda.is_available():
                    _model = _model.to("cpu")
                print(f"Qwen3-VL loaded on {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    return _model, _processor


def _format_detections_for_prompt(detections: list[dict[str, Any]], img_width: int = 640) -> str:
    """Format YOLO detections with spatial information for the LLM prompt."""
    if not detections:
        return "No objects detected."
    
    def get_horizontal_position(xyxy: list[float]) -> str:
        """Determine left/center/right based on bounding box center."""
        center_x = (xyxy[0] + xyxy[2]) / 2
        if center_x < img_width * 0.33:
            return "left"
        elif center_x > img_width * 0.66:
            return "right"
        return "center"
    
    def get_depth(xyxy: list[float]) -> str:
        """Estimate foreground/background based on box size (larger = closer)."""
        box_height = xyxy[3] - xyxy[1]
        if box_height > 200:
            return "foreground"
        elif box_height < 80:
            return "background"
        return "middle"
    
    # Format each detection with position
    items = []
    for det in sorted(detections, key=lambda d: -d.get("confidence", 0)):
        name = det.get("class_name", "unknown")
        xyxy = det.get("xyxy", [0, 0, 0, 0])
        h_pos = get_horizontal_position(xyxy)
        depth = get_depth(xyxy)
        items.append(f"{name} ({h_pos}, {depth})")
    
    return "; ".join(items)


def describe_scene_from_detections(detections: list[dict[str, Any]], *, max_new_tokens: int = 80) -> str:
    detection_summary = _format_detections_for_prompt(detections)

    prompt = (
        f"Detected objects with positions: {detection_summary}\n\n"
        "Describe the scene in ONE short factual sentence. Mention only detected objects and relative positions."
    )
    return describe_scene(prompt, max_new_tokens=max_new_tokens)


def _generate(messages: list[dict[str, Any]], image: Image.Image | None, max_new_tokens: int) -> str:
    model, processor = _get_model_and_processor()
    cfg = _load_runtime_config().get("llm", {})
    temperature = float(cfg.get("temperature", 0.2))
    top_p = float(cfg.get("top_p", 0.9))

    formatted_messages = messages
    if image is not None:
        formatted_messages = [
            {
                "role": m["role"],
                "content": [
                    {"type": "image", "image": image} if c.get("type") == "image" else c
                    for c in m.get("content", [])
                ],
            }
            for m in messages
        ]

    inputs = processor.apply_chat_template(
        formatted_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
        )

    generated_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
    ]
    return processor.batch_decode(
        generated_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def describe_scene(prompt: str, *, max_new_tokens: int = 80) -> str:
    messages = [
        {"role": "system", "content": "You are a concise scene description assistant."},
        {"role": "user", "content": prompt},
    ]
    return _generate(messages, image=None, max_new_tokens=max_new_tokens)


def describe_scene_from_frame(
    frame_bytes: bytes,
    detections: list[dict[str, Any]],
    *,
    max_new_tokens: int = 80,
) -> str:
    """Return safety text only when the frame is judged dangerous by the LLM."""
    img = Image.open(BytesIO(frame_bytes)).convert("RGB")
    detection_summary = _format_detections_for_prompt(detections, img_width=img.width)
    text_prompt = (
        "You are a safety judge for a pedestrian scene.\n"
        "Step 1: Decide if the current frame is dangerous right now (yes/no).\n"
        "Step 2: If dangerous=yes, provide one short factual warning sentence.\n"
        "Step 3: If dangerous=no, message must be empty.\n\n"
        f"Detections: {detection_summary}\n\n"
        "Return STRICT JSON only:\n"
        '{"dangerous":"yes|no","message":"..."}'
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
    raw = _generate(messages, image=img, max_new_tokens=max_new_tokens).strip()

    # Parse strict JSON response; keep robust fallback for minor format drift.
    obj: dict[str, Any] | None = None
    try:
        obj = json.loads(raw)
    except Exception:  # noqa: BLE001
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(raw[start : end + 1])
            except Exception:  # noqa: BLE001
                obj = None

    if isinstance(obj, dict):
        dangerous = str(obj.get("dangerous", "")).strip().lower()
        message = str(obj.get("message", "")).strip()
        if dangerous == "yes":
            return message
        return ""

    lower = raw.lower()
    if '"dangerous":"yes"' in lower or '"dangerous": "yes"' in lower:
        return raw
    return ""
