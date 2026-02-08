"""Gemini API backend for scene description."""

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

_MODULE_DIR = Path(__file__).resolve().parent
_RUNTIME_CONFIG_YAML = _MODULE_DIR.parent.parent / "config" / "runtime_config.yaml"


def _load_runtime_config() -> dict[str, Any]:
    if _RUNTIME_CONFIG_YAML.exists():
        data = yaml.safe_load(_RUNTIME_CONFIG_YAML.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    return {"llm": {"gemini_model": "gemini-2.0-flash", "gemini_api_key_env": "GEMINI_API_KEY"}}


def _format_detections_for_prompt(detections: list[dict[str, Any]], img_width: int = 640) -> str:
    if not detections:
        return "No objects detected."

    def get_horizontal_position(xyxy: list[float]) -> str:
        center_x = (xyxy[0] + xyxy[2]) / 2
        if center_x < img_width * 0.33:
            return "left"
        if center_x > img_width * 0.66:
            return "right"
        return "center"

    def get_depth(xyxy: list[float]) -> str:
        box_height = xyxy[3] - xyxy[1]
        if box_height > 200:
            return "foreground"
        if box_height < 80:
            return "background"
        return "middle"

    items = []
    for det in sorted(detections, key=lambda d: -d.get("confidence", 0)):
        name = det.get("class_name", "unknown")
        xyxy = det.get("xyxy", [0, 0, 0, 0])
        h_pos = get_horizontal_position(xyxy)
        depth = get_depth(xyxy)
        items.append(f"{name} ({h_pos}, {depth})")
    return "; ".join(items)


def _extract_text_from_response(resp_obj: dict[str, Any]) -> str:
    candidates = resp_obj.get("candidates", [])
    parts = []
    for cand in candidates:
        content = cand.get("content", {})
        for part in content.get("parts", []):
            txt = part.get("text")
            if txt:
                parts.append(str(txt))
    return "\n".join(parts).strip()


def _call_gemini(*, text_prompt: str, image_bytes: bytes | None, max_new_tokens: int) -> str:
    llm_cfg = _load_runtime_config().get("llm", {})
    model = str(llm_cfg.get("gemini_model", "gemini-2.0-flash"))
    api_key_env = str(llm_cfg.get("gemini_api_key_env", "GEMINI_API_KEY"))
    api_key = str(llm_cfg.get("gemini_api_key", "")).strip() or os.getenv(api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Gemini API key is missing. Set `{api_key_env}` or llm.gemini_api_key.")

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    parts: list[dict[str, Any]] = []
    if image_bytes is not None:
        parts.append(
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(image_bytes).decode("utf-8"),
                }
            }
        )
    parts.append({"text": text_prompt})

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"maxOutputTokens": int(max_new_tokens)},
    }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Gemini API HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Gemini API connection failed: {e}") from e

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Gemini API returned non-JSON response: {body[:300]}") from e

    return _extract_text_from_response(parsed)


def describe_scene(prompt: str, *, max_new_tokens: int = 80) -> str:
    return _call_gemini(text_prompt=prompt, image_bytes=None, max_new_tokens=max_new_tokens)


def describe_scene_from_detections(detections: list[dict[str, Any]], *, max_new_tokens: int = 80) -> str:
    detection_summary = _format_detections_for_prompt(detections)
    prompt = (
        f"Detected objects with positions: {detection_summary}\n\n"
        "Describe the scene in ONE short factual sentence. Mention only detected objects and relative positions."
    )
    return describe_scene(prompt, max_new_tokens=max_new_tokens)


def describe_scene_from_frame(
    frame_bytes: bytes,
    detections: list[dict[str, Any]],
    *,
    max_new_tokens: int = 80,
) -> str:
    detection_summary = _format_detections_for_prompt(detections)
    text_prompt = (
        "You are a live navigation assistant for a blind person walking.\n"
        "Give a SHORT, ACTIONABLE description of what's ahead. Be specific about:\n"
        "- Direction: left, right, ahead, approaching\n"
        "- Distance: close, nearby, ahead\n"
        "- Action needed: watch out, clear path, step aside\n\n"
        f"Objects detected: {detection_summary}\n\n"
        "Respond in ONE short sentence like a guide dog would alert. Examples:\n"
        "- 'Person approaching on your left.'\n"
        "- 'Clear path ahead, chair on right.'\n"
        "- 'Car nearby on left, stay right.'\n"
    )
    return _call_gemini(text_prompt=text_prompt, image_bytes=frame_bytes, max_new_tokens=max_new_tokens).strip()
