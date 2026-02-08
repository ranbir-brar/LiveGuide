"""LLM router module with provider selection from runtime config."""

import json
from pathlib import Path
from typing import Any, Optional

import yaml

from models.llm import qwen

try:
    from models.llm import gemini
except Exception:  # noqa: BLE001
    gemini = None

_MODULE_DIR = Path(__file__).resolve().parent
_RUNTIME_CONFIG_YAML = _MODULE_DIR.parent.parent / "config" / "runtime_config.yaml"
_RUNTIME_CONFIG_JSON = _MODULE_DIR.parent.parent / "config" / "runtime_config.json"


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
    return {"llm": {"provider": "qwen_local"}}


def _resolve_provider(provider: Optional[str]) -> str:
    if provider and str(provider).strip():
        return str(provider).strip().lower()
    cfg = _load_runtime_config().get("llm", {})
    return str(cfg.get("provider", "qwen_local")).strip().lower()


def describe_scene(prompt: str, *, max_new_tokens: int = 80, provider: Optional[str] = None) -> str:
    p = _resolve_provider(provider)
    if p == "gemini_api":
        if gemini is None:
            raise RuntimeError("gemini_api provider requested but gemini backend is unavailable.")
        return gemini.describe_scene(prompt, max_new_tokens=max_new_tokens)
    return qwen.describe_scene(prompt, max_new_tokens=max_new_tokens)


def describe_scene_from_detections(
    detections: list[dict[str, Any]],
    *,
    max_new_tokens: int = 80,
    provider: Optional[str] = None,
) -> str:
    p = _resolve_provider(provider)
    if p == "gemini_api":
        if gemini is None:
            raise RuntimeError("gemini_api provider requested but gemini backend is unavailable.")
        return gemini.describe_scene_from_detections(detections, max_new_tokens=max_new_tokens)
    return qwen.describe_scene_from_detections(detections, max_new_tokens=max_new_tokens)


def describe_scene_from_frame(
    frame_bytes: bytes,
    detections: list[dict[str, Any]],
    *,
    max_new_tokens: int = 80,
    provider: Optional[str] = None,
) -> str:
    p = _resolve_provider(provider)
    if p == "gemini_api":
        if gemini is None:
            raise RuntimeError("gemini_api provider requested but gemini backend is unavailable.")
        return gemini.describe_scene_from_frame(frame_bytes, detections, max_new_tokens=max_new_tokens)
    return qwen.describe_scene_from_frame(frame_bytes, detections, max_new_tokens=max_new_tokens)


__all__ = ["describe_scene", "describe_scene_from_detections", "describe_scene_from_frame"]
