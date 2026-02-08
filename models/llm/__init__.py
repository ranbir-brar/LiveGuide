"""LLM router module with provider selection from runtime config."""

from pathlib import Path
from typing import Any

import yaml

from models.llm import qwen

try:
    from models.llm import gemini
except Exception:  # noqa: BLE001
    gemini = None

_MODULE_DIR = Path(__file__).resolve().parent
_RUNTIME_CONFIG_YAML = _MODULE_DIR.parent.parent / "config" / "runtime_config.yaml"


def _load_runtime_config() -> dict[str, Any]:
    if _RUNTIME_CONFIG_YAML.exists():
        data = yaml.safe_load(_RUNTIME_CONFIG_YAML.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    return {"llm": {"provider": "qwen_local"}}


def _resolve_provider(provider: str | None) -> str:
    if provider and str(provider).strip():
        return str(provider).strip().lower()
    cfg = _load_runtime_config().get("llm", {})
    return str(cfg.get("provider", "qwen_local")).strip().lower()


def describe_scene(prompt: str, *, max_new_tokens: int = 80, provider: str | None = None) -> str:
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
    provider: str | None = None,
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
    provider: str | None = None,
) -> str:
    p = _resolve_provider(provider)
    if p == "gemini_api":
        if gemini is None:
            raise RuntimeError("gemini_api provider requested but gemini backend is unavailable.")
        return gemini.describe_scene_from_frame(frame_bytes, detections, max_new_tokens=max_new_tokens)
    return qwen.describe_scene_from_frame(frame_bytes, detections, max_new_tokens=max_new_tokens)


__all__ = ["describe_scene", "describe_scene_from_detections", "describe_scene_from_frame"]
