"""LLM module for scene description using Qwen2.5."""

from models.llm.qwen import describe_scene, describe_scene_from_detections

__all__ = ["describe_scene", "describe_scene_from_detections"]
