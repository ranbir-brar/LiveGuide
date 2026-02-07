"""
Qwen2.5 Scene Description Module

Converts YOLO detection results into natural language scene descriptions.
Uses Qwen2.5-0.5B-Instruct for lightweight, fast inference.
"""

import threading
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Use the smallest Qwen2.5 model for speed
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

_model = None
_tokenizer = None
_load_lock = threading.Lock()


def _get_model_and_tokenizer():
    """Lazy-load the model and tokenizer (thread-safe, cached)."""
    global _model, _tokenizer
    
    if _model is None:
        with _load_lock:
            if _model is None:
                print(f"Loading {MODEL_ID}...")
                _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
                _model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True,
                )
                if not torch.cuda.is_available():
                    _model = _model.to("cpu")
                print(f"Model loaded on {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    return _model, _tokenizer


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


def describe_scene_from_detections(
    detections: list[dict[str, Any]],
    *,
    max_new_tokens: int = 60,
) -> str:
    """
    Generate a concise scene description from YOLO detections for visually impaired users.
    
    Args:
        detections: List of detection dicts with 'class_name', 'confidence', and 'xyxy' keys.
        max_new_tokens: Maximum tokens to generate.
    
    Returns:
        A brief, factual description of the scene.
    """
    detection_summary = _format_detections_for_prompt(detections)
    
    prompt = f"""Detected objects with positions: {detection_summary}

Describe what's in the scene in ONE short sentence. Only mention what's detected. Use spatial terms like "beside", "behind", "in front of", "to the left/right". Be direct and factual - no creative additions."""

    return describe_scene(prompt, max_new_tokens=max_new_tokens)


def describe_scene(
    prompt: str,
    *,
    max_new_tokens: int = 100,
) -> str:
    """
    Generate text using Qwen2.5 given a prompt.
    
    Args:
        prompt: The input prompt.
        max_new_tokens: Maximum tokens to generate.
    
    Returns:
        Generated text response.
    """
    model, tokenizer = _get_model_and_tokenizer()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides concise scene descriptions."},
        {"role": "user", "content": prompt},
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer([text], return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    
    return response.strip()
