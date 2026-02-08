"""TTS module for LiveGuide."""

import json
from pathlib import Path
from typing import Any

from models.tts import piper, elevenlabs_tts

_MODULE_DIR = Path(__file__).resolve().parent
_RUNTIME_CONFIG_YAML = _MODULE_DIR.parent.parent / "config" / "runtime_config.yaml"

def _load_tts_config() -> dict[str, Any]:
    import yaml
    if _RUNTIME_CONFIG_YAML.exists():
        data = yaml.safe_load(_RUNTIME_CONFIG_YAML.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data.get("tts", {})
    return {}

def synthesize(text: str) -> bytes | None:
    cfg = _load_tts_config()
    provider = cfg.get("provider", "piper")
    
    if provider == "elevenlabs":
        # Request PCM 22050Hz raw bytes
        return elevenlabs_tts.synthesize(text, output_format="pcm_22050")
    else:
        # Piper returns WAV bytes
        return piper.synthesize(text)

def wav_bytes_to_numpy(audio_bytes: bytes) -> tuple[int, Any]:
    cfg = _load_tts_config()
    provider = cfg.get("provider", "piper")
    
    if provider == "elevenlabs":
        return elevenlabs_tts.wav_bytes_to_numpy(audio_bytes)
    else:
        return piper.wav_bytes_to_numpy(audio_bytes)

def speak(text: str, blocking: bool = True):
    # Fallback to local playback if needed (mostly for debugging)
    # This might need updates to handle ElevenLabs PCM locally if we ever use speak() directly
    # For now, let's just use piper for local speak debugging or implement if needed.
    # But Gradio uses synthesize() + wav_bytes_to_numpy(), so this is less critical.
    print("[TTS] speak() called - local playback not fully supported for ElevenLabs yet")
    piper.speak(text, blocking=blocking)

__all__ = ["speak", "synthesize", "wav_bytes_to_numpy"]
