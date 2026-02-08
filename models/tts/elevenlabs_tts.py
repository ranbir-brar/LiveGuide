"""ElevenLabs TTS module for LiveGuide."""

import io
import threading
import wave
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Lazy load client to avoid slow startup
_client = None
_client_lock = threading.Lock()

print("[ElevenLabs] Module loaded (v2 - client.text_to_speech.convert)")

_MODULE_DIR = Path(__file__).resolve().parent
_RUNTIME_CONFIG_YAML = _MODULE_DIR.parent.parent / "config" / "runtime_config.yaml"

# Audio cache (text -> pcm/wav bytes) to save credits
_audio_cache: dict[str, bytes] = {}
_audio_cache_lock = threading.Lock()


def _load_runtime_config() -> dict[str, Any]:
    import yaml
    if _RUNTIME_CONFIG_YAML.exists():
        data = yaml.safe_load(_RUNTIME_CONFIG_YAML.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    return {}


def _get_client_and_config():
    global _client
    cfg = _load_runtime_config()
    tts_cfg = cfg.get("tts", {}).get("elevenlabs", {})
    api_key = tts_cfg.get("api_key")
    
    if _client is None:
        with _client_lock:
            if _client is None:
                try:
                    from elevenlabs.client import ElevenLabs
                except ImportError as e:
                    print(f"[ElevenLabs] Error loading module: {e}")
                    raise ImportError(
                        "elevenlabs package not found. pip install elevenlabs"
                    ) from e
                
                if not api_key:
                    print("[ElevenLabs] Warning: No API key found in config.")
                    return None, {}
                
                _client = ElevenLabs(api_key=api_key)
                
    return _client, tts_cfg


def synthesize(text: str, output_format: str = "pcm_22050") -> bytes | None:
    """Synthesize text using ElevenLabs API (returns PCM bytes by default)."""
    if not text or not text.strip():
        return None

    client, cfg = _get_client_and_config()
    if client is None:
        return None

    # Check cache
    cache_key = f"{text}|{output_format}"
    with _audio_cache_lock:
        if cache_key in _audio_cache:
            return _audio_cache[cache_key]

    voice_id = cfg.get("voice_id", "21m00Tcm4TlvDq8ikWAM")  # Rachel

    try:
        # Generate audio generator (streaming response) - Correct API method for 1.x SDK
        # The method is usually client.text_to_speech.convert or client.generate (which might be legacy or async)
        # Standard client usage per docs:
        # audio = client.text_to_speech.convert(
        #     voice_id="...",
        #     output_format="...",
        #     text="..."
        # )
        
        # Try the modern method first
        if hasattr(client, "text_to_speech") and hasattr(client.text_to_speech, "convert"):
            audio_generator = client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_monolingual_v1",
                output_format=output_format
            )
        else:
            # Fallback to older .generate() if it exists (though error suggested it doesn't on 'ElevenLabs' object)
            # Maybe it is on the module level? "from elevenlabs import generate"
            # But we are using the client object. 
            # If the client object form `from elevenlabs.client import ElevenLabs` is used, it should have .text_to_speech
            
            # Let's try likely 1.x method
             audio_generator = client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_monolingual_v1",
                output_format=output_format
            )
        
        # Consume generator to get all bytes
        audio_bytes = b"".join(chunk for chunk in audio_generator)
        
        if not audio_bytes:
            return None

        # Cache result
        with _audio_cache_lock:
            _audio_cache[cache_key] = audio_bytes
            
        return audio_bytes
    except Exception as e:
        print(f"[ElevenLabs] Error synthesizing: {e}")
        # Detailed debugging
        try:
             print(f"[ElevenLabs Debug] client dirs: {dir(client)}")
        except:
             pass
        return None


def wav_bytes_to_numpy(audio_bytes: bytes) -> tuple[int, Any]:
    """
    Convert RAW PCM bytes (from ElevenLabs pcm_22050) to (sample_rate, numpy_array).
    Unlike Piper (which returns WAV), ElevenLabs returns raw PCM or MP3.
    We assume 'pcm_22050' format for simplicity and consistency.
    """
    # Sample rate from the format name
    sample_rate = 22050 
    
    # Raw PCM 16-bit little-endian mono is standard for ElevenLabs pcm_* formats
    dtype = np.int16
    
    try:
        audio_data = np.frombuffer(audio_bytes, dtype=dtype)
        return sample_rate, audio_data
    except Exception as e:
        print(f"[ElevenLabs] Error converting PCM to numpy: {e}")
        return sample_rate, np.zeros(0, dtype=np.int16)
