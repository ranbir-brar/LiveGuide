"""ElevenLabs TTS module for LiveGuide."""

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import yaml

_MODULE_DIR = Path(__file__).resolve().parent
_RUNTIME_CONFIG = _MODULE_DIR.parent.parent / "config" / "runtime_config.yaml"
_CACHE_LOCK = threading.Lock()
_AUDIO_CACHE: dict[str, bytes] = {}


def _load_tts_config() -> dict[str, Any]:
    if _RUNTIME_CONFIG.exists():
        data = yaml.safe_load(_RUNTIME_CONFIG.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            tts_cfg = data.get("tts", {})
            if isinstance(tts_cfg, dict):
                return tts_cfg
    return {}


def _api_settings() -> tuple[str, str, str, str, int]:
    cfg = _load_tts_config()
    api_key = str(cfg.get("elevenlabs_api_key", "")).strip()
    api_key_env = str(cfg.get("elevenlabs_api_key_env", "ELEVENLABS_API_KEY")).strip()
    if not api_key and api_key_env:
        import os

        api_key = os.getenv(api_key_env, "").strip()

    voice_id = str(cfg.get("voice_id", "EXAVITQu4vr4xnSDxMaL")).strip()
    model_id = str(cfg.get("model_id", "eleven_multilingual_v2")).strip()
    output_format = str(cfg.get("output_format", "pcm_22050")).strip()
    timeout = int(cfg.get("request_timeout_sec", 30))

    return api_key, voice_id, model_id, output_format, timeout


def synthesize(text: str, *, use_cache: bool = True) -> bytes:
    text = str(text or "").strip()
    if not text:
        return b""

    if use_cache:
        with _CACHE_LOCK:
            cached = _AUDIO_CACHE.get(text)
            if cached is not None:
                return cached

    api_key, voice_id, model_id, output_format, timeout = _api_settings()
    if not api_key:
        raise RuntimeError(
            "ElevenLabs API key is missing. Set `tts.elevenlabs_api_key` in runtime_config.yaml "
            "or environment variable defined by `tts.elevenlabs_api_key_env`."
        )

    query = urllib.parse.urlencode({"output_format": output_format})
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?{query}"
    payload = {"text": text, "model_id": model_id}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "xi-api-key": api_key,
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            audio = resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"ElevenLabs API HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"ElevenLabs API connection failed: {e}") from e

    if use_cache:
        with _CACHE_LOCK:
            _AUDIO_CACHE[text] = audio
    return audio


def speak(text: str, *, blocking: bool = True) -> None:
    """Call ElevenLabs TTS on every VLM response and optionally play audio locally."""
    audio = synthesize(text)
    if not audio:
        return

    cfg = _load_tts_config()
    if not bool(cfg.get("enable_playback", False)):
        return

    output_format = str(cfg.get("output_format", "pcm_22050")).strip().lower()
    if not output_format.startswith("pcm_"):
        return

    try:
        sample_rate = int(output_format.split("_", 1)[1])
    except Exception:
        sample_rate = 22050

    try:
        import numpy as np
        import sounddevice as sd
    except Exception:
        return

    pcm = np.frombuffer(audio, dtype=np.int16)
    sd.play(pcm, samplerate=sample_rate)
    if blocking:
        sd.wait()


def cache_size() -> int:
    with _CACHE_LOCK:
        return len(_AUDIO_CACHE)
