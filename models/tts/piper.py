"""Piper TTS module for LiveGuide.

Fast, offline text-to-speech using Piper neural TTS.
Auto-downloads voice model on first use.
"""

import io
import json
import threading
import wave
from pathlib import Path
from typing import Any, Optional

# Lazy imports to avoid slow startup
_piper = None
_sounddevice = None

_MODULE_DIR = Path(__file__).resolve().parent
_WEIGHTS_DIR = _MODULE_DIR / "weights"
_RUNTIME_CONFIG = _MODULE_DIR.parent.parent / "config" / "runtime_config.json"

_voice = None
_voice_lock = threading.Lock()

# Audio cache for common phrases (text -> WAV bytes)
_audio_cache: dict[str, bytes] = {}
_audio_cache_lock = threading.Lock()

# Common phrases to pre-cache
COMMON_PHRASES = [
    "Path is clear.",
    "Caution advised.",
    "WARNING: Obstacle detected.",
]

# Default voice model (fast, clear US English)
DEFAULT_MODEL = "en_US-lessac-medium"
MODEL_URLS = {
    "en_US-lessac-medium": (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
    ),
    "en_US-amy-medium": (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
    ),
    "en_GB-alan-medium": (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json",
    ),
}


def _load_runtime_config() -> dict[str, Any]:
    """Load TTS config from runtime_config.json."""
    if _RUNTIME_CONFIG.exists():
        cfg = json.loads(_RUNTIME_CONFIG.read_text(encoding="utf-8"))
        return cfg.get("tts", {})
    return {}


def _download_file(url: str, dest: Path) -> None:
    """Download a file from URL to destination."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest.name}...")
    req = urllib.request.Request(url, headers={"User-Agent": "LiveGuide-TTS/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())
    print(f"Downloaded {dest.name}")


def _ensure_model(model_name: str) -> Path:
    """Ensure voice model is downloaded and return path to .onnx file."""
    onnx_path = _WEIGHTS_DIR / f"{model_name}.onnx"
    json_path = _WEIGHTS_DIR / f"{model_name}.onnx.json"

    if not onnx_path.exists() or not json_path.exists():
        if model_name not in MODEL_URLS:
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {list(MODEL_URLS.keys())}"
            )
        onnx_url, json_url = MODEL_URLS[model_name]
        _download_file(onnx_url, onnx_path)
        _download_file(json_url, json_path)

    return onnx_path


def _get_voice():
    """Get or create the Piper voice instance (thread-safe, cached)."""
    global _voice, _piper

    if _voice is None:
        with _voice_lock:
            if _voice is None:
                # Lazy import
                if _piper is None:
                    from piper import PiperVoice
                    _piper = PiperVoice

                cfg = _load_runtime_config()
                model_name = cfg.get("model_name", DEFAULT_MODEL)
                model_path = _ensure_model(model_name)

                print(f"Loading Piper voice: {model_name}")
                _voice = _piper.load(str(model_path))
                print("Piper TTS ready")

    return _voice


def preload_cache() -> None:
    """Pre-synthesize common phrases to cache. Call at startup for faster responses."""
    for phrase in COMMON_PHRASES:
        if phrase not in _audio_cache:
            synthesize(phrase, use_cache=True)


def synthesize(
    text: str,
    *,
    speaker_id: Optional[int] = None,
    length_scale: Optional[float] = None,
    sentence_silence: Optional[float] = None,
    use_cache: bool = True,
) -> bytes:
    """
    Synthesize text to WAV audio bytes.

    Args:
        text: Text to speak.
        speaker_id: Speaker ID for multi-speaker models (default: from config or 0).
        length_scale: Speed factor (1.0 = normal, <1.0 = faster, >1.0 = slower).
        sentence_silence: Seconds of silence between sentences.
        use_cache: If True, use cached audio if available (default: True).

    Returns:
        WAV audio as bytes.
    """
    if not text or not text.strip():
        return b""
    
    # Check cache first
    text_key = text.strip()
    if use_cache and text_key in _audio_cache:
        return _audio_cache[text_key]

    voice = _get_voice()
    cfg = _load_runtime_config()

    # Use config defaults if not specified
    if speaker_id is None:
        speaker_id = cfg.get("speaker_id", 0)
    if length_scale is None:
        length_scale = cfg.get("length_scale", 1.0)
    if sentence_silence is None:
        sentence_silence = cfg.get("sentence_silence", 0.2)

    # Collect audio chunks from the iterator
    audio_bytes_list = []
    sample_rate = 22050  # default
    for audio_chunk in voice.synthesize(text):
        audio_bytes_list.append(audio_chunk.audio_int16_bytes)
        sample_rate = audio_chunk.sample_rate

    if not audio_bytes_list:
        return b""

    # Combine all audio data
    combined_audio = b"".join(audio_bytes_list)

    # Write to WAV format
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(combined_audio)

    wav_bytes = buffer.getvalue()
    
    # Store in cache
    if use_cache:
        with _audio_cache_lock:
            _audio_cache[text_key] = wav_bytes

    return wav_bytes


def wav_bytes_to_numpy(wav_bytes: bytes) -> tuple[int, Any]:
    """Convert WAV bytes to (sample_rate, numpy_array)."""
    import numpy as np

    buffer = io.BytesIO(wav_bytes)
    with wave.open(buffer, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width == 2:
        dtype = np.int16
    elif sample_width == 1:
        dtype = np.uint8
    else:
        dtype = np.int32

    audio_data = np.frombuffer(frames, dtype=dtype)
    if n_channels > 1:
        audio_data = audio_data.reshape(-1, n_channels)
    
    return sample_rate, audio_data


def speak(
    text: str,
    *,
    speaker_id: Optional[int] = None,
    length_scale: Optional[float] = None,
    sentence_silence: Optional[float] = None,
    blocking: bool = True,
) -> None:
    """
    Speak text through the default audio output.

    Args:
        text: Text to speak.
        speaker_id: Speaker ID for multi-speaker models.
        length_scale: Speed factor (1.0 = normal).
        sentence_silence: Seconds of silence between sentences.
        blocking: If True, wait for audio to finish. If False, play asynchronously.
    """
    global _sounddevice

    if not text or not text.strip():
        return

    wav_bytes = synthesize(
        text,
        speaker_id=speaker_id,
        length_scale=length_scale,
        sentence_silence=sentence_silence,
    )

    if not wav_bytes:
        return

    # Lazy import sounddevice
    if _sounddevice is None:
        import sounddevice as sd
        _sounddevice = sd

    sample_rate, audio_data = wav_bytes_to_numpy(wav_bytes)

    # Play audio
    _sounddevice.play(audio_data, samplerate=sample_rate)
    if blocking:
        _sounddevice.wait()
