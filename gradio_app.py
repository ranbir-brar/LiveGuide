import atexit
import base64
import html
import json
import re
import threading
import time
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Generator

import cv2
import gradio as gr
from gradio import processing_utils
from PIL import Image

from main import FrameSequencePipeline, load_runtime_config
from models.tts import synthesize

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
SAMPLE_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DEFAULT_SAMPLE_VIDEO_NAME = "Manhattan POV Walking Park Avenue New York City.mp4"


@dataclass
class SessionContext:
    pipeline: FrameSequencePipeline
    lock: threading.Lock
    event_log: Path
    last_active_ts: float
    runtime_detection_sensitivity: float


_SESSIONS_LOCK = threading.Lock()
_SESSIONS: dict[str, SessionContext] = {}
_SESSION_CLEANUP_STOP = threading.Event()

HIDDEN_TTS_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&family=Sora:wght@400;600;700;800&display=swap');

:root {
  --lg-bg-0: #f2f6fb;
  --lg-bg-1: #e6efff;
  --lg-bg-2: #fff4df;
  --lg-surface: rgba(255, 255, 255, 0.78);
  --lg-surface-strong: rgba(255, 255, 255, 0.92);
  --lg-border: rgba(17, 24, 39, 0.12);
  --lg-shadow: 0 24px 60px rgba(15, 23, 42, 0.12);
  --lg-text: #0f172a;
  --lg-text-soft: #475569;
  --lg-primary: #0f766e;
  --lg-primary-2: #2563eb;
}

:root,
html,
body,
.gradio-container,
html.dark,
body.dark,
.dark,
[data-theme="dark"] {
  color-scheme: light !important;
}

html.dark,
body.dark,
.dark,
[data-theme="dark"] {
  --body-background-fill: #f8fbff !important;
  --body-background-fill-subdued: #f2f6fb !important;
  --body-text-color: #0f172a !important;
  --body-text-color-subdued: #475569 !important;
  --block-background-fill: rgba(255, 255, 255, 0.86) !important;
  --block-border-color: rgba(17, 24, 39, 0.12) !important;
  --input-background-fill: #ffffff !important;
  --input-border-color: rgba(17, 24, 39, 0.2) !important;
  --button-secondary-background-fill: rgba(255, 255, 255, 0.9) !important;
  --button-secondary-text-color: #0f172a !important;
}

body {
  background:
    radial-gradient(80rem 40rem at -10% -20%, var(--lg-bg-1) 0%, transparent 60%),
    radial-gradient(60rem 30rem at 110% -10%, var(--lg-bg-2) 0%, transparent 65%),
    linear-gradient(180deg, #f8fbff 0%, var(--lg-bg-0) 100%);
}

html.dark body,
body.dark,
.dark body,
[data-theme="dark"] body {
  background:
    radial-gradient(80rem 40rem at -10% -20%, var(--lg-bg-1) 0%, transparent 60%),
    radial-gradient(60rem 30rem at 110% -10%, var(--lg-bg-2) 0%, transparent 65%),
    linear-gradient(180deg, #f8fbff 0%, var(--lg-bg-0) 100%);
}

.gradio-container {
  max-width: 1220px !important;
  margin: 0 auto !important;
  padding: 22px 20px 36px !important;
  color: var(--lg-text);
  font-family: "Manrope", "Segoe UI", sans-serif;
}

#app-shell {
  gap: 16px;
}

.glass-card {
  background: var(--lg-surface);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid var(--lg-border);
  border-radius: 22px;
  box-shadow: var(--lg-shadow);
  padding: 14px 16px;
}

#app-header {
  align-items: stretch;
  column-gap: 14px;
  margin-bottom: 4px;
}

#app-logo-col {
  display: flex;
  align-items: center;
  justify-content: center;
}

#app-logo {
  width: clamp(168px, 16vw, 230px) !important;
  height: 100% !important;
  margin: 0 auto;
  display: flex;
  align-items: stretch;
  justify-content: center;
}

#app-logo img {
  width: 100%;
  height: 100%;
  max-width: none !important;
  max-height: none !important;
  min-height: 180px;
  max-height: 252px;
  display: block;
  object-fit: contain;
  aspect-ratio: 1 / 1;
  border-radius: 18px;
  background: #ffffff;
  padding: 8px;
  border: 1px solid rgba(15, 23, 42, 0.08);
  box-shadow: 0 14px 28px rgba(15, 23, 42, 0.14);
}

#app-title-block {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

#app-title-block .eyebrow {
  margin: 0;
  font-family: "Sora", "Segoe UI", sans-serif;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--lg-primary-2);
}

#app-title-block h1 {
  margin: 0;
  font-family: "Sora", "Segoe UI", sans-serif;
  font-weight: 800;
  font-size: clamp(2rem, 4.2vw, 3.3rem);
  line-height: 1.08;
  letter-spacing: 0.01em;
  background: linear-gradient(120deg, #0f172a 0%, #1d4ed8 48%, #0f766e 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

#app-title-block .lead {
  margin: 0;
  color: var(--lg-text-soft);
  font-size: 0.98rem;
  line-height: 1.55;
  max-width: 68ch;
}

#capability-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 3px;
}

#capability-badges .badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  border: 1px solid rgba(15, 23, 42, 0.1);
  border-radius: 999px;
  padding: 7px 14px;
  background: var(--lg-surface-strong);
  color: #0f172a;
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 0.02em;
}

#control-strip {
  align-items: end;
  gap: 12px;
  overflow: visible !important;
}

#control-strip > div {
  padding-top: 2px;
}

/* Gradio dropdown options use position: fixed; backdrop-filter on parent
   can break popup positioning in some browsers, so disable it here only. */
#control-strip.glass-card {
  backdrop-filter: none !important;
  -webkit-backdrop-filter: none !important;
  background: rgba(255, 255, 255, 0.9);
}

#provider-select label,
#sensitivity-slider label,
#tts-toggle label,
#webcam-llm label,
#video-llm label,
#video-progress label {
  color: #0f172a !important;
  font-weight: 700 !important;
}

#provider-select,
#sensitivity-slider,
#tts-toggle,
#webcam-llm,
#video-llm,
#video-progress,
#video-input,
#video-frame,
#webcam-yolo,
#video-yolo {
  border-radius: 16px;
}

#provider-select select,
#sensitivity-slider input,
#video-progress textarea,
#webcam-llm textarea,
#video-llm textarea {
  border-radius: 12px !important;
}

#provider-select [role="combobox"],
#provider-select input,
#provider-select button {
  background: rgba(255, 255, 255, 0.95) !important;
  border: 1px solid rgba(15, 23, 42, 0.2) !important;
  color: #0f172a !important;
  font-weight: 700 !important;
  min-height: 44px !important;
  border-radius: 12px !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.88);
}

#provider-select [role="combobox"]:focus-within,
#provider-select input:focus,
#provider-select button:focus {
  border-color: rgba(37, 99, 235, 0.72) !important;
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.18) !important;
}

#provider-select svg {
  color: #1d4ed8 !important;
}

#provider-select [role="listbox"],
#provider-select ul {
  border-radius: 12px !important;
  border: 1px solid rgba(15, 23, 42, 0.16) !important;
  background: #ffffff !important;
  box-shadow: 0 16px 30px rgba(15, 23, 42, 0.18) !important;
}

/* Keep video-related errors readable in forced light mode. */
#video-input .wrap.translucent {
  background: rgba(255, 255, 255, 0.96) !important;
}

#video-input .wrap .error {
  color: #b91c1c !important;
  font-weight: 800 !important;
}

[data-testid="toast-body"].error {
  border-color: #dc2626 !important;
  background: #fef2f2 !important;
}

[data-testid="toast-body"] .toast-title.error,
[data-testid="toast-body"] .toast-text.error,
[data-testid="toast-body"] .toast-close.error,
[data-testid="toast-body"] .toast-icon.error {
  color: #b91c1c !important;
}

#main-tabs {
  margin-top: 2px;
}

#main-tabs .tab-nav {
  background: transparent;
  gap: 8px;
  padding: 2px;
}

#main-tabs .tab-nav button {
  border-radius: 999px !important;
  border: 1px solid rgba(15, 23, 42, 0.12) !important;
  background: rgba(255, 255, 255, 0.56) !important;
  color: #1e293b !important;
  font-weight: 700 !important;
  padding: 8px 14px !important;
}

#main-tabs .tab-nav button.selected {
  background: linear-gradient(120deg, #0f766e 0%, #2563eb 100%) !important;
  color: #f8fafc !important;
  border-color: transparent !important;
  box-shadow: 0 10px 24px rgba(37, 99, 235, 0.32);
}

#webcam-section,
#video-section {
  gap: 12px;
}

#webcam-wrap {
  position: relative;
  border-radius: 18px;
  overflow: hidden;
  border: 1px solid rgba(15, 23, 42, 0.12);
  background: rgba(255, 255, 255, 0.7);
}

#webcam-wrap img,
#video-frame img,
#video-input video {
  border-radius: 16px !important;
}

#webcam-toolbar {
  justify-content: flex-end;
}

#webcam-flip-btn {
  position: absolute;
  top: 12px;
  right: 12px;
  z-index: 12;
  transform: none;
  width: auto !important;
  min-width: 0 !important;
}

#webcam-flip-btn button {
  width: auto !important;
  min-width: 0 !important;
  padding: 7px 12px !important;
  border-radius: 999px !important;
  border: 1px solid rgba(15, 23, 42, 0.2) !important;
  background: rgba(255, 255, 255, 0.92) !important;
  color: #0f172a !important;
  font-weight: 700 !important;
}

#process-btn button {
  background: linear-gradient(120deg, #1d4ed8 0%, #0f766e 100%) !important;
  color: #f8fafc !important;
  border: 0 !important;
  border-radius: 14px !important;
  min-height: 44px !important;
  font-weight: 700 !important;
  letter-spacing: 0.01em;
  box-shadow: 0 14px 26px rgba(37, 99, 235, 0.28);
}

#video-input-row,
#video-output-row {
  gap: 12px;
}

#video-helper p {
  margin-top: 0;
  margin-bottom: 8px;
  color: var(--lg-text-soft);
}

.yolo-panel {
  border: 1px solid rgba(15, 23, 42, 0.14);
  border-radius: 14px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.92) 0%, rgba(248, 250, 252, 0.76) 100%);
  padding: 12px;
}

.yolo-panel__head {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 10px;
}

.yolo-panel__title {
  font-weight: 800;
  color: #0f172a;
  letter-spacing: 0.01em;
}

.yolo-panel__meta {
  font-family: "Sora", "Segoe UI", sans-serif;
  font-size: 0.8rem;
  color: #334155;
  background: rgba(226, 232, 240, 0.74);
  border: 1px solid rgba(148, 163, 184, 0.4);
  border-radius: 999px;
  padding: 4px 10px;
}

.yolo-panel__chips {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.det-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border-radius: 999px;
  padding: 5px 10px;
  border: 1px solid rgba(30, 64, 175, 0.26);
  background: rgba(219, 234, 254, 0.68);
  color: #1e3a8a;
  font-weight: 700;
  font-size: 0.8rem;
}

.det-name {
  max-width: 18ch;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.det-conf {
  font-family: "Sora", "Segoe UI", sans-serif;
  font-size: 0.75rem;
  color: #0f766e;
  background: rgba(204, 251, 241, 0.9);
  border-radius: 999px;
  padding: 1px 6px;
  border: 1px solid rgba(15, 118, 110, 0.25);
}

.det-empty {
  display: inline-flex;
  border-radius: 10px;
  padding: 6px 10px;
  font-weight: 700;
  color: #334155;
  background: rgba(226, 232, 240, 0.62);
}

#webcam-llm textarea,
#video-llm textarea {
  min-height: 110px !important;
  line-height: 1.52 !important;
  font-size: 0.96rem !important;
}

#video-progress textarea {
  min-height: 44px !important;
  line-height: 1.3 !important;
  font-family: "Sora", "Segoe UI", sans-serif !important;
  font-weight: 600 !important;
}

.tts-hidden {
  height: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  border: 0 !important;
  opacity: 0 !important;
  overflow: hidden !important;
}

.tts-hidden audio {
  height: 0 !important;
}

@media (max-width: 1120px) {
  #app-logo-col,
  #app-right-spacer {
    display: none !important;
  }
}

@media (max-width: 980px) {
  .gradio-container {
    padding: 16px 12px 24px !important;
  }

  #app-header {
    align-items: flex-start;
  }

  #app-title-block h1 {
    font-size: clamp(1.7rem, 7.3vw, 2.4rem);
  }

  #control-strip {
    flex-direction: column;
    align-items: stretch;
  }

  #video-input-row,
  #video-output-row {
    flex-direction: column;
  }

  #webcam-flip-btn {
    top: 10px;
    right: 10px;
  }
}
"""

FORCE_LIGHT_MODE_JS = """
() => {
  const root = document.documentElement;
  const body = document.body;
  const applyLightTheme = () => {
    if (root.classList.contains("dark")) root.classList.remove("dark");
    if (body && body.classList.contains("dark")) body.classList.remove("dark");
    if (root.getAttribute("data-theme") !== "light") root.setAttribute("data-theme", "light");
    if (body && body.getAttribute("data-theme") !== "light") body.setAttribute("data-theme", "light");
  };

  applyLightTheme();

  try {
    localStorage.setItem("theme", "light");
    localStorage.setItem("__theme", "light");
    localStorage.setItem("gradio-theme", "light");
  } catch (e) {
    // Ignore storage errors in restricted browser contexts.
  }

  const observer = new MutationObserver(() => applyLightTheme());
  observer.observe(root, { attributes: true, attributeFilter: ["class", "data-theme"] });
  if (body) observer.observe(body, { attributes: true, attributeFilter: ["class", "data-theme"] });
}
"""


def _new_pipeline() -> FrameSequencePipeline:
    gradio_cfg = load_runtime_config().get("gradio", {})
    if not isinstance(gradio_cfg, dict):
        gradio_cfg = {}
    llm_worker_mode = str(gradio_cfg.get("llm_worker_mode", "thread")).strip().lower()
    if llm_worker_mode not in {"thread", "process"}:
        llm_worker_mode = "thread"
    result_buffer_size = int(gradio_cfg.get("result_buffer_size", 300))
    return FrameSequencePipeline(
        context="general",
        use_depth=True,
        return_annotated=False,
        llm_worker_mode=llm_worker_mode,
        max_results=result_buffer_size,
    )


def _gradio_runtime_settings() -> tuple[float, float, int]:
    cfg = load_runtime_config().get("gradio", {})
    if not isinstance(cfg, dict):
        cfg = {}
    idle_timeout = max(30.0, float(cfg.get("session_idle_timeout_sec", 900)))
    cleanup_interval = max(5.0, float(cfg.get("session_cleanup_interval_sec", 30)))
    max_sessions = int(cfg.get("max_sessions", 32))
    return idle_timeout, cleanup_interval, max_sessions


def _default_detection_sensitivity() -> float:
    cfg = load_runtime_config().get("gradio", {})
    if not isinstance(cfg, dict):
        cfg = {}
    return _normalize_detection_sensitivity(cfg.get("detection_sensitivity", 5.0))


def _normalize_detection_sensitivity(value: Any, *, fallback: float = 5.0) -> float:
    try:
        normalized = float(value)
    except Exception:
        normalized = float(fallback)
    return min(10.0, max(1.0, normalized))


def _session_id(request: gr.Request | None) -> str:
    raw = getattr(request, "session_hash", None) or "anonymous"
    return re.sub(r"[^a-zA-Z0-9_-]", "_", str(raw))


def _get_or_create_session(request: gr.Request | None) -> tuple[str, SessionContext]:
    sid = _session_id(request)
    now = time.time()
    with _SESSIONS_LOCK:
        ctx = _SESSIONS.get(sid)
        if ctx is None:
            ctx = SessionContext(
                pipeline=_new_pipeline(),
                lock=threading.Lock(),
                event_log=LOG_DIR / f"webcam_events_{sid}.jsonl",
                last_active_ts=now,
                runtime_detection_sensitivity=_default_detection_sensitivity(),
            )
            _SESSIONS[sid] = ctx
        else:
            ctx.last_active_ts = now
    return sid, ctx


def _reset_session_pipeline(request: gr.Request | None) -> tuple[str, SessionContext]:
    sid, ctx = _get_or_create_session(request)
    with ctx.lock:
        ctx.pipeline.close()
        ctx.pipeline = _new_pipeline()
        ctx.pipeline.set_detection_sensitivity(ctx.runtime_detection_sensitivity)
        ctx.last_active_ts = time.time()
    return sid, ctx


def _cleanup_sessions_once() -> None:
    now = time.time()
    idle_timeout, _, max_sessions = _gradio_runtime_settings()
    stale: list[tuple[str, SessionContext]] = []

    with _SESSIONS_LOCK:
        for sid, ctx in list(_SESSIONS.items()):
            if now - ctx.last_active_ts > idle_timeout:
                stale.append((sid, ctx))

        if max_sessions > 0:
            alive_count = len(_SESSIONS) - len(stale)
            extra = alive_count - max_sessions
            if extra > 0:
                stale_ids = {sid for sid, _ in stale}
                survivors = [(sid, ctx) for sid, ctx in _SESSIONS.items() if sid not in stale_ids]
                survivors.sort(key=lambda x: x[1].last_active_ts)
                stale.extend(survivors[:extra])

        for sid, _ in stale:
            _SESSIONS.pop(sid, None)

    for _, ctx in stale:
        with ctx.lock:
            ctx.pipeline.close()


def _session_cleanup_loop() -> None:
    while not _SESSION_CLEANUP_STOP.is_set():
        _, cleanup_interval, _ = _gradio_runtime_settings()
        _SESSION_CLEANUP_STOP.wait(timeout=cleanup_interval)
        if _SESSION_CLEANUP_STOP.is_set():
            break
        _cleanup_sessions_once()


def _touch_session(ctx: SessionContext) -> None:
    ctx.last_active_ts = time.time()


def _set_runtime_detection_sensitivity(
    detection_sensitivity: float,
    request: gr.Request | None = None,
) -> None:
    sid = _session_id(request)
    sensitivity = _normalize_detection_sensitivity(detection_sensitivity)
    with _SESSIONS_LOCK:
        ctx = _SESSIONS.get(sid)
    if ctx is None:
        return
    with ctx.lock:
        ctx.runtime_detection_sensitivity = sensitivity
        ctx.pipeline.set_detection_sensitivity(sensitivity)
        _touch_session(ctx)


def _close_all_sessions() -> None:
    _SESSION_CLEANUP_STOP.set()
    with _SESSIONS_LOCK:
        items = list(_SESSIONS.items())
        _SESSIONS.clear()
    for _, ctx in items:
        try:
            ctx.pipeline.close()
        except Exception:
            pass


_SESSION_CLEANUP_THREAD = threading.Thread(target=_session_cleanup_loop, daemon=True)
_SESSION_CLEANUP_THREAD.start()


atexit.register(_close_all_sessions)


def _to_jpeg_bytes(frame: Any) -> bytes:
    img = Image.fromarray(frame).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _emit_log(event_log: Path, event: str, payload: dict[str, Any]) -> None:
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    with event_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _format_yolo_panel(result: dict[str, Any], *, flip_horizontal: bool = False) -> str:
    """Format detected objects panel for display."""
    detections = result.get("detections", [])
    if detections:
        obj_items = []
        for d in sorted(detections, key=lambda x: -x.get("confidence", 0)):
            name = html.escape(str(d.get("class_name", "unknown")))
            conf = float(d.get("confidence", 0.0))
            obj_items.append(
                "<span class='det-chip'>"
                f"<span class='det-name'>{name}</span>"
                f"<span class='det-conf'>{conf:.0%}</span>"
                "</span>"
            )
        obj_text = "".join(obj_items)
    else:
        obj_text = "<span class='det-empty'>No objects detected</span>"

    count = len(detections)
    flip_text = "ON" if flip_horizontal else "OFF"
    return (
        "<div class='yolo-panel'>"
        "<div class='yolo-panel__head'>"
        "<span class='yolo-panel__title'>Detected Objects</span>"
        f"<span class='yolo-panel__meta'>Count {count} | Flip {flip_text}</span>"
        "</div>"
        f"<div class='yolo-panel__chips'>{obj_text}</div>"
        "</div>"
    )


def _toggle_webcam_flip(enabled: bool) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    new_enabled = not bool(enabled)
    label = f"Flip: {'ON' if new_enabled else 'OFF'}"
    btn_update = gr.update(value=label, variant="primary" if new_enabled else "secondary")
    cam_update = gr.update(webcam_options=gr.WebcamOptions(mirror=new_enabled))
    return new_enabled, btn_update, cam_update


def _tts_to_audio_html(text: str) -> str | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        audio_bytes = synthesize(text, use_cache=True)
    except Exception as e:  # noqa: BLE001
        print(f"[TTS] Error: {e}")
        return None

    if not audio_bytes:
        return None

    tts_cfg = load_runtime_config().get("tts", {})
    output_format = str(tts_cfg.get("output_format", "pcm_22050")).strip().lower()
    if output_format.startswith("pcm_"):
        try:
            sample_rate = int(output_format.split("_", 1)[1])
        except Exception:
            sample_rate = 22050
        wav_buffer = BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)
        wav_b64 = base64.b64encode(wav_buffer.getvalue()).decode("utf-8")
        return (
            f"<audio autoplay playsinline src='data:audio/wav;base64,{wav_b64}'></audio>"
        )
    if output_format.startswith("mp3"):
        mp3_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        return (
            f"<audio autoplay playsinline src='data:audio/mpeg;base64,{mp3_b64}'></audio>"
        )
    return None


def _load_sample_video_for_input(sample_video_path: str | None) -> str | None:
    sample_video_path = (sample_video_path or "").strip()
    if not sample_video_path:
        return None

    test_videos_dir = Path(__file__).resolve().parent / "test" / "test_videos"
    default_video = test_videos_dir / DEFAULT_SAMPLE_VIDEO_NAME

    p = Path(sample_video_path)
    if not p.exists() and sample_video_path.lower() == "default" and default_video.exists():
        p = default_video

    if not p.exists() and test_videos_dir.exists():
        local_candidates = sorted(
            c for c in test_videos_dir.iterdir() if c.is_file() and c.suffix.lower() in SAMPLE_VIDEO_EXTENSIONS
        )
        query = sample_video_path.casefold()
        normalized_query = re.sub(r"[^a-z0-9]+", " ", query).strip()

        for candidate in local_candidates:
            if candidate.name.casefold() == query or candidate.stem.casefold() == query:
                p = candidate
                break
        else:
            fuzzy_matches: list[Path] = []
            for candidate in local_candidates:
                normalized_stem = re.sub(r"[^a-z0-9]+", " ", candidate.stem.casefold()).strip()
                if normalized_query and normalized_query in normalized_stem:
                    fuzzy_matches.append(candidate)
            if len(fuzzy_matches) == 1:
                p = fuzzy_matches[0]

    if p.exists():
        resolved = str(p.resolve())
        try:
            if processing_utils.ffmpeg_installed() and not processing_utils.video_is_playable(resolved):
                resolved = processing_utils.convert_video_to_playable_mp4(resolved)
        except Exception:
            pass
        return resolved
    return None


def _reset_video_file_ui() -> tuple[Any, Any, Any, str, Any, dict[str, Any], Any]:
    return (
        None,
        gr.update(visible=False),
        gr.update(value=""),
        "",
        gr.update(value="", visible=False),
        {"seen_llm_ids": [], "latest_llm_text": ""},
        gr.update(value=""),
    )


def process_stream(
    frame: Any,
    state: dict[str, Any],
    llm_provider: str,
    tts_enabled: bool,
    detection_sensitivity: float,
    flip_horizontal: bool,
    request: gr.Request | None = None,
) -> tuple[str, str, dict[str, Any], Any]:
    if frame is None:
        return "No frame", "", state, gr.update()
    if flip_horizontal:
        frame = cv2.flip(frame, 1)

    sid, ctx = _get_or_create_session(request)
    with ctx.lock:
        _touch_session(ctx)
        pipeline = ctx.pipeline
        pipeline.set_llm_provider(llm_provider)
        sensitivity = _normalize_detection_sensitivity(detection_sensitivity)
        ctx.runtime_detection_sensitivity = sensitivity
        pipeline.set_detection_sensitivity(sensitivity)
        frame_bytes = _to_jpeg_bytes(frame)
        result = pipeline.submit_frames([frame_bytes])[-1]
        pipeline.poll_llm()

        results = pipeline.get_results()
        seen_ids = set(state.get("seen_llm_ids", []))
        latest_llm_text = str(state.get("latest_llm_text", ""))
        latest_audio_html: str | None = None

        for r in results:
            fid = int(r["frame_id"])
            llm = r.get("llm", {})
            if fid in seen_ids:
                continue
            if llm.get("sent") and llm.get("done"):
                seen_ids.add(fid)
                desc = (llm.get("description") or "").strip()
                err = (llm.get("error") or "").strip()
                if desc:
                    latest_llm_text = f"frame={fid}: {desc}"
                    if tts_enabled:
                        print(f"[TTS] Generating browser audio: {desc[:30]}...")
                        latest_audio_html = _tts_to_audio_html(desc)
                if desc or err:
                    _emit_log(
                        ctx.event_log,
                        "llm",
                        {
                            "session_id": sid,
                            "frame_id": fid,
                            "sent": llm.get("sent"),
                            "done": llm.get("done"),
                            "description": desc,
                            "error": err,
                            "llm_ms": r.get("latency", {}).get("llm_ms", 0.0),
                        },
                    )
                    print(
                        f"[LLM ] frame={fid} sent={llm.get('sent')} done={llm.get('done')} "
                        f"text={desc} err={err}"
                    )

        fid = int(result["frame_id"])
        _emit_log(
            ctx.event_log,
            "yolo",
            {
                "session_id": sid,
                "frame_id": fid,
                "det_count": len(result.get("detections", [])),
                "llm_sent": result.get("llm", {}).get("sent"),
            },
        )

        q = pipeline.get_queue_status()
        print(
            "[QUEUE] "
            f"frames={q['total_frames']} yolo_queue={q['yolo_queue']} "
            f"llm_sent={q['llm_sent']} llm_done={q['llm_done']} "
            f"llm_queued={q['llm_queued']} llm_pending={q['llm_pending']} "
            f"provider={result.get('runtime', {}).get('llm_provider', llm_provider)}"
        )

        yolo_panel = _format_yolo_panel(result, flip_horizontal=flip_horizontal)
        
        new_state = {
            "seen_llm_ids": sorted(seen_ids),
            "latest_llm_text": latest_llm_text,
        }
        audio_update = latest_audio_html if latest_audio_html is not None else gr.update()
        return yolo_panel, latest_llm_text, new_state, audio_update


def process_video_file(
    video_path: str,
    state: dict[str, Any],
    llm_provider: str,
    tts_enabled: bool,
    detection_sensitivity: float,
    request: gr.Request | None = None,
) -> Generator[tuple[Any, Any, Any, Any, str, dict[str, Any], Any], None, None]:
    """
    Process a video file frame by frame at source FPS timing.
    Yields (current_frame, progress_update, yolo_container, yolo_panel, llm_text, state, audio_update).
    """
    if video_path is None:
        yield (
            None,
            gr.update(value="No video uploaded", visible=True),
            gr.update(visible=False),
            gr.update(value=""),
            "",
            state,
            gr.update(),
        )
        return

    # Reset only this user's pipeline for fresh processing
    sid, ctx = _reset_session_pipeline(request)
    initial_sensitivity = _normalize_detection_sensitivity(detection_sensitivity)
    with ctx.lock:
        ctx.runtime_detection_sensitivity = initial_sensitivity
        ctx.pipeline.set_detection_sensitivity(initial_sensitivity)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield (
            None,
            gr.update(value="Error: Cannot open video file", visible=True),
            gr.update(visible=False),
            gr.update(value=""),
            "",
            state,
            gr.update(),
        )
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = 1.0 / video_fps if video_fps > 0 else 1.0 / 30.0

    frame_idx = 0
    seen_ids: set[int] = set()
    latest_llm_text = ""
    yolo_panel = ""
    new_state = {
        "seen_llm_ids": sorted(seen_ids),
        "latest_llm_text": latest_llm_text,
    }

    # Reset panel state first so video tab always starts with only the VLM box.
    yield (
        None,
        gr.update(value=f"Frame 0/{total_frames} (0.0%)", visible=True),
        gr.update(visible=False),
        gr.update(value=""),
        latest_llm_text,
        new_state,
        gr.update(),
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            start_time = time.time()

            # Convert BGR to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with ctx.lock:
                _touch_session(ctx)
                current_audio_html: str | None = None
                pipeline = ctx.pipeline
                pipeline.set_llm_provider(llm_provider)
                runtime_sensitivity = _normalize_detection_sensitivity(ctx.runtime_detection_sensitivity)
                pipeline.set_detection_sensitivity(runtime_sensitivity)
                frame_bytes = _to_jpeg_bytes(frame_rgb)
                result = pipeline.submit_frames([frame_bytes])[-1]
                pipeline.poll_llm()

                results = pipeline.get_results()
                for r in results:
                    fid = int(r["frame_id"])
                    llm = r.get("llm", {})
                    if fid in seen_ids:
                        continue
                    if llm.get("sent") and llm.get("done"):
                        seen_ids.add(fid)
                        desc = (llm.get("description") or "").strip()
                        err = (llm.get("error") or "").strip()
                        if desc:
                            latest_llm_text = f"frame={fid}: {desc}"
                            if tts_enabled:
                                current_audio_html = _tts_to_audio_html(desc)
                        if desc or err:
                            _emit_log(
                                ctx.event_log,
                                "llm_video",
                                {
                                    "session_id": sid,
                                    "frame_id": fid,
                                    "video_frame": frame_idx,
                                    "sent": llm.get("sent"),
                                    "done": llm.get("done"),
                                    "description": desc,
                                    "error": err,
                                    "llm_ms": r.get("latency", {}).get("llm_ms", 0.0),
                                },
                            )

                fid = int(result["frame_id"])
                _emit_log(
                    ctx.event_log,
                    "yolo_video",
                    {
                        "session_id": sid,
                        "frame_id": fid,
                        "video_frame": frame_idx,
                        "det_count": len(result.get("detections", [])),
                        "llm_sent": result.get("llm", {}).get("sent"),
                        "llm_provider": result.get("runtime", {}).get("llm_provider"),
                    },
                )

                yolo_panel = _format_yolo_panel(result)

            progress = f"Frame {frame_idx}/{total_frames} ({100*frame_idx/total_frames:.1f}%)"
            new_state = {
                "seen_llm_ids": sorted(seen_ids),
                "latest_llm_text": latest_llm_text,
            }

            audio_update = current_audio_html if current_audio_html is not None else gr.update()
            yield (
                frame_rgb,
                gr.update(value=progress, visible=True),
                gr.update(visible=True),
                gr.update(value=yolo_panel),
                latest_llm_text,
                new_state,
                audio_update,
            )

            # Maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        cap.release()

    # Final yield with completion message
    final_container_update = gr.update(visible=frame_idx > 0)
    final_panel_update = gr.update(value=yolo_panel if frame_idx > 0 else "")
    yield (
        None,
        gr.update(value=f"Completed: {frame_idx} frames processed", visible=frame_idx > 0),
        final_container_update,
        final_panel_update,
        latest_llm_text,
        new_state,
        gr.update(),
    )


with gr.Blocks(title="LiveGuide Scene Assistant", js=FORCE_LIGHT_MODE_JS) as demo:
    logo_path = Path(__file__).resolve().parent / "ge.png"
    logo_b64 = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    default_provider = str(load_runtime_config().get("llm", {}).get("provider", "qwen_local"))

    with gr.Column(elem_id="app-shell"):
        with gr.Row(elem_id="app-header", elem_classes=["glass-card"]):
            with gr.Column(scale=2, min_width=188, elem_id="app-logo-col"):
                gr.HTML(
                    value=f"<img src='data:image/png;base64,{logo_b64}' alt='LiveGuide logo' />",
                    elem_id="app-logo",
                )
            with gr.Column(scale=8):
                gr.HTML(
                    """
                    <div id="app-title-block">
                      <p class="eyebrow">Real-time vision copilot</p>
                      <h1>LiveGuide Scene Assistant</h1>
                      <p class="lead">
                        Real-time scene descriptions for visual assistance. Objects are detected by YOLO and narrated by VLM.
                      </p>
                      <div id="capability-badges">
                        <span class="badge">Gemini-powered VLM</span>
                        <span class="badge">ElevenLabs TTS</span>
                        <span class="badge">YOLO</span>
                        <span class="badge">MiDaS</span>
                      </div>
                    </div>
                    """
                )
            with gr.Column(scale=1, min_width=96, elem_id="app-right-spacer"):
                gr.HTML("")

        with gr.Row(elem_id="control-strip", elem_classes=["glass-card"]):
            llm_provider = gr.Dropdown(
                choices=["qwen_local", "gemini_api"],
                value=default_provider if default_provider in {"qwen_local", "gemini_api"} else "qwen_local",
                label="VLM Model",
                info="Switch live between local Qwen and Gemini API",
                scale=5,
                elem_id="provider-select",
            )
            detection_sensitivity = gr.Slider(
                minimum=1.0,
                maximum=10.0,
                value=_default_detection_sensitivity(),
                step=0.5,
                label="Detection Sensitivity",
                info="Higher means more sensitive and more frequent VLM updates.",
                scale=4,
                elem_id="sensitivity-slider",
            )
            tts_enabled = gr.Checkbox(
                label="TTS",
                value=True,
                info="Auto play",
                scale=1,
                elem_id="tts-toggle",
            )
            detection_sensitivity.change(
                fn=_set_runtime_detection_sensitivity,
                inputs=[detection_sensitivity],
                outputs=[],
                show_progress="hidden",
            )

        with gr.Tabs(elem_id="main-tabs"):
            # Tab 1: Webcam (original functionality)
            with gr.TabItem("Webcam"):
                with gr.Column(elem_id="webcam-section", elem_classes=["glass-card"]):
                    with gr.Column(elem_id="webcam-wrap"):
                        webcam_flip_btn = gr.Button(
                            "Flip: OFF",
                            variant="secondary",
                            size="sm",
                            elem_id="webcam-flip-btn",
                            scale=0,
                            min_width=96,
                        )
                        cam = gr.Image(
                            sources=["webcam"],
                            streaming=True,
                            type="numpy",
                            label="Webcam",
                            webcam_options=gr.WebcamOptions(mirror=False),
                        )
                    webcam_flip_state = gr.State(False)
                    webcam_yolo_html = gr.HTML(label="Detected Objects", elem_id="webcam-yolo")
                    webcam_llm_text = gr.Textbox(label="Scene Description", lines=4, elem_id="webcam-llm")
                    webcam_tts_audio = gr.HTML(
                        value="",
                        elem_classes=["tts-hidden"],
                    )
                    webcam_state = gr.State({"seen_llm_ids": [], "latest_llm_text": ""})

                    cam.stream(
                        fn=process_stream,
                        inputs=[
                            cam,
                            webcam_state,
                            llm_provider,
                            tts_enabled,
                            detection_sensitivity,
                            webcam_flip_state,
                        ],
                        outputs=[webcam_yolo_html, webcam_llm_text, webcam_state, webcam_tts_audio],
                        show_progress="hidden",
                    )
                    webcam_flip_btn.click(
                        fn=_toggle_webcam_flip,
                        inputs=[webcam_flip_state],
                        outputs=[webcam_flip_state, webcam_flip_btn, cam],
                        show_progress="hidden",
                    )

            # Tab 2: Video File (for debugging)
            with gr.TabItem("Video File") as video_file_tab:
                with gr.Column(elem_id="video-section", elem_classes=["glass-card"]):
                    test_videos_dir = Path(__file__).resolve().parent / "test" / "test_videos"
                    test_videos_dir.mkdir(parents=True, exist_ok=True)
                    default_video_path = test_videos_dir / DEFAULT_SAMPLE_VIDEO_NAME

                    gr.Markdown(
                        "Upload a video file to process frame-by-frame for debugging. "
                        "Use the top Detection Sensitivity slider to control update sensitivity.",
                        elem_id="video-helper",
                    )

                    with gr.Row(elem_id="video-input-row"):
                        video_input = gr.Video(
                            sources=["upload"],
                            label="Upload Video",
                            elem_id="video-input",
                        )

                    process_btn = gr.Button("Process Video", variant="primary", elem_id="process-btn")

                    with gr.Row(elem_id="video-output-row"):
                        video_frame_display = gr.Image(label="Current Frame", type="numpy", elem_id="video-frame")
                        video_progress = gr.Textbox(label="Progress", lines=1, elem_id="video-progress", visible=False)

                    with gr.Column(visible=False) as video_yolo_wrap:
                        video_yolo_html = gr.HTML(label="Detected Objects", elem_id="video-yolo")
                    video_llm_text = gr.Textbox(label="Scene Description", lines=4, elem_id="video-llm")
                    video_tts_audio = gr.HTML(
                        value="",
                        elem_classes=["tts-hidden"],
                    )
                    video_state = gr.State({"seen_llm_ids": [], "latest_llm_text": ""})
                    sample_video_path_input = gr.Textbox(
                        value="",
                        visible=False,
                        label="sample_video_path",
                    )

                    sample_videos = sorted(
                        p for p in test_videos_dir.iterdir() if p.is_file() and p.suffix.lower() in SAMPLE_VIDEO_EXTENSIONS
                    )
                    if default_video_path.exists() and default_video_path in sample_videos:
                        sample_videos = [default_video_path] + [v for v in sample_videos if v != default_video_path]
                    if sample_videos:
                        sample_video_examples = [v.stem for v in sample_videos]
                        gr.Examples(
                            examples=[[v] for v in sample_video_examples],
                            inputs=[sample_video_path_input],
                            outputs=[video_input],
                            fn=_load_sample_video_for_input,
                            cache_examples=False,
                            run_on_click=True,
                            label="Local Sample Videos",
                        )

                    process_btn.click(
                        fn=process_video_file,
                        inputs=[video_input, video_state, llm_provider, tts_enabled, detection_sensitivity],
                        outputs=[
                            video_frame_display,
                            video_progress,
                            video_yolo_wrap,
                            video_yolo_html,
                            video_llm_text,
                            video_state,
                            video_tts_audio,
                        ],
                    )
                    video_input.change(
                        fn=_reset_video_file_ui,
                        inputs=[],
                        outputs=[
                            video_frame_display,
                            video_yolo_wrap,
                            video_yolo_html,
                            video_llm_text,
                            video_progress,
                            video_state,
                            video_tts_audio,
                        ],
                        show_progress="hidden",
                    )
                    video_file_tab.select(
                        fn=_reset_video_file_ui,
                        inputs=[],
                        outputs=[
                            video_frame_display,
                            video_yolo_wrap,
                            video_yolo_html,
                            video_llm_text,
                            video_progress,
                            video_state,
                            video_tts_audio,
                        ],
                        show_progress="hidden",
                    )


if __name__ == "__main__":
    gradio_cfg = load_runtime_config().get("gradio", {})
    if not isinstance(gradio_cfg, dict):
        gradio_cfg = {}
    queue_cfg = gradio_cfg.get("queue", {})
    if not isinstance(queue_cfg, dict):
        queue_cfg = {}

    default_concurrency_limit = queue_cfg.get("default_concurrency_limit", 2)
    max_size = queue_cfg.get("max_size", 32)

    try:
        default_concurrency_limit = int(default_concurrency_limit)
    except Exception:
        default_concurrency_limit = 2
    if default_concurrency_limit <= 0:
        default_concurrency_limit = None

    try:
        max_size = int(max_size)
    except Exception:
        max_size = 32
    if max_size <= 0:
        max_size = None

    demo.queue(
        default_concurrency_limit=default_concurrency_limit,
        max_size=max_size,
    )

    # server_name="0.0.0.0" allows access from other devices on same network
    # Access from phone using your PC's IP, e.g., http://192.168.1.x:8000
    demo.launch(server_name="0.0.0.0", server_port=8000, share=True, css=HIDDEN_TTS_CSS)
