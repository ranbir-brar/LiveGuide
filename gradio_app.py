import atexit
import json
import threading
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Generator

import cv2
import gradio as gr
from PIL import Image

from main import FrameSequencePipeline, load_runtime_config
from models.tts import synthesize, wav_bytes_to_numpy

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
EVENT_LOG = LOG_DIR / "webcam_events.jsonl"

_PIPELINE_LOCK = threading.Lock()
_PIPELINE: "FrameSequencePipeline | None" = None
_PIPELINE: "FrameSequencePipeline | None" = None
_LAST_SPOKEN_TEXT: str = ""  # Track last spoken text to avoid repeats
_LAST_SPOKEN_TIME: float = 0.0  # Track time of last speech to enforce interval


def _text_similarity(text1: str, text2: str) -> float:
    """Calculate word-overlap similarity (Jaccard) between two texts.
    Returns 0.0 to 1.0 where 1.0 means identical."""
    if not text1 or not text2:
        return 0.0
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0


def _should_speak(new_text: str) -> bool:
    """Check if we should speak this text (not too similar to last spoken AND not too soon)."""
    global _LAST_SPOKEN_TEXT, _LAST_SPOKEN_TIME
    
    cfg = load_runtime_config()
    tts_cfg = cfg.get("tts", {})
    threshold = float(tts_cfg.get("similarity_threshold", 0.7))
    min_interval = float(tts_cfg.get("min_tts_interval", 2.5))
    
    now = time.time()
    time_since_last = now - _LAST_SPOKEN_TIME
    
    # 1. Check time interval (Rate Limiting)
    if time_since_last < min_interval:
        print(f"[TTS] Skipping (too soon: {time_since_last:.1f}s < {min_interval:.1f}s)")
        return False

    # 2. Check similarity
    similarity = _text_similarity(new_text, _LAST_SPOKEN_TEXT)
    if similarity > threshold:
        print(f"[TTS] Skipping (similarity={similarity:.0%} > {threshold:.0%})")
        return False
        
    _LAST_SPOKEN_TEXT = new_text
    _LAST_SPOKEN_TIME = now
    return True


def get_pipeline() -> FrameSequencePipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = FrameSequencePipeline(
            context="general",
            use_depth=True,
            return_annotated=False,
            llm_worker_mode="process",
        )
    return _PIPELINE


def close_pipeline() -> None:
    global _PIPELINE
    if _PIPELINE is not None:
        _PIPELINE.close()
        _PIPELINE = None


def reset_pipeline() -> None:
    """Close and recreate the pipeline for fresh video processing."""
    global _PIPELINE
    with _PIPELINE_LOCK:
        close_pipeline()


atexit.register(close_pipeline)


def _to_jpeg_bytes(frame: Any) -> bytes:
    img = Image.fromarray(frame).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _emit_log(event: str, payload: dict[str, Any]) -> None:
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    with EVENT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _format_yolo_panel(result: dict[str, Any]) -> str:
    """Format detected objects panel for display."""
    detections = result.get("detections", [])
    if detections:
        obj_items = []
        for d in sorted(detections, key=lambda x: -x.get("confidence", 0)):
            name = d.get("class_name", "unknown")
            conf = float(d.get("confidence", 0.0))
            obj_items.append(f"{name} ({conf:.0%})")
        obj_text = ", ".join(obj_items)
    else:
        obj_text = "No objects detected"

    count = len(detections)
    return (
        "<div style='padding:10px;border:1px solid #d0d7de;border-radius:8px;'>"
        f"<div style='font-weight:600;'>Detected Objects ({count})</div>"
        f"<div style='margin-top:8px;'>{obj_text}</div>"
        "</div>"
    )

def process_stream(frame: Any, state: dict[str, Any], llm_provider: str) -> tuple[str, str, tuple[int, bytes] | Any, dict[str, Any]]:
    """Process webcam frame and return (yolo_panel, llm_text, audio, state)."""
    if frame is None:
        return "No frame", "", None, state

    audio_output = gr.skip()  # Default to skipping audio update

    with _PIPELINE_LOCK:
        pipeline = get_pipeline()
        pipeline.set_llm_provider(llm_provider)
        frame_bytes = _to_jpeg_bytes(frame)
        result = pipeline.submit_frames([frame_bytes])[-1]
        pipeline.poll_llm()

        results = pipeline.get_results()
        seen_ids = set(state.get("seen_llm_ids", []))
        latest_llm_text = str(state.get("latest_llm_text", ""))

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
                    # Synthesize TTS audio for browser playback (skip if too similar to last)
                    if _should_speak(desc):
                        try:
                            print(f"[TTS] Synthesizing for browser: {desc[:30]}...")
                            wav_bytes = synthesize(desc)
                            if wav_bytes:
                                audio_output = wav_bytes_to_numpy(wav_bytes)
                        except Exception as e:
                            print(f"[TTS] Error: {e}")
                if desc or err:
                    _emit_log(
                        "llm",
                        {
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
            "yolo",
            {
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

        yolo_panel = _format_yolo_panel(result)
        
        new_state = {
            "seen_llm_ids": sorted(seen_ids),
            "latest_llm_text": latest_llm_text,
        }
    
    return yolo_panel, latest_llm_text, audio_output, new_state


def process_video_file(
    video_path: str, fps: float, state: dict[str, Any], llm_provider: str
) -> Generator[tuple[Any, str, str, str, tuple[int, bytes] | Any, dict[str, Any]], None, None]:
    """
    Process a video file frame by frame at the specified FPS.
    Yields (current_frame, progress_text, yolo_panel, llm_text, audio, state) for each frame.
    """
    if video_path is None:
        yield None, "No video uploaded", "", "", None, state
        return

    # Indicate startup immediately
    yield None, "Initializing models (this may take a few seconds)...", "", "", gr.skip(), state

    # Reset pipeline for fresh processing
    reset_pipeline()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield None, "Error: Cannot open video file", "", "", None, state
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = 1.0 / fps if fps > 0 else 1.0 / video_fps

    frame_idx = 0
    seen_ids = set(state.get("seen_llm_ids", []))
    latest_llm_text = str(state.get("latest_llm_text", ""))
    audio_output = None  # Will hold (sample_rate, wav_bytes) if TTS generated
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            start_time = time.time()

            # Convert BGR to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with _PIPELINE_LOCK:
                pipeline = get_pipeline()
                pipeline.set_llm_provider(llm_provider)
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
                            # Synthesize TTS audio for browser playback (skip if too similar to last)
                            if _should_speak(desc):
                                try:
                                    print(f"[TTS] Synthesizing (video): {desc[:30]}...")
                                    wav_bytes = synthesize(desc)
                                    if wav_bytes:
                                        audio_output = wav_bytes_to_numpy(wav_bytes)
                                except Exception as e:
                                    print(f"[TTS] Error: {e}")
                        if desc or err:
                            _emit_log(
                                "llm_video",
                                {
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
                    "yolo_video",
                    {
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

            yield frame_rgb, progress, yolo_panel, latest_llm_text, audio_output if audio_output is not None else gr.skip(), new_state
            audio_output = None  # Reset after yielding to avoid replaying

            # Maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        cap.release()

    # Final yield with completion message
    yield None, f"Completed: {frame_idx} frames processed", yolo_panel, latest_llm_text, None, new_state



# CSS to hide elements visually but keep them functional
css = """
.hidden-audio { display: none !important; }
"""

with gr.Blocks(title="LiveGuide Scene Assistant", css=css) as demo:
    gr.Markdown("## LiveGuide Scene Assistant")
    gr.Markdown("Real-time scene descriptions for visual assistance. Objects detected by YOLO, scene narrated by VLM.")
    default_provider = str(load_runtime_config().get("llm", {}).get("provider", "qwen_local"))
    llm_provider = gr.Dropdown(
        choices=["qwen_local", "gemini_api"],
        value=default_provider if default_provider in {"qwen_local", "gemini_api"} else "qwen_local",
        label="VLM Model",
        info="Switch live between local Qwen and Gemini API",
    )

    with gr.Tabs():
        # Tab 1: Webcam (original functionality)
        with gr.TabItem("Webcam"):
            cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam")
            webcam_yolo_html = gr.HTML(label="Detected Objects")
            webcam_llm_text = gr.Textbox(label="Scene Description", lines=4)
            webcam_audio = gr.Audio(label="Scene Narration", autoplay=True, visible=True, elem_classes=["hidden-audio"])
            webcam_state = gr.State({"seen_llm_ids": [], "latest_llm_text": ""})

            cam.stream(
                fn=process_stream,
                inputs=[cam, webcam_state, llm_provider],
                outputs=[webcam_yolo_html, webcam_llm_text, webcam_audio, webcam_state],
                show_progress="hidden",
            )

        # Tab 2: Video File (for debugging)
        with gr.TabItem("Video File"):
            gr.Markdown("Upload a video file to process frame-by-frame for debugging.")
            
            with gr.Row():
                video_input = gr.Video(sources=["upload"], label="Upload Video")
                video_fps = gr.Slider(
                    minimum=0.5,
                    maximum=30,
                    value=6,
                    step=0.5,
                    label="Processing FPS",
                    info="Frames per second to process (lower = slower but more LLM calls)"
                )
            
            process_btn = gr.Button("Process Video", variant="primary")
            
            with gr.Row():
                video_frame_display = gr.Image(label="Current Frame", type="numpy")
                video_progress = gr.Textbox(label="Progress", lines=1)
            
            video_yolo_html = gr.HTML(label="Detected Objects")
            video_llm_text = gr.Textbox(label="Scene Description", lines=4)
            video_audio = gr.Audio(label="Scene Narration", autoplay=True, visible=True, elem_classes=["hidden-audio"])
            video_state = gr.State({"seen_llm_ids": [], "latest_llm_text": ""})

            test_videos_dir = Path(__file__).resolve().parent / "test" / "test_videos"
            test_videos_dir.mkdir(parents=True, exist_ok=True)
            sample_videos = sorted(
                str(p) for p in test_videos_dir.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
            )
            if sample_videos:
                gr.Examples(
                    examples=[[v] for v in sample_videos],
                    inputs=[video_input],
                    label="Local Sample Videos (test/test_videos)",
                )

            process_btn.click(
                fn=process_video_file,
                inputs=[video_input, video_fps, video_state, llm_provider],
                outputs=[video_frame_display, video_progress, video_yolo_html, video_llm_text, video_audio, video_state],
            )


if __name__ == "__main__":
    # server_name="0.0.0.0" allows access from other devices on same network
    # Access from phone using your PC's IP, e.g., http://192.168.1.x:7860
    demo.launch(server_name="0.0.0.0", share=True)
