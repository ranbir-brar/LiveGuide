import atexit
import json
import threading
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import gradio as gr
from PIL import Image

import time
import tempfile
from main import FrameSequencePipeline
from models.tts import speak, synthesize

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
EVENT_LOG = LOG_DIR / "webcam_events.jsonl"

_PIPELINE_LOCK = threading.Lock()
_PIPELINE: "FrameSequencePipeline | None" = None
_LAST_SPOKEN_ALERT: str = ""  # Track last spoken alert to avoid repeats


def get_pipeline() -> FrameSequencePipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = FrameSequencePipeline(
            context="walking",
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


def _format_yolo_panel(result: dict[str, Any]) -> tuple[str, bool]:
    """Returns (HTML panel, danger_active flag)."""
    danger_on = bool(result["danger"]["active"])
    lamp_color = "#E00000" if danger_on else "#3E4B59"
    lamp_text = "DANGER" if danger_on else "SAFE"
    detections = result.get("detections", [])
    if detections:
        obj_text = ", ".join(
            f"{d.get('class_name', 'unknown')}({float(d.get('confidence', 0.0)):.2f})" for d in detections
        )
    else:
        obj_text = "none"

    return (
        "<div style='padding:10px;border:1px solid #d0d7de;border-radius:8px;'>"
        f"<div style='display:flex;align-items:center;gap:8px;font-weight:600;'>"
        f"<span style='width:14px;height:14px;border-radius:50%;display:inline-block;background:{lamp_color};'></span>"
        f"<span>{lamp_text}</span></div>"
        f"<div style='margin-top:8px;'><b>Objects:</b> {obj_text}</div>"
        "</div>"
    ), danger_on


def process_stream(frame: Any, state: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    if frame is None:
        return "No frame", "", state

    with _PIPELINE_LOCK:
        pipeline = get_pipeline()
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
                    # Speak LLM description aloud
                    try:
                        print(f"[TTS] Speaking LLM: {desc[:30]}...")
                        speak(desc, blocking=False)
                    except Exception as e:
                        print(f"[TTS] Error: {e}")
                        pass
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
                "hazard_level": result.get("hazard", {}).get("hazard_level"),
                "hazard_score": result.get("hazard", {}).get("hazard_score"),
                "danger_active": result.get("danger", {}).get("active"),
                "llm_sent": result.get("llm", {}).get("sent"),
                "similarity_blocked": result.get("llm", {}).get("similarity_blocked"),
            },
        )

        q = pipeline.get_queue_status()
        print(
            "[QUEUE] "
            f"frames={q['total_frames']} yolo_queue={q['yolo_queue']} "
            f"llm_sent={q['llm_sent']} llm_done={q['llm_done']} "
            f"llm_queued={q['llm_queued']} llm_pending={q['llm_pending']}"
        )

        yolo_panel, danger_active = _format_yolo_panel(result)
        
        # Speak DANGER alert if status changed to danger
        global _LAST_SPOKEN_ALERT
        if danger_active:
            alert_text = "Warning: Hazard detected ahead."
            if _LAST_SPOKEN_ALERT != alert_text:
                _LAST_SPOKEN_ALERT = alert_text
                try:
                    print(f"[TTS] Speaking Alert: {alert_text}")
                    speak(alert_text, blocking=False)
                except Exception as e:
                    print(f"[TTS] Error: {e}")
                    pass
        else:
            _LAST_SPOKEN_ALERT = ""
        
        new_state = {
            "seen_llm_ids": sorted(seen_ids),
            "latest_llm_text": latest_llm_text,
        }
        return yolo_panel, latest_llm_text, new_state



def run_tts_manual(text: str, speed: float, silence: float) -> tuple[Optional[str], str]:
    if not text.strip():
        return None, "Error: No text provided"
    
    start = time.perf_counter()
    try:
        audio_bytes = synthesize(
            text,
            length_scale=speed,
            sentence_silence=silence,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Save to temp file for Gradio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            fp.write(audio_bytes)
            temp_path = fp.name
            
        msg = f"Synthesis complete\nTime: {elapsed_ms:.1f}ms\nSize: {len(audio_bytes)} bytes"
        return temp_path, msg
    except Exception as e:
        return None, f"Error: {str(e)}"


def run_latency_test() -> str:
    phrases = ["Path is clear.", "Warning: Hazard ahead."]
    results = ["--- Latency Test ---"]
    
    try:
        # Cold run
        start = time.perf_counter()
        synthesize(phrases[0], use_cache=False)
        cold_ms = (time.perf_counter() - start) * 1000
        results.append(f"Cold synthesis: {cold_ms:.1f}ms")
        
        # Cached run
        start = time.perf_counter()
        synthesize(phrases[0], use_cache=True)
        cached_ms = (time.perf_counter() - start) * 1000
        results.append(f"Cached retrieval: {cached_ms:.1f}ms")
        
        if cold_ms > 0:
            speedup = cold_ms / cached_ms if cached_ms > 0 else 0
            results.append(f"Speedup: {speedup:.1f}x")
            
    except Exception as e:
        results.append(f"Error: {str(e)}")
    
    return "\n".join(results)


def test_host_playback(text: str) -> str:
    if not text.strip():
        return "Error: No text"
    try:
        speak(text, blocking=True)
        return "Playback requested on host device."
    except Exception as e:
        return f"Error: {e}"


with gr.Blocks(title="LiveGuide Webcam Hazard Monitor") as demo:
    gr.Markdown("## LiveGuide Webcam Hazard Monitor")
    
    with gr.Tabs():
        with gr.Tab("Webcam Monitor"):
            gr.Markdown("YOLO: red light + objects. LLM: latest returned warning text.")
            cam = gr.Image(source="webcam", streaming=True, type="numpy", label="Webcam")
            yolo_html = gr.HTML(label="YOLO Status")
            llm_text = gr.Textbox(label="LLM Output", lines=4)
            state = gr.State({"seen_llm_ids": [], "latest_llm_text": ""})

            cam.stream(
                fn=process_stream,
                inputs=[cam, state],
                outputs=[yolo_html, llm_text, state],
                show_progress="hidden",
            )
            
        with gr.Tab("TTS Experiment"):
            gr.Markdown("### TTS Manual Control")
            with gr.Row():
                with gr.Column():
                    tts_input = gr.Textbox(label="Text to Speak", value="Warning: Hazard detected.")
                    tts_speed = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Speed (Length Scale)")
                    tts_silence = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, label="Sentence Silence")
                    tts_btn = gr.Button("Synthesize & Speak", variant="primary")
                with gr.Column():
                    tts_audio = gr.Audio(label="Output Audio", type="filepath")
                    tts_logs = gr.Textbox(label="Logs", lines=5)
            
            gr.Markdown("### Performance Test")
            with gr.Row():
                perf_btn = gr.Button("Run Latency Test")
                host_btn = gr.Button("Test Host Playback")
            
            tts_btn.click(
                fn=run_tts_manual,
                inputs=[tts_input, tts_speed, tts_silence],
                outputs=[tts_audio, tts_logs]
            )
            
            perf_btn.click(
                fn=run_latency_test,
                inputs=[],
                outputs=[tts_logs]
            )
            
            host_btn.click(
                fn=test_host_playback,
                inputs=[tts_input],
                outputs=[tts_logs]
            )



if __name__ == "__main__":
    demo.launch(share=True)
