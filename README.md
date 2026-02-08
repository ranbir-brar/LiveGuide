# LiveGuide

## 1. How multiprocessing is maintained

- The main flow is in `FrameSequencePipeline` inside `main.py`.
- Processing model:
  - YOLO: frame-by-frame in the main process.
  - LLM: run as an independent process.
- Queue structures:
  - `self._llm_in`: LLM input queue (only frames with `sent=True` are enqueued).
  - `self._llm_out`: LLM output queue (LLM pushes results back after completion).
- Real-time terminal output:
  - `inference.py` / `gradio_app.py` print YOLO results, LLM results (when content exists), and queue status in real time.
- Local video debugging:
  - Place files under `test/test_videos/` and run either:
    - Gradio `Video File (Debug)` tab, or
    - `python test/video_assist_demo.py`
- Log recording:
  - `logs/pipeline_events.jsonl` (batch inference)
  - `logs/webcam_events.jsonl` (real-time webcam)
  - Each log entry includes `frame_id`, so it can be mapped to a specific frame.

## 2. YOLO input / output

- Input: single-frame image bytes (internally resized, then sent to detection).
- Output (core):
  - `detections`: object list (`class_name`, `confidence`, `xyxy`)
  - `latency`: per-frame runtime metrics (`yolo_ms`, `llm_ms`, `total_ms`)
  - `runtime`: execution metadata (`yolo_pid`, `llm_pid`, `llm_provider`)

## 3. LLM input / output

- Input:
  - frame image bytes
  - YOLO detections for that frame (classes, boxes, confidences)
- Output:
  - Returns one SHORT, actionable navigation sentence
  - Focuses on direction, distance, and suggested action
  - Example style: “Person approaching on your left.”

## 4. When YOLO sends frames to LLM

A YOLO frame is sent to LLM when fixed interval gating allows it:
- `llm_call_interval_sec` defines the minimum time gap between two LLM calls.
- If enough time has elapsed since the last LLM call, current frame is sent.
- Otherwise, current frame is skipped for LLM.

Configuration is in `config/runtime_config.yaml`.
