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
- Log recording:
  - `logs/pipeline_events.jsonl` (batch inference)
  - `logs/webcam_events.jsonl` (real-time webcam)
  - Each log entry includes `frame_id`, so it can be mapped to a specific frame.

## 2. YOLO input / output

- Input: single-frame image bytes (internally resized, then sent to detection).
- Output (core):
  - `detections`: object list (`class_name`, `confidence`, `xyxy`)
  - `hazard`: hazard score and level (`hazard_score`, `hazard_level`)
  - `danger`: danger state machine status (whether danger is entered/exited)

## 3. LLM input / output

- Input:
  - frame image bytes
  - YOLO detections for that frame (classes, boxes, confidences)
- Output:
  - First does a binary decision: `dangerous = yes/no`
  - If `yes`: returns one short warning sentence
  - If `no`: returns empty text

## 4. When YOLO sends frames to LLM

A YOLO frame is sent to LLM only if all of the following are satisfied:

1. Not blocked by "similar frame suppression"
- If similarity with the previous frame's YOLO result is too high (default threshold `0.9`), it is not sent.

2. Probabilistic gate is hit
- `x = yolo_score * constant * probability_scale`
- `smooth(x) = x / (x + tau)`  (for `x >= 0`)
- `P = bias + (1-bias) * smooth(x)`
- It must pass random sampling before continuing.

3. Rate limit check passes
- Must satisfy both `min_interval_ms` and `max_calls_per_second` constraints.

Configuration is in `config/runtime_config.yaml`.
