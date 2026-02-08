# LiveGuide

LiveGuide is a real-time scene assistance system built with `YOLO (OpenImages)` + `VLM (Qwen/Gemini)` + `TTS (ElevenLabs)`.  
It supports webcam streaming and local video debugging, designed for low-latency, observable navigation guidance.

## Features

- Real-time Perception: Leverages `YOLOv8n-oiv7` (detecting 601 classes) enriched with `MiDaS` depth estimation for immediate object awareness.
- Adaptive AI Backends: Supports hot-switching between `qwen_local` and `gemini_api`, optimized by a unique VLM triggering algorithm including interval and similarity gating.
- Concurrency Control: Ensures stable multi-user support via Gradio through per-session pipeline isolation.

## Architecture

```text
Camera / Video
    -> preprocess (resize/jpeg)
    -> YOLO detect (+ MiDaS depth)
    -> gating (interval + similarity)
    -> LLM queue (thread/process worker)
    -> short actionable sentence
    -> TTS
    -> Gradio UI + JSONL logs
```

## Quick Start

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install opencv-python
```

### 2) Configure API Keys

Verify `config/runtime_config.yaml`:
- `llm.provider`: `gemini_api` or `qwen_local`
- `tts.provider`: `elevenlabs`

### 3) Run Web App

```bash
python gradio_app.py
```

By default it listens on `0.0.0.0:8000`.

## Usage

### Webcam (real-time)

- Open the `Webcam` tab and start streaming
- `Detection Sensitivity` dynamically controls LLM trigger behavior
- Switch `VLM Model` between `qwen_local` and `gemini_api`

### Video File (debug)

- Open the `Video File` tab and upload a video
- Videos under `test/test_videos/` appear as local examples
