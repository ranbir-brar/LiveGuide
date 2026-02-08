## Inspiration

We wanted to build a simple real-time helper for people with visual impairments.
Object labels alone are not enough, so we focused on short guidance plus voice.

## What it does

LiveGuide is a real-time scene assistant.

- Detects objects with YOLOv8n OpenImages.
- Adds depth hints with MiDaS.
- Generates one short guide sentence with Qwen or Gemini.
- Speaks the sentence with ElevenLabs TTS.
- Runs in a Gradio web app (webcam and video file).

## How we built it

Pipeline:
- Frame -> preprocess -> YOLO (+ MiDaS) -> LLM -> TTS -> UI.
- We use interval + similarity gating to avoid calling LLM on every frame.
- We support two LLM backends: `qwen_local` and `gemini_api`.
- We log events in JSONL for debugging.

## Challenges we ran into

- Keeping latency low while still giving useful guidance.
- Avoiding repeated LLM output on similar frames.
- Handling multi-user sessions safely in Gradio.
- Managing different backend behaviors (local Qwen vs Gemini API).

## Accomplishments that we're proud of

- End-to-end real-time pipeline works from camera to spoken output.
- Gating reduces unnecessary LLM calls.
- LLM backend switching works at runtime.
- Logging and queue stats make debugging easier.

## What we learned

- Scheduling is critical for real-time AI systems.
- Short, action-focused prompts work better than generic captions.
- Session and resource management matter early, not later.

## What's next for LiveGuide

- Better hazard priority and safer path guidance.
- Better temporal understanding across frames.
- More language/voice options.
- Stronger evaluation and edge-device optimization.
