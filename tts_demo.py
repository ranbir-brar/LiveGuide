"""Text-to-Speech Demo for LiveGuide Video Assist.

Run: python tts_demo.py
"""

import time
from pathlib import Path

from main import run_frame_sequence
from models.tts import speak, synthesize
from models.tts.piper import _audio_cache


def demo_tts_basic() -> None:
    print("=" * 50)
    print("TTS Basic Demo")
    print("=" * 50)
    test_phrases = [
        "Scene is stable.",
        "A bicycle is approaching from the left.",
        "A vehicle is close ahead.",
    ]
    for phrase in test_phrases:
        print(f"\nSpeaking: '{phrase}'")
        start = time.perf_counter()
        speak(phrase, blocking=True)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  Latency: {elapsed:.1f}ms")


def demo_tts_caching() -> None:
    print("\n" + "=" * 50)
    print("TTS Caching Demo")
    print("=" * 50)
    test_phrase = "Scene is stable."

    print(f"\nFirst call (cold cache): '{test_phrase}'")
    start = time.perf_counter()
    speak(test_phrase, blocking=True)
    first_time = (time.perf_counter() - start) * 1000
    print(f"  Latency: {first_time:.1f}ms")

    print(f"\nSecond call (cached): '{test_phrase}'")
    start = time.perf_counter()
    speak(test_phrase, blocking=True)
    second_time = (time.perf_counter() - start) * 1000
    print(f"  Latency: {second_time:.1f}ms")

    if first_time > 0:
        improvement = ((first_time - second_time) / first_time) * 100
        print(f"\n  Cache speedup: {improvement:.1f}% faster!")

    print(f"\n  Cached phrases: {len(_audio_cache)}")


def demo_image_to_speech() -> None:
    print("\n" + "=" * 50)
    print("Image -> VLM -> TTS Demo")
    print("=" * 50)

    test_dir = Path("test/test_images")
    image_files = sorted([*test_dir.glob("*.png"), *test_dir.glob("*.jpg"), *test_dir.glob("*.jpeg")])
    if not image_files:
        print("No images found in test/test_images/")
        return

    frames = [p.read_bytes() for p in image_files]
    results = run_frame_sequence(
        frames,
        context="general",
        use_depth=False,
        return_annotated=False,
        wait_for_llm=True,
        llm_worker_mode="process",
    )

    for img_path, res in zip(image_files, results, strict=True):
        llm = res.get("llm", {})
        dets = res.get("detections", [])
        print(f"\n>> {img_path.name}")
        print(f"   Objects: {', '.join(d.get('class_name', 'unknown') for d in dets) if dets else 'none'}")
        if llm.get("description"):
            text = llm["description"]
            print(f"   VLM: {text}")
            speak(text, blocking=False)
        elif llm.get("error"):
            print(f"   VLM Error: {llm['error']}")
        else:
            print("   VLM: (no message)")


def measure_synthesis_latency() -> None:
    print("\n" + "=" * 50)
    print("TTS Synthesis Latency Measurement")
    print("=" * 50)
    test_phrases = [
        "Scene is stable.",
        "A person is on the right.",
        "A vehicle is close ahead.",
    ]

    for phrase in test_phrases:
        start = time.perf_counter()
        wav_bytes = synthesize(phrase, use_cache=False)
        elapsed = (time.perf_counter() - start) * 1000
        audio_duration = len(wav_bytes) / (22050 * 2) * 1000
        print(f"\n  '{phrase[:30]}...'")
        print(f"    Synthesis: {elapsed:.1f}ms")
        print(f"    Audio size: {len(wav_bytes) / 1024:.1f} KB")
        print(f"    Audio duration: ~{audio_duration:.0f}ms")


def main() -> None:
    print("\n" + "=" * 60)
    print("  LiveGuide Text-to-Speech Demo")
    print("=" * 60)
    demo_tts_basic()
    demo_tts_caching()
    measure_synthesis_latency()
    demo_image_to_speech()
    print("\n" + "=" * 60)
    print("  Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
