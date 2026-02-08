"""Text-to-Speech Demo for LiveGuide.

Demonstrates the Piper TTS integration for hazard alert audio output.
Run: python tts_demo.py

This file contains all audio-related functionality for the LiveGuide project.
"""

import time
from pathlib import Path

from models.tts import speak, synthesize
from models.tts.piper import preload_cache, COMMON_PHRASES, _audio_cache
from models.yolo import detect_image_bytes, classify_hazard, format_hazard_alert


def demo_tts_basic():
    """Basic TTS demonstration - speak test phrases."""
    print("=" * 50)
    print("TTS Basic Demo")
    print("=" * 50)
    
    test_phrases = [
        "Path is clear.",
        "Caution: Person detected ahead.",
        "Warning: Car approaching at high speed.",
    ]
    
    for phrase in test_phrases:
        print(f"\nSpeaking: '{phrase}'")
        start = time.perf_counter()
        speak(phrase, blocking=True)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  Latency: {elapsed:.1f}ms")


def demo_tts_caching():
    """Demonstrate audio caching performance improvement."""
    print("\n" + "=" * 50)
    print("TTS Caching Demo")
    print("=" * 50)
    
    test_phrase = "Path is clear."
    
    # First call - synthesizes and caches
    print(f"\nFirst call (cold cache): '{test_phrase}'")
    start = time.perf_counter()
    speak(test_phrase, blocking=True)
    first_time = (time.perf_counter() - start) * 1000
    print(f"  Latency: {first_time:.1f}ms")
    
    # Second call - uses cache
    print(f"\nSecond call (cached): '{test_phrase}'")
    start = time.perf_counter()
    speak(test_phrase, blocking=True)
    second_time = (time.perf_counter() - start) * 1000
    print(f"  Latency: {second_time:.1f}ms")
    
    # Show improvement
    if first_time > 0:
        improvement = ((first_time - second_time) / first_time) * 100
        print(f"\n  Cache speedup: {improvement:.1f}% faster!")
    
    print(f"\n  Cached phrases: {len(_audio_cache)}")


def demo_hazard_to_speech():
    """End-to-end demo: Image -> Hazard Detection -> TTS."""
    print("\n" + "=" * 50)
    print("Hazard Detection + TTS Demo")
    print("=" * 50)
    
    test_dir = Path("data/test_images")
    image_files = list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg"))
    
    if not image_files:
        print("No images found in data/test_images/")
        return
    
    print(f"Found {len(image_files)} image(s)\n")
    
    for img_path in sorted(image_files):
        print(f">> {img_path.name}")
        
        # Detect objects
        img_bytes = img_path.read_bytes()
        yolo_start = time.perf_counter()
        result = detect_image_bytes(img_bytes, use_depth=False)
        yolo_time = (time.perf_counter() - yolo_start) * 1000
        
        # Classify hazards
        detections = result["detections"]
        hazard = classify_hazard(detections, context="walking")
        alert = format_hazard_alert(hazard)
        
        # Speak alert
        tts_start = time.perf_counter()
        speak(alert, blocking=False)
        tts_time = (time.perf_counter() - tts_start) * 1000
        
        # Display results
        if detections:
            objects = [d["class_name"] for d in detections]
            print(f"   Objects: {', '.join(objects)}")
        print(f"   Hazard: {hazard['hazard_level'].upper()}")
        print(f"   Alert: {alert}")
        print(f"   Latency: YOLO={yolo_time:.1f}ms | TTS={tts_time:.1f}ms")
        print()


def measure_synthesis_latency():
    """Measure pure synthesis latency (no playback)."""
    print("\n" + "=" * 50)
    print("TTS Synthesis Latency Measurement")
    print("=" * 50)
    
    test_phrases = [
        "Path is clear.",
        "Caution advised.",
        "Warning: Car at seventy percent confidence.",
    ]
    
    for phrase in test_phrases:
        # Clear cache to measure fresh synthesis
        start = time.perf_counter()
        wav_bytes = synthesize(phrase, use_cache=False)
        elapsed = (time.perf_counter() - start) * 1000
        
        audio_duration = len(wav_bytes) / (22050 * 2) * 1000  # approx ms
        print(f"\n  '{phrase[:30]}...'")
        print(f"    Synthesis: {elapsed:.1f}ms")
        print(f"    Audio size: {len(wav_bytes) / 1024:.1f} KB")
        print(f"    Audio duration: ~{audio_duration:.0f}ms")


def main():
    """Run all TTS demos."""
    print("\n" + "=" * 60)
    print("  LiveGuide Text-to-Speech Demo")
    print("  Using Piper TTS (en_US-lessac-medium)")
    print("=" * 60)
    
    print("\nInitializing TTS (first call loads model)...")
    
    # Run demos
    demo_tts_basic()
    demo_tts_caching()
    measure_synthesis_latency()
    demo_hazard_to_speech()
    
    print("\n" + "=" * 60)
    print("  Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
