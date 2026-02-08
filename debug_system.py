import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

def test_tts():
    print("\n--- Testing TTS (Audio) ---")
    try:
        from models.tts.piper import speak
        print("Attempting to speak: 'System check initiated.'")
        speak("System check initiated.", blocking=True)
        print("TTS command sent. Did you hear audio?")
        return True
    except Exception as e:
        print(f"TTS Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gemini():
    print("\n--- Testing Gemini API ---")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = input("Enter Gemini API Key for testing: ").strip()
        os.environ["GEMINI_API_KEY"] = api_key
    
    if not api_key:
        print("Skipping Gemini test (no key provided)")
        return False

    try:
        from models.llm.gemini import describe_scene
        print("Sending test prompt to Gemini...")
        desc = describe_scene("Describe a peaceful garden in one sentence.", max_new_tokens=20)
        print(f"Gemini Response: {desc}")
        return True
    except Exception as e:
        print(f"Gemini Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_imports():
    print("\n--- Checking Imports ---")
    try:
        import gradio
        print(f"Gradio: {gradio.__version__}")
        import sounddevice
        print(f"SoundDevice: {sounddevice.__version__}")
        import piper
        print("Piper: Installed")
    except ImportError as e:
        print(f"Import Error: {e}")

if __name__ == "__main__":
    check_imports()
    test_tts()
    test_gemini()
    print("\nDebug complete.")
