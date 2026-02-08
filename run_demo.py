"""Simple LiveGuide demo - YOLO objects + VLM description + optional TTS.

Run: python run_demo.py
"""

from pathlib import Path

from main import run_frame_sequence
from models.tts import speak


def main() -> int:
    test_dir = Path("test/test_images")
    image_files = sorted([*test_dir.glob("*.png"), *test_dir.glob("*.jpg"), *test_dir.glob("*.jpeg")])

    if not image_files:
        print("No images found in test/test_images/")
        return 1

    print(f"Found {len(image_files)} image(s)")
    print("=" * 50)

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
        print(f"\n>> {img_path.name}")
        detections = res.get("detections", [])
        if detections:
            objects = [d.get("class_name", "unknown") for d in detections]
            print(f"   Objects: {', '.join(objects)}")
        else:
            print("   Objects: none")

        llm = res.get("llm", {})
        if llm.get("description"):
            text = llm["description"]
            print(f"   VLM: {text}")
            speak(text, blocking=False)
        elif llm.get("error"):
            print(f"   VLM Error: {llm['error']}")
        else:
            print("   VLM: (no message)")

    print("\n" + "=" * 50)
    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
