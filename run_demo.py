"""Simple LiveGuide demo - YOLO hazard detection + TTS speech.

Run: python run_demo.py
"""
from pathlib import Path

from models.tts import speak
from models.yolo import classify_hazard, detect_image_bytes, format_hazard_alert


def main():
    test_dir = Path("data/test_images")
    image_files = list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg"))
    
    if not image_files:
        print("No images found in data/test_images/")
        return
    
    print(f"Found {len(image_files)} image(s)")
    print("=" * 50)
    
    for img_path in sorted(image_files):
        print(f"\n>> {img_path.name}")
        
        # Detect objects with YOLO (no depth to speed things up)
        img_bytes = img_path.read_bytes()
        result = detect_image_bytes(img_bytes, use_depth=False)
        detections = result["detections"]
        
        # Classify hazards
        hazard = classify_hazard(detections, context="walking")
        alert = format_hazard_alert(hazard)
        
        # Show results
        if detections:
            objects = [d["class_name"] for d in detections]
            print(f"   Objects: {', '.join(objects)}")
        print(f"   Hazard: {hazard['hazard_level'].upper()}")
        print(f"   Alert: {alert}")
        
        # SPEAK the alert!
        speak(alert)
    
    print("\n" + "=" * 50)
    print("Done!")


if __name__ == "__main__":
    main()
