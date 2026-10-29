import os
import wave
import struct
import math
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(".")
AUDIO_DIR = os.path.join(PROJECT_ROOT, "data", "sample_multimodal", "audio")
IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "sample_multimodal", "images")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)


def generate_synthetic_audio(filename: str, duration_sec: float = 2.5, freq: float = 440.0):
    """Generates a valid mono 16kHz PCM WAV file with audio cadence."""
    sample_rate = 16000
    num_samples = int(sample_rate * duration_sec)
    path = os.path.join(AUDIO_DIR, filename)

    with wave.open(path, "w") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        # Modulate frequency to mimic speech pitch fluctuations
        for i in range(num_samples):
            t = float(i) / sample_rate
            pitch = freq + 40.0 * math.sin(2 * math.pi * 3.0 * t)
            # Syllabic amplitude envelope
            envelope = 0.5 * (1.0 + math.sin(2 * math.pi * 4.0 * t))
            sample = int(32767.0 * 0.4 * envelope * math.sin(2 * math.pi * pitch * t))
            wav_file.writeframes(struct.pack("<h", sample))

    print(f"Generated audio: {path}")


def generate_pothole_image(filename: str):
    """Creates a synthetic photo of an asphalt road with a prominent pothole."""
    path = os.path.join(IMAGE_DIR, filename)
    w, h = 640, 480
    # Dark grey asphalt road texture
    img = np.random.randint(60, 95, (h, w, 3), dtype=np.uint8)

    # Road lane marking (white/yellow dashed line on side)
    cv2.line(img, (w // 2, 0), (w // 2, h), (200, 200, 200), 4)

    # Pothole cavity (dark irregular elliptical crater in center)
    center = (w // 2 + 30, int(h * 0.65))
    axes = (110, 65)
    cv2.ellipse(img, center, axes, 15, 0, 360, (25, 25, 28), -1)
    # Cracks and water reflection inside pothole
    cv2.ellipse(img, (center[0] - 10, center[1] + 5), (60, 30), 10, 0, 360, (40, 45, 55), -1)
    # Asphalt cracking lines radiating outward
    for angle in [0, 45, 90, 135, 180, 220, 300]:
        rad = math.radians(angle)
        pt1 = (int(center[0] + axes[0] * 0.8 * math.cos(rad)), int(center[1] + axes[1] * 0.8 * math.sin(rad)))
        pt2 = (int(center[0] + (axes[0] + 50) * math.cos(rad)), int(center[1] + (axes[1] + 35) * math.sin(rad)))
        cv2.line(img, pt1, pt2, (30, 30, 30), 2)

    cv2.imwrite(path, img)
    print(f"Generated pothole image: {path}")


def generate_garbage_image(filename: str):
    """Creates a synthetic photo of overflowing garbage dump."""
    path = os.path.join(IMAGE_DIR, filename)
    w, h = 640, 480
    # Pavement background
    img = np.full((h, w, 3), 110, dtype=np.uint8)

    # Dumpster / garbage container outline (dark blue metal)
    cv2.rectangle(img, (120, 160), (520, 400), (95, 70, 45), -1)
    cv2.rectangle(img, (120, 160), (520, 400), (60, 45, 25), 4)

    # Overflowing multicolored garbage bags and waste
    colors = [(220, 220, 220), (30, 30, 200), (200, 180, 30), (50, 120, 200), (40, 40, 40)]
    for _ in range(60):
        cx = np.random.randint(150, 490)
        cy = np.random.randint(130, 420)
        rad = np.random.randint(15, 40)
        c = colors[np.random.randint(0, len(colors))]
        cv2.circle(img, (cx, cy), rad, c, -1)
        cv2.circle(img, (cx, cy), rad, (30, 30, 30), 1)

    cv2.imwrite(path, img)
    print(f"Generated garbage image: {path}")


def generate_fallen_tree_image(filename: str):
    """Creates a synthetic photo of a fallen tree across a road."""
    path = os.path.join(IMAGE_DIR, filename)
    w, h = 640, 480
    # Road background
    img = np.full((h, w, 3), 75, dtype=np.uint8)

    # Large fallen trunk across the road
    cv2.line(img, (40, 380), (600, 150), (35, 55, 85), 36)  # Brown trunk
    # Foliage and dense branches
    for _ in range(45):
        bx = np.random.randint(100, 550)
        by = np.random.randint(120, 360)
        rad = np.random.randint(20, 50)
        cv2.circle(img, (bx, by), rad, (25, 110, 35), -1)

    cv2.imwrite(path, img)
    print(f"Generated fallen tree image: {path}")


if __name__ == "__main__":
    generate_synthetic_audio("BBMP-SAMPLE-006.wav", duration_sec=2.0, freq=380.0)
    generate_synthetic_audio("BBMP-SAMPLE-009.wav", duration_sec=2.0, freq=410.0)
    generate_pothole_image("BBMP-SAMPLE-007.jpg")
    generate_garbage_image("BBMP-SAMPLE-008.jpg")
    generate_fallen_tree_image("BBMP-SAMPLE-010.jpg")
