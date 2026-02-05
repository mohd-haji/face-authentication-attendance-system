"""
Face registration for the Face Authentication Attendance System.
Captures multiple face samples per user, computes and stores face embeddings.
Rejects registration if face is not clearly detected.
"""

import sys
import cv2
import numpy as np
import face_recognition
from pathlib import Path

from utils import (
    ensure_dirs,
    save_face_encodings,
    load_face_encodings,
    FACES_DIR,
    open_camera,
    bgr_to_rgb,
    normalize_lighting,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NUM_SAMPLES = 5
SAMPLE_INTERVAL_MS = 400
MIN_FACE_SIZE = 40
FACE_DETECTION_MODEL = "hog"  # "hog" (faster) or "cnn" (more accurate, needs GPU)

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def detect_face_and_encoding(frame_rgb: np.ndarray) -> tuple[list, list]:
    """
    Detect faces and compute encodings.
    Returns (face_locations, encodings). Empty if no face or multiple faces.
    """
    locations = face_recognition.face_locations(
        frame_rgb, model=FACE_DETECTION_MODEL, number_of_times_to_upsample=1
    )
    if len(locations) != 1:
        return [], []
    top, right, bottom, left = locations[0]
    h, w = bottom - top, right - left
    if h < MIN_FACE_SIZE or w < MIN_FACE_SIZE:
        return [], []
    encodings = face_recognition.face_encodings(frame_rgb, known_face_locations=locations)
    if not encodings:
        return [], []
    return locations, encodings


def collect_samples(camera_index: int = 0) -> list:
    """
    Collect NUM_SAMPLES face encodings from the camera.
    Uses both raw and lighting-normalized frames for robustness.
    Returns list of 128-dim encodings.
    """
    cap = open_camera(camera_index)
    if not cap.isOpened():
        print("Error: Could not open camera.", file=sys.stderr)
        return []

    collected = []
    sample_count = 0
    last_capture_time = 0

    print(f"Look at the camera. We need {NUM_SAMPLES} clear face samples.")
    print("Press SPACE to capture a sample, Q to finish early.")

    try:
        while sample_count < NUM_SAMPLES:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            frame_rgb = bgr_to_rgb(frame)
            frame_normalized = bgr_to_rgb(normalize_lighting(frame))
            locations, encodings = detect_face_and_encoding(frame_rgb)
            if not encodings:
                locations, encodings = detect_face_and_encoding(frame_normalized)
            if encodings:
                # Draw face box
                top, right, bottom, left = locations[0]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"Face detected - Sample {sample_count + 1}/{NUM_SAMPLES}",
                    (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
            else:
                cv2.putText(
                    frame, "No face or multiple faces. Position clearly.",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                )
            cv2.putText(
                frame, f"SPACE=capture | Q=quit | Samples: {sample_count}/{NUM_SAMPLES}",
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
            cv2.imshow("Registration - Face samples", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                if encodings:
                    collected.append(encodings[0])
                    sample_count += 1
                    print(f"  Sample {sample_count}/{NUM_SAMPLES} captured.")
                else:
                    print("  No clear face. Try again.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return collected


def register_user(user_id: str, name: str, camera_index: int = 0) -> bool:
    """
    Register a new user: collect samples, compute encodings, save to data/faces/.
    Returns True on success. Rejects if face not clearly detected (insufficient samples).
    """
    if not user_id.strip() or not name.strip():
        print("Error: user_id and name must be non-empty.", file=sys.stderr)
        return False
    if load_face_encodings(user_id.strip()) is not None:
        print(f"User '{user_id}' already registered. Delete {FACES_DIR / (user_id.strip() + '.json')} to re-register.", file=sys.stderr)
        return False

    ensure_dirs()
    encodings = collect_samples(camera_index)
    if len(encodings) < max(2, NUM_SAMPLES // 2):
        print("Registration failed: not enough clear face samples.", file=sys.stderr)
        return False
    save_face_encodings(user_id.strip(), name.strip(), encodings)
    print(f"Registered: {name} ({user_id}) with {len(encodings)} encodings.")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Register a user's face for attendance.")
    parser.add_argument("--user-id", required=True, help="Unique user ID")
    parser.add_argument("--name", required=True, help="Display name")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    args = parser.parse_args()
    ok = register_user(args.user_id, args.name, args.camera)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
