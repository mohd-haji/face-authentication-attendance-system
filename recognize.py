"""
Real-time face recognition and punch-in / punch-out for the Face Authentication Attendance System.
Uses live webcam, identifies user, runs basic anti-spoof, then records punch.
"""

import sys
from typing import Optional
import cv2
import numpy as np
import face_recognition
from collections import deque
from pathlib import Path

from utils import (
    ensure_dirs,
    list_registered_users,
    load_face_encodings,
    open_camera,
    bgr_to_rgb,
    normalize_lighting,
    get_storage,
)
from attendance import get_punch_state, get_next_punch_type, record_punch
from anti_spoof import (
    get_eye_landmarks_from_bbox,
    eye_aspect_ratio,
    init_blink_state,
    detect_blink_this_frame,
    head_movement_score,
    update_head_movement_history,
    has_recent_head_movement,
    face_texture_score,
    is_likely_real_texture,
    check_liveness,
    LivenessResult,
)

# Optional dlib predictor for EAR-based blink
try:
    import dlib
    PREDICTOR_PATH = Path(__file__).resolve().parent / "shape_predictor_68_face_landmarks.dat"
    if PREDICTOR_PATH.exists():
        _predictor = dlib.shape_predictor(str(PREDICTOR_PATH))
    else:
        _predictor = None
except Exception:
    _predictor = None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FACE_DETECTION_MODEL = "hog"
TOLERANCE = 0.5
MIN_FACE_SIZE = 40
LIVENESS_REQUIRE_BLINK = True
LIVENESS_REQUIRE_HEAD_MOVE = True
LIVENESS_REQUIRE_TEXTURE = False
LIVENESS_FRAMES = 30  # collect liveness over this many frames before allowing punch
HEAD_MOVEMENT_FRAMES = 15


# ---------------------------------------------------------------------------
# Load all known encodings
# ---------------------------------------------------------------------------

def load_all_encodings() -> tuple[list, list, list]:
    """Returns (encodings_list, names_list, user_ids_list)."""
    users = list_registered_users()
    all_encodings = []
    names = []
    user_ids = []
    for u in users:
        uid, name = u["user_id"], u["name"]
        data = load_face_encodings(uid)
        if data is None:
            continue
        for enc in data["encodings"]:
            all_encodings.append(enc)
            names.append(name)
            user_ids.append(uid)
    return all_encodings, names, user_ids


def match_face(encoding, known_encodings, known_names, known_user_ids, tolerance: float = TOLERANCE) -> tuple[Optional[str], Optional[str], float]:
    """
    Match one encoding against known encodings.
    Returns (user_id, name, distance). name/user_id None if no match.
    """
    if not known_encodings:
        return None, None, float("inf")
    distances = face_recognition.face_distance(known_encodings, encoding)
    i = int(np.argmin(distances))
    d = float(distances[i])
    if d <= tolerance:
        return known_user_ids[i], known_names[i], d
    return None, None, d


# ---------------------------------------------------------------------------
# Real-time loop: detect, recognize, liveness, punch
# ---------------------------------------------------------------------------

def run_recognition(camera_index: int = 0):
    ensure_dirs()
    get_storage().init_schema()
    known_encodings, known_names, known_user_ids = load_all_encodings()
    if not known_encodings:
        print("No registered users. Run register.py first.", file=sys.stderr)
        return

    cap = open_camera(camera_index)
    if not cap.isOpened():
        print("Could not open camera.", file=sys.stderr)
        return

    blink_state = init_blink_state()
    prev_bbox = None
    head_movement_history = deque(maxlen=HEAD_MOVEMENT_FRAMES)
    frame_index = 0
    liveness_collected = []
    current_uid = None
    current_name = None
    last_punch_frame = -100

    print("Real-time recognition. Face the camera. Press P to punch, Q to quit.")
    print("Unknown faces are rejected. Liveness (blink + head move) required before punch.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            frame_index += 1
            frame_rgb = bgr_to_rgb(frame)
            frame_norm = bgr_to_rgb(normalize_lighting(frame))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            locations = face_recognition.face_locations(
                frame_rgb, model=FACE_DETECTION_MODEL, number_of_times_to_upsample=1
            )
            encodings = face_recognition.face_encodings(frame_rgb, known_face_locations=locations)
            if not encodings:
                encodings = face_recognition.face_encodings(frame_norm, known_face_locations=face_recognition.face_locations(frame_norm, model=FACE_DETECTION_MODEL))

            display_text = "No face"
            confidence = 0.0
            uid, name = None, None
            curr_bbox = None
            ear_left = ear_right = 0.2
            texture_ok = False

            if locations and encodings:
                top, right, bottom, left = locations[0]
                curr_bbox = (top, right, bottom, left)
                h, w = bottom - top, right - left
                if h >= MIN_FACE_SIZE and w >= MIN_FACE_SIZE:
                    uid, name, dist = match_face(
                        encodings[0], known_encodings, known_names, known_user_ids
                    )
                    confidence = max(0, 1 - dist / TOLERANCE) if dist is not None else 0
                    if uid is not None:
                        display_text = f"{name} ({confidence:.2f})"
                        current_uid, current_name = uid, name
                        # Head movement
                        move_score = head_movement_score(prev_bbox, curr_bbox)
                        update_head_movement_history(head_movement_history, move_score)
                        head_moved = has_recent_head_movement(head_movement_history)
                        # Texture
                        face_roi = gray[top:bottom, left:right]
                        texture_ok = is_likely_real_texture(face_roi)
                        # Blink (EAR)
                        if _predictor is not None:
                            eyes = get_eye_landmarks_from_bbox(curr_bbox, gray, _predictor)
                            if eyes is not None:
                                left_eye, right_eye = eyes
                                ear_left = eye_aspect_ratio(left_eye)
                                ear_right = eye_aspect_ratio(right_eye)
                                ear = (ear_left + ear_right) / 2.0
                                blink_detected, blink_state = detect_blink_this_frame(
                                    blink_state, ear, frame_index
                                )
                            else:
                                blink_detected = False
                        else:
                            blink_detected = head_moved  # fallback: use head move as one check
                        liveness = check_liveness(
                            blink_detected=blink_detected,
                            head_moved=head_moved,
                            texture_ok=texture_ok,
                            require_blink=LIVENESS_REQUIRE_BLINK,
                            require_head_move=LIVENESS_REQUIRE_HEAD_MOVE,
                            require_texture=LIVENESS_REQUIRE_TEXTURE,
                        )
                        liveness_collected.append(liveness)
                        if len(liveness_collected) > LIVENESS_FRAMES:
                            liveness_collected.pop(0)
                        # Draw box and status
                        color = (0, 255, 0) if uid else (0, 0, 255)
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        cv2.putText(
                            frame, display_text, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                        )
                        passed_count = sum(1 for l in liveness_collected if l.passed)
                        cv2.putText(
                            frame, f"Liveness: {passed_count}/{len(liveness_collected)}",
                            (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        )
                    else:
                        display_text = "Unknown"
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(
                            frame, "Unknown", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                        )
                prev_bbox = curr_bbox
            else:
                prev_bbox = None
                current_uid, current_name = None, None

            cv2.putText(
                frame, "P=punch | Q=quit", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
            cv2.imshow("Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                if current_uid is None or current_name is None:
                    print("Cannot punch: no recognized user.")
                    continue
                if frame_index - last_punch_frame < 30:
                    print("Wait a moment before punching again.")
                    continue
                # Require at least one passed liveness in recent frames
                recent_passed = any(l.passed for l in liveness_collected[-10:]) if liveness_collected else False
                if not recent_passed and (LIVENESS_REQUIRE_BLINK or LIVENESS_REQUIRE_HEAD_MOVE):
                    print("Complete liveness (blink + head movement) before punching.")
                    continue
                punch_type = get_next_punch_type(current_uid, current_name)
                if record_punch(current_uid, current_name, punch_type):
                    print(f"Punch-{punch_type.upper()} recorded for {current_name} ({current_uid}).")
                    last_punch_frame = frame_index
                else:
                    print("Punch not recorded (duplicate or invalid state).")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Real-time face recognition and attendance punch.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    args = parser.parse_args()
    run_recognition(args.camera)


if __name__ == "__main__":
    main()
