"""
Basic anti-spoofing / liveness checks for the Face Authentication Attendance System.

Implements:
1. Eye-blink detection (EAR-based)
2. Head movement detection (frame-to-frame landmark shift)
3. Face texture consistency (Laplacian variance as simple "real vs flat" heuristic)

DISCLAIMER: This is BASIC anti-spoofing for assessment purposes. It is NOT foolproof.
Sophisticated spoofs (high-quality prints, screens, 3D masks) may bypass these checks.
For production, use dedicated liveness SDKs or hardware (e.g., depth cameras).
"""

import cv2
import numpy as np
from collections import deque
from typing import Optional
from dataclasses import dataclass

# Optional: use dlib for landmarks if available (better EAR); fallback to Haar for eyes
try:
    import dlib
    HAS_DLIB = True
except ImportError:
    HAS_DLIB = False


# ---------------------------------------------------------------------------
# Eye Aspect Ratio (EAR) for blink detection
# ---------------------------------------------------------------------------

def eye_aspect_ratio(eye_points: np.ndarray) -> float:
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    eye_points: 6 (x,y) points in order [p1..p6] (left to right for one eye).
    """
    if eye_points is None or len(eye_points) < 6:
        return 0.0
    p1, p2, p3, p4, p5, p6 = eye_points[:6]
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p4)
    if h < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def get_eye_landmarks_dlib(face_rect, gray: np.ndarray, predictor) -> Optional[tuple]:
    """Get left and right eye 6-point arrays from dlib predictor. Returns (left_eye, right_eye) or None."""
    if not HAS_DLIB or predictor is None:
        return None
    # dlib 68-point: left eye 36-41, right eye 42-47
    landmarks = predictor(gray, face_rect)
    left = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(36, 42)])
    right = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(42, 48)])
    return (left, right)


def get_eye_landmarks_from_bbox(face_bbox: tuple, gray: np.ndarray, predictor) -> Optional[tuple]:
    """face_bbox = (top, right, bottom, left) in face_recognition convention."""
    if not HAS_DLIB or predictor is None:
        return None
    top, right, bottom, left = face_bbox
    from dlib import rectangle
    rect = rectangle(int(left), int(top), int(right), int(bottom))
    return get_eye_landmarks_dlib(rect, gray, predictor)


# ---------------------------------------------------------------------------
# Blink detection (EAR drops below threshold then rises)
# ---------------------------------------------------------------------------

EAR_THRESHOLD = 0.2
EAR_CONSEC_FRAMES = 2
BLINK_WINDOW = 30  # frames to look for one blink


@dataclass
class BlinkState:
    ear_history: deque
    consecutive_low: int
    blink_count: int
    last_blink_frame: int


def init_blink_state() -> BlinkState:
    return BlinkState(
        ear_history=deque(maxlen=BLINK_WINDOW),
        consecutive_low=0,
        blink_count=0,
        last_blink_frame=-999,
    )


def check_blink(
    state: BlinkState,
    current_ear: float,
    frame_index: int,
    min_ear_threshold: float = EAR_THRESHOLD,
    consec_frames: int = EAR_CONSEC_FRAMES,
) -> tuple[bool, BlinkState]:
    """
    Returns (blink_detected, updated_state).
    A blink is counted when EAR goes low for consec_frames then goes high again.
    """
    state.ear_history.append(current_ear)
    if current_ear < min_ear_threshold:
        state.consecutive_low += 1
        return False, state
    # EAR is above threshold
    if state.consecutive_low >= consec_frames and (frame_index - state.last_blink_frame) > 5:
        state.blink_count += 1
        state.last_blink_frame = frame_index
    state.consecutive_low = 0
    return False, state


def detect_blink_this_frame(
    state: BlinkState,
    current_ear: float,
    frame_index: int,
) -> tuple[bool, BlinkState]:
    """Returns True in the frame when a blink is completed (EAR just went back up after low)."""
    prev_low = state.consecutive_low
    blink_detected = False
    if current_ear < EAR_THRESHOLD:
        state.consecutive_low += 1
    else:
        if prev_low >= EAR_CONSEC_FRAMES and (frame_index - state.last_blink_frame) > 5:
            blink_detected = True
            state.blink_count += 1
            state.last_blink_frame = frame_index
        state.consecutive_low = 0
    state.ear_history.append(current_ear)
    return blink_detected, state


# ---------------------------------------------------------------------------
# Head movement detection (face bbox / landmark shift between frames)
# ---------------------------------------------------------------------------

def head_movement_score(prev_bbox: Optional[tuple], curr_bbox: tuple) -> float:
    """
    prev/curr_bbox: (top, right, bottom, left).
    Returns a normalized movement score (0 = no movement, higher = more movement).
    """
    if prev_bbox is None:
        return 0.0
    t1, r1, b1, l1 = prev_bbox
    t2, r2, b2, l2 = curr_bbox
    cx1, cy1 = (l1 + r1) / 2, (t1 + b1) / 2
    cx2, cy2 = (l2 + r2) / 2, (t2 + b2) / 2
    w1, h1 = r1 - l1, b1 - t1
    size = max(w1, h1, 1)
    dx = (cx2 - cx1) / size
    dy = (cy2 - cy1) / size
    return np.sqrt(dx * dx + dy * dy)


HEAD_MOVEMENT_THRESHOLD = 0.08  # minimum movement over recent frames to count as "moved"
HEAD_MOVEMENT_FRAMES = 10


def update_head_movement_history(history: deque, score: float, maxlen: int = HEAD_MOVEMENT_FRAMES) -> deque:
    if len(history) >= maxlen:
        history.popleft()
    history.append(score)
    return history


def has_recent_head_movement(history: deque, threshold: float = HEAD_MOVEMENT_THRESHOLD) -> bool:
    return len(history) >= 3 and sum(history) >= threshold


# ---------------------------------------------------------------------------
# Face texture consistency (Laplacian variance — real faces have more detail)
# ---------------------------------------------------------------------------

def face_texture_score(face_roi: np.ndarray) -> float:
    """
    Laplacian variance: higher for detailed (real) textures, lower for flat/printed.
    face_roi: BGR or grayscale patch of the face.
    """
    if face_roi is None or face_roi.size == 0:
        return 0.0
    if len(face_roi.shape) == 3:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_roi
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


TEXTURE_MIN_REAL = 80.0  # below this often flat/print; tune per camera


def is_likely_real_texture(face_roi: np.ndarray, min_variance: float = TEXTURE_MIN_REAL) -> bool:
    return face_texture_score(face_roi) >= min_variance


# ---------------------------------------------------------------------------
# Combined liveness result
# ---------------------------------------------------------------------------

@dataclass
class LivenessResult:
    passed: bool
    blink_detected: bool
    head_moved: bool
    texture_ok: bool
    message: str


def check_liveness(
    blink_detected: bool,
    head_moved: bool,
    texture_ok: bool,
    require_blink: bool = True,
    require_head_move: bool = True,
    require_texture: bool = False,
) -> LivenessResult:
    """
    At least two checks: blink + head movement by default; optionally texture.
    require_texture = True uses all three (still basic anti-spoof).
    """
    checks = []
    if require_blink:
        checks.append(("blink", blink_detected))
    if require_head_move:
        checks.append(("head_move", head_moved))
    if require_texture:
        checks.append(("texture", texture_ok))
    passed_count = sum(1 for _, v in checks if v)
    passed = passed_count >= 2 if len(checks) >= 2 else passed_count >= 1
    msg = "Liveness OK" if passed else f"Need 2+ liveness checks (blink={blink_detected}, head={head_moved}, texture={texture_ok})"
    return LivenessResult(
        passed=passed,
        blink_detected=blink_detected,
        head_moved=head_moved,
        texture_ok=texture_ok,
        message=msg,
    )
