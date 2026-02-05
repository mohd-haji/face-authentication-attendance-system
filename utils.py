"""
Shared utilities for the Face Authentication Attendance System.
Handles storage abstraction, camera capture, and lighting normalization.
"""

import os
import json
import csv
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
FACES_DIR = DATA_DIR / "faces"
ATTENDANCE_DB = DATA_DIR / "attendance.db"
ATTENDANCE_CSV = DATA_DIR / "attendance.csv"

# Storage backend: "sqlite" or "csv"
STORAGE_BACKEND = os.environ.get("ATTENDANCE_STORAGE", "sqlite")


def ensure_dirs():
    """Create data directories if they do not exist."""
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Face encodings storage (JSON per user)
# ---------------------------------------------------------------------------


def get_face_encoding_path(user_id: str) -> Path:
    return FACES_DIR / f"{user_id}.json"


def save_face_encodings(user_id: str, name: str, encodings: list) -> None:
    """Save face encodings as JSON (list of 128-dim arrays)."""
    ensure_dirs()
    path = get_face_encoding_path(user_id)
    data = {
        "user_id": user_id,
        "name": name,
        "encodings": [enc.tolist() if hasattr(enc, "tolist") else enc for enc in encodings],
        "created_at": datetime.now().isoformat(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_face_encodings(user_id: str) -> Optional[dict]:
    """Load face encodings for a user. Returns None if not found."""
    path = get_face_encoding_path(user_id)
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    data["encodings"] = [np.array(enc) for enc in data["encodings"]]
    return data


def list_registered_users() -> list[dict]:
    """List all registered users (user_id, name) from faces dir."""
    ensure_dirs()
    users = []
    for path in FACES_DIR.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
            users.append({"user_id": data["user_id"], "name": data["name"]})
        except (json.JSONDecodeError, KeyError):
            continue
    return users


# ---------------------------------------------------------------------------
# Attendance storage (abstracted: SQLite or CSV)
# ---------------------------------------------------------------------------


class StorageBackend:
    """Abstract interface for attendance record storage."""

    def init_schema(self) -> None:
        raise NotImplementedError

    def append_record(
        self,
        user_id: str,
        name: str,
        punch_type: str,
        timestamp: Optional[datetime] = None,
        session_minutes: Optional[float] = None,
    ) -> None:
        raise NotImplementedError

    def get_records(
        self,
        user_id: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> list[dict]:
        raise NotImplementedError


class SQLiteBackend(StorageBackend):
    def __init__(self, path: Path = ATTENDANCE_DB):
        self.path = path

    def _conn(self):
        ensure_dirs()
        return sqlite3.connect(str(self.path))

    def init_schema(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    punch_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    session_minutes REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_attendance_user_date ON attendance(user_id, timestamp)"
            )

    def append_record(
        self,
        user_id: str,
        name: str,
        punch_type: str,
        timestamp: Optional[datetime] = None,
        session_minutes: Optional[float] = None,
    ) -> None:
        self.init_schema()
        ts = timestamp or datetime.now()
        ts_str = ts.isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO attendance (user_id, name, punch_type, timestamp, session_minutes) VALUES (?, ?, ?, ?, ?)",
                (user_id, name, punch_type, ts_str, session_minutes),
            )

    def get_records(
        self,
        user_id: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> list[dict]:
        self.init_schema()
        query = "SELECT user_id, name, punch_type, timestamp, session_minutes FROM attendance WHERE 1=1"
        params = []
        if user_id is not None:
            query += " AND user_id = ?"
            params.append(user_id)
        if date_from is not None:
            query += " AND timestamp >= ?"
            params.append(date_from.isoformat())
        if date_to is not None:
            query += " AND timestamp <= ?"
            params.append(date_to.isoformat())
        query += " ORDER BY timestamp ASC"
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


class CSVBackend(StorageBackend):
    def __init__(self, path: Path = ATTENDANCE_CSV):
        self.path = path

    def init_schema(self) -> None:
        ensure_dirs()
        headers = ["user_id", "name", "punch_type", "timestamp", "session_minutes"]
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(headers)
            return
        # File exists: ensure it has a proper header. If not, repair by inserting header and preserving rows.
        with open(self.path, "r", newline="", encoding="utf-8") as f:
            lines = f.readlines()
        # Strip leading blank lines
        idx = 0
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
        if idx >= len(lines):
            # empty file, write header
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(headers)
            return
        first_non_empty = lines[idx].strip()
        # If first non-empty line contains header keywords, assume header present
        if "user_id" in first_non_empty and "timestamp" in first_non_empty:
            return
        # Otherwise, repair: assume file has rows without header. Parse and rewrite with header.
        import io
        reader = csv.reader(io.StringIO("".join(lines)))
        rows = [r for r in reader if any(cell.strip() for cell in r)]
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
            for row in rows:
                # Ensure each row has exactly 5 columns (pad or truncate)
                row_to_write = row[:5] + [""] * max(0, 5 - len(row))
                w.writerow(row_to_write)

    def append_record(
        self,
        user_id: str,
        name: str,
        punch_type: str,
        timestamp: Optional[datetime] = None,
        session_minutes: Optional[float] = None,
    ) -> None:
        self.init_schema()
        ts = timestamp or datetime.now()
        ts_str = ts.isoformat()
        session_str = "" if session_minutes is None else str(session_minutes)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([user_id, name, punch_type, ts_str, session_str])

    def get_records(
        self,
        user_id: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> list[dict]:
        self.init_schema()
        if not self.path.exists():
            return []
        records = []
        with open(self.path, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            # If header is missing or malformed, DictReader.fieldnames may be None or not contain expected keys.
            # We rely on init_schema to repair header, but be defensive here: skip blank/malformed rows and strip fields.
            for row in r:
                # Skip entirely blank rows
                if not any((v or "").strip() for v in row.values()):
                    continue
                # Normalize/strip values
                row = {k: (v.strip() if v is not None else "") for k, v in row.items()}
                if user_id is not None and row.get("user_id") != user_id:
                    continue
                ts = row.get("timestamp", "")
                # Skip records without timestamp
                if not ts:
                    continue
                if date_from is not None and ts < date_from.isoformat():
                    continue
                if date_to is not None and ts > date_to.isoformat():
                    continue
                sm = row.get("session_minutes", "")
                try:
                    row["session_minutes"] = float(sm) if sm else None
                except Exception:
                    row["session_minutes"] = None
                records.append(row)
        records.sort(key=lambda x: x.get("timestamp", ""))
        return records


def get_storage() -> StorageBackend:
    if STORAGE_BACKEND.lower() == "csv":
        return CSVBackend()
    return SQLiteBackend()


# ---------------------------------------------------------------------------
# Camera & image preprocessing (lighting robustness)
# ---------------------------------------------------------------------------


def capture_frame(camera_index: int = 0) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Capture a single frame from webcam.
    Returns (original_bgr, normalized_bgr) for lighting robustness.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    normalized = normalize_lighting(frame)
    return (frame, normalized)


def normalize_lighting(frame: np.ndarray) -> np.ndarray:
    """
    Improve robustness to varying lighting:
    - Grayscale conversion for luminance
    - CLAHE (Contrast Limited Adaptive Histogram Equalization) on luminance
    - Merge back to BGR for face_recognition (expects RGB elsewhere we convert)
    """
    if len(frame.shape) == 2:
        gray = frame
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    # Keep BGR for display/consistency; face_recognition uses RGB
    return cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)


def open_camera(camera_index: int = 0):
    """Open VideoCapture for streaming. Caller must release."""
    cap = cv2.VideoCapture(camera_index)
    return cap


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
