"""
Unified GUI for Face Authentication Attendance System.
All modules in one place: Register, Punch In/Out, data saved to CSV.
"""

import os
os.environ["ATTENDANCE_STORAGE"] = "csv"

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
from collections import deque
from pathlib import Path
from datetime import datetime
import threading

# Project modules (import after env set)
from utils import (
    ensure_dirs,
    open_camera,
    bgr_to_rgb,
    normalize_lighting,
    save_face_encodings,
    load_face_encodings,
    list_registered_users,
    get_storage,
    ATTENDANCE_CSV,
)
from attendance import get_next_punch_type, record_punch, get_attendance_summary
import face_recognition

# Optional: recognition + liveness (same as recognize.py)
try:
    from recognize import (
        load_all_encodings,
        match_face,
        FACE_DETECTION_MODEL,
        TOLERANCE,
        MIN_FACE_SIZE,
    )
    from anti_spoof import (
        head_movement_score,
        update_head_movement_history,
        has_recent_head_movement,
        is_likely_real_texture,
        check_liveness,
        init_blink_state,
        detect_blink_this_frame,
        get_eye_landmarks_from_bbox,
        eye_aspect_ratio,
    )
    try:
        import dlib
        _PREDICTOR_PATH = Path(__file__).resolve().parent / "shape_predictor_68_face_landmarks.dat"
        _predictor = dlib.shape_predictor(str(_PREDICTOR_PATH)) if _PREDICTOR_PATH.exists() else None
    except Exception:
        _predictor = None
    HEAD_MOVEMENT_FRAMES = 15
    LIVENESS_FRAMES = 20
    HAS_RECOGNITION = True
except ImportError:
    HAS_RECOGNITION = False

# Register: reuse detection from register module
try:
    from register import detect_face_and_encoding, MIN_FACE_SIZE as REG_MIN_FACE, FACE_DETECTION_MODEL as REG_MODEL
except ImportError:
    REG_MIN_FACE = 40
    REG_MODEL = "hog"
    def detect_face_and_encoding(frame_rgb):
        locs = face_recognition.face_locations(frame_rgb, model=REG_MODEL, number_of_times_to_upsample=1)
        if len(locs) != 1:
            return [], []
        top, right, bottom, left = locs[0]
        if (bottom - top) < REG_MIN_FACE or (right - left) < REG_MIN_FACE:
            return [], []
        encs = face_recognition.face_encodings(frame_rgb, known_face_locations=locs)
        return (locs, encs) if encs else ([], [])

NUM_SAMPLES = 5
CAMERA_INDEX = 0
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
REG_TOLERANCE = 0.5  # face match threshold for "already registered" check


def _find_existing_user_by_face(encoding) -> tuple:
    """Compare one face encoding against all registered users. Returns (user_id, name) if match else (None, None)."""
    users = list_registered_users()
    if not users:
        return (None, None)
    all_encodings = []
    user_ids = []
    names = []
    for u in users:
        data = load_face_encodings(u["user_id"])
        if data is None:
            continue
        for enc in data["encodings"]:
            all_encodings.append(enc)
            user_ids.append(u["user_id"])
            names.append(u["name"])
    if not all_encodings:
        return (None, None)
    distances = face_recognition.face_distance(all_encodings, encoding)
    i = int(np.argmin(distances))
    if distances[i] <= REG_TOLERANCE:
        return (user_ids[i], names[i])
    return (None, None)


class AttendanceGUI:
    # Theme colors
    BG = "#e8eef4"
    CARD_BG = "#ffffff"
    ACCENT = "#0d9488"
    ACCENT_HOVER = "#0f766e"
    TEXT = "#1e293b"
    TEXT_MUTED = "#64748b"
    SUCCESS = "#059669"
    WARNING = "#d97706"

    def __init__(self):
        ensure_dirs()
        get_storage().init_schema()
        self.root = tk.Tk()
        self.root.title("Face Authentication Attendance")
        self.root.geometry("920x720")
        self.root.minsize(820, 620)
        self.root.configure(bg=self.BG)

        # Camera state
        self.cap = None
        self.camera_active = None  # "register" | "attendance" | None
        self.register_encodings = []
        self.attendance_state = None  # recognition state for attendance tab
        self._after_id = None
        self._camera_thread = None

        self._apply_styles()
        self._build_ui()

    def _apply_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=self.BG)
        style.configure("TLabel", background=self.BG, foreground=self.TEXT, font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=self.BG, foreground=self.TEXT, font=("Segoe UI", 14, "bold"))
        style.configure("Card.TFrame", background=self.CARD_BG)
        style.configure("Card.TLabel", background=self.CARD_BG, foreground=self.TEXT)
        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        style.configure("Accent.TButton", background=self.ACCENT, foreground="white", font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", self.ACCENT_HOVER), ("pressed", self.ACCENT_HOVER)])
        style.configure("TNotebook", background=self.BG)
        style.configure("TNotebook.Tab", padding=[12, 8], font=("Segoe UI", 10))
        style.configure("Treeview", rowheight=24, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

    def _build_ui(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ---- Tab: Register ----
        reg_frame = ttk.Frame(nb, padding=12)
        nb.add(reg_frame, text="  Register User  ")
        self._build_register_tab(reg_frame)

        # ---- Tab: Attendance (Punch In / Out) ----
        att_frame = ttk.Frame(nb, padding=12)
        nb.add(att_frame, text="  Punch In / Out  ")
        self._build_attendance_tab(att_frame)

        # ---- Tab: View Data (CSV) ----
        data_frame = ttk.Frame(nb, padding=12)
        nb.add(data_frame, text="  Attendance Log (CSV)  ")
        self._build_data_tab(data_frame)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_register_tab(self, parent):
        ttk.Label(parent, text="Register new user", style="Header.TLabel").grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 12))
        card = ttk.Frame(parent, style="Card.TFrame", padding=12)
        card.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=4)
        parent.columnconfigure(1, weight=1)
        ttk.Label(card, text="User ID:", style="Card.TLabel").grid(row=0, column=0, sticky=tk.W, pady=6)
        self.reg_user_id = ttk.Entry(card, width=28)
        self.reg_user_id.grid(row=0, column=1, sticky=tk.W, pady=6, padx=8)
        ttk.Label(card, text="Full Name:", style="Card.TLabel").grid(row=1, column=0, sticky=tk.W, pady=6)
        self.reg_name = ttk.Entry(card, width=28)
        self.reg_name.grid(row=1, column=1, sticky=tk.W, pady=6, padx=8)

        btn_frame = ttk.Frame(card, style="Card.TFrame")
        btn_frame.grid(row=2, column=0, columnspan=2, pady=12)
        self.reg_start_btn = ttk.Button(btn_frame, text="Start Camera", style="Accent.TButton", command=self._reg_start_camera)
        self.reg_start_btn.pack(side=tk.LEFT, padx=4)
        self.reg_capture_btn = ttk.Button(btn_frame, text="Capture Sample", command=self._reg_capture, state=tk.DISABLED)
        self.reg_capture_btn.pack(side=tk.LEFT, padx=4)
        self.reg_save_btn = ttk.Button(btn_frame, text="Save Registration", command=self._reg_save, state=tk.DISABLED)
        self.reg_save_btn.pack(side=tk.LEFT, padx=4)
        self.reg_stop_btn = ttk.Button(btn_frame, text="Stop Camera", command=self._reg_stop_camera, state=tk.DISABLED)
        self.reg_stop_btn.pack(side=tk.LEFT, padx=4)

        self.reg_status = ttk.Label(card, text="Samples: 0 / 5 — Start camera and capture 5 face samples.", style="Card.TLabel")
        self.reg_status.grid(row=3, column=0, columnspan=2, pady=6, sticky=tk.W)

        self.reg_video = ttk.Label(parent, text="[Camera off]", relief=tk.SUNKEN, anchor=tk.CENTER)
        self.reg_video.grid(row=4, column=0, columnspan=2, pady=12)

    def _build_attendance_tab(self, parent):
        ttk.Label(parent, text="Punch In / Out", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 10))
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=6)
        self.att_start_btn = ttk.Button(btn_frame, text="Start Camera", style="Accent.TButton", command=self._att_start_camera)
        self.att_start_btn.pack(side=tk.LEFT, padx=4)
        self.att_stop_btn = ttk.Button(btn_frame, text="Stop Camera", command=self._att_stop_camera, state=tk.DISABLED)
        self.att_stop_btn.pack(side=tk.LEFT, padx=4)

        self.att_status = ttk.Label(parent, text="Status: —")
        self.att_status.pack(anchor=tk.W, pady=4)
        self.att_punch_btn = ttk.Button(parent, text="Punch In", style="Accent.TButton", command=self._att_punch, state=tk.DISABLED)
        self.att_punch_btn.pack(pady=8)
        self.att_message = scrolledtext.ScrolledText(parent, height=5, width=70, state=tk.DISABLED, font=("Consolas", 9), bg="#f8fafc", fg=self.TEXT)
        self.att_message.pack(fill=tk.X, pady=8)
        self.att_video = ttk.Label(parent, text="[Camera off]", relief=tk.SUNKEN, anchor=tk.CENTER)
        self.att_video.pack(pady=8)

    def _build_data_tab(self, parent):
        ttk.Label(parent, text="Attendance Log (CSV)", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 8))
        ttk.Button(parent, text="Refresh", style="Accent.TButton", command=self._data_refresh).pack(anchor=tk.W, pady=4)
        ttk.Label(parent, text=f"Data file: {ATTENDANCE_CSV}", font=("Segoe UI", 9)).pack(anchor=tk.W)
        columns = ("user_id", "name", "punch_type", "timestamp", "session_minutes")
        self.data_tree = ttk.Treeview(parent, columns=columns, show="headings", height=22)
        for c in columns:
            self.data_tree.heading(c, text=c.replace("_", " ").title())
            self.data_tree.column(c, width=130)
        self.data_tree.pack(fill=tk.BOTH, expand=True, pady=10)
        self._data_refresh()

    def _on_close(self):
        self._stop_all_camera()
        self.root.destroy()

    def _stop_all_camera(self):
        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.camera_active = None
        self.reg_capture_btn.config(state=tk.DISABLED)
        self.reg_save_btn.config(state=tk.DISABLED)
        self.reg_stop_btn.config(state=tk.DISABLED)
        self.reg_start_btn.config(state=tk.NORMAL)
        self.reg_status.config(text="Samples: 0 / 5 — Camera stopped.")
        self.att_start_btn.config(state=tk.NORMAL)
        self.att_stop_btn.config(state=tk.DISABLED)
        self.att_punch_btn.config(state=tk.DISABLED)
        self.att_status.config(text="Status: —")
        self.att_video.config(text="[Camera off]")
        self.reg_video.config(text="[Camera off]")

    def _reg_start_camera(self):
        self._stop_all_camera()
        self.reg_start_btn.config(state=tk.DISABLED)
        self.reg_status.config(text="Opening camera…")
        self.reg_video.config(text="[Opening camera…]")

        def open_in_thread():
            cap = open_camera(CAMERA_INDEX)
            self.root.after(0, self._reg_camera_ready, cap)

        t = threading.Thread(target=open_in_thread, daemon=True)
        t.start()

    def _reg_camera_ready(self, cap):
        if cap is None or not cap.isOpened():
            self.reg_start_btn.config(state=tk.NORMAL)
            self.reg_status.config(text="Samples: 0 / 5 — Camera failed.")
            self.reg_video.config(text="[Camera off]")
            messagebox.showerror("Error", "Could not open camera.")
            return
        self.cap = cap
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
        self.register_encodings = []
        self.camera_active = "register"
        self.reg_capture_btn.config(state=tk.NORMAL)
        self.reg_save_btn.config(state=tk.NORMAL)
        self.reg_stop_btn.config(state=tk.NORMAL)
        self.reg_status.config(text="Samples: 0 / 5 — Position face and click Capture Sample.")
        self._reg_update_frame()

    def _reg_update_frame(self):
        if self.camera_active != "register" or self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self._after_id = self.root.after(30, self._reg_update_frame)
            return
        frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        frame_rgb = bgr_to_rgb(frame)
        frame_norm = bgr_to_rgb(normalize_lighting(frame))
        locs, encs = detect_face_and_encoding(frame_rgb)
        if not encs:
            locs, encs = detect_face_and_encoding(frame_norm)
        if encs:
            top, right, bottom, left = locs[0]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, "Face detected - Click Capture", (left, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "No face / multiple faces", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        self._show_frame(frame, self.reg_video)
        self._after_id = self.root.after(30, self._reg_update_frame)

    def _reg_capture(self):
        if self.camera_active != "register" or self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return
        frame_rgb = bgr_to_rgb(frame)
        frame_norm = bgr_to_rgb(normalize_lighting(frame))
        locs, encs = detect_face_and_encoding(frame_rgb)
        if not encs:
            locs, encs = detect_face_and_encoding(frame_norm)
        if not encs:
            messagebox.showwarning("No face", "No single face detected. Position clearly and try again.")
            return
        # Check if this face is already registered
        existing_id, existing_name = _find_existing_user_by_face(encs[0])
        if existing_id is not None:
            messagebox.showwarning(
                "Already registered",
                f"This person is already registered as:\n\n{existing_name} ({existing_id})\n\nCannot register the same face again with a new user ID.",
            )
            return
        self.register_encodings.append(encs[0])
        n = len(self.register_encodings)
        self.reg_status.config(text=f"Samples: {n} / {NUM_SAMPLES}")
        if n >= NUM_SAMPLES:
            self.reg_capture_btn.config(state=tk.DISABLED)
            self.reg_status.config(text=f"Samples: {n} / {NUM_SAMPLES} — Ready to save.")

    def _reg_save(self):
        uid = (self.reg_user_id.get() or "").strip()
        name = (self.reg_name.get() or "").strip()
        if not uid or not name:
            messagebox.showwarning("Missing fields", "Enter User ID and Full Name.")
            return
        if load_face_encodings(uid) is not None:
            messagebox.showwarning("Exists", f"User '{uid}' already registered. Use a different ID or delete existing data.")
            return
        if len(self.register_encodings) < 2:
            messagebox.showwarning("Too few samples", "Capture at least 2 face samples before saving.")
            return
        # Double-check: face may already be registered under another ID
        existing_id, existing_name = _find_existing_user_by_face(self.register_encodings[0])
        if existing_id is not None:
            messagebox.showwarning(
                "Already registered",
                f"This person is already registered as:\n\n{existing_name} ({existing_id})\n\nCannot register the same face again.",
            )
            return
        save_face_encodings(uid, name, self.register_encodings)
        messagebox.showinfo("Saved", f"Registered: {name} ({uid}) with {len(self.register_encodings)} samples.")
        self.register_encodings = []
        self.reg_user_id.delete(0, tk.END)
        self.reg_name.delete(0, tk.END)
        self.reg_status.config(text="Samples: 0 / 5 — Registration saved.")
        self.reg_save_btn.config(state=tk.DISABLED)
        self.reg_capture_btn.config(state=tk.NORMAL)

    def _reg_stop_camera(self):
        self._stop_all_camera()

    def _att_start_camera(self):
        if not HAS_RECOGNITION:
            messagebox.showerror("Error", "Recognition module not available. Install dependencies.")
            return
        users = list_registered_users()
        if not users:
            messagebox.showwarning("No users", "Register at least one user first.")
            return
        self._stop_all_camera()
        self.att_start_btn.config(state=tk.DISABLED)
        self.att_status.config(text="Opening camera…")
        self.att_video.config(text="[Opening camera…]")

        def open_in_thread():
            cap = open_camera(CAMERA_INDEX)
            self.root.after(0, self._att_camera_ready, cap)

        threading.Thread(target=open_in_thread, daemon=True).start()

    def _att_camera_ready(self, cap):
        if cap is None or not cap.isOpened():
            self.att_start_btn.config(state=tk.NORMAL)
            self.att_status.config(text="Status: —")
            self.att_video.config(text="[Camera off]")
            messagebox.showerror("Error", "Could not open camera.")
            return
        self.cap = cap
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
        self.camera_active = "attendance"
        self.attendance_state = {
            "encodings": None,
            "names": None,
            "user_ids": None,
            "blink_state": init_blink_state(),
            "prev_bbox": None,
            "head_history": deque(maxlen=HEAD_MOVEMENT_FRAMES),
            "liveness_list": [],
            "frame_index": 0,
            "uid": None,
            "name": None,
            "liveness_ok": False,
        }
        self._attendance_load_encodings()
        self.att_stop_btn.config(state=tk.NORMAL)
        self.att_status.config(text="Status: Camera ready.")
        self._att_update_frame()

    def _attendance_load_encodings(self):
        enc, names, uids = load_all_encodings()
        if self.attendance_state is not None:
            self.attendance_state["encodings"] = enc
            self.attendance_state["names"] = names
            self.attendance_state["user_ids"] = uids

    def _att_update_frame(self):
        if self.camera_active != "attendance" or self.cap is None or self.attendance_state is None:
            return
        st = self.attendance_state
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self._after_id = self.root.after(30, self._att_update_frame)
            return
        frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        frame_rgb = bgr_to_rgb(frame)
        frame_norm = bgr_to_rgb(normalize_lighting(frame))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        st["frame_index"] += 1
        locs = face_recognition.face_locations(frame_rgb, model=FACE_DETECTION_MODEL, number_of_times_to_upsample=1)
        encs = face_recognition.face_encodings(frame_rgb, known_face_locations=locs)
        if not encs:
            locs = face_recognition.face_locations(frame_norm, model=FACE_DETECTION_MODEL, number_of_times_to_upsample=1)
            encs = face_recognition.face_encodings(frame_norm, known_face_locations=locs)
        uid, name, conf = None, None, 0.0
        if locs and encs and st["encodings"]:
            top, right, bottom, left = locs[0]
            if (bottom - top) >= MIN_FACE_SIZE and (right - left) >= MIN_FACE_SIZE:
                uid, name, dist = match_face(encs[0], st["encodings"], st["names"], st["user_ids"], TOLERANCE)
                conf = max(0, 1 - dist / TOLERANCE) if dist is not None else 0
                curr_bbox = (top, right, bottom, left)
                move_score = head_movement_score(st["prev_bbox"], curr_bbox)
                update_head_movement_history(st["head_history"], move_score)
                head_moved = has_recent_head_movement(st["head_history"])
                face_roi = gray[top:bottom, left:right]
                texture_ok = is_likely_real_texture(face_roi)
                blink_detected = head_moved
                if _predictor is not None:
                    try:
                        eyes = get_eye_landmarks_from_bbox(curr_bbox, gray, _predictor)
                        if eyes is not None:
                            ear = (eye_aspect_ratio(eyes[0]) + eye_aspect_ratio(eyes[1])) / 2.0
                            blink_detected, st["blink_state"] = detect_blink_this_frame(st["blink_state"], ear, st["frame_index"])
                    except Exception:
                        pass
                liveness = check_liveness(blink_detected=blink_detected, head_moved=head_moved, texture_ok=texture_ok,
                                         require_blink=True, require_head_move=True, require_texture=False)
                st["liveness_list"].append(liveness)
                if len(st["liveness_list"]) > LIVENESS_FRAMES:
                    st["liveness_list"].pop(0)
                st["liveness_ok"] = any(l.passed for l in st["liveness_list"][-10:])
                st["prev_bbox"] = curr_bbox
                color = (0, 255, 0) if uid else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                label = f"{name} ({conf:.2f})" if uid else "Unknown"
                cv2.putText(frame, label, (left, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                st["prev_bbox"] = None
        else:
            st["prev_bbox"] = None
        st["uid"] = uid
        st["name"] = name
        # Update UI
        if uid and name:
            punch_type = get_next_punch_type(uid, name)
            self.att_status.config(text=f"Recognized: {name} — Next: Punch {punch_type.upper()}  |  Liveness: {'OK' if st['liveness_ok'] else 'Move/blink'}")
            self.att_punch_btn.config(state=tk.NORMAL if st["liveness_ok"] else tk.DISABLED, text=f"Punch {punch_type.upper()}")
        else:
            self.att_status.config(text="Status: No recognized face.")
            self.att_punch_btn.config(state=tk.DISABLED, text="Punch In / Out")
        self._show_frame(frame, self.att_video)
        self._after_id = self.root.after(30, self._att_update_frame)

    def _att_punch(self):
        if self.attendance_state is None or not self.attendance_state.get("liveness_ok"):
            self._att_log("Complete liveness (blink + head movement) first.")
            return
        uid = self.attendance_state.get("uid")
        name = self.attendance_state.get("name")
        if not uid or not name:
            self._att_log("No recognized user.")
            return
        punch_type = get_next_punch_type(uid, name)
        if record_punch(uid, name, punch_type):
            self._att_log(f"Punch-{punch_type.upper()} recorded for {name} ({uid}).")
            self._attendance_load_encodings()
        else:
            self._att_log("Punch not recorded (duplicate or invalid state).")

    def _att_log(self, msg: str):
        self.att_message.config(state=tk.NORMAL)
        self.att_message.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} — {msg}\n")
        self.att_message.see(tk.END)
        self.att_message.config(state=tk.DISABLED)

    def _att_stop_camera(self):
        self._stop_all_camera()

    def _show_frame(self, frame_bgr, label_widget: ttk.Label):
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img)
        photo = ImageTk.PhotoImage(image=pil)
        label_widget.config(image=photo)
        label_widget.image = photo

    def _data_refresh(self):
        for i in self.data_tree.get_children():
            self.data_tree.delete(i)
        records = get_attendance_summary()
        for r in records:
            self.data_tree.insert("", tk.END, values=(
                r.get("user_id", ""),
                r.get("name", ""),
                r.get("punch_type", ""),
                r.get("timestamp", ""),
                r.get("session_minutes") if r.get("session_minutes") is not None else "",
            ))

    def run(self):
        self.root.mainloop()


def main():
    app = AttendanceGUI()
    app.run()


if __name__ == "__main__":
    main()
