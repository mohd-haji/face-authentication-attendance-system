## Face Authentication Attendance System

### Overview

This project is a **face-recognition–based attendance system** built with Python.  
It allows you to **register users with their face**, then **punch in / punch out** using a webcam.  
Attendance is stored persistently (SQLite or CSV), and simple **liveness / anti-spoof checks** help reduce fake logins (e.g. photos on a phone).

You can use the system in two ways:
- **Unified GUI (`gui.py`)** – modern Tkinter interface for registering users, punching in/out, and browsing attendance.
- **CLI tools (`register.py`, `recognize.py`)** – command-line registration and real‑time recognition windows.

### Features

- **User registration**
  - Capture multiple face samples per user.
  - Stores 128‑D face embeddings to `data/faces/<user_id>.json`.
  - Prevents duplicate user IDs and attempts to block re‑registering the same face under a different ID.

- **Attendance tracking**
  - Punch **IN / OUT** with a recognized face.
  - Supports **multiple sessions per day**.
  - Computes **per‑session duration** and **total working time per day**.

- **Liveness / anti‑spoof checks** (basic, for learning/demo purposes)
  - **Blink detection** based on Eye Aspect Ratio (EAR).
  - **Head movement** detection across recent frames.
  - **Face texture heuristic** (Laplacian variance) to distinguish flat prints from real faces.
  - GUI and CLI require liveness (blink + head move, optionally texture) before accepting a punch.

- **Storage backends**
  - **SQLite** database (`data/attendance.db`) – default backend.
  - **CSV** file (`data/attendance.csv`) – used when `ATTENDANCE_STORAGE=csv` (GUI sets this automatically).

- **Unified GUI**
  - Tabbed interface:
    - **Register User** – create new users and capture samples.
    - **Punch In / Out** – live camera with recognition + liveness, single button to punch.
    - **Attendance Log (CSV)** – tabular view of stored records.

### Tech Stack

- **Language**: Python 3.10+
- **Computer vision & ML**: `face_recognition`, `dlib`, `opencv-python`, `numpy`
- **GUI**: Tkinter, `Pillow`
- **Data & analysis (optional)**: `pandas`, `matplotlib`, `seaborn`, Jupyter

See `requirements.txt` for exact versions.

### Installation

1. **Clone the repository**
   ```bash
   git clone <this-repo-url>
   cd face-authentication-attendance-system
   ```

2. **Create and activate a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # on Windows
   # source .venv/bin/activate  # on Linux/macOS
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **dlib note**
   - On Windows, installing `dlib` can require build tools or a precompiled wheel.  
   - If `dlib` is not available, the system still works, but blink detection falls back to simpler heuristics and liveness becomes weaker.

### Running the unified GUI

The GUI uses **CSV storage** internally (it sets `ATTENDANCE_STORAGE=csv` for its own process).

```bash
python gui.py
```

In the GUI:
- Go to **Register User**:
  - Enter **User ID** and **Full Name**.
  - Click **Start Camera**, position your face, and click **Capture Sample** several times.
  - When enough samples are collected, click **Save Registration**.
- Go to **Punch In / Out**:
  - Click **Start Camera**.
  - Look at the camera and blink / move your head slightly for liveness.
  - When recognized and liveness is satisfied, click **Punch IN / OUT**.
- Go to **Attendance Log (CSV)**:
  - Click **Refresh** to reload records from `data/attendance.csv`.

### Using the CLI tools

#### 1. Register a user (CLI)

```bash
python register.py --user-id USER001 --name "Alice Example" --camera 0
```

- Follow the on‑screen instructions to capture multiple clear face samples.
- Encodings are saved to `data/faces/USER001.json`.

#### 2. Run real‑time recognition & punching (CLI)

```bash
python recognize.py --camera 0
```

- A window opens showing the camera feed.
- Keys:
  - **P** – attempt to punch IN/OUT for the currently recognized and live user.
  - **Q** – quit.

By default, the CLI uses **SQLite** (`data/attendance.db`) via the storage abstraction in `utils.py`.  
You can force CSV instead by setting:

```bash
set ATTENDANCE_STORAGE=csv   # Windows PowerShell/CMD
# export ATTENDANCE_STORAGE=csv  # Linux/macOS
```

### Data & file structure (high level)

- `gui.py` – Tkinter GUI (register, punch, attendance viewer).
- `register.py` – CLI tool for capturing samples and registering users.
- `recognize.py` – CLI real‑time recognition + liveness + punching.
- `attendance.py` – business logic for punch state, sessions, and working hours.
- `anti_spoof.py` – blink, head movement, and texture‑based liveness checks.
- `utils.py` – shared utilities (storage backends, camera helpers, lighting normalization).
- `data/` – created at runtime:
  - `faces/` – JSON files with face encodings per user.
  - `attendance.db` – SQLite database (when using SQLite backend).
  - `attendance.csv` – CSV log (when using CSV backend).

### Limitations & disclaimer

- The anti‑spoofing module is **basic and educational**, not production‑grade security.
- Good‑quality photos, videos, or 3D masks can still bypass the system.
- For real‑world deployments, you should use **professional liveness SDKs / hardware** and add further security layers (e.g. multi‑factor authentication).

---

### 📌 Future Improvements

- Advanced liveness detection (blink / depth-based)

- Cloud-based database

- Web or mobile deployment

- Shift-based attendance rules

- Admin analytics dashboard

---

### 👤 Author

Mohd Haji
AI/ML Intern Candidate

GitHub: https://github.com/mohd-haji

---

### 📝 License

This project is licensed under the MIT License.

