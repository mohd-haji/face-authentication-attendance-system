# face-authentication-attendance-system
🧑‍💻 Face Authentication Attendance System

An AI-based Face Authentication Attendance System that uses real-time camera input to register users, identify faces, and mark attendance with Punch In / Punch Out, including worked hours calculation and basic spoof prevention.

This project is built as part of the AI/ML Intern assignment and focuses on practical implementation, system reliability, and understanding real-world ML limitations.

---

🚀 Features
✅ Core Requirements

- Face registration using webcam

- Face recognition for authentication

- Punch In / Punch Out attendance

- Real-time camera input

- Works under varying lighting conditions

- Basic spoof prevention

  ---


⭐ Additional Enhancements (to stand out)

- GUI-based desktop application (Tkinter)

- Automatic working hours calculation (session minutes)

- Attendance stored in CSV format

- Attendance dashboard (table view)

- Multiple face samples per user for better accuracy

- Clear status feedback during punch actions

---

🖥️ Application Screens
🔹 User Registration

- Enter User ID and Full Name

- Capture multiple face samples (5 samples)

- Store encoded facial features

---

🔹 Punch In / Punch Out

- Face verification before marking attendance

- Prevents duplicate punch-in without punch-out

- Live camera preview

---

🔹 Attendance Log

- View attendance records in tabular format

- Stored as CSV for easy export and analysis

- Displays:

  - User ID

  - Name

  - Punch Type

  - Timestamp

  - Session Minutes (worked duration)

---

🧠 ML Model & Approach
Face Detection & Recognition

- Library: face_recognition (dlib-based)

- Uses HOG-based face detection

- Facial embeddings generated using a pre-trained deep learning model

- Face matching done via Euclidean distance threshold

Why this approach?

- Lightweight and fast for real-time systems

- No heavy training required

- Suitable for desktop applications

---

🛡️ Spoof Prevention (Basic)

Implemented basic anti-spoofing techniques:

- Live camera requirement (no static image input)

- Multiple frame validation

- Face movement consistency checks

⚠️ Note: This is a basic approach and not as robust as IR/depth-based systems used in enterprise setups.

---

📊 Accuracy Expectations

- Expected Accuracy: ~85–90% in normal lighting conditions

- Accuracy depends on:

  - Lighting conditions

  - Camera quality

  - Face angle and occlusion

  - Number of samples captured during registration

---

⚠️ Known Limitations

- May struggle in very low light

- Cannot fully prevent high-quality photo/video spoofing

- Single-camera system (no depth sensing)

- Desktop-only (not deployed as web/mobile app)

These limitations are typical for software-only face recognition systems.

---

🧩 Project Structure

## face-authentication-attendance-system/
│

├── register.py          # Face registration logic

├── recognize.py         # Face recognition logic

├── attendance.py        # Punch in / punch out handling

├── anti_spoof.py        # Basic spoof prevention checks

├── gui.py               # Tkinter GUI application

├── utils.py             # Helper functions

├── dashboard.ipynb      # Attendance analysis notebook

├── requirements.txt     # Python dependencies

├── README.md            # Project documentation

├── data/

│     ├── faces/           # Stored face encodings

│     └── attendance.csv   # Attendance records


---


⚙️ Installation & Setup

1️⃣ Clone the Repository

`git clone https://github.com/mohd-haji/face-authentication-attendance-system.git
cd face-authentication-attendance-system`

2️⃣ Create Virtual Environment (Recommended)

`python -m venv venv
venv\Scripts\activate `  # Windows

3️⃣ Install Dependencies

`pip install -r requirements.txt`



4️⃣ Run the Application

`python gui.py`

---

📁 Attendance Output

- Attendance is saved in:

`data/attendance.csv`


- Each punch-out calculates:

  - Total session duration (in minutes)

  - Based on punch-in and punch-out timestamps

---

🧪 Evaluation Criteria Mapping
Requirement	Status
- Functional Accuracy	✅
- System Reliability	✅
- ML Limitations Awareness	✅
- Practical Implementation	✅
- Real Camera Input	✅
- Spoof Prevention	✅ (Basic)

---

📌 Future Improvements

- Advanced liveness detection (blink / depth-based)

- Cloud-based database

- Web or mobile deployment

- Shift-based attendance rules

- Admin analytics dashboard

---

👤 Author

Mohd Haji
AI/ML Intern Candidate

GitHub: https://github.com/mohd-haji

---

📝 License

This project is licensed under the MIT License.

