# 🛡️ HARIS / VIGIL·AI Surveillance System

A professional, real-time AI-powered security camera dashboard built on modern architectural principles. The system uses YOLOv8 for object detection and MediaPipe for full-body tracking, streaming intelligent analytics over MJPEG to a decoupled React Dashboard. 

## 🌟 Key Features
* **Live AI Vision:** Detects persons, objects, and deeply specific security behaviors (e.g. "Hands in Pockets") via OpenCV arrays.
* **Cyber-Noir Dashboard:** A high-performance React front-end inspired by cinematic security systems using custom CSS and `framer-motion` for complex animations.
* **Intelligent Alerts Log:** Automatically records anomalies to an SQLite database, accessible via an interactive historical log list.
* **Search & Filter Engine:** Fully instantaneous client-side searching, querying events natively down to the millisecond.
* **Analytics Timeline:** Visualize security threats over time with interactive, glowing responsive `recharts` diagrams.

---

## 📂 Project Structure
```text
/haris-OTC
│
├── Haris_Brian-main/       # Python Flask API & AI Inference Engine
│   ├── app.py              # Main REST API and MJPEG stream endpoints
│   ├── haris.py            # Computer Vision generator (YOLOv8 & MediaPipe)
│   └── instance/           # Secure location for the SQLite database
│
└── haris/                  # The React Frontend Application
    ├── src/pages/          # React views (AlertsPage, LivePage)
    ├── src/hooks/          # Custom data-fetching controllers
    └── src/components/     # UI building blocks (AnalyticsChart, AlertCard)
```

---

## 🚀 How to Run Locally

### 1. Start the AI Backend (Python)
The backend requires a webcam to be connected to your PC to process detections.

```bash
# Navigate to the Python directory
cd Haris_Brian-main

# Create and activate a Virtual Environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install the required AI & Backend libraries
pip install -r requirements.txt

# Boot the engine!
python app.py
```
*(The backend AI REST API runs silently on `http://127.0.0.1:5000`)*

### 2. Start the Frontend Dashboard (React)
Ensure you have Node.js installed on your machine. Open a **new, separate terminal tab** for this.

```bash
# Navigate to the React directory
cd haris

# Install UI dependencies
npm install

# Start the dashboard
npm run dev
```
*(The UI runs on `http://localhost:5173`)*

Open your browser to the React frontend URL to view the live dashboard!

---

## 🤖 Technology Stack
**Frontend:** React, Vite, TailwindCSS, Framer Motion, Recharts, Lucide-React. \
**Backend:** Python 3, Flask, SQLAlchemy, OpenCV (`cv2`). \
**Artificial Intelligence:** YOLOv8 (Ultralytics), MediaPipe Pose Estimation, PyTorch.
