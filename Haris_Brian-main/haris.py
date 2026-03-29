import cv2
import numpy as np
import mediapipe as mp
import os
import urllib.request
import math
import time


from datetime import datetime
from ultralytics import YOLO
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from app import db, Photo, app

with app.app_context():
    db.create_all()

# --- 1. CONFIGURATION & MODEL PATHS ---
YOLO_MODEL_NAME = "yolov8n.pt"
POSE_MODEL_PATH = "pose_landmarker_full.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
PROXIMITY_RATIO = 0.15 # For YOLO holding detection
HOLD_LABEL = "Object Detected Near Person"

# --- 2. SKELETON CONNECTIONS (MediaPipe) ---
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

# --- 3. DOWNLOAD POSE MODEL IF MISSING ---
if not os.path.exists(POSE_MODEL_PATH):
    print("Downloading MediaPipe pose model...")
    urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)

# --- 4. INITIALIZE MODELS ---
# Load YOLO
yolo_model = YOLO(YOLO_MODEL_NAME)

# Load MediaPipe Pose Landmarker
base_options = mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
options = mp_vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = mp_vision.PoseLandmarker.create_from_options(options)

# --- 5. HELPER FUNCTIONS ---

def boxes_are_close(person_box, obj_box, proximity_px):
    """Check if object bounding box is within 'proximity_px' of person box."""
    px1, py1, px2, py2 = person_box
    ox1, oy1, ox2, oy2 = obj_box
    # Expand person box by proximity
    return not (ox1 > px2 + proximity_px or ox2 < px1 - proximity_px or
                oy1 > py2 + proximity_px or oy2 < py1 - proximity_px)

def draw_pose(frame, landmarks, h, w):
    """Draw skeleton lines and joint circles on the frame."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (255, 0, 0), 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1, cv2.LINE_AA)

def check_pocket(landmarks):
    """Detect if hands are close to hip landmarks."""
    r_wrist, r_hip = landmarks[16], landmarks[24]
    l_wrist, l_hip = landmarks[15], landmarks[23]
    dist_r = math.sqrt((r_wrist.x - r_hip.x)**2 + (r_wrist.y - r_hip.y)**2)
    dist_l = math.sqrt((l_wrist.x - l_hip.x)**2 + (l_wrist.y - l_hip.y)**2)
    return dist_r < 0.20, dist_l < 0.20

def capture_and_save(frame):
    os.makedirs("static/screenshots", exist_ok=True)

    # 2. Create a unique filename using a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.png"
    filepath = f"static/screenshots/{filename}"

    # 3. Save the physical file to your folder
    cv2.imwrite(filepath, frame)

    # 4. Save the reference to the Database
    with app.app_context():       
        new_entry = Photo(filename=filename,filepath= filepath, upload_time=datetime.utcnow())
        db.session.add(new_entry)
        db.session.commit()

    print(f"Saved {filename} to database!")


# --- 6. MAIN LOOP ---

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_index = 0

    print("Live Stream Detection Running... Streaming to web.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame_index += 1
            h, w = frame.shape[:2]
            timestamp_ms = int(time.time() * 1000)

            # --- PHASE A: YOLO OBJECT DETECTION ---
            yolo_results = yolo_model(frame, conf=0.4, verbose=False)[0]
            persons = []
            objects = []

            for box in yolo_results.boxes:
                cls_id = int(box.cls[0])
                label = yolo_model.names[cls_id]
                coords = map(int, box.xyxy[0])
                x1, y1, x2, y2 = coords

                if label == "person":
                    persons.append((x1, y1, x2, y2))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                else:
                    objects.append((x1, y1, x2, y2, label))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 1)

            # Check YOLO holding logic
            holding_detected = False
            for (px1, py1, px2, py2) in persons:
                prox_px = int((py2 - py1) * PROXIMITY_RATIO)
                for (ox1, oy1, ox2, oy2, _) in objects:
                    if boxes_are_close((px1, py1, px2, py2), (ox1, oy1, ox2, oy2), prox_px):
                        holding_detected = True
                        break
            
            if holding_detected:
                cv2.putText(frame, "HOLDING OBJECT", (20, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)

            # --- PHASE B: MEDIAPIPE POSE DETECTION ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            pose_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if pose_result.pose_landmarks:
                for pose_landmarks in pose_result.pose_landmarks:
                    draw_pose(frame, pose_landmarks, h, w)
                    right_in, left_in = check_pocket(pose_landmarks)

                    # Hand in pocket alerts
                    if right_in and left_in:
                        cv2.putText(frame, 'BOTH HANDS IN POCKETS!', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        capture_and_save(frame)
                    elif right_in or left_in:
                        cv2.putText(frame, 'ONE HAND IN POCKET', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                        capture_and_save(frame)

            # --- PHASE C: FINAL STREAM OUTPUT ---
            ret_enc, buffer = cv2.imencode('.jpg', frame)
            if ret_enc:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        # Cleanup when client disconnects
        cap.release()

