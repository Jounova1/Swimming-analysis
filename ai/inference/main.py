import time
from collections import deque
import importlib
import importlib.util
import subprocess
import sys
import os
import numpy as np

# ================== INSTALL & IMPORT ==================
if importlib.util.find_spec("ultralytics") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])

if importlib.util.find_spec("cv2") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])

if importlib.util.find_spec("mediapipe") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mediapipe"])

from ultralytics import YOLO
import cv2
import mediapipe as mp

# لو detection_smoother مش موجود
# ارفع الملف أو عدّل path
from detection_smoother import DetectionSmoother


# ================== CONFIG ==================
MODEL_PATH   = r"C:\Users\sigma\OneDrive\Desktop\Swimming-analysis\ai\training\best (1).pt"
VIDEO_SOURCE = r"C:\Users\sigma\OneDrive\Desktop\Swimming-analysis\ai\inference\WhatsApp Video 2026-03-28 at 7.18.43 PM.mp4"
OUTPUT_DIR   = r"C:\Users\sigma\OneDrive\Desktop"
OUTPUT_NAME  = "breast_annotated.mp4"
save_path    = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
CONF         = 0.5
INPUT_WIDTH  = 640  # ↓ أسرع
INPUT_HEIGHT = 360

SAVE_OUTPUT  = True

pool_length = 50.0

# ================== HELPER FUNCTIONS ==================
def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

print("=" * 50)
print("SWIMMER ANALYSIS SYSTEM")
print("=" * 50)

# ================== LOAD ==================
model = YOLO(MODEL_PATH)
cap   = cv2.VideoCapture(VIDEO_SOURCE)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    fps = 30

print(f"Video FPS: {fps}")

# ================== MEDIAPIPE SETUP ==================
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ================== FRAME SKIP ==================
target_fps = 30
frame_skip = int(fps / target_fps)
if frame_skip < 1:
    frame_skip = 1

# ================== SMOOTHER ==================
smoother = DetectionSmoother(
    min_consecutive_frames=3,
    confidence_window_frames=5,
    confidence_accept_threshold=0.20,
    max_lost_frames=10,
)

# ================== VIDEO WRITER ==================
os.makedirs(os.path.dirname(save_path), exist_ok=True)
writer = None
if SAVE_OUTPUT:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(save_path, fourcc, fps, (INPUT_WIDTH, INPUT_HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for path: {save_path}")
    print(f"Saving output video to: {save_path}")

# ================== TRACKING ==================
frame_count = 0
timer_state = "idle"
swim_start_frame = 0

stroke_count = 0
distance_traveled = 0.0
laps_completed = 0

y_hist = deque(maxlen=5)

# ================== STROKE COUNTING (MediaPipe) ==================
stroke_stage = None
mono_angle_history = deque(maxlen=10)
stroke_cooldown = 0  # Prevents rapid re-counting

# ================== MAIN LOOP ==================
while True:
    t_start = time.time()

    # Frame skip
    for _ in range(frame_skip):
        ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))

    # ================== POSE DETECTION & STROKE COUNT ==================
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    pose_results = pose.process(image_rgb)
    image_rgb.flags.writeable = True
    
    # Decrement cooldown for stroke counting
    if stroke_cooldown > 0:
        stroke_cooldown -= 1
    
    # Process pose landmarks for stroke counting
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        
        try:
            # Get arm coordinates for stroke counting (using right arm)
            hips = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # Calculate arm angle
            mono_angle = calculate_angle(hips, knee, ankle)
            mono_angle_history.append(mono_angle)
            
            # Improved stroke counting logic with debouncing
            avg_angle = np.mean(mono_angle_history) if len(mono_angle_history) > 0 else mono_angle
            
            if avg_angle >= 160:
                stroke_stage = "mafrood"
            elif avg_angle < 120 and stroke_stage == "mafrood" and stroke_cooldown == 0:
                stroke_stage = "matny"
                stroke_count += 1
                stroke_cooldown = 15  # Prevent double-counting for 15 frames (~0.5 seconds)
        except:
            pass

    # YOLO tracking
    results = model.track(
        frame,
        conf=CONF,
        classes=[0],
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False,
    )

    boxes = results[0].boxes if results[0].boxes is not None else []
    smoother.add_detections(boxes, frame_count)

    smoothed = smoother.get_smoothed_detections()

    annotated = frame.copy()

    has_detection = False

    for track_id, conf, xyxy, is_pred in smoothed:
        x1, y1, x2, y2 = map(int, xyxy)

        has_detection = True

        # draw box
        color = (0,255,0)
        cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)

    # ================= TIMER =================
    if has_detection:
        if timer_state == "idle":
            timer_state = "swimming"
            swim_start_frame = frame_count

    if timer_state == "swimming":
        time_sec = (frame_count - swim_start_frame) / fps
    else:
        time_sec = 0

    # ================= DRAW TEXT =================
    cv2.putText(annotated, f"Time: {time_sec:.2f}s", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)

    cv2.putText(annotated, f"Strokes: {stroke_count}", (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
    
    cv2.putText(annotated, f"Stage: {stroke_stage if stroke_stage else 'N/A'}", (10,90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),2)

    cv2.putText(annotated, f"Distance: {distance_traveled:.1f}m", (10,120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)

    # ================= DRAW POSE LANDMARKS ==================
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated, 
            pose_results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

    # ================= FPS =================
    processing_fps = 1.0 / (time.time() - t_start)
    cv2.putText(annotated, f"FPS: {processing_fps:.1f}", (10,150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

    # ================= SAVE =================
    if writer:
        writer.write(annotated)

    frame_count += 1

cap.release()
if writer:
    writer.release()

pose.close()

print("DONE ✅")