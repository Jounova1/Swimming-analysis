import time
from collections import deque
import importlib
import importlib.util
import subprocess
import sys
import os

# ================== INSTALL & IMPORT ==================
if importlib.util.find_spec("ultralytics") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])

if importlib.util.find_spec("cv2") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])

from ultralytics import YOLO
import cv2

# لو detection_smoother مش موجود
# ارفع الملف أو عدّل path
from detection_smoother import DetectionSmoother


# ================== CONFIG ==================
MODEL_PATH   = r"C:\Swimming-analysis\ai\training\runs\train\yolo11m_swimmer_finetune_v2\weights\best.pt"
VIDEO_SOURCE = r"C:\Swimming-analysis\ai\videos\flip_turn.mp4"
OUTPUT_DIR   = r"C:\Swimming-analysis\ai\outputs"
OUTPUT_NAME  = "flip_turn_annotated.mp4"
save_path    = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
CONF         = 0.5
INPUT_WIDTH  = 640  # ↓ أسرع
INPUT_HEIGHT = 360

SAVE_OUTPUT  = True

pool_length = 50.0

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

# ================== MAIN LOOP ==================
while True:
    t_start = time.time()

    # Frame skip
    for _ in range(frame_skip):
        ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))

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

        # center for stroke
        center_y = (y1 + y2) / 2
        y_hist.append(center_y)

        if len(y_hist) >= 3:
            if y_hist[-2] < y_hist[-3] and y_hist[-2] < y_hist[-1]:
                pass
            elif y_hist[-2] > y_hist[-3] and y_hist[-2] > y_hist[-1]:
                stroke_count += 1

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

    cv2.putText(annotated, f"Distance: {distance_traveled:.1f}m", (10,90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)

    # ================= FPS =================
    processing_fps = 1.0 / (time.time() - t_start)
    cv2.putText(annotated, f"FPS: {processing_fps:.1f}", (10,120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

    # ================= SAVE =================
    if writer:
        writer.write(annotated)

    frame_count += 1

cap.release()
if writer:
    writer.release()

print("DONE ✅")