import time
import cv2
from ultralytics import YOLO

# --- Config ---
MODEL_PATH   = r"C:\clean-repo\ai\training\runs\runs\train\yolo11m_swimmer_finetune_v2\weights\best.pt"
VIDEO_SOURCE = r"C:\clean-repo\ai\training\dataset\images\videos\surface_5.mp4"   # or 0 for Pi camera live feed
CONF         = 0.25       # confidence threshold
INPUT_WIDTH  = 640               # resize frame before inference
INPUT_HEIGHT = 360
SAVE_OUTPUT  = False             # set True to save result video

# --------------

model = YOLO(r"C:\clean-repo\ai\training\runs\runs\train\yolo11m_swimmer_finetune_v2\weights\best.pt")
cap   = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_SOURCE}")

# Optional: save output video
writer = None
if SAVE_OUTPUT:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("output.mp4", fourcc, 20, (INPUT_WIDTH, INPUT_HEIGHT))

print("Running inference — press ESC to quit\n")

while True:
    t_start = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # FIX 1: resize to 640x360 before inference (big speed boost on Pi)
    frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))

    # FIX 2: use tracker (persist=True keeps swimmer ID across frames)
    # FIX 3: conf=0.45 filters out false positives
    # FIX 4: classes=[0] — only detect swimmer class, nothing else
    results = model.track(
        frame,
        conf=CONF,
        classes=[0],
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False,
    )

    # Draw detections on frame
    annotated = results[0].plot()

    # FIX 5: show detection info for debugging
    num_detections = len(results[0].boxes) if results[0].boxes is not None else 0 
    if num_detections > 0:
        for box in results[0].boxes:
            conf_score = float(box.conf)
            track_id   = int(box.id) if box.id is not None else -1
            x, y, w, h = box.xywhn[0].tolist()
            print(f"  Swimmer | conf={conf_score:.2f} | id={track_id} | "
                  f"box=({x:.2f}, {y:.2f}, {w:.2f}, {h:.2f})")
    else:
        print("  No swimmer detected this frame")

    # FIX 6: show FPS on screen so you can measure performance
    fps = 1.0 / (time.time() - t_start)
    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated, f"Detections: {num_detections}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    if writer:
        writer.write(annotated)

    cv2.imshow("SwimAI", annotated)

    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()
print("\nDone.")
