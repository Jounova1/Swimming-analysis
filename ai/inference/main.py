import time
from collections import deque
import importlib
import importlib.util
import subprocess
import sys
from detection_smoother import DetectionSmoother


def _load_yolo_class():
    """
    Dynamically load ultralytics.YOLO.

    This avoids static import resolution errors in editors when the current
    interpreter environment doesn't yet have ultralytics installed.
    """
    if importlib.util.find_spec("ultralytics") is None:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "ultralytics"]
        )

    ultra = importlib.import_module("ultralytics")
    return ultra.YOLO


YOLO = _load_yolo_class()


def _load_cv2():
    """
    Dynamically load OpenCV (cv2).

    Prevents editor/static-analysis "unresolved import" warnings when cv2
    isn't installed for the currently analyzed interpreter.
    """
    if importlib.util.find_spec("cv2") is None:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "opencv-python"]
        )
    return importlib.import_module("cv2")


cv2 = _load_cv2()

# --- Config ---
MODEL_PATH   = r"ai/training/runs/train/yolo11m_swimmer_finetune_v7/best.pt"
VIDEO_SOURCE = r"C:\Swimming-analysis\ai\videos\flip_turn.mp4"   # or 0 for Pi camera live feed
CONF         = 0.5             # confidence threshold (accept down to 20%)
INPUT_WIDTH  = 640              # resize frame before inference
INPUT_HEIGHT = 360
SAVE_OUTPUT  = False             # set True to save result video

# --- Bug Fix Params ---
MIN_CONSECUTIVE_FRAMES = 3  # Y=3: require detection in 3 consecutive frames to accept
STOP_FRAMES_THRESHOLD  = 10 # X=10: stop timer after 10 frames without detection

# --- Distance / Pool length (NO interactive input) ---
# Default pool length (edit this constant if needed).
pool_length = 50.0

print("=" * 60)
print("SWIMMER ANALYSIS SYSTEM")
print("=" * 60)
print(f"Pool length set to: {pool_length}m")
print("=" * 60 + "\n")

model = YOLO(MODEL_PATH)
cap   = cv2.VideoCapture(VIDEO_SOURCE)

# Get real video FPS
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0 or fps is None:
    fps = 30  # fallback

print(f"Video FPS (REAL): {fps}")

# Initialize bug-fix components
# Note: max_age_frames=3 means tracks are forgotten after 3 frames of not being detected
smoother = DetectionSmoother(
    min_consecutive_frames=MIN_CONSECUTIVE_FRAMES,
    confidence_window_frames=5,
    confidence_accept_threshold=0.20,
    max_lost_frames=10,  # keep tracking while missing <= 10 frames
)

# ========== EXPLICIT TIMER STATE MANAGEMENT ==========
# Spec:
# idle -> swimming (timer running)
# swimming -> stopped (after 10 consecutive missing frames; freeze display)
# stopped -> lost (after >10 consecutive missing frames)
# lost -> swimming (reset to 0 only when re-detected after lost)
timer_state = "idle"
swim_start_frame = None
frames_without_detection = 0
swim_duration = 0.0  # frozen elapsed time when timer is stopped

# ========== STROKE COUNTER ==========
stroke_count = 0
stroke_threshold = 20  # Minimum up/down amplitude (bbox center Y) to treat as a stroke
MIN_FRAMES_BETWEEN_STROKES = 15
y_hist = deque(maxlen=5)  # rolling buffer of bbox center Y for extrema detection
seen_trough_since_last_stroke = False
trough_y = None
last_stroke_frame = -10_000_000

# ========== DISTANCE TRACKER ==========
distance_traveled = 0.0  # in meters
laps_completed = 0
last_center_x = None  # Track horizontal position for lap detection
lap_threshold = INPUT_WIDTH * 0.3  # Threshold for detecting crossing (30% of frame width)
in_left_zone = None  # Track which end of pool swimmer is in

# Lap crossing boundaries (use hysteresis to avoid double-counting).
mid_x = INPUT_WIDTH / 2.0
lap_margin_px = int(INPUT_WIDTH * 0.05)
left_boundary = mid_x - lap_margin_px
right_boundary = mid_x + lap_margin_px

frame_count = 0

print(f"Video FPS: {fps}")

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_SOURCE}")

# Optional: save output video
writer = None
if SAVE_OUTPUT:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("output.mp4", fourcc, 20, (INPUT_WIDTH, INPUT_HEIGHT))

print("Running inference — press ESC to quit\n")

# ========== HELPER: Draw Clean Top-Right Overlay ==========
def draw_stats_overlay(frame, time_s: float, strokes: int, distance_m: float) -> None:
    """
    Required overlay (TOP-RIGHT), EXACT 3 lines, NO emojis:
      Time:     0.00s
      Strokes:  0
      Distance: 0.0m
    """
    lines = [
        f"Time:     {time_s:.2f}s",
        f"Strokes:  {strokes:d}",
        f"Distance: {distance_m:.1f}m",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    font_thickness = 1
    line_spacing = 24
    pad = 12

    text_sizes = [cv2.getTextSize(line, font, font_scale, font_thickness)[0] for line in lines]
    box_w = int(max(w for (w, _) in text_sizes) + pad * 2)
    box_h = int(len(lines) * line_spacing + pad * 2 - 6)

    x0 = int(frame.shape[1] - box_w - 15)
    y0 = 15

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), 1)

    for i, line in enumerate(lines):
        y_text = int(y0 + pad + (i + 1) * line_spacing - 8)
        x_text = x0 + pad

        # Thin black outline
        cv2.putText(frame, line, (x_text - 1, y_text), font, font_scale, (0, 0, 0), font_thickness + 1)
        cv2.putText(frame, line, (x_text + 1, y_text), font, font_scale, (0, 0, 0), font_thickness + 1)
        cv2.putText(frame, line, (x_text, y_text - 1), font, font_scale, (0, 0, 0), font_thickness + 1)
        cv2.putText(frame, line, (x_text, y_text + 1), font, font_scale, (0, 0, 0), font_thickness + 1)

        cv2.putText(frame, line, (x_text, y_text), font, font_scale, (255, 255, 255), font_thickness)

while True:
    t_start = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # FIX 1: resize to 640x360 before inference (big speed boost on Pi)
    frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))

    # FIX 2: use tracker (persist=True keeps swimmer ID across frames)
    # FIX 3: conf=0.10 captures more detections (lowered from 0.35)
    # FIX 4: classes=[0] — only detect swimmer class, nothing else
    results = model.track(
        frame,
        conf=CONF,
        classes=[0],
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False,
    )

    # ========== BUG FIX #1 & #2: Temporal Smoothing ==========
    # CRITICAL: Always call add_detections, even with empty list, so aging logic runs
    # This is how old detections get removed when swimmer is no longer visible
    boxes_to_add = results[0].boxes if results[0].boxes is not None else []
    smoother.add_detections(boxes_to_add, frame_count)
    
    # Temporal smoothing + prediction (up to N missing frames).
    # We must distinguish real detections from predicted ones:
    # - predicted => is_interpolated=True (no real detector hit this frame)
    smoothed_dets = smoother.get_smoothed_detections()
    best_det = None
    detected_now = False
    if len(smoothed_dets) > 0:
        real_dets = [d for d in smoothed_dets if not d[3]]
        if len(real_dets) > 0:
            best_det = max(real_dets, key=lambda d: d[1])
            detected_now = True
        else:
            best_det = max(smoothed_dets, key=lambda d: d[1])
            detected_now = False

    # For the timer logic, "detection present" means REAL detection this frame.
    has_detection = detected_now
    
    # Calculate detection counts for display
    num_raw_detections = len(results[0].boxes) if results[0].boxes is not None else 0
    num_smoothed_detections = len(smoothed_dets)
    
    # DEBUG: Show smoother state
    stats = smoother.get_stats()
    if num_raw_detections == 0 or has_detection == False:  # Show when no real detections
        print(
            f"[FRAME {frame_count}] Smoother: active_tracks={stats.get('active_tracks')} "
            f"tracks_total={stats.get('tracks_total')} has_detection={has_detection}"
        )
    
    # ========== EXPLICIT TIMER STATE MACHINE ==========
    if timer_state == "idle":
        if has_detection:
            timer_state = "swimming"
            swim_start_frame = frame_count
            frames_without_detection = 0
            swim_duration = 0.0

            # Reset per-spec stats when swimmer first appears.
            stroke_count = 0
            laps_completed = 0
            distance_traveled = 0.0
            last_center_x = None  # prev center_x for lap crossing
            y_hist.clear()
            seen_trough_since_last_stroke = False
            trough_y = None
            last_stroke_frame = -10_000_000
            in_left_zone = None

            print(f"[TIMER] ✓ Swim STARTED (frame {frame_count})")

    elif timer_state == "swimming":
        if has_detection:
            frames_without_detection = 0
        else:
            frames_without_detection += 1
            if frames_without_detection >= STOP_FRAMES_THRESHOLD:
                # Stop timer after 10 consecutive missing frames.
                swim_duration = (frame_count - swim_start_frame) / fps
                timer_state = "stopped"

                print(
                    f"[TIMER] ✗ Swim STOPPED (elapsed: {swim_duration:.2f}s, frames: {frame_count - swim_start_frame})"
                )
                print(
                    f"[SUMMARY] Final Stats: Time={swim_duration:.2f}s | Strokes={stroke_count} | Distance={distance_traveled:.1f}m | Laps={laps_completed}"
                )

    elif timer_state == "stopped":
        # Displayed time freezes in this state.
        if has_detection:
            frames_without_detection = 0
        else:
            frames_without_detection += 1
            if frames_without_detection > STOP_FRAMES_THRESHOLD:
                timer_state = "lost"

    elif timer_state == "lost":
        # Timer resets to 0 ONLY after being lost and re-detected.
        if has_detection:
            timer_state = "swimming"
            swim_start_frame = frame_count
            frames_without_detection = 0
            swim_duration = 0.0

            stroke_count = 0
            laps_completed = 0
            distance_traveled = 0.0
            last_center_x = None  # prev center_x for lap crossing
            y_hist.clear()
            seen_trough_since_last_stroke = False
            trough_y = None
            last_stroke_frame = -10_000_000
            in_left_zone = None

            print(f"[TIMER] ⟳ Swim RESET and restarted (frame {frame_count})")
    
    # ========== STROKE + LAP DETECTION (ONLY DURING ACTIVE SWIMMING) ==========
    if timer_state == "swimming" and has_detection and best_det is not None:
        _track_id, _conf, xyxy, _is_pred = best_det
        x1, y1, x2, y2 = xyxy
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        # ---- Stroke counting (up/down cycle = trough -> peak) ----
        y_hist.append(center_y)
        if len(y_hist) >= 3:
            y_a = y_hist[-3]
            y_b = y_hist[-2]
            y_c = y_hist[-1]

            is_local_min = (y_b < y_a) and (y_b < y_c)
            is_local_max = (y_b > y_a) and (y_b > y_c)

            if is_local_min:
                trough_y = y_b
                seen_trough_since_last_stroke = True

            if (
                is_local_max
                and seen_trough_since_last_stroke
                and trough_y is not None
            ):
                amplitude = abs(y_b - trough_y)
                enough_amp = amplitude >= stroke_threshold
                enough_time = (frame_count - last_stroke_frame) >= MIN_FRAMES_BETWEEN_STROKES
                if enough_amp and enough_time:
                    stroke_count += 1
                    last_stroke_frame = frame_count
                    seen_trough_since_last_stroke = False
                    print(f"  [STROKE] Stroke #{stroke_count} detected")

        # ---- Lap detection (cross from left side to right side) ----
        if last_center_x is None:
            last_center_x = center_x
        else:
            if last_center_x < left_boundary and center_x > right_boundary:
                laps_completed += 1
                distance_traveled = laps_completed * pool_length
                print(
                    f"  [LAP] Lap #{laps_completed} completed! Distance: {distance_traveled:.1f}m"
                )
            elif last_center_x > right_boundary and center_x < left_boundary:
                laps_completed += 1
                distance_traveled = laps_completed * pool_length
                print(
                    f"  [LAP] Lap #{laps_completed} completed! Distance: {distance_traveled:.1f}m"
                )

            last_center_x = center_x

    # When not actively swimming, clear motion buffers (prevents carry-over noise).
    if timer_state != "swimming":
        y_hist.clear()
        last_center_x = None
        seen_trough_since_last_stroke = False
        trough_y = None
    
    # Timer display is rendered via the spec-required overlay later.
    
    # Draw detections on frame (from smoothed detections)
    annotated = frame.copy()
    
    # Draw YOLO boxes for all smoothed detections
    for track_id, conf, xyxy, is_interpolated in smoothed_dets:
        x1, y1, x2, y2 = xyxy
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Color: green for high-confidence, yellow for lower confidence
        color = (0, 255, 0) if conf > 0.5 else (0, 255, 255)
        thickness = 2
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label with confidence and track ID
        label = f"ID:{track_id} Conf:{conf:.2f}"
        (text_width, text_height) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(annotated, (x1, y1 - text_height - 4), (x1 + text_width, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Print detection info for debugging
    if num_raw_detections > 0:
        for box in results[0].boxes:
            conf_score = float(box.conf)
            track_id   = int(box.id) if box.id is not None else -1
            x, y, w, h = box.xywhn[0].tolist()
            print(f"  Raw Detection | conf={conf_score:.2f} | id={track_id} | "
                  f"box=({x:.2f}, {y:.2f}, {w:.2f}, {h:.2f})")
    
    if num_smoothed_detections > 0:
        print(f"  ✓ Smoothed: {num_smoothed_detections} detection(s) passed temporal filter")
    else:
        print(f"  ✗ No detections passed temporal filter (raw: {num_raw_detections})")

    # FIX 6: show FPS on screen so you can measure performance
    # NOTE: Using processing_fps to avoid overwriting video fps (needed for accurate timer)
    processing_fps = 1.0 / (time.time() - t_start)
    cv2.putText(annotated, f"FPS: {processing_fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated, f"Raw Detections: {num_raw_detections}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(annotated, f"Smoothed: {num_smoothed_detections}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # ===== CLEAN STATS OVERLAY (spec-required) =====
    if timer_state == "swimming":
        display_time_s = (frame_count - swim_start_frame) / fps
    elif timer_state in ("stopped", "lost"):
        display_time_s = swim_duration
    else:
        display_time_s = 0.0

    draw_stats_overlay(annotated, display_time_s, stroke_count, distance_traveled)

    if writer:
        writer.write(annotated)

    cv2.imshow("SwimAI", annotated)
    
    frame_count += 1

    delay = int(1000 / fps)
    if cv2.waitKey(delay) == 27:
        break

cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()
print("\nDone.")
