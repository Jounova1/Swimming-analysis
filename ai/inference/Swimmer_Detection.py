import time
from collections import deque

import cv2
from ultralytics import YOLO

from detection_smoother import DetectionSmoother

# --- Config ---
MODEL_PATH = r"C:\Swimming-analysis\ai\training\runs\train\yolo11m_swimmer_finetune_v2\weights\best.pt"
VIDEO_SOURCE = r"C:\Swimming-analysis\ai\training\dataset\images\videos\surface_6.mp4"  # or 0 for live feed

# --- Required bug-fix parameters ---
CONF = 0.20  # Lower confidence threshold to 20%
INPUT_WIDTH = 640
INPUT_HEIGHT = 360
SAVE_OUTPUT = False

MIN_CONSECUTIVE_FRAMES = 3  # for low-confidence consistency
STOP_FRAMES_THRESHOLD = 10  # stop timer after 10 missing (no detection) frames
MAX_MISSING_FRAMES = 10  # keep tracking while missing <= 10 frames

# --- Stroke detection params (bbox vertical oscillation) ---
STROKE_AMPLITUDE_PX = 20
MIN_FRAMES_BETWEEN_STROKES = 15

# --------------

# Distance uses laps * pool_length (NO interactive input).
pool_length = 50.0


def draw_stats_overlay_top_right(frame, time_s: float, strokes: int, distance_m: float) -> None:
    """
    Draw the required overlay box in the TOP-RIGHT corner.
    Must show exactly:
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

    # Semi-transparent black background.
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    # Subtle border.
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), 1)

    # White text with thin black outline.
    for i, line in enumerate(lines):
        y_text = int(y0 + pad + (i + 1) * (line_spacing) - 8)
        x_text = x0 + pad

        # Outline (thin)
        cv2.putText(frame, line, (x_text - 1, y_text), font, font_scale, (0, 0, 0), font_thickness + 1)
        cv2.putText(frame, line, (x_text + 1, y_text), font, font_scale, (0, 0, 0), font_thickness + 1)
        cv2.putText(frame, line, (x_text, y_text - 1), font, font_scale, (0, 0, 0), font_thickness + 1)
        cv2.putText(frame, line, (x_text, y_text + 1), font, font_scale, (0, 0, 0), font_thickness + 1)

        # Main text
        cv2.putText(frame, line, (x_text, y_text), font, font_scale, (255, 255, 255), font_thickness)


model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_SOURCE)

# Get video FPS for frame-based timing
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps < 1:
    fps = 30.0

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_SOURCE}")

smoother = DetectionSmoother(
    min_consecutive_frames=MIN_CONSECUTIVE_FRAMES,
    confidence_window_frames=5,
    confidence_accept_threshold=0.20,
    max_lost_frames=MAX_MISSING_FRAMES,
)

frame_count = 0

print(f"Video FPS: {fps}")
print("Running inference — press ESC to quit\n")

writer = None
if SAVE_OUTPUT:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("output.mp4", fourcc, 20, (INPUT_WIDTH, INPUT_HEIGHT))

# ==========================
# Timer / Stroke / Lap state
# ==========================
timer_state = "idle"  # idle -> swimming -> stopped -> lost
start_frame = None
final_elapsed_s = 0.0
missing_no_detect_frames = 0  # consecutive frames with NO accepted detection

strokes_count = 0
laps_completed = 0
distance_traveled_m = 0.0

# Lap crossing tracking (center_x crossing mid-frame with hysteresis)
mid_x = INPUT_WIDTH / 2.0
side_margin_px = int(INPUT_WIDTH * 0.05)  # hysteresis
left_boundary = mid_x - side_margin_px
right_boundary = mid_x + side_margin_px
prev_center_x_real = None  # previous real detection center_x

# Stroke detection buffers
y_deque = deque(maxlen=5)  # rolling center_y values
seen_trough_since_last_stroke = False
last_trough_y = None
last_stroke_frame = -10_000


def xyxy_to_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


while True:
    t_start = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))

    results = model.track(
        frame,
        conf=CONF,
        classes=[0],
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False,
    )

    boxes_to_add = results[0].boxes if results[0].boxes is not None else []
    smoother.add_detections(boxes_to_add, frame_count)
    smoothed_dets = smoother.get_smoothed_detections()

    # Choose the best current (or predicted) track.
    best_det = None
    if smoothed_dets:
        # Prefer real detections over predicted when conf is similar.
        best_det = sorted(
            smoothed_dets,
            key=lambda d: (d[3], d[1]),  # is_interpolated, conf
            reverse=False,
        )[0]
        # Actually, the sort above picks interpolated first due to bool ordering.
        # Fix: prefer !is_interpolated then higher conf.
        best_det = max(smoothed_dets, key=lambda d: ((not d[3]), d[1]))

    detected_now = False  # real detection present this frame (not predicted)
    if best_det is not None:
        detected_now = not best_det[3]

    annotated = frame.copy()

    # Draw YOLO/smoothed boxes.
    for track_id, conf, xyxy, is_interpolated in smoothed_dets:
        x1, y1, x2, y2 = xyxy
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1 = max(0, min(INPUT_WIDTH - 1, x1))
        y1 = max(0, min(INPUT_HEIGHT - 1, y1))
        x2 = max(0, min(INPUT_WIDTH - 1, x2))
        y2 = max(0, min(INPUT_HEIGHT - 1, y2))

        color = (0, 255, 0) if not is_interpolated else (0, 165, 255)
        thickness = 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label = f"ID:{track_id} Conf:{conf:.2f}"
        (text_width, text_height) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(annotated, (x1, y1 - text_height - 4), (x1 + text_width, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # ==========================
    # Timer + missing frame logic
    # ==========================
    if timer_state == "idle":
        if detected_now:
            timer_state = "swimming"
            start_frame = frame_count
            final_elapsed_s = 0.0
            missing_no_detect_frames = 0

            # Reset per-spec stats when swimmer first appears.
            strokes_count = 0
            laps_completed = 0
            distance_traveled_m = 0.0
            prev_center_x_real = None

            y_deque.clear()
            seen_trough_since_last_stroke = False
            last_trough_y = None
            last_stroke_frame = -10_000

    elif timer_state == "swimming":
        if detected_now:
            missing_no_detect_frames = 0
        else:
            missing_no_detect_frames += 1

            if missing_no_detect_frames >= STOP_FRAMES_THRESHOLD:
                # STOP THE TIMER HERE and freeze time.
                if start_frame is not None:
                    final_elapsed_s = (frame_count - start_frame) / fps

                timer_state = "stopped"

                # Final outputs when swimmer stops.
                print(
                    "\n[SUMMARY] Swim stopped."
                    f" Time={final_elapsed_s:.2f}s | Strokes={strokes_count} |"
                    f" Distance={distance_traveled_m:.1f}m | Laps={laps_completed}"
                )

    elif timer_state == "stopped":
        # Display time freezes; only enter lost after >10 missing frames.
        if detected_now:
            missing_no_detect_frames = 0
        else:
            missing_no_detect_frames += 1
            if missing_no_detect_frames > STOP_FRAMES_THRESHOLD:
                timer_state = "lost"

    elif timer_state == "lost":
        if detected_now:
            # Reset to 0 ONLY after being lost.
            timer_state = "swimming"
            start_frame = frame_count
            final_elapsed_s = 0.0
            missing_no_detect_frames = 0

            strokes_count = 0
            laps_completed = 0
            distance_traveled_m = 0.0
            prev_center_x_real = None

            y_deque.clear()
            seen_trough_since_last_stroke = False
            last_trough_y = None
            last_stroke_frame = -10_000

        # else remain lost

    # Current displayed time (freeze when stopped/lost)
    if timer_state == "swimming" and start_frame is not None:
        display_time_s = (frame_count - start_frame) / fps
    else:
        display_time_s = final_elapsed_s

    # ==========================
    # Stroke + lap/distance logic (ONLY during active swimming)
    # ==========================
    if timer_state == "swimming" and best_det is not None and detected_now:
        _, _, xyxy, _is_pred = best_det
        cx, cy = xyxy_to_center(xyxy)

        # ---- Stroke detection (bbox Y oscillation; trough -> peak) ----
        y_deque.append((frame_count, cy))
        if len(y_deque) >= 3:
            y_a = y_deque[-3][1]
            y_b = y_deque[-2][1]
            y_c = y_deque[-1][1]

            is_local_max = y_b > y_a and y_b > y_c
            is_local_min = y_b < y_a and y_b < y_c

            if is_local_min:
                last_trough_y = y_b
                seen_trough_since_last_stroke = True

            if (
                is_local_max
                and seen_trough_since_last_stroke
                and last_trough_y is not None
            ):
                amplitude = abs(y_b - last_trough_y)
                enough_amp = amplitude >= STROKE_AMPLITUDE_PX
                enough_time = (frame_count - last_stroke_frame) >= MIN_FRAMES_BETWEEN_STROKES
                if enough_amp and enough_time:
                    strokes_count += 1
                    last_stroke_frame = frame_count
                    seen_trough_since_last_stroke = False

        # ---- Lap detection (mid-frame crossing with hysteresis) ----
        if prev_center_x_real is None:
            prev_center_x_real = cx
        else:
            if prev_center_x_real < left_boundary and cx > right_boundary:
                laps_completed += 1
                distance_traveled_m = laps_completed * pool_length
            elif prev_center_x_real > right_boundary and cx < left_boundary:
                laps_completed += 1
                distance_traveled_m = laps_completed * pool_length
            prev_center_x_real = cx

    # Reset buffers outside active swimming (prevents carry-over noise).
    if timer_state != "swimming":
        y_deque.clear()
        prev_center_x_real = None
        seen_trough_since_last_stroke = False
        last_trough_y = None

    # ==========================
    # Required clean overlay UI (top-right)
    # ==========================
    draw_stats_overlay_top_right(annotated, display_time_s, strokes_count, distance_traveled_m)

    if writer:
        writer.write(annotated)

    cv2.imshow("SwimAI", annotated)

    frame_count += 1

    # ESC to quit
    if cv2.waitKey(1) == 27:
        break

cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()
print("\nDone.")
