import time
import cv2
from ultralytics import YOLO
from detection_smoother import DetectionSmoother
import numpy as np

# --- Config ---
MODEL_PATH   = r"C:\Swimming-analysis\Swimming-analysis\ai\training\best (1).pt"
VIDEO_SOURCE = r"C:\Swimming-analysis\Swimming-analysis\ai\training\videoplayback (1).mp4"   # or 0 for Pi camera live feed
CONF         = 0.10             # confidence threshold (lowered to 0.10 to capture more detections)
INPUT_WIDTH  = 640              # resize frame before inference
INPUT_HEIGHT = 360
SAVE_OUTPUT  = False             # set True to save result video

# --- Bug Fix Params ---
MIN_CONSECUTIVE_FRAMES = 3  # Y=3: require detection in 3 consecutive frames to accept
STOP_FRAMES_THRESHOLD  = 10 # X=10: stop timer after 10 frames without detection

# --- User Input for Pool Length ---
print("=" * 60)
print("SWIMMER ANALYSIS SYSTEM")
print("=" * 60)
pool_length = float(input("Enter pool length in meters: "))
print(f"Pool length set to: {pool_length}m")
print("=" * 60 + "\n")

# --------------

model = YOLO(MODEL_PATH)
cap   = cv2.VideoCapture(VIDEO_SOURCE)

# Get video FPS for frame-based timing
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps < 1:
    fps = 30.0  # Default to 30 FPS if unable to detect

# Initialize bug-fix components
# Note: max_age_frames=3 means tracks are forgotten after 3 frames of not being detected
smoother = DetectionSmoother(
    min_consecutive_frames=MIN_CONSECUTIVE_FRAMES,
    max_age_frames=3  # REDUCED from default 5 to 3 for faster timeout
)

# ========== EXPLICIT TIMER STATE MANAGEMENT ==========
# Timer states: "idle", "swimming", "paused"
timer_state = "idle"  # idle -> swimming -> paused -> idle/swimming
swim_start_frame = None
swim_end_frame = None
frames_without_detection = 0
swim_duration = 0.0

# ========== STROKE COUNTER ==========
stroke_count = 0
last_center_y = None  # Track vertical position for stroke detection
stroke_threshold = 20  # pixels of movement to detect a stroke
frames_since_last_stroke = 0

# ========== DISTANCE TRACKER ==========
distance_traveled = 0.0  # in meters
laps_completed = 0
last_center_x = None  # Track horizontal position for lap detection
lap_threshold = INPUT_WIDTH * 0.3  # Threshold for detecting crossing (30% of frame width)
in_left_zone = None  # Track which end of pool swimmer is in

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

# ========== HELPER: Draw Semi-Transparent Stats Box ==========
def draw_stats_overlay(frame, timer_display, stroke_count, distance_traveled, position="top-left"):
    """
    Draw a clean, semi-transparent stats box on the frame.
    
    Args:
        frame: The image to draw on
        timer_display: Time string (e.g., "6.17s")
        stroke_count: Number of strokes
        distance_traveled: Distance in meters
        position: "top-left" or "top-right"
    """
    # Prepare text lines
    lines = [
        f"⏱ Time:     {timer_display}",
        f"🏊 Strokes:  {stroke_count}",
        f"📏 Distance: {distance_traveled:.1f}m"
    ]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    font_thickness = 1
    line_spacing = 28
    box_padding = 12
    
    # Calculate box dimensions
    text_width = 0
    for line in lines:
        (w, h) = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
        text_width = max(text_width, w)
    
    box_width = text_width + box_padding * 2
    box_height = len(lines) * line_spacing + box_padding * 2
    
    # Position box
    if position == "top-right":
        x_start = frame.shape[1] - box_width - 15
    else:  # top-left
        x_start = 15
    y_start = 15
    
    # Draw semi-transparent black background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_start, y_start), 
                  (x_start + box_width, y_start + box_height), 
                  (0, 0, 0), -1)  # Black fill
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw border
    cv2.rectangle(frame, (x_start, y_start), 
                  (x_start + box_width, y_start + box_height), 
                  (200, 200, 200), 2)  # Light gray border
    
    # Draw text lines
    for i, line in enumerate(lines):
        y_text = y_start + box_padding + (i + 1) * line_spacing - 8
        
        # Draw black outline for better readability
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    cv2.putText(frame, line, (x_start + box_padding + dx, y_text + dy),
                                font, font_scale, (0, 0, 0), font_thickness + 1)
        
        # Draw white text
        cv2.putText(frame, line, (x_start + box_padding, y_text),
                    font, font_scale, (255, 255, 255), font_thickness)
    
    return frame

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
    
    # Get only detections that passed temporal consistency filter
    smoothed_dets = smoother.get_smoothed_detections()
    has_detection = len(smoothed_dets) > 0
    
    # Calculate detection counts for display
    num_raw_detections = len(results[0].boxes) if results[0].boxes is not None else 0
    num_smoothed_detections = len(smoothed_dets)
    
    # DEBUG: Show smoother state
    stats = smoother.get_stats()
    if num_raw_detections == 0 or has_detection == False:  # Show when no detections
        print(f"[FRAME {frame_count}] Smoother state: active_tracks={stats['active_tracks']}, "
              f"pending={stats['pending_tracks']}, inactive={stats['inactive_tracks']}, "
              f"has_detection={has_detection}")
    
    # ========== EXPLICIT TIMER STATE MACHINE ==========
    # Three states: idle (waiting for first detection), swimming (active), paused (stopped)
    
    if has_detection:
        # Swimmer detected in this frame
        if timer_state == "idle":
            # NEW SWIM STARTING
            timer_state = "swimming"
            swim_start_frame = frame_count
            swim_end_frame = None
            frames_without_detection = 0
            swim_duration = 0.0
            print(f"[TIMER] ✓ Swim STARTED (frame {frame_count})")
        
        elif timer_state == "paused":
            # RESUMING AFTER PAUSE (NEW SWIM SESSION)
            timer_state = "swimming"
            swim_start_frame = frame_count
            swim_end_frame = None
            frames_without_detection = 0
            swim_duration = 0.0  # RESET to 0!
            print(f"[TIMER] ⟳ Swim RESET and restarted (frame {frame_count})")
        
        else:  # timer_state == "swimming"
            # Continue swimming, reset no-detection counter
            frames_without_detection = 0
    
    else:
        # No swimmer detected in this frame
        if timer_state == "swimming":
            # Start counting frames without detection
            frames_without_detection += 1
            print(f"  No detection for {frames_without_detection} frames (threshold: {STOP_FRAMES_THRESHOLD})")
            
            if frames_without_detection >= STOP_FRAMES_THRESHOLD:
                # STOP THE TIMER
                timer_state = "paused"
                swim_end_frame = frame_count - STOP_FRAMES_THRESHOLD
                swim_duration = (swim_end_frame - swim_start_frame) / fps
                print(f"[TIMER] ✗ Swim STOPPED (elapsed: {swim_duration:.2f}s, frames: {swim_end_frame - swim_start_frame})")
                print(f"[SUMMARY] Final Stats: Time={swim_duration:.2f}s | Strokes={stroke_count} | Distance={distance_traveled:.2f}m | Laps={laps_completed}")
    
    # ========== STROKE DETECTION ==========
    # Detect strokes by monitoring vertical (Y) movement of swimmer's bounding box
    if has_detection and timer_state == "swimming":
        for track_id, conf, xyxy, is_interpolated in smoothed_dets:
            x1, y1, x2, y2 = xyxy
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Detect vertical movement (stroke indication)
            if last_center_y is not None:
                y_movement = abs(center_y - last_center_y)
                # Always increment debounce counter (tracks time since last stroke)
                frames_since_last_stroke += 1
                # Count stroke when: (1) significant movement AND (2) debounce interval passed
                if y_movement > stroke_threshold and frames_since_last_stroke >= 15:  # At least 15 frames between strokes
                    stroke_count += 1
                    frames_since_last_stroke = 0
                    print(f"  [STROKE] Stroke #{stroke_count} detected")
            else:
                frames_since_last_stroke = 0
            
            # ========== LAP/DISTANCE DETECTION ==========
            # Detect when swimmer crosses from one end to the other
            if last_center_x is not None:
                # Determine which zone swimmer is in (left or right)
                if center_x < INPUT_WIDTH * 0.4:  # Left zone
                    current_zone = "left"
                elif center_x > INPUT_WIDTH * 0.6:  # Right zone
                    current_zone = "right"
                else:
                    current_zone = None
                
                # Detect crossing (change zones)
                if current_zone and in_left_zone and current_zone != in_left_zone:
                    laps_completed += 1
                    distance_traveled = laps_completed * pool_length
                    print(f"  [LAP] Lap #{laps_completed} completed! Distance: {distance_traveled:.2f}m")
                
                if current_zone:
                    in_left_zone = current_zone
            else:
                # Initialize zone tracking
                if center_x < INPUT_WIDTH * 0.4:
                    in_left_zone = "left"
                elif center_x > INPUT_WIDTH * 0.6:
                    in_left_zone = "right"
            
            last_center_y = center_y
            last_center_x = center_x
    
    # Reset position tracking when not swimming
    if not has_detection or timer_state != "swimming":
        last_center_y = None
        last_center_x = None
        frames_since_last_stroke = 0
    
    # Calculate current elapsed time based on timer state
    if timer_state == "swimming":
        current_elapsed = (frame_count - swim_start_frame) / fps
        timer_display = f"Swimming: {current_elapsed:.2f}s"
        timer_color = (0, 255, 0)  # Green
    elif timer_state == "paused":
        timer_display = f"Stopped: {swim_duration:.2f}s"
        timer_color = (0, 165, 255)  # Orange
    else:  # idle
        timer_display = "Ready: 0.00s"
        timer_color = (0, 0, 255)  # Red
    
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
    
    # ===== CLEAN STATS OVERLAY =====
    # Extract just the time value for display
    if timer_state == "swimming":
        time_value = f"{(frame_count - swim_start_frame) / fps:.2f}s"
    elif timer_state == "paused":
        time_value = f"{swim_duration:.2f}s"
    else:  # idle
        time_value = "0.00s"
    
    # Draw the stats box overlay
    annotated = draw_stats_overlay(annotated, time_value, stroke_count, distance_traveled, position="top-left")

    if writer:
        writer.write(annotated)

    cv2.imshow("SwimAI", annotated)
    
    frame_count += 1

    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()
print("\nDone.")
