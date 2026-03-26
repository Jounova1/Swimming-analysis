"""
Swim Timer: Tracks swimming duration with automatic start/stop.

Starts when swimmer is first detected.
Runs while swimmer is actively visible.
Stops after X consecutive frames with no detection.

FIXED: Uses frame-based timing (not wall-clock) for accuracy with slow-playing videos.
"""


class SwimTimer:
    """
    Manages swim session timing using frame counts (not wall-clock time).
    
    Args:
        stop_frames: Number of consecutive frames without detection before stopping (default: 10)
    """
    
    def __init__(self, stop_frames: int = 10):
        self.stop_frames = stop_frames
        self.start_frame = None
        self.is_active = False
        self.frames_without_detection = 0
        self.total_elapsed = 0.0
        self.current_frame_count = 0
    
    def update(self, has_detection: bool, current_frame_count: int, fps: float = 30.0) -> tuple[bool, float]:
        """
        Update timer state based on current frame's detection status.
        
        Args:
            has_detection: True if a swimmer was detected in this frame
            current_frame_count: Current frame number in the video
            fps: Frames per second (default: 30.0)
            
        Returns:
            (is_active, elapsed_seconds) tuple
        """
        self.current_frame_count = current_frame_count
        self.fps = fps
        
        if has_detection:
            # Swimmer detected
            if not self.is_active:
                # Start the timer
                self.start_frame = current_frame_count
                self.is_active = True
                self.frames_without_detection = 0
                self.total_elapsed = 0.0
                print(f"[TIMER] ✓ Swim started (frame {current_frame_count})")
            else:
                # Swimmer still detected, reset counter
                self.frames_without_detection = 0
        else:
            # No swimmer detected
            if self.is_active:
                self.frames_without_detection += 1
                
                if self.frames_without_detection >= self.stop_frames:
                    # Stop the timer
                    if self.start_frame is not None:
                        frame_diff = current_frame_count - self.start_frame
                        self.total_elapsed = frame_diff / fps
                    self.is_active = False
                    self.frames_without_detection = 0
                    print(f"[TIMER] ✗ Swim stopped (elapsed: {self.total_elapsed:.2f}s, frames: {frame_diff})")
        
        # Calculate current elapsed time based on frame count
        if self.is_active and self.start_frame is not None:
            frame_diff = current_frame_count - self.start_frame
            elapsed = frame_diff / fps
        else:
            elapsed = self.total_elapsed
        
        return self.is_active, elapsed
    
    def get_display_text(self) -> str:
        """
        Get formatted timer text for on-screen overlay.
        Uses frame-based elapsed time (not wall-clock).
        
        Returns:
            String like "Swimming: 12.34s" or "Not swimming"
        """
        if self.is_active and self.start_frame is not None:
            frame_diff = self.current_frame_count - self.start_frame
            elapsed = frame_diff / self.fps
            status = "Swimming"
        else:
            elapsed = self.total_elapsed
            status = "Not swimming" if self.total_elapsed == 0.0 else "Finished"
        
        return f"{status}: {elapsed:.2f}s"
    
    def reset(self):
        """Reset the timer to initial state."""
        self.start_frame = None
        self.is_active = False
        self.frames_without_detection = 0
        self.total_elapsed = 0.0
        self.current_frame_count = 0
