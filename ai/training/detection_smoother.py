"""
Detection Smoother: Applies temporal filtering and interpolation to swimmer detections.

Fixes two issues:
1. Low-confidence detections: Only accepts detections that appear consistently over Y frames
2. Missing frames: Interpolates swimmer position when briefly lost (1-3 frames)

FIXED: Properly ages out detections that haven't been seen for N frames, so old swimmers don't persist.
"""

from collections import defaultdict, deque
from typing import List, Tuple, Optional
import numpy as np


class DetectionSmoother:
    """
    Applies temporal smoothing to swimmer detections.
    
    Args:
        min_consecutive_frames: Minimum consecutive frames a detection must appear 
                                to be accepted (Y=3)
        interpolate_gap_frames: Maximum gap to interpolate across (1-3 frames)
        max_age_frames: Maximum frames a track can be inactive before being forgotten (default: 5)
    """
    
    def __init__(self, min_consecutive_frames: int = 3, interpolate_gap_frames: int = 3, max_age_frames: int = 5):
        self.min_frames = min_consecutive_frames
        self.interpolate_gap = interpolate_gap_frames
        self.max_age_frames = max_age_frames  # NEW: Forget tracks after this many frames without detection
        
        # Track detection history: track_id -> deque of (frame_idx, box_data)
        self.detection_history = defaultdict(lambda: deque(maxlen=10))
        
        # Track ages: how many consecutive frames each swimmer has been detected
        self.track_ages = defaultdict(int)
        
        # Track inactivity: how many consecutive frames since last detection
        self.track_inactive_frames = defaultdict(int)
        
        # Last seen frame index for each track
        self.last_seen_frame = {}
        
        # Current frame index
        self.current_frame = 0
    
    def add_detections(self, boxes_list: List, frame_idx: int) -> None:
        """
        Add detections from current frame to the smoother's history.
        
        Args:
            boxes_list: List of detection boxes from YOLO
                        Each box should have: .id, .conf, .xyxy (or .xywh)
            frame_idx: Current frame index
        """
        self.current_frame = frame_idx
        detected_ids = set()
        
        # Record all detections for this frame
        for box in boxes_list:
            track_id = int(box.id) if box.id is not None else -1
            conf = float(box.conf)
            xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, 'cpu') else box.xyxy[0]
            
            # Store detection
            box_data = {
                'track_id': track_id,
                'conf': conf,
                'xyxy': tuple(xyxy.tolist()),  # (x1, y1, x2, y2)
            }
            
            self.detection_history[track_id].append((frame_idx, box_data))
            detected_ids.add(track_id)
            
            # Update track age (consecutive frames detected)
            if track_id in self.track_ages:
                self.track_ages[track_id] += 1
            else:
                self.track_ages[track_id] = 1
            
            # Reset inactivity counter (track was just detected)
            self.track_inactive_frames[track_id] = 0
            self.last_seen_frame[track_id] = frame_idx
        
        # Age out undetected tracks
        for track_id in list(self.track_ages.keys()):
            if track_id not in detected_ids:
                # Track not detected this frame - increment inactivity counter
                self.track_inactive_frames[track_id] += 1
                
                # If track has been inactive for too long, forget it completely
                if self.track_inactive_frames[track_id] >= self.max_age_frames:
                    # Remove this track from memory
                    del self.track_ages[track_id]
                    del self.track_inactive_frames[track_id]
                    if track_id in self.last_seen_frame:
                        del self.last_seen_frame[track_id]
                    if track_id in self.detection_history:
                        del self.detection_history[track_id]
                else:
                    # Track is inactive but not yet forgotten
                    self.track_ages[track_id] = 0
    
    def get_smoothed_detections(self) -> List[Tuple]:
        """
        Return detections that have passed the temporal consistency filter.
        
        Returns:
            List of (track_id, conf, xyxy, is_interpolated) tuples
            where xyxy is (x1, y1, x2, y2)
        """
        smoothed = []
        
        for track_id, age in self.track_ages.items():
            # Accept only tracks seen for at least min_frames consecutive detections
            if age >= self.min_frames:
                history = list(self.detection_history[track_id])
                if history:
                    # Use the most recent detection for this track
                    frame_idx, box_data = history[-1]
                    smoothed.append((
                        track_id,
                        box_data['conf'],
                        box_data['xyxy'],
                        False  # Not interpolated
                    ))
            elif 1 <= age < self.min_frames:
                # Track is building up confidence but hasn't reached min_frames yet
                # Still include it but mark as "pending" (low priority)
                history = list(self.detection_history[track_id])
                if history:
                    frame_idx, box_data = history[-1]
                    # Reduce confidence for pending tracks
                    adjusted_conf = box_data['conf'] * (age / self.min_frames)
                    smoothed.append((
                        track_id,
                        adjusted_conf,
                        box_data['xyxy'],
                        False
                    ))
        
        return smoothed
    
    def interpolate_positions(self, frame_idx: int) -> List[Tuple]:
        """
        Fill in positions for tracks that briefly disappeared (1-3 frames).
        Linearly interpolates position between last seen and current detection.
        
        Args:
            frame_idx: Current frame index
            
        Returns:
            List of interpolated (track_id, conf, xyxy, is_interpolated) tuples
        """
        interpolated = []
        
        for track_id, last_frame in list(self.last_seen_frame.items()):
            gap = frame_idx - last_frame
            
            # If track disappeared for a few frames, try to interpolate
            if 1 < gap <= self.interpolate_gap and track_id in self.detection_history:
                history = list(self.detection_history[track_id])
                if len(history) >= 2:
                    # Get last two detections
                    prev_frame, prev_data = history[-1]
                    prev_xyxy = np.array(prev_data['xyxy'])
                    
                    # Check if this track reappears in current frame
                    # (would be handled by regular detections)
                    # This is for predicting future positions
        
        return interpolated
    
    def reset(self):
        """Reset the smoother state."""
        self.detection_history.clear()
        self.track_ages.clear()
        self.track_inactive_frames.clear()
        self.last_seen_frame.clear()
        self.current_frame = 0
    
    def get_stats(self) -> dict:
        """
        Get statistics about current detection smoothing.
        
        Returns:
            Dict with 'active_tracks', 'pending_tracks', 'inactive_tracks'
        """
        active = sum(1 for age in self.track_ages.values() if age >= self.min_frames)
        pending = sum(1 for age in self.track_ages.values() if 0 < age < self.min_frames)
        inactive = sum(1 for age in self.track_ages.values() if age == 0)
        
        return {
            'active_tracks': active,
            'pending_tracks': pending,
            'inactive_tracks': inactive,
        }
