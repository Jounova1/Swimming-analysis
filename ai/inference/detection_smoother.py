"""
Temporal detection smoother + continuity predictor.

Required behaviors:
1) Low-confidence acceptance:
   - Maintain a rolling confidence buffer per track.
   - Compute averaged confidence over recent detections.
   - Consider a track "detected" when it appears consistently AND its
     averaged confidence passes the acceptance threshold (default: 0.20).

2) Dropped/missing frames continuity:
   - If a confirmed swimmer track is missing for <= max_lost_frames,
     keep it by predicting its bounding box via constant-velocity
     (linear) extrapolation from the last 2 detections.
   - If missing for > max_lost_frames, the track is removed ("lost").
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import numpy as np


@dataclass
class _TrackState:
    # Rolling buffer of recent detections (for confidence averaging + velocity estimation).
    history: Deque[Tuple[int, float, Tuple[float, float, float, float]]]  # (frame_idx, conf, xyxy)
    last_det_frame: int
    consecutive_det_frames: int
    inactive_frames: int
    active: bool  # becomes True once confidence+consistency criteria pass


class DetectionSmoother:
    def __init__(
        self,
        *,
        min_consecutive_frames: int = 3,
        confidence_window_frames: int = 5,
        confidence_accept_threshold: float = 0.20,
        max_lost_frames: int = 10,
    ):
        """
        Args:
            min_consecutive_frames: how many consecutive frames a track must be
                detected for before it becomes active.
            confidence_window_frames: rolling buffer length for confidence averaging.
            confidence_accept_threshold: acceptance threshold (e.g. 0.20 for 20%).
            max_lost_frames: if missing for > this many frames, track is removed.
        """
        self.min_frames = int(min_consecutive_frames)
        self.conf_window = int(confidence_window_frames)
        self.conf_thresh = float(confidence_accept_threshold)
        self.max_lost_frames = int(max_lost_frames)

        self.current_frame = 0
        self.track_states: Dict[int, _TrackState] = {}

    @staticmethod
    def _box_to_xyxy(box) -> Tuple[float, float, float, float]:
        xyxy_t = box.xyxy[0]
        if hasattr(xyxy_t, "cpu"):
            xyxy_np = xyxy_t.cpu().numpy()
        else:
            xyxy_np = np.asarray(xyxy_t)
        x1, y1, x2, y2 = xyxy_np.tolist()
        return float(x1), float(y1), float(x2), float(y2)

    @staticmethod
    def _center_from_xyxy(xyxy: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def add_detections(self, boxes_list: List, frame_idx: int) -> None:
        """
        Record detections for the current frame.
        Must be called every frame (even with an empty list) so that inactivity counters age correctly.
        """
        self.current_frame = int(frame_idx)

        detected_ids = set()
        for box in boxes_list:
            track_id = int(box.id) if box.id is not None else -1
            if track_id < 0:
                continue
            detected_ids.add(track_id)

        # Age all existing tracks that were not detected this frame.
        for track_id in list(self.track_states.keys()):
            if track_id in detected_ids:
                continue

            st = self.track_states[track_id]
            st.inactive_frames += 1

            # Remove tracks that have been missing for too long.
            if st.inactive_frames > self.max_lost_frames:
                del self.track_states[track_id]

        # Insert/update detections for tracks present in this frame.
        for box in boxes_list:
            track_id = int(box.id) if box.id is not None else -1
            if track_id < 0:
                continue

            conf = float(box.conf)
            xyxy = self._box_to_xyxy(box)

            if track_id not in self.track_states:
                st = _TrackState(
                    history=deque(maxlen=self.conf_window),
                    last_det_frame=self.current_frame,
                    consecutive_det_frames=1,
                    inactive_frames=0,
                    active=False,
                )
                self.track_states[track_id] = st
            else:
                st = self.track_states[track_id]

                # consecutive means "no gaps" between frames
                if self.current_frame - st.last_det_frame == 1:
                    st.consecutive_det_frames += 1
                else:
                    st.consecutive_det_frames = 1

                st.last_det_frame = self.current_frame
                st.inactive_frames = 0

            st.history.append((self.current_frame, conf, xyxy))

            # Become active only when both consistency and averaged confidence criteria are met.
            if not st.active:
                if st.consecutive_det_frames >= self.min_frames:
                    confs = [entry[1] for entry in st.history]
                    avg_conf = float(np.mean(confs)) if confs else 0.0
                    if avg_conf >= self.conf_thresh:
                        st.active = True

    def get_smoothed_detections(self) -> List[Tuple[int, float, Tuple[float, float, float, float], bool]]:
        """
        Returns:
            List of (track_id, conf_avg, xyxy, is_interpolated/predicted).

        Notes:
            - Only active tracks are returned.
            - When a track is inactive (missing detections), its bbox is predicted
              via constant-velocity extrapolation from the last 2 detections.
        """
        out: List[Tuple[int, float, Tuple[float, float, float, float], bool]] = []

        for track_id, st in self.track_states.items():
            if not st.active:
                continue

            if not st.history:
                continue

            # Confidence averaging (rolling buffer).
            confs = [entry[1] for entry in st.history]
            avg_conf = float(np.mean(confs)) if confs else 0.0

            last_frame, last_conf, last_xyxy = st.history[-1]
            x1, y1, x2, y2 = last_xyxy
            last_w = x2 - x1
            last_h = y2 - y1
            last_cx, last_cy = self._center_from_xyxy(last_xyxy)

            if st.inactive_frames <= 0:
                out.append((track_id, avg_conf, last_xyxy, False))
                continue

            # Predict center using last 2 detections (constant velocity).
            if len(st.history) >= 2:
                prev_frame, _, prev_xyxy = st.history[-2]
                prev_cx, prev_cy = self._center_from_xyxy(prev_xyxy)
                dt = max(1, last_frame - prev_frame)

                vx = (last_cx - prev_cx) / dt
                vy = (last_cy - prev_cy) / dt

                pred_cx = last_cx + vx * st.inactive_frames
                pred_cy = last_cy + vy * st.inactive_frames

                pred_xyxy = (
                    pred_cx - last_w / 2.0,
                    pred_cy - last_h / 2.0,
                    pred_cx + last_w / 2.0,
                    pred_cy + last_h / 2.0,
                )
            else:
                pred_xyxy = last_xyxy

            # Decay confidence when predicting.
            pred_conf = avg_conf * (0.92 ** st.inactive_frames)
            out.append((track_id, float(pred_conf), pred_xyxy, True))

        return out

    def reset(self) -> None:
        self.current_frame = 0
        self.track_states.clear()

    def get_stats(self) -> dict:
        active = sum(1 for st in self.track_states.values() if st.active)
        return {
            "active_tracks": active,
            "tracks_total": len(self.track_states),
        }
