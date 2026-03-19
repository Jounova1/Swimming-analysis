from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class TrackObservation:
	frame: int
	swimmer_id: int
	x: float
	y: float
	bbox: Tuple[int, int, int, int]


class SwimmerTracker:
	"""Simple centroid-based multi-swimmer tracker.

	This tracker uses background subtraction for detection and nearest-neighbor
	assignment to keep swimmer IDs stable across frames.
	"""

	def __init__(
		self,
		min_area: int = 500,
		max_lost: int = 15,
		distance_threshold: float = 80.0,
	) -> None:
		self.min_area = min_area
		self.max_lost = max_lost
		self.distance_threshold = distance_threshold

		self._next_id = 1
		self._tracks: Dict[int, Dict[str, object]] = {}
		self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
			history=500, varThreshold=25, detectShadows=True
		)

	def _detect_swimmers(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
		fg_mask = self._bg_subtractor.apply(frame)
		_, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
		fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
		fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

		contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		detections: List[Tuple[int, int, int, int]] = []

		for contour in contours:
			area = cv2.contourArea(contour)
			if area < self.min_area:
				continue

			x, y, w, h = cv2.boundingRect(contour)
			detections.append((x, y, w, h))

		return detections

	@staticmethod
	def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
		x, y, w, h = bbox
		return x + (w / 2.0), y + (h / 2.0)

	def _update_tracks(
		self, detections: List[Tuple[int, int, int, int]]
	) -> List[Tuple[int, Tuple[int, int, int, int], Tuple[float, float]]]:
		matched_track_ids = set()
		matched_detection_indices = set()

		track_centers = {
			track_id: self._tracks[track_id]["center"] for track_id in self._tracks
		}
		detection_centers = [self._bbox_center(bbox) for bbox in detections]

		for track_id, t_center_obj in track_centers.items():
			t_center = t_center_obj  # type: ignore[assignment]
			if not detection_centers:
				continue

			best_idx = -1
			best_distance = float("inf")

			for idx, d_center in enumerate(detection_centers):
				if idx in matched_detection_indices:
					continue

				distance = float(np.linalg.norm(np.array(t_center) - np.array(d_center)))
				if distance < best_distance:
					best_distance = distance
					best_idx = idx

			if best_idx >= 0 and best_distance <= self.distance_threshold:
				bbox = detections[best_idx]
				center = detection_centers[best_idx]
				self._tracks[track_id] = {"bbox": bbox, "center": center, "lost": 0}
				matched_track_ids.add(track_id)
				matched_detection_indices.add(best_idx)

		for track_id in list(self._tracks.keys()):
			if track_id in matched_track_ids:
				continue

			lost = int(self._tracks[track_id]["lost"]) + 1
			self._tracks[track_id]["lost"] = lost
			if lost > self.max_lost:
				del self._tracks[track_id]

		for idx, bbox in enumerate(detections):
			if idx in matched_detection_indices:
				continue

			center = detection_centers[idx]
			self._tracks[self._next_id] = {"bbox": bbox, "center": center, "lost": 0}
			self._next_id += 1

		visible_tracks = []
		for track_id, data in self._tracks.items():
			if int(data["lost"]) == 0:
				visible_tracks.append((track_id, data["bbox"], data["center"]))

		return visible_tracks

	def track_video(
		self, video_path: str, visualize: bool = False
	) -> List[TrackObservation]:
		capture = cv2.VideoCapture(video_path)
		if not capture.isOpened():
			raise FileNotFoundError(f"Could not open video: {video_path}")

		observations: List[TrackObservation] = []
		frame_idx = 0

		while True:
			success, frame = capture.read()
			if not success:
				break

			detections = self._detect_swimmers(frame)
			tracked_swimmers = self._update_tracks(detections)

			for swimmer_id, bbox, center in tracked_swimmers:
				x_center, y_center = center
				observations.append(
					TrackObservation(
						frame=frame_idx,
						swimmer_id=swimmer_id,
						x=float(x_center),
						y=float(y_center),
						bbox=bbox,
					)
				)

				if visualize:
					x, y, w, h = bbox
					cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 205, 50), 2)
					cv2.putText(
						frame,
						f"ID {swimmer_id}",
						(x, max(20, y - 8)),
						cv2.FONT_HERSHEY_SIMPLEX,
						0.6,
						(255, 255, 255),
						2,
					)

			if visualize:
				cv2.imshow("Swimmer Tracking", frame)
				if cv2.waitKey(1) & 0xFF == ord("q"):
					break

			frame_idx += 1

		capture.release()
		if visualize:
			cv2.destroyAllWindows()

		return observations
