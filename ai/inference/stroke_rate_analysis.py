from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.signal import find_peaks

try:
	from inference.track_swimmer import TrackObservation
except ModuleNotFoundError:
	from track_swimmer import TrackObservation


class StrokeRateAnalyzer:
	"""Estimate stroke rate by finding peaks in vertical swimmer motion."""

	def __init__(
		self,
		min_peak_prominence: float = 2.0,
		min_peak_distance_seconds: float = 0.35,
	) -> None:
		self.min_peak_prominence = min_peak_prominence
		self.min_peak_distance_seconds = min_peak_distance_seconds

	def estimate(
		self, observations: Iterable[TrackObservation], fps: float
	) -> Dict[Tuple[int, int], float]:
		by_swimmer: Dict[int, List[TrackObservation]] = defaultdict(list)
		for obs in observations:
			by_swimmer[obs.swimmer_id].append(obs)

		stroke_rate_by_frame_and_id: Dict[Tuple[int, int], float] = {}
		min_distance_frames = max(1, int(self.min_peak_distance_seconds * fps))

		for swimmer_id, swimmer_observations in by_swimmer.items():
			ordered = sorted(swimmer_observations, key=lambda item: item.frame)
			frames = np.array([obs.frame for obs in ordered], dtype=float)
			y_positions = np.array([obs.y for obs in ordered], dtype=float)

			if len(y_positions) < 3:
				stroke_rate_spm = 0.0
			else:
				peaks, _ = find_peaks(
					y_positions,
					prominence=self.min_peak_prominence,
					distance=min_distance_frames,
				)

				if len(peaks) < 2:
					stroke_rate_spm = 0.0
				else:
					elapsed_minutes = (
						(frames[peaks[-1]] - frames[peaks[0]]) / fps
					) / 60.0
					stroke_rate_spm = (
						float(len(peaks)) / elapsed_minutes if elapsed_minutes > 0 else 0.0
					)

			for obs in ordered:
				stroke_rate_by_frame_and_id[(obs.frame, swimmer_id)] = float(stroke_rate_spm)

		return stroke_rate_by_frame_and_id
