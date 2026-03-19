from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

try:
	from inference.track_swimmer import TrackObservation
except ModuleNotFoundError:
	from track_swimmer import TrackObservation


class VelocityEstimator:
	"""Estimate swimmer velocity from tracked positions.

	Formula:
	v = delta_x * pixel_to_meter * fps
	"""

	def __init__(self, pixel_to_meter: float = 0.01) -> None:
		self.pixel_to_meter = pixel_to_meter

	def estimate(
		self, observations: Iterable[TrackObservation], fps: float
	) -> Dict[Tuple[int, int], float]:
		by_swimmer: Dict[int, List[TrackObservation]] = defaultdict(list)
		for obs in observations:
			by_swimmer[obs.swimmer_id].append(obs)

		velocity_by_frame_and_id: Dict[Tuple[int, int], float] = {}

		for swimmer_id, swimmer_observations in by_swimmer.items():
			ordered = sorted(swimmer_observations, key=lambda item: item.frame)
			previous = None

			for current in ordered:
				if previous is None:
					velocity = 0.0
				else:
					delta_x = current.x - previous.x
					velocity = abs(delta_x) * self.pixel_to_meter * fps

				velocity_by_frame_and_id[(current.frame, swimmer_id)] = float(velocity)
				previous = current

		return velocity_by_frame_and_id
