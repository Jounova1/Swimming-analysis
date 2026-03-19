from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import cv2

try:
	from inference.stroke_rate_analysis import StrokeRateAnalyzer
	from inference.track_swimmer import SwimmerTracker, TrackObservation
	from inference.velocity_estimation import VelocityEstimator
except ModuleNotFoundError:
	from stroke_rate_analysis import StrokeRateAnalyzer
	from track_swimmer import SwimmerTracker, TrackObservation
	from velocity_estimation import VelocityEstimator


class SwimmerAnalysisPipeline:
	"""Run swimmer tracking and derive kinematic metrics per tracked swimmer."""

	def __init__(
		self,
		tracker: SwimmerTracker,
		velocity_estimator: VelocityEstimator,
		stroke_analyzer: StrokeRateAnalyzer,
	) -> None:
		self.tracker = tracker
		self.velocity_estimator = velocity_estimator
		self.stroke_analyzer = stroke_analyzer

	@staticmethod
	def _get_video_fps(video_path: str) -> float:
		capture = cv2.VideoCapture(video_path)
		if not capture.isOpened():
			raise FileNotFoundError(f"Could not open video: {video_path}")

		fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
		capture.release()
		return fps if fps > 0 else 30.0

	def run(
		self,
		video_path: str,
		output_csv: str = "analysis.csv",
		visualize: bool = False,
	) -> List[Dict[str, float]]:
		fps = self._get_video_fps(video_path)

		observations: List[TrackObservation] = self.tracker.track_video(
			video_path, visualize=visualize
		)

		velocity_map = self.velocity_estimator.estimate(observations, fps=fps)
		stroke_rate_map = self.stroke_analyzer.estimate(observations, fps=fps)

		rows = []
		for obs in observations:
			rows.append(
				{
					"frame": obs.frame,
					"swimmer_id": obs.swimmer_id,
					"x": round(obs.x, 3),
					"y": round(obs.y, 3),
					"velocity_m_s": round(
						float(velocity_map.get((obs.frame, obs.swimmer_id), 0.0)), 4
					),
					"stroke_rate_spm": round(
						float(stroke_rate_map.get((obs.frame, obs.swimmer_id), 0.0)), 2
					),
				}
			)

		rows.sort(key=lambda item: (int(item["frame"]), int(item["swimmer_id"])))

		headers = [
			"frame",
			"swimmer_id",
			"x",
			"y",
			"velocity_m_s",
			"stroke_rate_spm",
		]

		with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
			writer = csv.DictWriter(csv_file, fieldnames=headers)
			writer.writeheader()
			writer.writerows(rows)

		print(f"Saved analysis results to {output_csv}")

		if not rows:
			print("No swimmers were tracked; result file contains headers only.")
		else:
			for row in rows:
				print(
					f"frame={row['frame']} swimmer_id={row['swimmer_id']} "
					f"x={row['x']} y={row['y']} "
					f"velocity_m_s={row['velocity_m_s']} "
					f"stroke_rate_spm={row['stroke_rate_spm']}"
				)

		return rows


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Run AI swimmer analysis pipeline")
	parser.add_argument("video_path", help="Path to input video")
	parser.add_argument(
		"--pixel-to-meter",
		type=float,
		default=0.01,
		help="Conversion factor from pixels to meters",
	)
	parser.add_argument(
		"--output-csv",
		default="analysis.csv",
		help="Output CSV path (default: analysis.csv)",
	)
	parser.add_argument(
		"--visualize",
		action="store_true",
		help="Display tracking visualization while processing video",
	)
	return parser


def main() -> None:
	args = build_arg_parser().parse_args()
	video_path = str(Path(args.video_path))

	tracker = SwimmerTracker()
	velocity_estimator = VelocityEstimator(pixel_to_meter=args.pixel_to_meter)
	stroke_analyzer = StrokeRateAnalyzer()

	pipeline = SwimmerAnalysisPipeline(
		tracker=tracker,
		velocity_estimator=velocity_estimator,
		stroke_analyzer=stroke_analyzer,
	)
	pipeline.run(
		video_path=video_path,
		output_csv=args.output_csv,
		visualize=args.visualize,
	)


if __name__ == "__main__":
	main()