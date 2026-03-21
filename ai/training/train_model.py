"""
YOLOv8 small-dataset training script (Google Colab ready).

Key goals:
- Transfer learning from a large pretrained detector (yolov8l.pt by default)
- Strong underwater/swimming augmentation
- Hyperparameter choices tuned for ~200 images
- Overfitting control (regularization + early stopping + close_mosaic)
- GPU-memory-aware batch sizing
- Cosine learning rate schedule
- Proper metric logging (plots, CSV, checkpoints, TensorBoard-compatible logs)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def ensure_ultralytics() -> None:
	"""Install ultralytics if missing (useful in fresh Colab runtimes)."""
	try:
		import ultralytics  # noqa: F401  # type: ignore[import-not-found]
	except ImportError:
		subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])


def main() -> None:
	ensure_ultralytics()

	from ultralytics import YOLO  # type: ignore[import-not-found]

	# -----------------------------
	# Paths (Colab + local friendly)
	# -----------------------------
	script_dir = Path(__file__).resolve().parent
	default_data_yaml = script_dir / "dataset.yaml"

	# Use your dataset yaml location; for Colab, this is commonly under /content/...
	data_yaml = "/content/dataset.yaml"

	# --------------------------------------------
	# Model choice for transfer learning
	# --------------------------------------------
	# yolov8l.pt: best speed/accuracy balance on common Colab GPUs (T4/L4)
	# yolov8x.pt: higher potential accuracy, but much heavier memory/compute
	model_weights = "yolov8l.pt"  # Change to "yolov8x.pt" if GPU memory is ample.

	model = YOLO(model_weights)

	# --------------------------------------------
	# Small-dataset tuned hyperparameters (suggested)
	# --------------------------------------------
	# lr0 = 0.002:
	#   Lower than common defaults to avoid unstable updates when only ~200 images are available.
	lr0 = 0.002

	# lrf = 0.01:
	#   Final LR = lr0 * lrf with cosine schedule; decays to 1% of initial LR for stable convergence.
	lrf = 0.01

	# batch = 0.70 (AutoBatch mode):
	#   Uses ~70% of available GPU memory automatically, maximizing throughput while reducing OOM risk.
	#   If you prefer fixed values, a good starting point is:
	#   - 8 for yolov8l on Colab T4
	#   - 4 for yolov8x on Colab T4
	batch = 0.70

	# --------------------------------------------
	# Train
	# --------------------------------------------
	results = model.train(
		# Data and model IO
		data=data_yaml,
		imgsz=960,
		epochs=300,
		batch=batch,
		project="runs/train",
		name="yolov8_small_underwater",
		exist_ok=True,

		# Optimization
		optimizer="AdamW",   # More stable than SGD for small datasets + transfer learning.
		lr0=lr0,
		lrf=lrf,
		cos_lr=True,          # Cosine schedule as requested.
		momentum=0.90,
		weight_decay=0.001,   # Slightly stronger regularization to reduce overfitting.
		warmup_epochs=5.0,
		warmup_momentum=0.80,
		warmup_bias_lr=0.05,

		# Overfitting control
		patience=40,          # Early stopping if no val improvement for 40 epochs.
		freeze=10,            # Freeze early backbone layers; helps small data generalization.
		close_mosaic=20,      # Turn off mosaic in final epochs to match real image distribution.
		label_smoothing=0.05, # Slight target smoothing improves robustness on noisy labels.

		# Strong augmentation for underwater/swimming scenes
		hsv_h=0.02,           # Mild hue shift for underwater color cast changes.
		hsv_s=0.75,           # Strong saturation variation (water lighting/color variability).
		hsv_v=0.50,           # Brightness variation for reflections/shadows.
		degrees=7.5,          # Small rotation; swimmers/camera angle variation.
		translate=0.15,       # Position jitter to improve localization robustness.
		scale=0.55,           # Scale variation for distance/crop differences.
		shear=2.0,            # Mild geometric distortion.
		perspective=0.0007,   # Slight perspective variation for camera viewpoint shifts.
		fliplr=0.50,          # Left-right flip usually valid in swimming footage.
		flipud=0.10,          # Vertical flip is less realistic; keep low but non-zero for robustness.
		mosaic=1.0,           # Strong data mixing helps when dataset is very small.
		mixup=0.20,           # Additional blending for regularization.
		copy_paste=0.10,      # Adds object-level composition diversity.
		erasing=0.25,         # Simulates occlusions/splashes/partial visibility.

		# Runtime + reproducibility
		device=0,
		workers=2,            # Conservative worker count for Colab memory stability.
		cache="ram",         # Faster epochs if RAM allows; change to False if RAM is limited.
		amp=True,             # Mixed precision for speed + lower VRAM.
		seed=42,
		deterministic=False,

		# Validation and logging
		val=True,
		save=True,
		save_period=10,
		plots=True,           # Generates PR/F1/confusion/result curves.
		verbose=True,
	)

	print("Training complete.")
	print("Best checkpoint:", model.trainer.best)
	print("Results saved under:", model.trainer.save_dir)
	print("Metrics object:", results)


if __name__ == "__main__":
	main()
