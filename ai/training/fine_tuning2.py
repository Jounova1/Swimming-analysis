"""Train YOLO for swimmer detection with generalization-focused settings.

This script is configured for an RTX 3050 target profile and uses
carefully balanced augmentations for robust detection training.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path


def install_ultralytics_if_missing() -> None:
	"""Install Ultralytics automatically if missing, or upgrade to latest if present."""
	if importlib.util.find_spec("ultralytics") is None:
		print("ultralytics not found. Installing latest version...")
		subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
		print("ultralytics installed successfully.")
		return

	# Keep package current so script uses latest Ultralytics features/fixes.
	print("ultralytics is installed. Upgrading to latest version...")
	subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "ultralytics"])
	print("ultralytics upgraded successfully.")


def resolve_path(path_str: str, base_dir: Path) -> Path:
	"""Resolve relative paths from the script directory for convenience."""
	path = Path(path_str)
	if path.is_absolute():
		return path
	return (base_dir / path).resolve()


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments so switching model source is easy."""
	parser = argparse.ArgumentParser(description="Ultralytics YOLO swimmer training")
	parser.add_argument(
		"--mode",
		choices=["finetune", "scratch"],
		default="finetune",
		help="finetune: load best.pt, scratch: load yolo11m.pt pretrained backbone.",
	)
	parser.add_argument(
		"--data",
		default="dataset_1.yaml",
		help="Path to dataset YAML file.",
	)
	return parser.parse_args()


def main() -> None:
	"""Run YOLO training with RTX 3050-friendly defaults."""
	args = parse_args()
	install_ultralytics_if_missing()

	# Import only after optional installation succeeds.
	from ultralytics import YOLO  # pylint: disable=import-outside-toplevel

	script_dir = Path(__file__).resolve().parent

	# Fine-tune from an existing task-specific checkpoint or start from base model.
	model_file = "best.pt" if args.mode == "finetune" else "yolo11m.pt"
	model_path = resolve_path(model_file, script_dir)

	# Path to dataset YAML describing train/val/test image and label folders.
	data_yaml = resolve_path(args.data, script_dir)

	if not model_path.exists():
		raise FileNotFoundError(
			f"Model checkpoint not found: {model_path}. "
			"Place the selected checkpoint in the training folder or provide a valid path."
		)

	if not data_yaml.exists():
		raise FileNotFoundError(
			f"Dataset YAML not found: {data_yaml}. "
			"Place dataset YAML in the training folder or edit data_yaml."
		)

	# Load YOLO model from selected checkpoint (fine-tune or scratch-pretrained).
	model = YOLO(str(model_path))

	# All training arguments are explicit and documented for clarity.
	train_args = {
		"data": str(data_yaml),  # Custom dataset config file in YOLO YAML format.
		"single_cls": True,  # Force one-class training for unified "Swimmer" labels.
		"epochs": 80,  # Total number of training passes over the dataset.
		"imgsz": 640,  # Input image size; 640 is a good speed/accuracy point for RTX 3050.
		"batch": 8,  # Batch size tuned to typical 4GB RTX 3050 VRAM constraints.
		"optimizer": "SGD",  # Requested optimizer for stable, well-understood convergence.
		"lr0": 0.003,  # Initial learning rate at epoch 0.
		"momentum": 0.937,  # SGD momentum to smooth updates and accelerate training.
		"weight_decay": 0.0005,  # L2 regularization to reduce overfitting.
		"cos_lr": True,  # Enable cosine learning rate decay schedule.
		"warmup_epochs": 3,  # Gradually ramp up training during early epochs.
		"patience": 25,  # Early stopping patience to reduce overfitting on repeated frames.
		"close_mosaic": 15,  # Disable mosaic in final epochs to stabilize box refinement.
		"hsv_h": 0.015,  # Hue jitter for mild color robustness.
		"hsv_s": 0.5,  # Saturation jitter for stronger color augmentation.
		"hsv_v": 0.35,  # Value/brightness jitter for illumination changes.
		"degrees": 3.0,  # Small random rotation to handle camera angle shifts.
		"translate": 0.08,  # Random translation for positional robustness.
		"scale": 0.35,  # Scale jitter to simulate swimmer size variation.
		"shear": 0.5,  # Small shear to model perspective-like geometric changes.
		"perspective": 0.0003,  # Very light perspective transform to avoid label distortion.
		"fliplr": 0.5,  # Horizontal flip probability.
		"flipud": 0.0,  # Vertical flip disabled for realistic swimming orientation.
		"mosaic": 0.25,  # Balanced mosaic to improve context diversity.
		"mixup": 0.05,  # Light mixup to regularize without over-blending samples.
		"erasing": 0.1,  # Random erasing to improve occlusion robustness.
		"amp": True,  # Automatic mixed precision for faster training and lower VRAM usage.
		"workers": 4,  # Data loader workers to keep GPU fed without oversubscription.
		"save_period": 10,  # Save checkpoint every 10 epochs.
		"val": True,  # Run validation during training.
		"plots": True,  # Generate training curves and diagnostics plots.
		"device": 0,  # Use first CUDA GPU (expected RTX 3050 on single-GPU setup).
		"project": "runs/train",  # Parent folder for experiment outputs.
		"name": f"yolo_swimmer_{args.mode}_rtx3050",  # Distinct run name per training mode.
		"exist_ok": True,  # Reuse run folder name if it already exists.
	}

	print("Starting YOLO training with the following setup:")
	print(f"  Mode:             {args.mode}")
	print(f"  Model checkpoint: {model_path}")
	print(f"  Dataset YAML:     {data_yaml}")
	print(f"  Epochs:           {train_args['epochs']}")
	print(f"  Image size:       {train_args['imgsz']}")
	print(f"  Batch size:       {train_args['batch']}")

	# Launch training and capture results metadata.
	results = model.train(**train_args)

	# Ultralytics returns save_dir, where weights/best.pt is stored.
	save_dir = Path(results.save_dir).resolve()
	best_model_path = (save_dir / "weights" / "best.pt").resolve()

	print("\nTraining finished.")
	print(f"Results directory: {save_dir}")
	print(f"Best model path:   {best_model_path}")


if __name__ == "__main__":
	main()
