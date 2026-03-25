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

	from ultralytics import YOLO  # pylint: disable=import-outside-toplevel

	script_dir = Path(__file__).resolve().parent

	# finetune → نكمل من best.pt (أسرع convergence)
	# scratch  → نبدأ من yolo11m.pt الرسمي (أفضل لو best.pt overfit)
	model_file = "best.pt" if args.mode == "finetune" else "yolo11m.pt"
	model_path = resolve_path(model_file, script_dir)

	# مسار ملف الداتا اللي فيه train/val paths وأسماء الـ classes
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

	model = YOLO(str(model_path))

	train_args = {
		# ── داتا ──────────────────────────────────────────────────────────
		"data": str(data_yaml),
		"single_cls": True,        # class واحدة بس (Swimmer) → يحسن التدريب

		# ── جدول التدريب ──────────────────────────────────────────────────
		"epochs": 80,
		"imgsz": 640,              # توازن بين الدقة والسرعة على RTX 3050
		"batch": 8,                # آمن لـ 4GB VRAM

		# ── الـ optimizer ─────────────────────────────────────────────────
		"optimizer": "SGD",        # أفضل من Adam للـ generalization في التدريب الطويل
		"lr0": 0.003,              # منخفض لأننا بنعمل fine-tune
		"momentum": 0.937,
		"weight_decay": 0.0005,    # يقلل overfitting

		# ── جدول الـ learning rate ────────────────────────────────────────
		"cos_lr": True,            # يقلل الـ LR بشكل سلس على شكل منحنى
		"warmup_epochs": 3,        # يبدأ ببطء لتفادي instability في الأول

		# ── منع الـ overfitting ───────────────────────────────────────────
		"patience": 25,            # يوقف التدريب لو مفيش تحسن لـ 25 epoch
		"close_mosaic": 15,        # يوقف mosaic في آخر 15 epoch لتثبيت الـ detections

		# ── تعديلات الألوان والإضاءة ──────────────────────────────────────
		"hsv_h": 0.015,
		"hsv_s": 0.5,              # يحاكي بيئات إضاءة مختلفة
		"hsv_v": 0.35,             # مهم جداً للمسابح المختلفة

		# ── التحويلات الهندسية ────────────────────────────────────────────
		"degrees": 3.0,            # السباحون دايماً أفقيين تقريباً → دوران بسيط بس
		"translate": 0.08,         # يحاكي تحرك السباح في الإطار
		"scale": 0.35,             # يتعامل مع السباحين في مسافات مختلفة
		"shear": 0.5,
		"perspective": 0.0003,
		"fliplr": 0.5,             # يضاعف الداتا (يسار = يمين)
		"flipud": 0.0,             # سباح مقلوب مش منطقي → معطل

		# ── augmentations متقدمة ──────────────────────────────────────────
		"mosaic": 0.25,            # مخفف عن الـ default (1.0) عشان الفريمات متكررة
		"mixup": 0.05,
		"erasing": 0.1,            # يحاكي إخفاء السباح بالمياه أو حبال الـ lanes

		# ── هاردوير ───────────────────────────────────────────────────────
		"amp": True,               # يوفر VRAM ويسرع التدريب
		"workers": 4,
		"device": 0,

		# ── مخرجات ────────────────────────────────────────────────────────
		"save_period": 10,         # حفظ checkpoint كل 10 epochs (حماية من الـ crashes)
		"val": True,
		"plots": True,
		"project": "runs/train",
		"name": f"yolo_swimmer_{args.mode}_rtx3050",
		"exist_ok": True,
	}

	print("Starting YOLO training with the following setup:")
	print(f"  Mode:             {args.mode}")
	print(f"  Model checkpoint: {model_path}")
	print(f"  Dataset YAML:     {data_yaml}")
	print(f"  Epochs:           {train_args['epochs']}")
	print(f"  Image size:       {train_args['imgsz']}")
	print(f"  Batch size:       {train_args['batch']}")

	results = model.train(**train_args)

	save_dir = Path(results.save_dir).resolve()
	best_model_path = (save_dir / "weights" / "best.pt").resolve()

	print("\nTraining finished.")
	print(f"Results directory: {save_dir}")
	print(f"Best model path:   {best_model_path}")


if __name__ == "__main__":
	main()