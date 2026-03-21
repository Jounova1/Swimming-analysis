"""
Practical YOLOv8 improvement pipeline for swimmer detection on small datasets.

Commands:
1) split-video-wise
   Create train/val splits grouped by video prefix to avoid frame leakage.

2) train
   Two-stage transfer-learning training with strong underwater augmentation,
   cosine LR scheduler, early stopping, and small-data regularization.

3) tune-thresholds
   Grid-search inference conf/iou thresholds on validation images to reduce
   false detections while preserving recall.

4) track-video
   Run detection + ByteTrack for temporal stability in videos.

Google Colab friendly:
- Automatically installs ultralytics if missing.
"""

from __future__ import annotations

import argparse
import itertools
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def ensure_ultralytics() -> None:
	"""Install ultralytics lazily for fresh Colab/runtime environments."""
	try:
		import ultralytics  # noqa: F401  # type: ignore[import-not-found]
	except ImportError:
		subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])


def ensure_yaml() -> None:
	"""Install pyyaml if missing so config files can be read/written."""
	try:
		import yaml  # noqa: F401  # type: ignore[import-not-found]
	except ImportError:
		subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyyaml"])


def ensure_opencv() -> None:
	"""Install OpenCV if missing (used in threshold tuning utility)."""
	try:
		import cv2  # noqa: F401  # type: ignore[import-not-found]
	except ImportError:
		subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "opencv-python"])


def stem_video_id(file_stem: str) -> str:
	"""
	Extract a clip/group id from filename stem.

	Example:
	- bf_0017 -> bf
	- surface_2_0006 -> surface_2
	- flip_turn_0001 -> flip_turn
	"""
	parts = file_stem.split("_")
	if len(parts) <= 1:
		return file_stem
	if parts[-1].isdigit():
		return "_".join(parts[:-1])
	return file_stem


def copy_pair(src_img: Path, src_lbl: Path, dst_img_dir: Path, dst_lbl_dir: Path) -> None:
	dst_img_dir.mkdir(parents=True, exist_ok=True)
	dst_lbl_dir.mkdir(parents=True, exist_ok=True)
	shutil.copy2(src_img, dst_img_dir / src_img.name)
	if src_lbl.exists():
		shutil.copy2(src_lbl, dst_lbl_dir / src_lbl.name)
	else:
		# Keep explicit empty label for negatives; YOLO treats empty txt as background.
		(dst_lbl_dir / f"{src_img.stem}.txt").write_text("", encoding="utf-8")


def split_video_wise(
	dataset_root: Path,
	val_ratio: float,
	seed: int,
	image_extensions: tuple[str, ...] = ("*.jpg", "*.jpeg", "*.png"),
) -> None:
	"""
	Build video-wise train/val split to prevent train/val leakage from adjacent frames.
	"""
	random.seed(seed)

	img_dir = dataset_root / "images" / "all"
	lbl_dir = dataset_root / "labels" / "all"

	if not img_dir.exists():
		raise FileNotFoundError(f"Expected source images at: {img_dir}")

	images: list[Path] = []
	for ext in image_extensions:
		images.extend(sorted(img_dir.glob(ext)))

	if not images:
		raise RuntimeError(f"No images found under: {img_dir}")

	grouped: dict[str, list[Path]] = {}
	for img in images:
		key = stem_video_id(img.stem)
		grouped.setdefault(key, []).append(img)

	groups = sorted(grouped.keys())
	random.shuffle(groups)
	n_val_groups = max(1, int(len(groups) * val_ratio))
	val_groups = set(groups[:n_val_groups])

	train_img = dataset_root / "images" / "train"
	val_img = dataset_root / "images" / "val"
	train_lbl = dataset_root / "labels" / "train"
	val_lbl = dataset_root / "labels" / "val"

	for d in [train_img, val_img, train_lbl, val_lbl]:
		if d.exists():
			shutil.rmtree(d)
		d.mkdir(parents=True, exist_ok=True)

	n_train = 0
	n_val = 0
	for group_name, imgs in grouped.items():
		to_val = group_name in val_groups
		for img in imgs:
			lbl = lbl_dir / f"{img.stem}.txt"
			if to_val:
				copy_pair(img, lbl, val_img, val_lbl)
				n_val += 1
			else:
				copy_pair(img, lbl, train_img, train_lbl)
				n_train += 1

	print(f"[split] train images: {n_train}, val images: {n_val}")
	print(f"[split] train groups: {len(groups) - len(val_groups)}, val groups: {len(val_groups)}")


def update_dataset_yaml(dataset_yaml: Path, dataset_root: Path, class_name: str = "swimmer") -> None:
	"""Write a clean dataset YAML with train/val paths from dataset_root."""
	ensure_yaml()
	import yaml  # type: ignore[import-not-found]

	payload = {
		"path": str(dataset_root.resolve()),
		"train": "images/train",
		"val": "images/val",
		"nc": 1,
		"names": [class_name],
	}

	dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
	with dataset_yaml.open("w", encoding="utf-8") as f:
		yaml.safe_dump(payload, f, sort_keys=False)

	print(f"[yaml] Updated: {dataset_yaml}")


def train_small_data(
	dataset_yaml: Path,
	model_size: str,
	epochs_stage1: int,
	epochs_stage2: int,
	batch: float,
	imgsz: int,
	project: str,
	run_name: str,
) -> None:
	"""
	Two-stage transfer-learning strategy for ~200 images.

	Stage 1 (frozen backbone): stabilizes head adaptation and reduces early overfitting.
	Stage 2 (unfrozen): full fine-tuning for final accuracy.
	"""
	ensure_ultralytics()
	from ultralytics import YOLO  # type: ignore[import-not-found]

	if model_size not in {"l", "x"}:
		raise ValueError("model_size must be one of: 'l' or 'x'")

	weights = f"yolov8{model_size}.pt"
	print(f"[train] Using pretrained weights: {weights}")

	# Recommended small-data learning rates:
	# lr0=0.002: conservative start to avoid unstable updates on tiny datasets.
	# lrf=0.01: cosine anneal down to 1% of initial LR for smooth convergence.
	lr0 = 0.002
	lrf = 0.01

	# Batch recommendation:
	# batch=0.70 uses AutoBatch and targets 70% VRAM usage (safer on Colab GPUs).
	# If you prefer fixed integers, start with 8 (v8l) or 4 (v8x).
	# This script defaults to AutoBatch behavior for portability.

	# Stage 1: frozen backbone
	model = YOLO(weights)
	model.train(
		data=str(dataset_yaml),
		epochs=epochs_stage1,
		batch=batch,
		imgsz=imgsz,
		optimizer="AdamW",   # Small-data stable optimizer.
		lr0=lr0,
		lrf=lrf,
		cos_lr=True,          # Requirement: cosine schedule enabled.
		weight_decay=0.001,   # Regularization to reduce overfitting.
		warmup_epochs=5.0,
		patience=max(15, epochs_stage1 // 2),
		freeze=10,            # Freeze early layers for transfer-learning stability.
		close_mosaic=min(10, max(1, epochs_stage1 // 3)),
		label_smoothing=0.05,
		hsv_h=0.02,
		hsv_s=0.75,
		hsv_v=0.50,
		degrees=7.5,
		translate=0.15,
		scale=0.55,
		shear=2.0,
		perspective=0.0007,
		fliplr=0.50,
		flipud=0.10,
		mosaic=1.0,
		mixup=0.20,
		copy_paste=0.10,
		erasing=0.25,
		amp=True,
		cache="ram",
		workers=2,
		seed=42,
		val=True,
		save=True,
		plots=True,
		project=project,
		name=f"{run_name}_stage1",
		exist_ok=True,
	)

	# Stage 2: unfreeze and continue from best checkpoint
	stage1_best = Path(project) / f"{run_name}_stage1" / "weights" / "best.pt"
	if not stage1_best.exists():
		raise FileNotFoundError(f"Stage 1 best checkpoint not found: {stage1_best}")

	finetune = YOLO(str(stage1_best))
	finetune.train(
		data=str(dataset_yaml),
		epochs=epochs_stage2,
		batch=batch,
		imgsz=imgsz,
		optimizer="AdamW",
		lr0=lr0 * 0.6,        # Slightly lower LR for full-model fine-tuning.
		lrf=lrf,
		cos_lr=True,
		weight_decay=0.0012,  # Slightly stronger regularization in full unfreeze stage.
		warmup_epochs=3.0,
		patience=40,          # Requirement: early stopping enabled.
		freeze=0,
		close_mosaic=20,      # Disable mosaic near end for real-distribution alignment.
		label_smoothing=0.05,
		hsv_h=0.02,
		hsv_s=0.70,
		hsv_v=0.45,
		degrees=5.0,
		translate=0.12,
		scale=0.45,
		shear=1.5,
		perspective=0.0005,
		fliplr=0.50,
		flipud=0.05,
		mosaic=0.8,
		mixup=0.15,
		copy_paste=0.08,
		erasing=0.20,
		amp=True,
		cache="ram",
		workers=2,
		seed=42,
		val=True,
		save=True,
		save_period=10,
		plots=True,
		project=project,
		name=f"{run_name}_stage2",
		exist_ok=True,
	)

	print("[train] Completed Stage 1 + Stage 2 training")
	print(f"[train] Final best checkpoint: {Path(project) / f'{run_name}_stage2' / 'weights' / 'best.pt'}")


@dataclass
class Box:
	cls_id: int
	x1: float
	y1: float
	x2: float
	y2: float
	conf: float


def yolo_txt_to_boxes(label_file: Path, w: int, h: int) -> list[Box]:
	"""Parse YOLO label txt to absolute xyxy boxes."""
	if not label_file.exists():
		return []

	out: list[Box] = []
	for line in label_file.read_text(encoding="utf-8").strip().splitlines():
		parts = line.strip().split()
		if len(parts) != 5:
			continue
		cls_id = int(float(parts[0]))
		cx, cy, bw, bh = map(float, parts[1:])
		x1 = (cx - bw / 2.0) * w
		y1 = (cy - bh / 2.0) * h
		x2 = (cx + bw / 2.0) * w
		y2 = (cy + bh / 2.0) * h
		out.append(Box(cls_id, x1, y1, x2, y2, conf=1.0))
	return out


def iou_xyxy(a: Box, b: Box) -> float:
	ix1 = max(a.x1, b.x1)
	iy1 = max(a.y1, b.y1)
	ix2 = min(a.x2, b.x2)
	iy2 = min(a.y2, b.y2)
	iw = max(0.0, ix2 - ix1)
	ih = max(0.0, iy2 - iy1)
	inter = iw * ih
	if inter <= 0:
		return 0.0
	area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
	area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
	union = area_a + area_b - inter
	return inter / union if union > 0 else 0.0


def match_counts(preds: list[Box], gts: list[Box], iou_thr: float) -> tuple[int, int, int]:
	"""Greedy matching to compute TP/FP/FN."""
	used_gt = set()
	tp = 0
	fp = 0

	preds_sorted = sorted(preds, key=lambda b: b.conf, reverse=True)
	for p in preds_sorted:
		best_j = -1
		best_iou = 0.0
		for j, g in enumerate(gts):
			if j in used_gt:
				continue
			if p.cls_id != g.cls_id:
				continue
			iou = iou_xyxy(p, g)
			if iou > best_iou:
				best_iou = iou
				best_j = j
		if best_j >= 0 and best_iou >= iou_thr:
			tp += 1
			used_gt.add(best_j)
		else:
			fp += 1

	fn = len(gts) - len(used_gt)
	return tp, fp, fn


def tune_thresholds(
	model_path: Path,
	dataset_root: Path,
	image_size: int,
	conf_grid: Iterable[float],
	iou_grid: Iterable[float],
	match_iou: float = 0.5,
) -> None:
	"""
	Find practical conf/iou inference thresholds that reduce false positives.
	"""
	ensure_ultralytics()
	ensure_opencv()
	from ultralytics import YOLO  # type: ignore[import-not-found]

	import cv2  # type: ignore[import-not-found]

	model = YOLO(str(model_path))
	val_img_dir = dataset_root / "images" / "val"
	val_lbl_dir = dataset_root / "labels" / "val"

	images = sorted(
		list(val_img_dir.glob("*.jpg"))
		+ list(val_img_dir.glob("*.jpeg"))
		+ list(val_img_dir.glob("*.png"))
	)
	if not images:
		raise RuntimeError(f"No validation images found under: {val_img_dir}")

	best = None
	rows: list[tuple[float, float, int, int, int, float, float, float]] = []

	for conf, iou in itertools.product(conf_grid, iou_grid):
		total_tp = 0
		total_fp = 0
		total_fn = 0

		for img_path in images:
			img = cv2.imread(str(img_path))
			if img is None:
				continue
			h, w = img.shape[:2]

			gt = yolo_txt_to_boxes(val_lbl_dir / f"{img_path.stem}.txt", w=w, h=h)

			preds_raw = model.predict(
				source=str(img_path),
				imgsz=image_size,
				conf=conf,
				iou=iou,
				verbose=False,
				max_det=50,
				device=0,
			)

			pred_boxes: list[Box] = []
			if preds_raw and preds_raw[0].boxes is not None:
				b = preds_raw[0].boxes
				xyxy = b.xyxy.cpu().numpy()
				cls = b.cls.cpu().numpy().astype(int)
				confs = b.conf.cpu().numpy()
				for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, confs):
					pred_boxes.append(Box(int(c), float(x1), float(y1), float(x2), float(y2), float(cf)))

			tp, fp, fn = match_counts(pred_boxes, gt, iou_thr=match_iou)
			total_tp += tp
			total_fp += fp
			total_fn += fn

		precision = total_tp / max(1, total_tp + total_fp)
		recall = total_tp / max(1, total_tp + total_fn)
		f1 = 2 * precision * recall / max(1e-9, precision + recall)

		# Score prioritizes F1 and penalizes false positives slightly.
		score = f1 - 0.15 * (total_fp / max(1, len(images)))
		rows.append((conf, iou, total_tp, total_fp, total_fn, precision, recall, f1))

		if best is None or score > best["score"]:
			best = {
				"score": score,
				"conf": conf,
				"iou": iou,
				"tp": total_tp,
				"fp": total_fp,
				"fn": total_fn,
				"precision": precision,
				"recall": recall,
				"f1": f1,
			}

	print("\n[tune] Top threshold candidates (sorted by F1):")
	for r in sorted(rows, key=lambda x: x[-1], reverse=True)[:5]:
		conf, iou, tp, fp, fn, p, rec, f1 = r
		print(
			f"conf={conf:.2f}, iou={iou:.2f} | "
			f"TP={tp}, FP={fp}, FN={fn} | "
			f"P={p:.3f}, R={rec:.3f}, F1={f1:.3f}"
		)

	if best:
		print("\n[tune] Recommended operating point (F1 + FP penalty):")
		print(
			f"conf={best['conf']:.2f}, iou={best['iou']:.2f} | "
			f"TP={best['tp']}, FP={best['fp']}, FN={best['fn']} | "
			f"P={best['precision']:.3f}, R={best['recall']:.3f}, F1={best['f1']:.3f}"
		)


def track_video(
	model_path: Path,
	video_path: Path,
	out_dir: Path,
	conf: float,
	iou: float,
) -> None:
	"""
	Run detection + tracking for temporal stability and fewer one-frame false positives.
	"""
	ensure_ultralytics()
	from ultralytics import YOLO  # type: ignore[import-not-found]

	if not video_path.exists():
		raise FileNotFoundError(f"Video not found: {video_path}")

	model = YOLO(str(model_path))
	out_dir.mkdir(parents=True, exist_ok=True)

	model.track(
		source=str(video_path),
		conf=conf,
		iou=iou,
		tracker="bytetrack.yaml",  # Tracking improves stability in partial visibility.
		save=True,
		project=str(out_dir),
		name="tracked",
		exist_ok=True,
		show=False,
		verbose=True,
	)

	print(f"[track] Saved tracked output under: {out_dir / 'tracked'}")


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Swimmer detection enhancement pipeline")
	sub = parser.add_subparsers(dest="cmd", required=True)

	p_split = sub.add_parser("split-video-wise", help="Create video-wise train/val split")
	p_split.add_argument("--dataset-root", type=Path, required=True)
	p_split.add_argument("--val-ratio", type=float, default=0.2)
	p_split.add_argument("--seed", type=int, default=42)
	p_split.add_argument("--dataset-yaml", type=Path, default=Path("ai/training/dataset.yaml"))
	p_split.add_argument("--class-name", type=str, default="swimmer")

	p_train = sub.add_parser("train", help="Two-stage small-data transfer training")
	p_train.add_argument("--dataset-yaml", type=Path, default=Path("ai/training/dataset.yaml"))
	p_train.add_argument("--model-size", type=str, choices=["l", "x"], default="l")
	p_train.add_argument("--epochs-stage1", type=int, default=50)
	p_train.add_argument("--epochs-stage2", type=int, default=180)
	p_train.add_argument("--batch", type=float, default=0.70)
	p_train.add_argument("--imgsz", type=int, default=960)
	p_train.add_argument("--project", type=str, default="runs/train")
	p_train.add_argument("--run-name", type=str, default="swimmer_small_data")

	p_tune = sub.add_parser("tune-thresholds", help="Tune conf/iou to reduce false detections")
	p_tune.add_argument("--model", type=Path, required=True)
	p_tune.add_argument("--dataset-root", type=Path, required=True)
	p_tune.add_argument("--imgsz", type=int, default=960)
	p_tune.add_argument("--match-iou", type=float, default=0.5)

	p_track = sub.add_parser("track-video", help="Run ByteTrack video tracking")
	p_track.add_argument("--model", type=Path, required=True)
	p_track.add_argument("--video", type=Path, required=True)
	p_track.add_argument("--out-dir", type=Path, default=Path("runs/track"))
	p_track.add_argument("--conf", type=float, default=0.40)
	p_track.add_argument("--iou", type=float, default=0.55)

	return parser


def main() -> None:
	args = build_parser().parse_args()

	if args.cmd == "split-video-wise":
		split_video_wise(
			dataset_root=args.dataset_root,
			val_ratio=args.val_ratio,
			seed=args.seed,
		)
		update_dataset_yaml(
			dataset_yaml=args.dataset_yaml,
			dataset_root=args.dataset_root,
			class_name=args.class_name,
		)
		return

	if args.cmd == "train":
		train_small_data(
			dataset_yaml=args.dataset_yaml,
			model_size=args.model_size,
			epochs_stage1=args.epochs_stage1,
			epochs_stage2=args.epochs_stage2,
			batch=args.batch,
			imgsz=args.imgsz,
			project=args.project,
			run_name=args.run_name,
		)
		return

	if args.cmd == "tune-thresholds":
		conf_grid = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
		iou_grid = [0.45, 0.50, 0.55, 0.60]
		tune_thresholds(
			model_path=args.model,
			dataset_root=args.dataset_root,
			image_size=args.imgsz,
			conf_grid=conf_grid,
			iou_grid=iou_grid,
			match_iou=args.match_iou,
		)
		return

	if args.cmd == "track-video":
		track_video(
			model_path=args.model,
			video_path=args.video,
			out_dir=args.out_dir,
			conf=args.conf,
			iou=args.iou,
		)
		return

	raise ValueError(f"Unsupported command: {args.cmd}")


if __name__ == "__main__":
	main()

