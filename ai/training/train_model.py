"""
YOLO training script with automatic JSON -> YOLO TXT conversion before training.

What this script does:
1) Ensures ultralytics is installed
2) Converts annotations from JSON to YOLO TXT
3) Creates labels folders automatically
4) Starts YOLO training

Supported JSON formats:
- LabelMe-style per-image JSON:
  {
    "imageWidth": ...,
    "imageHeight": ...,
    "shapes": [{"label": "swimmer", "points": [[x1,y1],[x2,y2]]}, ...]
  }

- COCO-style JSON:
  {
    "images": [...],
    "annotations": [...],
    "categories": [...]
  }
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def ensure_ultralytics() -> None:
    """Install ultralytics if missing."""
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clamp(value: float, min_v: float = 0.0, max_v: float = 1.0) -> float:
    return max(min_v, min(max_v, value))


def bbox_to_yolo(img_w: float, img_h: float, x: float, y: float, w: float, h: float):
    """Convert top-left bbox to YOLO normalized bbox."""
    x_center = (x + w / 2.0) / img_w
    y_center = (y + h / 2.0) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    return (
        clamp(x_center),
        clamp(y_center),
        clamp(w_norm),
        clamp(h_norm),
    )


def write_yolo_txt(txt_path: Path, lines: list[str]) -> None:
    ensure_dir(txt_path.parent)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def convert_labelme_json(json_path: Path, txt_path: Path, class_map: dict[str, int]) -> bool:
    """Convert one LabelMe-style JSON file into one YOLO txt file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not all(k in data for k in ("imageWidth", "imageHeight", "shapes")):
        return False

    img_w = float(data["imageWidth"])
    img_h = float(data["imageHeight"])
    lines: list[str] = []

    for shape in data.get("shapes", []):
        label = str(shape.get("label", "")).strip()
        if label not in class_map:
            continue

        points = shape.get("points", [])
        if len(points) < 2:
            continue

        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]

        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs)
        y_max = max(ys)

        bw = x_max - x_min
        bh = y_max - y_min

        if bw <= 0 or bh <= 0:
            continue

        x_c, y_c, w_n, h_n = bbox_to_yolo(img_w, img_h, x_min, y_min, bw, bh)
        lines.append(f"{class_map[label]} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

    write_yolo_txt(txt_path, lines)
    return True


def convert_coco_json(json_path: Path, output_labels_dir: Path, class_map: dict[str, int]) -> bool:
    """Convert one COCO-style JSON file into many YOLO txt files."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not all(k in data for k in ("images", "annotations")):
        return False

    images = {img["id"]: img for img in data.get("images", [])}
    categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}

    image_lines: dict[str, list[str]] = {}

    for ann in data.get("annotations", []):
        image_id = ann.get("image_id")
        category_id = ann.get("category_id")
        bbox = ann.get("bbox")

        if image_id not in images or not bbox or len(bbox) != 4:
            continue

        cat_name = categories.get(category_id, "swimmer")
        if cat_name not in class_map:
            continue

        img = images[image_id]
        img_w = float(img["width"])
        img_h = float(img["height"])
        file_name = Path(img["file_name"]).stem + ".txt"

        x, y, w, h = map(float, bbox)
        if w <= 0 or h <= 0:
            continue

        x_c, y_c, w_n, h_n = bbox_to_yolo(img_w, img_h, x, y, w, h)
        line = f"{class_map[cat_name]} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"

        image_lines.setdefault(file_name, []).append(line)

    ensure_dir(output_labels_dir)
    for txt_name, lines in image_lines.items():
        write_yolo_txt(output_labels_dir / txt_name, lines)

    return True


def convert_json_annotations(
    dataset_root: Path,
    class_map: dict[str, int],
    split_names: tuple[str, ...] = ("Trainset", "Valset", "Testset"),
) -> None:
    """
    Convert JSON annotations under:
      dataset_root / annotations / <split>
    into:
      dataset_root / labels / <split>
    """
    annotations_root = dataset_root / "annotations"
    labels_root = dataset_root / "labels"

    print("\n[INFO] Starting JSON -> YOLO TXT conversion...")

    for split in split_names:
        ann_dir = annotations_root / split
        label_dir = labels_root / split
        ensure_dir(label_dir)

        if not ann_dir.exists():
            print(f"[WARN] Annotation folder not found, skipping: {ann_dir}")
            continue

        json_files = sorted(ann_dir.glob("*.json"))
        print(f"[INFO] {split}: found {len(json_files)} JSON file(s) in {ann_dir}")

        for json_file in json_files:
            # Try LabelMe-style first
            txt_path = label_dir / f"{json_file.stem}.txt"
            converted = False

            try:
                converted = convert_labelme_json(json_file, txt_path, class_map)
                if converted:
                    continue
            except Exception as e:
                print(f"[WARN] LabelMe parse failed for {json_file.name}: {e}")

            # Try COCO-style
            try:
                converted = convert_coco_json(json_file, label_dir, class_map)
                if converted:
                    continue
            except Exception as e:
                print(f"[WARN] COCO parse failed for {json_file.name}: {e}")

            print(f"[ERROR] Unsupported or failed JSON format: {json_file}")

    print("[INFO] Conversion finished.\n")


def main() -> None:
    ensure_ultralytics()
    from ultralytics import YOLO

    # =============================
    # PATHS
    # =============================
    # غيّر المسار ده حسب مشروعك
    dataset_root = Path(r"D:\Swimming-analysis\ai\training\data")

    # لو dataset.yaml عندك في مكان تاني، عدله هنا
    data_yaml = r"D:\Swimming-analysis\ai\training\dataset.yaml"

    # =============================
    # CLASSES
    # =============================
    # لو اسم الكلاس في JSON مختلف، زوده هنا
    class_map = {
        "swimmer": 0,
        "Swimmer": 0,
    }

    # =============================
    # AUTO CONVERT JSON -> TXT
    # =============================
    convert_json_annotations(dataset_root=dataset_root, class_map=class_map)

    # =============================
    # MODEL
    # =============================
    model_weights = "yolo11m.pt"
    model = YOLO(model_weights)

    # =============================
    # TRAINING SETTINGS
    # مناسب أكثر لـ ~2000 صورة + RTX 3070
    # =============================
    results = model.train(
        # Data
        data=data_yaml,
        imgsz=640,
        epochs=300,
        batch=8,                  # جرّب 12 بعدين لو VRAM مستحملة
        project="runs/train",
        name="yolo11m_swimmer_2000imgs",
        exist_ok=True,

        # Optimization
        optimizer="AdamW",
        lr0=0.0015,
        lrf=0.01,
        cos_lr=True,
        momentum=0.90,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.80,
        warmup_bias_lr=0.05,

        # Regularization / stability
        patience=40,
        freeze=0,
        close_mosaic=15,
        label_smoothing=0.0,

        # Augmentation
        hsv_h=0.015,
        hsv_s=0.50,
        hsv_v=0.35,
        degrees=5.0,
        translate=0.10,
        scale=0.35,
        shear=1.0,
        perspective=0.0003,
        fliplr=0.50,
        flipud=0.0,
        mosaic=0.50,
        mixup=0.05,
        copy_paste=0.0,
        erasing=0.15,

        # Runtime
        device=0,
        workers=4,
        cache=False,
        amp=True,
        seed=42,
        deterministic=False,

        # Validation / logs
        val=True,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    print("Training complete.")
    print("Best checkpoint:", model.trainer.best)
    print("Results saved under:", model.trainer.save_dir)
    print("Metrics object:", results)


if __name__ == "__main__":
    main()