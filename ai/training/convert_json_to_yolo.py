import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable


SPLITS = ("Trainset", "Valset", "Testset")
VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_ultralytics() -> None:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))



def list_image_files(folder: Path) -> list[Path]:
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_SUFFIXES]



def yolo_line_from_vertices(vertices: Iterable[float]) -> str | None:
    vals = list(vertices)
    if len(vals) != 4:
        return None

    x1, y1, x2, y2 = [float(v) for v in vals]
    x_min, x_max = sorted((x1, x2))
    y_min, y_max = sorted((y1, y2))

    width = x_max - x_min
    height = y_max - y_min
    if width <= 0 or height <= 0:
        return None

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0

    x_center = clamp(x_center)
    y_center = clamp(y_center)
    width = clamp(width)
    height = clamp(height)

    if width <= 0 or height <= 0:
        return None

    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"



def convert_one_json(json_path: Path, txt_path: Path) -> tuple[bool, int, str | None]:
    """Return (success, number_of_boxes_written, error_message)."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, 0, f"read/json error: {e}"

    annotations = data.get("annotations", [])
    if not isinstance(annotations, list):
        return False, 0, "annotations field is not a list"

    lines: list[str] = []

    for ann in annotations:
        if not isinstance(ann, dict):
            continue

        category = str(ann.get("category", "")).strip().lower()
        if category != "swimmer":
            continue

        geometry = ann.get("geometry", {})
        if not isinstance(geometry, dict):
            continue

        if str(geometry.get("type", ann.get("type", "rectangle"))).lower() not in {"rectangle", "bbox", "box"}:
            # Keep going if type is unusual but vertices still exist.
            pass

        vertices = geometry.get("vertices")
        if vertices is None:
            continue

        line = yolo_line_from_vertices(vertices)
        if line is not None:
            lines.append(line)

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return True, len(lines), None



def clear_yolo_cache(labels_root: Path) -> None:
    removed = 0
    for cache_file in labels_root.rglob("*.cache"):
        try:
            cache_file.unlink()
            removed += 1
        except OSError:
            pass
    print(f"[INFO] Removed {removed} old cache file(s).")



def convert_json_to_labels(dataset_root: Path) -> dict[str, dict[str, int]]:
    annotations_root = dataset_root / "annotations"
    labels_root = dataset_root / "labels"
    images_root = dataset_root / "images"

    summary: dict[str, dict[str, int]] = {}

    clear_yolo_cache(labels_root)

    print("\n[INFO] Starting JSON -> YOLO TXT conversion...")
    for split in SPLITS:
        ann_dir = annotations_root / split
        label_dir = labels_root / split
        image_dir = images_root / split
        label_dir.mkdir(parents=True, exist_ok=True)

        split_stats = {
            "json_files": 0,
            "txt_files": 0,
            "labeled_files": 0,
            "boxes": 0,
            "errors": 0,
            "images": 0,
        }

        if image_dir.exists():
            split_stats["images"] = len(list_image_files(image_dir))

        if not ann_dir.exists():
            print(f"[WARN] Missing annotations folder: {ann_dir}")
            summary[split] = split_stats
            continue

        json_files = sorted(ann_dir.glob("*.json"))
        split_stats["json_files"] = len(json_files)
        print(f"\n[INFO] {split}: found {len(json_files)} JSON file(s)")

        for json_file in json_files:
            txt_file = label_dir / f"{json_file.stem}.txt"
            ok, n_boxes, err = convert_one_json(json_file, txt_file)
            if not ok:
                split_stats["errors"] += 1
                print(f"[ERROR] {json_file.name}: {err}")
                continue

            split_stats["txt_files"] += 1
            split_stats["boxes"] += n_boxes
            if n_boxes > 0:
                split_stats["labeled_files"] += 1

        summary[split] = split_stats
        print(
            f"[INFO] {split} done | images={split_stats['images']} json={split_stats['json_files']} "
            f"txt={split_stats['txt_files']} labeled_txt={split_stats['labeled_files']} boxes={split_stats['boxes']} "
            f"errors={split_stats['errors']}"
        )

    total_boxes = sum(v["boxes"] for v in summary.values())
    total_labeled = sum(v["labeled_files"] for v in summary.values())
    print(f"\n[INFO] Conversion finished | labeled files={total_labeled}, total boxes={total_boxes}\n")
    return summary



def validate_labels(summary: dict[str, dict[str, int]]) -> None:
    train_boxes = summary.get("Trainset", {}).get("boxes", 0)
    val_boxes = summary.get("Valset", {}).get("boxes", 0)
    train_labeled = summary.get("Trainset", {}).get("labeled_files", 0)
    val_labeled = summary.get("Valset", {}).get("labeled_files", 0)

    if train_boxes <= 0 or train_labeled <= 0:
        raise RuntimeError(
            "No valid training labels were created. Check your JSON format and the conversion logic before training."
        )
    if val_boxes <= 0 or val_labeled <= 0:
        raise RuntimeError(
            "No valid validation labels were created. Check your JSON format and the conversion logic before training."
        )