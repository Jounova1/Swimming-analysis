from __future__ import annotations

import json
from pathlib import Path
from collections import Counter


# =========================================
# CONFIG
# =========================================
DATA_ROOT = Path(r"D:\Swimming-analysis\ai\training\data")

IMAGES_DIR = DATA_ROOT / "images"
ANNOTATIONS_DIR = DATA_ROOT / "annotations"
LABELS_DIR = DATA_ROOT / "labels"

SPLITS = ["Trainset", "Valset", "Testset"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_MAP = {
    "swimmer": 0,
    "Swimmer": 0,
}


# =========================================
# HELPERS
# =========================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def find_images(split_dir: Path) -> dict[str, Path]:
    images = {}
    for p in split_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            images[p.stem] = p
    return images


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def convert_vertices_to_yolo(vertices: list[float]) -> tuple[float, float, float, float] | None:
    """
    Input custom format:
    [x1, y1, x2, y2] already normalized in range [0,1]

    Output YOLO format:
    x_center, y_center, width, height
    """
    if len(vertices) != 4:
        return None

    x1, y1, x2, y2 = map(float, vertices)

    # sort just in case points are reversed
    x_min, x_max = sorted([x1, x2])
    y_min, y_max = sorted([y1, y2])

    w = x_max - x_min
    h = y_max - y_min

    if w <= 0 or h <= 0:
        return None

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0

    return (
        clamp(x_center),
        clamp(y_center),
        clamp(w),
        clamp(h),
    )


def parse_custom_json(json_path: Path) -> list[str]:
    """
    Expected JSON shape:
    {
      "annotations": [
        {
          "geometry": {"vertices": [x1, y1, x2, y2]},
          "category": "swimmer"
        }
      ]
    }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines: list[str] = []

    for ann in data.get("annotations", []):
        category = str(ann.get("category", "")).strip()
        if category not in CLASS_MAP:
            continue

        geometry = ann.get("geometry", {})
        vertices = geometry.get("vertices", None)
        if not vertices:
            continue

        yolo_box = convert_vertices_to_yolo(vertices)
        if yolo_box is None:
            continue

        xc, yc, w, h = yolo_box
        class_id = CLASS_MAP[category]
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    return lines


def write_txt(txt_path: Path, lines: list[str]) -> None:
    ensure_dir(txt_path.parent)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def clear_yolo_cache() -> None:
    for split in SPLITS:
        cache_file = LABELS_DIR / f"{split}.cache"
        remove_if_exists(cache_file)

        split_cache = LABELS_DIR / split.with_suffix(".cache") if isinstance(Path(split), Path) else None
        if split_cache and split_cache.exists():
            split_cache.unlink()

    # also remove common cache names inside labels/
    for p in LABELS_DIR.rglob("*.cache"):
        remove_if_exists(p)


# =========================================
# MAIN PREP
# =========================================
def prepare_split(split: str) -> dict[str, int]:
    image_split_dir = IMAGES_DIR / split
    ann_split_dir = ANNOTATIONS_DIR / split
    label_split_dir = LABELS_DIR / split

    ensure_dir(label_split_dir)

    stats = Counter()
    stats["split_found"] = int(image_split_dir.exists() and ann_split_dir.exists())

    if not image_split_dir.exists():
        print(f"[WARN] Missing images folder: {image_split_dir}")
        return dict(stats)

    if not ann_split_dir.exists():
        print(f"[WARN] Missing annotations folder: {ann_split_dir}")
        return dict(stats)

    images = find_images(image_split_dir)
    json_files = sorted(ann_split_dir.glob("*.json"))

    stats["images_found"] = len(images)
    stats["json_found"] = len(json_files)

    print(f"\n===== {split} =====")
    print(f"Images found: {len(images)}")
    print(f"JSON files found: {len(json_files)}")

    for json_file in json_files:
        stem = json_file.stem
        txt_path = label_split_dir / f"{stem}.txt"

        if stem not in images:
            stats["json_without_image"] += 1
            print(f"[WARN] No matching image for JSON: {json_file.name}")
            continue

        try:
            lines = parse_custom_json(json_file)
            write_txt(txt_path, lines)

            if lines:
                stats["txt_with_boxes"] += 1
                stats["boxes_total"] += len(lines)
            else:
                stats["txt_empty"] += 1

        except Exception as e:
            stats["failed_json"] += 1
            print(f"[ERROR] Failed on {json_file.name}: {e}")

    # images without json
    for image_stem in images:
        json_path = ann_split_dir / f"{image_stem}.json"
        if not json_path.exists():
            stats["image_without_json"] += 1

    # txt files check
    txt_files = list(label_split_dir.glob("*.txt"))
    stats["txt_total"] = len(txt_files)

    # empty txt count after write
    empty_count = 0
    for txt_file in txt_files:
        try:
            content = txt_file.read_text(encoding="utf-8").strip()
            if not content:
                empty_count += 1
        except Exception:
            pass
    stats["txt_empty_after_write"] = empty_count

    print(f"TXT files created: {stats['txt_total']}")
    print(f"TXT with boxes: {stats['txt_with_boxes']}")
    print(f"Empty TXT: {stats['txt_empty_after_write']}")
    print(f"Total boxes: {stats['boxes_total']}")
    print(f"Images without JSON: {stats['image_without_json']}")
    print(f"JSON without image: {stats['json_without_image']}")
    print(f"Failed JSON: {stats['failed_json']}")

    return dict(stats)


def validate_dataset() -> None:
    print("\n===== FINAL VALIDATION =====")
    for split in SPLITS:
        image_split_dir = IMAGES_DIR / split
        label_split_dir = LABELS_DIR / split

        if not image_split_dir.exists() or not label_split_dir.exists():
            print(f"[WARN] Skipping validation for {split} (missing folder)")
            continue

        images = find_images(image_split_dir)
        txts = {p.stem: p for p in label_split_dir.glob("*.txt")}

        matched = 0
        missing_labels = 0
        nonempty_labels = 0

        for stem in images:
            txt = txts.get(stem)
            if txt is None:
                missing_labels += 1
                continue

            matched += 1
            try:
                if txt.read_text(encoding="utf-8").strip():
                    nonempty_labels += 1
            except Exception:
                pass

        print(f"\n[{split}]")
        print(f"Images: {len(images)}")
        print(f"Labels: {len(txts)}")
        print(f"Matched image-label pairs: {matched}")
        print(f"Missing labels: {missing_labels}")
        print(f"Non-empty labels: {nonempty_labels}")


def write_dataset_yaml() -> None:
    yaml_path = DATA_ROOT.parent / "dataset.yaml"
    content = (
        f"path: {DATA_ROOT.as_posix()}\n"
        f"train: images/Trainset\n"
        f"val: images/Valset\n"
        f"test: images/Testset\n\n"
        f"names:\n"
        f"  0: swimmer\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    print(f"\n[INFO] dataset.yaml written to: {yaml_path}")


def main() -> None:
    print("Starting dataset preparation...")

    ensure_dir(LABELS_DIR)

    overall = Counter()

    for split in SPLITS:
        split_stats = prepare_split(split)
        overall.update(split_stats)

    clear_yolo_cache()
    print("\n[INFO] Old YOLO cache files removed.")

    validate_dataset()
    write_dataset_yaml()

    print("\n===== SUMMARY =====")
    print(f"Total images found: {overall['images_found']}")
    print(f"Total JSON found: {overall['json_found']}")
    print(f"Total TXT created: {overall['txt_total']}")
    print(f"Total TXT with boxes: {overall['txt_with_boxes']}")
    print(f"Total empty TXT: {overall['txt_empty_after_write']}")
    print(f"Total boxes: {overall['boxes_total']}")
    print(f"Total images without JSON: {overall['image_without_json']}")
    print(f"Total JSON without image: {overall['json_without_image']}")
    print(f"Total failed JSON: {overall['failed_json']}")

    print("\nDone.")


if __name__ == "__main__":
    main()