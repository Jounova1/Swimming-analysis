from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

# =========================================
# CONFIG
# =========================================
DATA_ROOT = Path(r"D:\Swimming-analysis\ai\training\dataset")

ANNOTATIONS_DIR = DATA_ROOT / "labels"
LABELS_DIR = DATA_ROOT / "labels"

SPLITS = ["train", "val"]

CLASS_MAP = {
    "swimmer": 0,
    "Swimmer": 0,
}

# =========================================
# HELPERS
# =========================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def voc_to_yolo(size_w: float, size_h: float, xmin: float, ymin: float, xmax: float, ymax: float):
    # safety ترتيب
    x_min, x_max = sorted([xmin, xmax])
    y_min, y_max = sorted([ymin, ymax])

    bw = x_max - x_min
    bh = y_max - y_min
    if bw <= 0 or bh <= 0 or size_w <= 0 or size_h <= 0:
        return None

    x_center = (x_min + x_max) / 2.0 / size_w
    y_center = (y_min + y_max) / 2.0 / size_h
    w = bw / size_w
    h = bh / size_h

    return (
        clamp(x_center),
        clamp(y_center),
        clamp(w),
        clamp(h),
    )


def convert_xml_file(xml_path: Path, txt_path: Path) -> tuple[int, int]:
    """
    Converts one Pascal VOC XML file to one YOLO TXT file.
    Returns:
        objects_found, objects_written
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        raise ValueError("Missing <size> in XML")

    width = float(size.findtext("width", default="0"))
    height = float(size.findtext("height", default="0"))
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image size in XML")

    lines = []
    objects_found = 0
    objects_written = 0

    for obj in root.findall("object"):
        objects_found += 1

        name = obj.findtext("name", default="").strip()
        if name not in CLASS_MAP:
            continue

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = float(bndbox.findtext("xmin", default="0"))
        ymin = float(bndbox.findtext("ymin", default="0"))
        xmax = float(bndbox.findtext("xmax", default="0"))
        ymax = float(bndbox.findtext("ymax", default="0"))

        converted = voc_to_yolo(width, height, xmin, ymin, xmax, ymax)
        if converted is None:
            continue

        xc, yc, w, h = converted
        class_id = CLASS_MAP[name]
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        objects_written += 1

    ensure_dir(txt_path.parent)
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    return objects_found, objects_written


def remove_cache_files() -> None:
    for p in LABELS_DIR.rglob("*.cache"):
        try:
            p.unlink()
            print(f"[INFO] Removed cache: {p}")
        except Exception as e:
            print(f"[WARN] Could not remove cache {p}: {e}")


# =========================================
# MAIN
# =========================================
def main() -> None:
    total_xml = 0
    total_written_txt = 0
    total_found_objects = 0
    total_written_objects = 0
    total_failed = 0

    print("Starting XML -> YOLO TXT conversion...\n")

    for split in SPLITS:
        ann_dir = ANNOTATIONS_DIR / split
        lbl_dir = LABELS_DIR / split
        ensure_dir(lbl_dir)

        if not ann_dir.exists():
            print(f"[WARN] Missing folder: {ann_dir}")
            continue

        xml_files = sorted(ann_dir.glob("*.xml"))
        print(f"{split}: found {len(xml_files)} XML files")

        for xml_file in xml_files:
            total_xml += 1
            txt_file = lbl_dir / f"{xml_file.stem}.txt"

            try:
                found_objs, written_objs = convert_xml_file(xml_file, txt_file)
                total_found_objects += found_objs
                total_written_objects += written_objs
                total_written_txt += 1
            except Exception as e:
                total_failed += 1
                print(f"[ERROR] {xml_file.name}: {e}")

    remove_cache_files()

    print("\nDone.")
    print(f"XML files found: {total_xml}")
    print(f"TXT files created: {total_written_txt}")
    print(f"Objects found in XML: {total_found_objects}")
    print(f"Objects written to TXT: {total_written_objects}")
    print(f"Failed files: {total_failed}")


if __name__ == "__main__":
    main()