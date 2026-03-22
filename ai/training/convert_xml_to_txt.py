from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def parse_object_to_yolo_line(obj: ET.Element, img_w: int, img_h: int, class_id: int) -> str | None:
    bndbox = obj.find("bndbox")
    if bndbox is None:
        return None

    try:
        xmin = float(bndbox.findtext("xmin", default=""))
        ymin = float(bndbox.findtext("ymin", default=""))
        xmax = float(bndbox.findtext("xmax", default=""))
        ymax = float(bndbox.findtext("ymax", default=""))
    except ValueError:
        return None

    x_min, x_max = sorted((xmin, xmax))
    y_min, y_max = sorted((ymin, ymax))

    box_w = x_max - x_min
    box_h = y_max - y_min
    if box_w <= 0 or box_h <= 0:
        return None

    x_center = ((x_min + x_max) / 2.0) / img_w
    y_center = ((y_min + y_max) / 2.0) / img_h
    norm_w = box_w / img_w
    norm_h = box_h / img_h

    x_center = clamp(x_center)
    y_center = clamp(y_center)
    norm_w = clamp(norm_w)
    norm_h = clamp(norm_h)

    if norm_w <= 0 or norm_h <= 0:
        return None

    return f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"


def convert_xml_file(xml_path: Path, out_txt_path: Path, target_class: str = "swimmer", class_id: int = 0) -> tuple[bool, int, str | None]:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        return False, 0, f"XML parse error: {exc}"
    except OSError as exc:
        return False, 0, f"File error: {exc}"

    size_node = root.find("size")
    if size_node is None:
        return False, 0, "Missing <size> tag"

    try:
        img_w = int(size_node.findtext("width", default=""))
        img_h = int(size_node.findtext("height", default=""))
    except ValueError:
        return False, 0, "Invalid image width/height"

    if img_w <= 0 or img_h <= 0:
        return False, 0, "Image width/height must be positive"

    lines: list[str] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name", default="") or "").strip().lower()
        if name != target_class.lower():
            continue

        line = parse_object_to_yolo_line(obj, img_w, img_h, class_id)
        if line is not None:
            lines.append(line)

    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    out_txt_path.write_text("\n".join(lines), encoding="utf-8")

    return True, len(lines), None


def convert_path(input_path: Path, output_dir: Path | None, target_class: str, class_id: int, recursive: bool) -> None:
    if input_path.is_file():
        xml_files = [input_path]
    else:
        pattern = "**/*.xml" if recursive else "*.xml"
        xml_files = sorted(input_path.glob(pattern))

    if not xml_files:
        print(f"[WARN] No XML files found in: {input_path}")
        return

    ok_count = 0
    error_count = 0
    total_boxes = 0

    for xml_file in xml_files:
        if output_dir is None:
            out_txt = xml_file.with_suffix(".txt")
        else:
            if input_path.is_file():
                rel = Path(xml_file.name)
            else:
                rel = xml_file.relative_to(input_path)
            out_txt = (output_dir / rel).with_suffix(".txt")

        ok, box_count, err = convert_xml_file(
            xml_path=xml_file,
            out_txt_path=out_txt,
            target_class=target_class,
            class_id=class_id,
        )

        if ok:
            ok_count += 1
            total_boxes += box_count
            print(f"[OK] {xml_file} -> {out_txt} | boxes={box_count}")
        else:
            error_count += 1
            print(f"[ERROR] {xml_file}: {err}")

    print(
        f"\n[INFO] Done. converted={ok_count}, errors={error_count}, total_boxes={total_boxes}, files_scanned={len(xml_files)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Pascal VOC XML annotations to YOLO TXT labels.")
    parser.add_argument("input", type=Path, help="XML file or directory containing XML files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. If omitted, TXT files are written next to XML files.",
    )
    parser.add_argument("--class-name", type=str, default="swimmer", help="Object class name to export")
    parser.add_argument("--class-id", type=int, default=0, help="YOLO class id for class-name")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for XML files when input is a directory",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input path does not exist: {args.input}")

    convert_path(
        input_path=args.input,
        output_dir=args.output_dir,
        target_class=args.class_name,
        class_id=args.class_id,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()
