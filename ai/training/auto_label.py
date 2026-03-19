import argparse
from pathlib import Path

from ultralytics import YOLO


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-label swimmers by detecting only person class with YOLOv8."
    )
    parser.add_argument(
        "--image-folder",
        type=Path,
        default=Path("frames/frames2/project4"),
        help="Folder containing images to label.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="Path to YOLOv8 model weights.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size for inference (higher can be faster if memory allows).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Include images in subfolders.",
    )
    return parser.parse_args()


def gather_images(folder: Path, recursive: bool) -> list[Path]:
    images: list[Path] = []
    iterator = folder.rglob if recursive else folder.glob
    for pattern in IMAGE_EXTENSIONS:
        images.extend(iterator(pattern))
    return sorted(set(images))


def get_person_class_id(model: YOLO) -> int:
    names = model.names
    for class_id, class_name in names.items():
        if str(class_name).lower() == "person":
            return int(class_id)
    raise RuntimeError("The loaded model does not have a 'person' class.")


def main() -> None:
    args = parse_args()
    image_folder = args.image_folder

    if not image_folder.exists() or not image_folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    model = YOLO(args.model)
    person_class_id = get_person_class_id(model)

    all_images = gather_images(image_folder, recursive=args.recursive)
    if not all_images:
        print(f"No images found in: {image_folder}")
        return

    to_process: list[Path] = []
    skipped_existing = 0
    for img_path in all_images:
        label_path = img_path.with_suffix(".txt")
        if label_path.exists():
            skipped_existing += 1
            continue
        to_process.append(img_path)

    print(
        f"Found {len(all_images)} images | "
        f"skip existing labels: {skipped_existing} | "
        f"to process: {len(to_process)}"
    )

    if not to_process:
        print("Nothing to process. All images already have label files.")
        return

    written_labels = 0
    processed = 0

    results = model.predict(
        source=[str(p) for p in to_process],
        conf=args.conf,
        stream=True,
        verbose=False,
        batch=args.batch,
    )

    for result, img_path in zip(results, to_process):
        label_path = img_path.with_suffix(".txt")

        with open(label_path, "w", encoding="utf-8") as f:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes_xywhn = result.boxes.xywhn
                classes = result.boxes.cls

                for box, cls in zip(boxes_xywhn, classes):
                    if int(cls.item()) != person_class_id:
                        continue

                    x_center, y_center, width, height = box.tolist()
                    # Map every detected person to class 0 (swimmer).
                    f.write(
                        f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                    )

        written_labels += 1
        processed += 1

        if processed % 50 == 0 or processed == len(to_process):
            print(f"Processed {processed}/{len(to_process)} images")

    print(
        f"Done. Wrote {written_labels} label files in image folders "
        f"(confidence={args.conf})."
    )


if __name__ == "__main__":
    main()
