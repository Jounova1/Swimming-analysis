import argparse
import re
import sys
import random
from pathlib import Path

import cv2


# Split ratios
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.15
TEST_RATIO  = 0.05


def get_split(index: int, total: int) -> str:
    """Deterministic split based on position after shuffle."""
    ratio = index / total
    if ratio < TRAIN_RATIO:
        return "train"
    elif ratio < TRAIN_RATIO + VAL_RATIO:
        return "val"
    else:
        return "test"


def extract_frames(video_path: Path, interval: int, dataset_root: Path, class_name: str) -> dict:
    # create all folders
    for split in ["train", "val", "test"]:
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps                = cap.get(cv2.CAP_PROP_FPS)
    estimated          = total_video_frames // interval if total_video_frames > 0 else 0

    print(f"\n{'='*52}")
    print(f"  Video    : {video_path.name}")
    print(f"  Class    : {class_name}")
    print(f"  Interval : every {interval} frames  |  FPS: {fps:.0f}")
    if total_video_frames > 0:
        dur = total_video_frames / fps if fps > 0 else 0
        print(f"  Duration : {dur:.0f}s ({dur/60:.1f} min)  |  {total_video_frames} total frames")
        print(f"  Estimated output → train ~{int(estimated*TRAIN_RATIO)} | val ~{int(estimated*VAL_RATIO)} | test ~{int(estimated*TEST_RATIO)}")
    print(f"{'='*52}")

    # --- pass 1: extract all frames into memory ---
    all_frames = []
    processed  = 0

    try:
        while True:
            grabbed = cap.grab()
            if not grabbed:
                break
            if processed % interval == 0:
                ok, frame = cap.retrieve()
                if ok:
                    all_frames.append(frame)
            processed += 1
            if processed % 300 == 0:
                pct = f"{processed/total_video_frames*100:.0f}%" if total_video_frames > 0 else "?"
                print(f"  Extracting... {processed} frames  ({pct})  |  collected {len(all_frames)}")
    finally:
        cap.release()

    total = len(all_frames)
    if total == 0:
        print("  Warning: no frames extracted!")
        return {"train": 0, "val": 0, "test": 0}

    # --- shuffle with fixed seed so results are reproducible ---
    random.seed(42)
    random.shuffle(all_frames)

    # --- pass 2: save each frame to correct split ---
    counts = {"train": 0, "val": 0, "test": 0}

    for i, frame in enumerate(all_frames):
        split     = get_split(i, total)
        split_dir = dataset_root / "images" / split
        n         = len(list(split_dir.glob(f"{class_name}_*.jpg")))
        filename  = split_dir / f"{class_name}_{n:04d}.jpg"

        ok = cv2.imwrite(str(filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError(f"Failed to write: {filename}")

        counts[split] += 1

    print(f"\n  Saved {total} frames:")
    print(f"    train → {counts['train']}  ({counts['train']/total*100:.0f}%)")
    print(f"    val   → {counts['val']}   ({counts['val']/total*100:.0f}%)")
    print(f"    test  → {counts['test']}  ({counts['test']/total*100:.0f}%)")

    return counts


def print_summary(dataset_root: Path) -> None:
    print(f"\n{'='*52}")
    print("  FULL DATASET SUMMARY")
    print(f"{'='*52}")
    total_all = 0
    for split in ["train", "val", "test"]:
        d = dataset_root / "images" / split
        n = len(list(d.glob("*.jpg"))) if d.exists() else 0
        total_all += n
        print(f"    {split:<6} : {n} images")
    print(f"    {'TOTAL':<6} : {total_all} images")
    print(f"  Saved to : {dataset_root.resolve()}")
    print(f"{'='*52}\n")


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV")


def class_from_filename(stem: str) -> str:
    """
    Extract class name from video filename.
    Takes everything before the first underscore.
    Example:
      normal_video1     → normal
      splitfins_session2 → splitfins
      monofin_above_A3  → monofin
      apnea_B4_underwater → apnea
    """
    return stem.split("_")[0]


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from videos — auto split 80% train / 15% val / 5% test."
    )

    # can pass either a folder OR individual video files
    parser.add_argument(
        "input",
        type=Path,
        nargs="+",
        help="Folder containing videos OR one or more video files directly"
    )
    parser.add_argument("--interval",   type=int,  default=30,             help="Extract every N frames (default: 30)")
    parser.add_argument("--output",     type=Path, default=Path("dataset"), help="Dataset root folder (default: dataset)")
    parser.add_argument("--class-name", type=str,  default=None,           help="Override class name for all videos (optional)")

    args = parser.parse_args()

    if args.interval <= 0:
        raise ValueError("--interval must be a positive integer")

    # --- collect all video files ---
    video_files: list[Path] = []

    for inp in args.input:
        if inp.is_dir():
            # folder mode — grab all video files inside
            found = [f for f in sorted(inp.iterdir()) if f.suffix in VIDEO_EXTENSIONS]
            if not found:
                print(f"  Warning: no videos found in folder: {inp}", file=sys.stderr)
            else:
                print(f"\n  Found {len(found)} videos in folder: {inp}")
                for f in found:
                    print(f"    • {f.name}")
                video_files.extend(found)

        elif inp.is_file():
            if inp.suffix in VIDEO_EXTENSIONS:
                video_files.append(inp)
            else:
                print(f"  Warning: not a video file, skipping: {inp}", file=sys.stderr)
        else:
            print(f"  Warning: not found, skipping: {inp}", file=sys.stderr)

    if not video_files:
        print("\nNo videos to process. Exiting.")
        raise SystemExit(0)

    print(f"\n  Total videos to process: {len(video_files)}")
    print(f"  Output dataset folder  : {args.output}")
    print(f"  Interval               : every {args.interval} frames")

    # --- process each video ---
    total_counts = {"train": 0, "val": 0, "test": 0}

    for video_path in video_files:
        # class name: use override if given, else extract from filename
        if args.class_name:
            class_name = args.class_name
        else:
            class_name = class_from_filename(video_path.stem)

        counts = extract_frames(
            video_path   = video_path,
            interval     = args.interval,
            dataset_root = args.output,
            class_name   = class_name,
        )
        for s in total_counts:
            total_counts[s] += counts[s]

    print_summary(args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise SystemExit(1)