import argparse
import re
import sys
from pathlib import Path

import cv2


def get_next_frame_index(output_dir: Path) -> int:
    pattern = re.compile(r"^frame_(\d+)\.jpg$", re.IGNORECASE)
    max_index = 0

    for image_path in output_dir.glob("frame_*.jpg"):
        match = pattern.match(image_path.name)
        if match:
            index = int(match.group(1))
            if index > max_index:
                max_index = index

    return max_index + 1


def extract_frames(video_path: Path, interval: int, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    next_index = get_next_frame_index(output_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_count = 0
    saved_count = 0

    print(f"Processing video: {video_path}")
    print(f"Saving frames to: {output_dir}")
    print(f"Starting file index: {next_index:04d}")
    print(f"Extraction interval: every {interval} frame(s)")
    if total_frames > 0:
        print(f"Total frames in video: {total_frames}")

    try:
        while True:
            grabbed = cap.grab()
            if not grabbed:
                break

            if processed_count % interval == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    processed_count += 1
                    continue

                filename = output_dir / f"frame_{next_index + saved_count:04d}.jpg"
                written = cv2.imwrite(str(filename), frame)
                if not written:
                    raise RuntimeError(f"Failed to write image: {filename}")
                saved_count += 1

            processed_count += 1

            if processed_count % 200 == 0:
                if total_frames > 0:
                    print(
                        f"Processed {processed_count}/{total_frames} frames | "
                        f"Saved {saved_count}"
                    )
                else:
                    print(f"Processed {processed_count} frames | Saved {saved_count}")
    finally:
        cap.release()

    if processed_count % 200 != 0:
        if total_frames > 0:
            print(f"Processed {processed_count}/{total_frames} frames | Saved {saved_count}")
        else:
            print(f"Processed {processed_count} frames | Saved {saved_count}")

    print("Finished processing video.")
    print(f"Total frames extracted: {saved_count}")
    return saved_count


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from a video at a fixed interval and save as JPEG images."
    )

    parser.add_argument(
        "video",
        type=Path,
        help="Path to input video file"
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Extract every N frames (default: 5)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("frames"),
        help="Folder to save extracted frames (default: frames)"
    )

    args = parser.parse_args()

    if args.interval <= 0:
        raise ValueError("--interval must be a positive integer")

    if not args.video.exists():
        raise FileNotFoundError(f"Video file not found: {args.video}")

    video_output_dir = args.output / args.video.stem
    extract_frames(args.video, args.interval, video_output_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    