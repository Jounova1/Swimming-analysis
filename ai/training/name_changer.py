from __future__ import annotations

import argparse
from pathlib import Path


SUPPORTED_EXTENSIONS = {
	".jpg",
	".jpeg",
	".png",
	".bmp",
	".tif",
	".tiff",
	".webp",
	".txt",
}


def rename_files(root_dir: Path, dry_run: bool = False) -> None:
	if not root_dir.exists() or not root_dir.is_dir():
		raise ValueError(f"Directory does not exist or is not a folder: {root_dir}")

	video_folders = sorted([p for p in root_dir.iterdir() if p.is_dir()])
	if not video_folders:
		print(f"No subfolders found in: {root_dir}")
		return

	total_seen = 0
	total_renamed = 0
	total_skipped = 0

	print(f"Scanning root folder: {root_dir}")
	print(f"Found {len(video_folders)} subfolder(s).")

	for video_folder in video_folders:
		prefix = video_folder.name
		folder_seen = 0
		folder_renamed = 0
		folder_skipped = 0

		print(f"\nProcessing subfolder: {video_folder}")

		for file_path in sorted(video_folder.rglob("*")):
			if not file_path.is_file():
				continue

			folder_seen += 1
			total_seen += 1

			ext = file_path.suffix.lower()
			if ext not in SUPPORTED_EXTENSIONS:
				folder_skipped += 1
				total_skipped += 1
				print(f"  [skip] unsupported extension: {file_path.name}")
				continue

			if file_path.name.startswith(f"{prefix}_"):
				folder_skipped += 1
				total_skipped += 1
				print(f"  [skip] already prefixed: {file_path.name}")
				continue

			new_name = f"{prefix}_{file_path.name}"
			target_path = file_path.with_name(new_name)

			if target_path.exists():
				folder_skipped += 1
				total_skipped += 1
				print(f"  [skip] target already exists: {target_path.name}")
				continue

			if dry_run:
				print(f"  [dry-run] {file_path.name} -> {target_path.name}")
			else:
				file_path.rename(target_path)
				print(f"  [renamed] {file_path.name} -> {target_path.name}")

			folder_renamed += 1
			total_renamed += 1

		print(
			f"Subfolder summary ({prefix}): seen={folder_seen}, "
			f"renamed={folder_renamed}, skipped={folder_skipped}"
		)

	print("\nDone.")
	print(
		f"Overall summary: seen={total_seen}, "
		f"renamed={total_renamed}, skipped={total_skipped}"
	)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Rename files inside each subfolder by prefixing the parent "
			"subfolder name (recursive)."
		)
	)
	parser.add_argument(
		"root",
		type=Path,
		help="Path to the root directory containing subfolders like Project, project2, ...",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Show what would be renamed without changing files.",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	rename_files(args.root, dry_run=args.dry_run)
