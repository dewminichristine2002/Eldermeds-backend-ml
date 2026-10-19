from __future__ import annotations

import argparse
import csv
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_yolo_objects(label_path: Path) -> int:
    if not label_path.exists():
        return 0

    count = 0
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 5:
                count += 1
    return count


def find_label_for_image(image_path: Path, dataset_root: Path, labels_dir: Path) -> Path:
    relative = image_path.relative_to(dataset_root)
    parts = list(relative.parts)
    if parts and parts[0] == "images":
        parts[0] = "labels"
    label_relative = Path(*parts).with_suffix(".txt")
    candidate = dataset_root / label_relative
    if candidate.exists():
        return candidate

    return labels_dir / image_path.parent.name / f"{image_path.stem}.txt"


def collect_split_rows(dataset_root: Path, split: str) -> list[dict[str, str | int]]:
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels"
    if not images_dir.exists():
        return []

    rows: list[dict[str, str | int]] = []
    for image_path in sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS):
        label_path = find_label_for_image(image_path, dataset_root, labels_dir)
        rows.append(
            {
                "image_path": str(image_path),
                "count": count_yolo_objects(label_path),
                "split": split,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a tablet-count manifest from a YOLO pill detection dataset.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--max-count", type=int, default=-1)
    args = parser.parse_args()

    rows: list[dict[str, str | int]] = []
    for split in args.splits:
        rows.extend(collect_split_rows(args.dataset_root, split))
    if args.max_count >= 0:
        rows = [row for row in rows if int(row["count"]) <= args.max_count]

    if not rows:
        raise ValueError(f"No image rows found under {args.dataset_root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "count", "split"])
        writer.writeheader()
        writer.writerows(rows)

    max_count = max(int(row["count"]) for row in rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Max visible pill count: {max_count}")


if __name__ == "__main__":
    main()
