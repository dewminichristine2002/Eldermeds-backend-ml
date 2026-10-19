from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IMAGE_COLUMN_CANDIDATES = [
    "image_path",
    "path",
    "filepath",
    "file_path",
    "filename",
    "file",
    "image",
    "image_name",
    "imageid",
    "image_id",
]

LABEL_COLUMN_CANDIDATES = [
    "medicine_name",
    "drug_name",
    "name",
    "label",
    "class",
    "class_name",
    "appearance",
    "appearance_id",
    "pill_id",
    "spl_id",
    "ndc",
]


def normalize_column_name(value: str) -> str:
    return "".join(char.lower() for char in str(value or "") if char.isalnum() or char == "_")


def resolve_column(frame: pd.DataFrame, requested: str, candidates: list[str]) -> str | None:
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"Requested column '{requested}' was not found. Available columns: {', '.join(frame.columns)}")
        return requested

    normalized = {normalize_column_name(column): column for column in frame.columns}
    for candidate in candidates:
        resolved = normalized.get(normalize_column_name(candidate))
        if resolved:
            return resolved

    return None


def find_metadata_files(dataset_root: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".tsv"}
    )


def read_table(path: Path) -> pd.DataFrame:
    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=separator)


def resolve_image_path(value: str, dataset_root: Path, metadata_path: Path) -> Path | None:
    raw = str(value or "").strip().strip('"').strip("'")
    if not raw:
        return None

    candidate = Path(raw)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    search_paths = [
        metadata_path.parent / raw,
        dataset_root / raw,
        dataset_root / "classification_data" / raw,
        dataset_root / "images" / raw,
        dataset_root / "imgs" / raw,
    ]
    for path in search_paths:
        if path.exists():
            return path.resolve()

    filename_matches = list(dataset_root.rglob(Path(raw).name))
    for path in filename_matches:
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            return path.resolve()

    return None


def rows_from_metadata(
    dataset_root: Path,
    metadata_path: Path,
    image_column: str,
    label_column: str,
) -> list[dict[str, str]]:
    frame = read_table(metadata_path)
    resolved_image_column = resolve_column(frame, image_column, IMAGE_COLUMN_CANDIDATES)
    resolved_label_column = resolve_column(frame, label_column, LABEL_COLUMN_CANDIDATES)

    if not resolved_image_column or not resolved_label_column:
        return []

    rows = []
    for _, row in frame.iterrows():
        image_path = resolve_image_path(str(row[resolved_image_column]), dataset_root, metadata_path)
        label = str(row[resolved_label_column] or "").strip()
        if not image_path or not label:
            continue
        rows.append(
            {
                "image_path": str(image_path),
                "medicine_name": label,
            }
        )
    return rows


def rows_from_directory_names(dataset_root: Path) -> list[dict[str, str]]:
    rows = []
    for image_path in sorted(dataset_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = image_path.parent.name.strip()
        if not label or label.lower() in {"images", "imgs", "train", "val", "valid", "validation", "test"}:
            continue
        rows.append(
            {
                "image_path": str(image_path.resolve()),
                "medicine_name": label,
            }
        )
    return rows


def filter_rows(rows: Iterable[dict[str, str]], min_images_per_class: int, max_classes: int) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        raise ValueError("No image rows were found. Check --dataset-root and metadata columns.")

    frame["medicine_name"] = frame["medicine_name"].astype(str).str.strip()
    frame = frame[frame["medicine_name"] != ""].copy()
    counts = frame["medicine_name"].value_counts()
    keep_labels = counts[counts >= min_images_per_class].index.tolist()
    frame = frame[frame["medicine_name"].isin(keep_labels)].copy()

    if max_classes > 0:
        top_labels = frame["medicine_name"].value_counts().head(max_classes).index.tolist()
        frame = frame[frame["medicine_name"].isin(top_labels)].copy()

    if frame["medicine_name"].nunique() < 2:
        raise ValueError("At least two classes are required after filtering.")

    return frame.sample(frac=1, random_state=42).reset_index(drop=True)


def add_split(frame: pd.DataFrame, val_split: float) -> pd.DataFrame:
    class_counts = frame["medicine_name"].value_counts()
    requested_val_rows = int(round(len(frame) * val_split))

    if class_counts.min() >= 2 and requested_val_rows >= frame["medicine_name"].nunique():
        train_frame, val_frame = train_test_split(
            frame,
            test_size=val_split,
            random_state=42,
            stratify=frame["medicine_name"],
        )
        train_frame = train_frame.copy()
        val_frame = val_frame.copy()
    elif class_counts.min() >= 2:
        train_parts = []
        val_parts = []
        for _, group in frame.groupby("medicine_name"):
            shuffled = group.sample(frac=1, random_state=42)
            val_count = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * val_split))))
            val_parts.append(shuffled.iloc[:val_count])
            train_parts.append(shuffled.iloc[val_count:])
        train_frame = pd.concat(train_parts, ignore_index=True)
        val_frame = pd.concat(val_parts, ignore_index=True)
    else:
        train_frame, val_frame = train_test_split(frame, test_size=val_split, random_state=42)
        train_frame = train_frame.copy()
        val_frame = val_frame.copy()

    train_frame["split"] = "train"
    val_frame["split"] = "val"
    return pd.concat([train_frame, val_frame], ignore_index=True).sample(frac=1, random_state=42)


def write_manifest(frame: pd.DataFrame, output: Path, relative_to: Path | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    out_frame = frame.copy()
    if relative_to:
        base = relative_to.resolve()
        out_frame["image_path"] = out_frame["image_path"].map(
            lambda value: str(Path(value).resolve().relative_to(base))
            if Path(value).resolve().is_relative_to(base)
            else str(Path(value).resolve())
        )
    out_frame[["image_path", "medicine_name", "split"]].to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)


def build_manifest(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise ValueError(f"Dataset root does not exist: {dataset_root}")

    metadata_files = [args.metadata.resolve()] if args.metadata else find_metadata_files(dataset_root)
    rows: list[dict[str, str]] = []
    for metadata_path in metadata_files:
        if not metadata_path.exists():
            continue
        rows.extend(rows_from_metadata(dataset_root, metadata_path, args.image_column, args.label_column))

    if not rows:
        rows = rows_from_directory_names(dataset_root)

    frame = filter_rows(rows, args.min_images_per_class, args.max_classes)
    frame = add_split(frame, args.val_split)
    write_manifest(frame, args.output, args.relative_paths_from)

    print(
        f"Wrote {len(frame)} rows across {frame['medicine_name'].nunique()} classes to {args.output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a tablet identity manifest from extracted ePillID data.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--image-column", default="")
    parser.add_argument("--label-column", default="")
    parser.add_argument("--min-images-per-class", type=int, default=2)
    parser.add_argument("--max-classes", type=int, default=0)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--relative-paths-from", type=Path, default=None)
    args = parser.parse_args()
    build_manifest(args)


if __name__ == "__main__":
    main()
