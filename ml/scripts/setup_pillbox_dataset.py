from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET


XLSX_NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def column_to_index(cell_ref: str) -> int:
    letters = "".join(char for char in cell_ref if char.isalpha())
    value = 0
    for char in letters:
        value = value * 26 + ord(char.upper()) - 64
    return value - 1


def cell_text(cell: ET.Element) -> str:
    if cell.attrib.get("t") == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(".//main:t", XLSX_NS)).strip()

    value = cell.find("main:v", XLSX_NS)
    return "" if value is None else str(value.text or "").strip()


def read_first_sheet_rows(xlsx_path: Path) -> list[dict[str, str]]:
    with ZipFile(xlsx_path) as archive:
        worksheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        raw_rows: list[list[str]] = []
        for row in worksheet.findall(".//main:sheetData/main:row", XLSX_NS):
            values: list[str] = []
            for cell in row.findall("main:c", XLSX_NS):
                index = column_to_index(cell.attrib.get("r", "A1"))
                while len(values) <= index:
                    values.append("")
                values[index] = cell_text(cell)
            raw_rows.append(values)

    if not raw_rows:
        return []

    headers = [value.strip() for value in raw_rows[0]]
    rows: list[dict[str, str]] = []
    for raw_row in raw_rows[1:]:
        row = {
            header: raw_row[index].strip() if index < len(raw_row) else ""
            for index, header in enumerate(headers)
            if header
        }
        rows.append(row)
    return rows


def normalize_space(value: str) -> str:
    return " ".join(str(value or "").replace("\n", " ").replace("\r", " ").split())


def clean_filename(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return cleaned.strip("._") or "pill"


def truthy(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def image_candidates(row: dict[str, str]) -> list[str]:
    candidates: list[str] = []
    for key in ("splimage", "ndc9", "product_code"):
        value = normalize_space(row.get(key, ""))
        if not value:
            continue
        candidates.extend([f"{value}.jpg", f"{value.replace('-', '')}.jpg"])
    return candidates


def display_medicine_name(row: dict[str, str]) -> str:
    rxstring = normalize_space(row.get("rxstring", ""))
    if rxstring:
        return rxstring

    name = normalize_space(row.get("medicine_name", ""))
    strength = normalize_space(row.get("spl_strength", "")).strip(";")
    if name and strength:
        return f"{name} {strength}"
    return name or normalize_space(row.get("product_code", "")) or "Unknown medicine"


def split_for_label(label: str, val_split: float) -> str:
    rng = random.Random(label)
    return "val" if rng.random() < val_split else "train"


def build_manifest(args: argparse.Namespace) -> None:
    rows = read_first_sheet_rows(args.metadata)
    if not rows:
        raise ValueError(f"No rows found in metadata file: {args.metadata}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.image_output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(args.image_archive) as image_archive:
        archive_names = {name.lower(): name for name in image_archive.namelist()}
        manifest_rows: list[dict[str, str]] = []
        seen_images: set[str] = set()

        for row in rows:
            if args.only_has_image and not truthy(row.get("has_image", "")):
                continue

            source_name = ""
            for candidate in image_candidates(row):
                source_name = archive_names.get(candidate.lower(), "")
                if source_name:
                    break
            if not source_name:
                continue

            source_stem = Path(source_name).stem
            output_name = f"{clean_filename(source_stem)}.jpg"
            output_path = args.image_output_dir / output_name
            if source_name not in seen_images or not output_path.exists():
                with image_archive.open(source_name) as source, output_path.open("wb") as target:
                    target.write(source.read())
                seen_images.add(source_name)

            medicine_name = display_medicine_name(row)
            product_code = normalize_space(row.get("product_code", ""))
            ndc9 = normalize_space(row.get("ndc9", ""))
            manifest_rows.append(
                {
                    "image_path": str(output_path.resolve().relative_to(args.relative_paths_from.resolve())),
                    "medicine_name": medicine_name,
                    "generic_name": normalize_space(row.get("medicine_name", "")),
                    "rxstring": normalize_space(row.get("rxstring", "")),
                    "rxcui": normalize_space(row.get("rxcui", "")),
                    "product_code": product_code,
                    "ndc9": ndc9,
                    "size": normalize_space(row.get("splsize", "") or row.get("pillbox_size", "")),
                    "shape": normalize_space(row.get("splshape_text", "") or row.get("pillbox_shape_text", "")),
                    "score": normalize_space(row.get("splscore", "") or row.get("pillbox_score", "")),
                    "imprint": normalize_space(row.get("splimprint", "") or row.get("pillbox_imprint", "")),
                    "color": normalize_space(row.get("splcolor_text", "") or row.get("pillbox_color_text", "")),
                    "strength": normalize_space(row.get("spl_strength", "")).strip(";"),
                    "ingredients": normalize_space(row.get("spl_ingredients", "")).strip(";"),
                    "dosage_form": normalize_space(row.get("dosage_form", "")),
                    "author": normalize_space(row.get("author", "")),
                    "image_source": normalize_space(row.get("image_source", "")),
                    "source_image": source_name,
                    "identity_label": product_code or ndc9 or medicine_name,
                    "split": split_for_label(product_code or ndc9 or medicine_name, args.val_split),
                }
            )

            if args.max_rows > 0 and len(manifest_rows) >= args.max_rows:
                break

    if not manifest_rows:
        raise ValueError("No Pillbox metadata rows matched images in the archive.")

    fieldnames = [
        "image_path",
        "medicine_name",
        "generic_name",
        "rxstring",
        "rxcui",
        "product_code",
        "ndc9",
        "size",
        "shape",
        "score",
        "imprint",
        "color",
        "strength",
        "ingredients",
        "dosage_form",
        "author",
        "image_source",
        "source_image",
        "identity_label",
        "split",
    ]
    with args.manifest_output.open("w", newline="", encoding="utf-8") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(
        f"Saved Pillbox manifest with {len(manifest_rows)} rows "
        f"and {len({row['identity_label'] for row in manifest_rows})} identities to {args.manifest_output}"
    )
    print(f"Extracted matched images to {args.image_output_dir}")


def main() -> None:
    ml_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build a rich Pillbox image+metadata manifest for tablet identity.")
    parser.add_argument("--metadata", required=True, type=Path, help="Pillbox archived XLSX metadata file.")
    parser.add_argument("--image-archive", required=True, type=Path, help="Pillbox production images ZIP file.")
    parser.add_argument("--output-dir", type=Path, default=ml_root / "data" / "public" / "pillbox")
    parser.add_argument("--image-output-dir", type=Path, default=ml_root / "data" / "public" / "pillbox" / "images")
    parser.add_argument("--manifest-output", type=Path, default=ml_root / "data" / "public" / "pillbox-identity-manifest.csv")
    parser.add_argument("--relative-paths-from", type=Path, default=ml_root)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--only-has-image", action="store_true", default=True)
    args = parser.parse_args()
    build_manifest(args)


if __name__ == "__main__":
    main()
