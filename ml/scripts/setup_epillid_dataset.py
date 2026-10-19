from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path


ARCHIVE_SUFFIXES = (".zip", ".tar.gz", ".tgz", ".tar")


def extract_archive(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(archive_path.suffixes).lower()
    if suffixes.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(output_dir)
        return

    if suffixes.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path) as archive:
            archive.extractall(output_dir)
        return

    raise ValueError(f"Unsupported archive type: {archive_path.name}")


def run_manifest_builder(args: argparse.Namespace, dataset_root: Path) -> None:
    command = [
        sys.executable,
        str(Path(__file__).with_name("build_identity_manifest_from_epillid.py")),
        "--dataset-root",
        str(dataset_root),
        "--output",
        str(args.manifest_output),
        "--relative-paths-from",
        str(args.relative_paths_from),
        "--min-images-per-class",
        str(args.min_images_per_class),
        "--val-split",
        str(args.val_split),
    ]

    if args.metadata:
        command.extend(["--metadata", str(args.metadata)])
    if args.image_column:
        command.extend(["--image-column", args.image_column])
    if args.label_column:
        command.extend(["--label-column", args.label_column])
    if args.max_classes > 0:
        command.extend(["--max-classes", str(args.max_classes)])

    subprocess.run(command, check=True)


def setup(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    archive_path = args.archive.resolve() if args.archive else None

    if output_dir.exists() and args.force_extract:
        shutil.rmtree(output_dir)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.force_extract:
        print(f"Using existing extracted dataset: {output_dir}")
    else:
        if not archive_path:
            raise ValueError(
                "No local ePillID dataset was found. Put the extracted dataset at "
                f"{output_dir} or pass a local archive with --archive."
            )
        if not archive_path.exists():
            raise FileNotFoundError(f"Local ePillID archive not found: {archive_path}")

        with tempfile.TemporaryDirectory(prefix="eldermeds-epillid-") as temp_dir:
            temp_extract = Path(temp_dir) / "extract"
            print(f"Extracting {archive_path}...")
            extract_archive(archive_path, temp_extract)

            output_dir.parent.mkdir(parents=True, exist_ok=True)
            if output_dir.exists():
                shutil.rmtree(output_dir)
            entries = list(temp_extract.iterdir())
            if len(entries) == 1 and entries[0].is_dir():
                shutil.move(str(entries[0]), str(output_dir))
            else:
                shutil.move(str(temp_extract), str(output_dir))

    if not args.skip_manifest:
        run_manifest_builder(args, output_dir)

    print(f"ePillID dataset ready at: {output_dir}")


def main() -> None:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Prepare a local ePillID dataset and build an ElderMeds identity manifest.")
    parser.add_argument("--output-dir", type=Path, default=default_root / "data" / "public" / "ePillID_data")
    parser.add_argument("--archive", type=Path, default=None, help="Use a local ePillID archive.")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--skip-manifest", action="store_true")
    parser.add_argument("--manifest-output", type=Path, default=default_root / "data" / "public" / "epillid-identity-manifest.csv")
    parser.add_argument("--relative-paths-from", type=Path, default=default_root)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--image-column", default="")
    parser.add_argument("--label-column", default="")
    parser.add_argument("--min-images-per-class", type=int, default=2)
    parser.add_argument("--max-classes", type=int, default=0)
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()
    setup(args)


if __name__ == "__main__":
    main()
