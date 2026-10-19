"""Create a reproducible, mapped turn-level DailyDialog research dataset."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .dailydialog import (ARCHIVE_URL, CITATION, DATASET_ORIGIN, LABEL_MAP, LICENSE,
                          SOURCE_URL, load_daily_dialog_archive, prepare_dialogues,
                          sha256_file)

ROOT = Path(__file__).resolve().parent


def prepare(archive_path: Path, output_dir: Path):
    archive_path, output_dir = Path(archive_path), Path(output_dir)
    dialogues = load_daily_dialog_archive(archive_path)
    records, counts = prepare_dialogues(dialogues)
    output_dir.mkdir(parents=True, exist_ok=False)
    processed_path = output_dir / "dailydialog_mapped_turns.jsonl"
    with processed_path.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "experiment_dataset_id": "dailydialog_public_emotion_context_v1",
        "dataset_origin": DATASET_ORIGIN,
        "source_url": SOURCE_URL,
        "source_archive_url": ARCHIVE_URL,
        "source_license": LICENSE,
        "source_citation": CITATION,
        "archive_sha256": sha256_file(archive_path),
        "prepared_file": processed_path.name,
        "prepared_sha256": sha256_file(processed_path),
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "label_mapping": LABEL_MAP,
        "native_label_preserved": True,
        "dialogue_splits_preserved": True,
        "population_status": "public_nonclinical_not_eldermeds_data",
        "metrics_status": "public_nonclinical_only_not_clinical_or_elderly_validation",
        "counts": counts,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data" / "raw" / "dailydialog_data.zip")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed" / "dailydialog_public_v1")
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.archive, args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
