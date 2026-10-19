"""Prepare a local human-review queue from explicitly authorized JSONL sessions."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from context_dataset_schema import validate_context_dataset, validate_context_record

QUEUE_FIELDS = [
    "sample_id", "source_file", "source_row_1based", "existing_label", "reason",
    "participant_group_id", "session_id", "turn_index", "question_id", "current_question",
    "current_answer", "previous_turns_json", "consent_or_use_authorization_reference",
    "question_bank_version", "data_origin", "source_provenance",
]


def load_collection_jsonl(source_path: Path):
    records = []
    for line_number, line in enumerate(source_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            record = validate_context_record(record, require_annotation=False)
        except Exception as exc:
            raise ValueError(f"invalid collection record on line {line_number}: {exc}") from exc
        records.append(record)
    if not records:
        raise ValueError("collection file contains no records")
    validate_context_dataset(records, require_training_eligible=False, require_previous_turns_present=True)
    for record in records:
        if record.get("data_origin") != "consented_participant" or not record.get("consent_or_use_authorization_reference"):
            raise ValueError("review queues require explicit consented-participant origin and authorization references")
        if not record.get("question_id") or not record.get("question_bank_version") or not record.get("source_provenance"):
            raise ValueError("review queues require question-bank identity and source provenance")
    return records


def build_review_queue(records, source_file):
    rows = []
    for row_number, record in enumerate(records, 1):
        rows.append({
            "sample_id": record["sample_id"], "source_file": str(source_file),
            "source_row_1based": row_number, "existing_label": record.get("emotion_label", ""),
            "reason": record.get("annotation_reason") or "contextual_emotion_annotation",
            "participant_group_id": record["participant_group_id"], "session_id": record["session_id"],
            "turn_index": record["turn_index"], "question_id": record.get("question_id", ""),
            "current_question": record["current_question"], "current_answer": record["current_answer"],
            "previous_turns_json": json.dumps(record.get("previous_turns", []), ensure_ascii=False),
            "consent_or_use_authorization_reference": record["consent_or_use_authorization_reference"],
            "question_bank_version": record.get("question_bank_version", ""),
            "data_origin": record["data_origin"],
            "source_provenance": record["source_provenance"],
        })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Access-controlled authorized session JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Access-controlled review queue CSV")
    parser.add_argument("--assert-consent-confirmed", action="store_true",
                        help="Required explicit assertion that approved consent/governance exists")
    args = parser.parse_args(argv)
    if not args.assert_consent_confirmed:
        parser.error("refusing to create a review queue without --assert-consent-confirmed")
    records = load_collection_jsonl(args.source)
    rows = build_review_queue(records, args.source.name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=QUEUE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if os.name != "nt":
        args.output.chmod(0o600)
    print(json.dumps({"queue_rows": len(rows), "output": str(args.output),
                      "raw_conversation_data_written": True,
                      "note": "Store this file in an approved access-controlled location."}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
