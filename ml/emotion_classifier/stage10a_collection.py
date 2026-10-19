"""Validate and locally export approved, already de-identified check-in JSONL.

This utility never connects to PostgreSQL. Inputs are immutable. The CSV output
is created exclusively and contains only the explicit collection allowlist.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from context_dataset_schema import CONTEXT_SCHEMA_VERSION, validate_context_dataset, validate_context_record

CSV_FIELDS = [
    "schema_version", "sample_id", "participant_group_id", "session_id", "turn_number",
    "question_id", "question_bank_version", "question", "answer", "previous_turns_json",
    "consent_or_use_authorization_reference", "data_origin", "source_provenance", "dataset_split",
    "emotion_label", "annotation_source", "annotation_status", "annotation_guideline_version",
    "annotation_reviewers_json", "annotation_reason",
]
LEDGER_FIELDS = {"participant_group_id", "session_id", "consent_status", "research_scope",
                 "withdrawn", "authorization_reference"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_records(source_path: Path):
    records = []
    for line_number, line in enumerate(source_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = validate_context_record(json.loads(line), require_annotation=False)
        except Exception as exc:
            raise ValueError(f"invalid de-identified session record at line {line_number}: {exc}") from exc
        if item.get("schema_version") != CONTEXT_SCHEMA_VERSION:
            raise ValueError(f"line {line_number}: Stage 10A requires schema {CONTEXT_SCHEMA_VERSION}")
        if item.get("data_origin") != "consented_participant":
            raise ValueError(f"line {line_number}: production export only accepts consented_participant rows")
        records.append(item)
    if not records:
        raise ValueError("source JSONL has no conversation turns")
    last_turn_by_session = {}
    for item in records:
        for field in ("question_id", "question_bank_version", "source_provenance",
                      "consent_or_use_authorization_reference"):
            if not item.get(field):
                raise ValueError(f"session {item['session_id']} turn {item['turn_index']}: {field} is required")
        key = (item["participant_group_id"], item["session_id"])
        prior_turn = last_turn_by_session.get(key, 0)
        if item["turn_index"] <= prior_turn:
            raise ValueError(f"session {item['session_id']}: source turns are not in chronological order")
        last_turn_by_session[key] = item["turn_index"]
    # Require each session to be contiguous, and each history payload to be the
    # exact prior two (or fewer) turns. This excludes future-turn leakage.
    validate_context_dataset(records, require_previous_turns_present=True)
    groups = {}
    for item in records:
        groups.setdefault((item["participant_group_id"], item["session_id"]), []).append(item)
    for (_participant, session), rows in groups.items():
        rows.sort(key=lambda row: row["turn_index"])
        indexes = [row["turn_index"] for row in rows]
        if indexes != list(range(1, len(rows) + 1)):
            raise ValueError(f"session {session}: turn numbers must be contiguous from 1")
        question_ids = [row["question_id"] for row in rows]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError(f"session {session}: adaptive assessment question IDs must not repeat")
        for position, row in enumerate(rows):
            expected = rows[max(0, position - 2):position]
            actual = row.get("previous_turns", [])
            if len(actual) != len(expected):
                raise ValueError(f"session {session}: turn {row['turn_index']} must contain only its prior two (or fewer) turns")
            for prior, actual_prior in zip(expected, actual):
                if (actual_prior["turn_index"] != prior["turn_index"]
                        or actual_prior["participant_group_id"] != prior["participant_group_id"]
                        or actual_prior["session_id"] != prior["session_id"]
                        or actual_prior["question"] != prior["current_question"]
                        or actual_prior["answer"] != prior["current_answer"]):
                    raise ValueError(f"session {session}: invalid/future/cross-session history at turn {row['turn_index']}")
    return records


def load_consent_ledger(ledger_path: Path):
    with ledger_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not LEDGER_FIELDS.issubset(reader.fieldnames):
            raise ValueError(f"consent ledger must contain fields: {', '.join(sorted(LEDGER_FIELDS))}")
        entries = {}
        for line_number, row in enumerate(reader, 2):
            key = (row["participant_group_id"].strip(), row["session_id"].strip())
            if not all(key) or key in entries:
                raise ValueError(f"consent ledger row {line_number}: pseudonymous participant/session key must be present and unique")
            if row["consent_status"].strip().casefold() != "consented":
                raise ValueError(f"consent ledger row {line_number}: consent_status must be consented")
            if row["research_scope"].strip() != "emotion_model_research":
                raise ValueError(f"consent ledger row {line_number}: consent does not cover emotion_model_research")
            if row["withdrawn"].strip().casefold() not in {"false", "no", "0"}:
                raise ValueError(f"consent ledger row {line_number}: withdrawn or unknown consent is not exportable")
            authorization_reference = row["authorization_reference"].strip()
            if not authorization_reference:
                raise ValueError(f"consent ledger row {line_number}: authorization_reference is required")
            entries[key] = authorization_reference
    if not entries:
        raise ValueError("consent ledger contains no authorized sessions")
    return entries


def validate_authorization(records, ledger):
    record_sessions = {(item["participant_group_id"], item["session_id"]) for item in records}
    if not record_sessions.issubset(ledger):
        raise ValueError("one or more source sessions have no matching consent-ledger entry")
    for item in records:
        key = (item["participant_group_id"], item["session_id"])
        if item.get("consent_or_use_authorization_reference") != ledger[key]:
            raise ValueError(f"session {item['session_id']}: authorization reference does not match consent ledger")


def validate_question_bank(records, bank_path: Path):
    try:
        bank = json.loads(Path(bank_path).read_text(encoding="utf-8-sig"))
        questions = {str(item["question_id"]): item for item in bank["questions"]}
        version = bank["question_bank_version"]
    except Exception as exc:
        raise ValueError(f"invalid frozen question-bank export: {exc}") from exc
    for item in records:
        if item["question_bank_version"] != version:
            raise ValueError(f"session {item['session_id']}: question-bank version does not match the supplied frozen export")
        question = questions.get(str(item["question_id"]))
        if (not question or not question.get("is_active") or not question.get("is_assessment")
                or question.get("question_text") != item["current_question"]):
            raise ValueError(f"session {item['session_id']} turn {item['turn_index']}: question does not match an active assessment item in the frozen bank")


def _csv_row(item):
    return {
        "schema_version": item["schema_version"], "sample_id": item["sample_id"],
        "participant_group_id": item["participant_group_id"], "session_id": item["session_id"],
        "turn_number": item["turn_index"], "question_id": item["question_id"],
        "question_bank_version": item["question_bank_version"], "question": item["current_question"],
        "answer": item["current_answer"],
        "previous_turns_json": json.dumps(item.get("previous_turns", []), ensure_ascii=False),
        "consent_or_use_authorization_reference": item["consent_or_use_authorization_reference"],
        "data_origin": item["data_origin"], "source_provenance": item["source_provenance"],
        "dataset_split": item.get("dataset_split", "unassigned"),
        "emotion_label": item.get("emotion_label", ""),
        "annotation_source": item.get("annotation_source", ""),
        "annotation_status": item.get("annotation_status", "pending"),
        "annotation_guideline_version": item.get("annotation_guideline_version", ""),
        "annotation_reviewers_json": json.dumps(item.get("annotation_reviewers", [])),
        "annotation_reason": item.get("annotation_reason", ""),
    }


def export_csv(source_path: Path, ledger_path: Path, bank_path: Path, output_path: Path):
    source_path, ledger_path, bank_path, output_path = map(Path, (source_path, ledger_path, bank_path, output_path))
    records = load_records(source_path)
    ledger = load_consent_ledger(ledger_path)
    validate_authorization(records, ledger)
    validate_question_bank(records, bank_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_row(item) for item in records)
    if os.name != "nt":
        output_path.chmod(0o600)
    sessions = {(item["participant_group_id"], item["session_id"]) for item in records}
    return {
        "status": "EXPORTED_AUTHORIZED_DEIDENTIFIED_DATA",
        "records": len(records), "sessions": len(sessions), "turns_per_session_max": 5,
        "source_sha256": _sha256(source_path), "consent_ledger_sha256": _sha256(ledger_path),
        "question_bank_sha256": _sha256(bank_path), "question_bank_version": records[0]["question_bank_version"],
        "output": str(output_path), "raw_identifiers_exported": False,
        "database_accessed": False,
    }


def validate_synthetic_fixture(path: Path):
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            try:
                records.append(validate_context_record(json.loads(line), require_annotation=False))
            except Exception as exc:
                raise ValueError(f"invalid synthetic fixture line {line_number}: {exc}") from exc
    if not records or any(item.get("data_origin") != "synthetic_test" for item in records):
        raise ValueError("fixture validation accepts only explicitly synthetic_test rows")
    validate_context_dataset(records, require_previous_turns_present=True)
    if len(records) != 5 or [item["turn_index"] for item in records] != [1, 2, 3, 4, 5]:
        raise ValueError("synthetic fixture must contain exactly one ordered five-turn session")
    return {"status": "SYNTHETIC_FORMAT_FIXTURE_VALID", "records": 5,
            "sessions": 1, "research_data": False, "training_eligible": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export", help="Export approved, already de-identified JSONL to a new CSV")
    export_parser.add_argument("--source", type=Path, required=True)
    export_parser.add_argument("--consent-ledger", type=Path, required=True)
    export_parser.add_argument("--question-bank-export", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--assert-authorization-verified", action="store_true")
    fixture_parser = subparsers.add_parser("validate-synthetic", help="Validate the checked-in synthetic format fixture only")
    fixture_parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "validate-synthetic":
        print(json.dumps(validate_synthetic_fixture(args.source), indent=2))
        return 0
    if not args.assert_authorization_verified:
        parser.error("refusing export without --assert-authorization-verified")
    print(json.dumps(export_csv(args.source, args.consent_ledger, args.question_bank_export, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
