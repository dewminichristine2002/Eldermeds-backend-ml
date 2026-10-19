"""Validation and audit-safe storage for human emotion annotation decisions."""
from __future__ import annotations

import csv
import hashlib
import hmac
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

ALLOWED_LABELS = {
    "happiness", "sadness", "loneliness", "anxiety", "anger", "cognitive_fog", "neutral"
}
REVIEW_STATUSES = {"pending", "reviewed", "ambiguous", "needs_context", "rejected", "adjudicated"}
GUIDELINE_VERSION = "eldermeds-emotion-guidelines-1.0"
CONTEXT_GUIDELINE_VERSION = "eldermeds-context-emotion-guidelines-1.0"


class ReviewValidationError(ValueError):
    """Raised when a review event could corrupt or ambiguously change labels."""


def load_queue(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_id = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id or sample_id in by_id:
            raise ReviewValidationError("queue sample IDs must be present and unique")
        original = str(row.get("existing_label") or "").strip().casefold()
        if original and original not in ALLOWED_LABELS:
            raise ReviewValidationError(f"queue has unsupported original label for {sample_id}")
        row["existing_label"] = original
        if row.get("current_question"):
            try:
                from context_dataset_schema import validate_context_record
                previous_turns = json.loads(row.get("previous_turns_json") or "[]")
                validate_context_record({
                    "schema_version": "eldermeds-context-emotion-1.1",
                    "sample_id": sample_id,
                    "participant_group_id": row.get("participant_group_id"),
                    "session_id": row.get("session_id"),
                    "turn_index": int(row.get("turn_index")),
                    "question_id": row.get("question_id"),
                    "question_bank_version": row.get("question_bank_version"),
                    "current_question": row["current_question"],
                    "current_answer": row.get("current_answer"),
                    "previous_turns": previous_turns,
                    "consent_or_use_authorization_reference": row.get("consent_or_use_authorization_reference"),
                    "data_origin": row.get("data_origin"),
                    "source_provenance": row.get("source_provenance"),
                    "dataset_split": "unassigned",
                }, require_annotation=False)
            except Exception as exc:
                raise ReviewValidationError(f"invalid contextual queue record {sample_id}: {exc}") from exc
        by_id[sample_id] = row
    return by_id


def reviewer_pseudonym(reviewer_identity: str, secret: str | None = None) -> str:
    identity = str(reviewer_identity or "").strip()
    key = secret or os.environ.get("ELDERMEDS_REVIEWER_KEY", "")
    if not identity or len(key) < 16:
        raise ReviewValidationError("reviewer identity and a 16+ character ELDERMEDS_REVIEWER_KEY are required")
    digest = hmac.new(key.encode("utf-8"), identity.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"reviewer_{digest[:16]}"


def validate_review_event(event: dict, queue_item: dict):
    if not isinstance(event, dict):
        raise ReviewValidationError("review event must be an object")
    sample_id = str(event.get("sample_id") or "")
    if sample_id != queue_item.get("sample_id"):
        raise ReviewValidationError("review event sample does not match the source queue item")
    if event.get("original_label", "") != queue_item.get("existing_label", ""):
        raise ReviewValidationError("original label must be copied unchanged from the queue")
    if event.get("dataset_source") != queue_item.get("source_file"):
        raise ReviewValidationError("dataset_source must match the queue provenance")
    if event.get("review_reason") != queue_item.get("reason"):
        raise ReviewValidationError("review_reason must match the queue flag")
    status = event.get("review_status")
    if status not in REVIEW_STATUSES - {"pending"}:
        raise ReviewValidationError("review_status must be a completed review status")
    label = event.get("reviewed_label")
    if label is not None and label not in ALLOWED_LABELS:
        raise ReviewValidationError("reviewed_label is not one of the seven supported labels")
    if status in {"reviewed", "adjudicated"} and label is None:
        raise ReviewValidationError("reviewed and adjudicated decisions require a valid label")
    if status in {"ambiguous", "needs_context", "rejected"} and label is not None:
        raise ReviewValidationError(f"{status} records must not silently assign a replacement label")
    if status == "adjudicated" and not str(event.get("reviewer_notes") or "").strip():
        raise ReviewValidationError("adjudicated records require a rationale")
    if not str(event.get("reviewer_id") or "").startswith("reviewer_"):
        raise ReviewValidationError("reviewer_id must be a privacy-preserving pseudonym")
    expected_guideline = CONTEXT_GUIDELINE_VERSION if queue_item.get("current_question") else GUIDELINE_VERSION
    if event.get("annotation_guideline_version") != expected_guideline:
        raise ReviewValidationError("annotation guideline version is missing or unsupported")
    notes = str(event.get("reviewer_notes") or "")
    if len(notes) > 1000:
        raise ReviewValidationError("reviewer_notes exceed the 1000 character limit")
    if re.search(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", notes) or re.search(r"(?:\+?\d[\s().-]*){8,}", notes):
        raise ReviewValidationError("reviewer notes must not contain email addresses or phone-like identifiers")
    # Text is deliberately not accepted in events or written to the audit log.
    forbidden_text_fields = {"text", "answer", "question", "current_question", "current_answer",
                              "previous_turns", "previous_turns_json"}
    if forbidden_text_fields & set(event):
        raise ReviewValidationError("review events must not duplicate source text")
    return event


def append_review_event(events_path, event: dict, queue_item: dict):
    validate_review_event(event, queue_item)
    path = Path(events_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if event["review_status"] == "adjudicated":
        prior = []
        if path.exists():
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ReviewValidationError(f"invalid event at line {line_no}: {exc}") from exc
                if record.get("sample_id") == queue_item["sample_id"]:
                    validate_review_event(record, queue_item)
                    prior.append(record)
        independent = {record["reviewer_id"]: record for record in prior
                       if record["sample_id"] == queue_item["sample_id"]
                       and record["review_status"] == "reviewed"}
        if len(independent) < 2:
            raise ReviewValidationError("adjudication requires two prior independent reviewer decisions")
        if event["reviewer_id"] in independent:
            raise ReviewValidationError("adjudication must be completed by a third independent reviewer")
    elif event["review_status"] in {"rejected", "ambiguous", "needs_context"}:
        prior = []
        if path.exists():
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ReviewValidationError(f"invalid event at line {line_no}: {exc}") from exc
                if record.get("sample_id") == queue_item["sample_id"]:
                    validate_review_event(record, queue_item)
                    prior.append(record)
        prior_reviewers = {record["reviewer_id"] for record in prior
                           if record["review_status"] == event["review_status"]}
        if event["reviewer_id"] in prior_reviewers:
            raise ReviewValidationError("terminal decisions require a distinct independent reviewer")
    payload = dict(event)
    payload.setdefault("reviewed_at", datetime.now(timezone.utc).isoformat())
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return payload


def read_review_events(events_path, queue_by_id):
    path = Path(events_path)
    if not path.exists():
        return []
    events = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            event = json.loads(line)
            item = queue_by_id[event["sample_id"]]
            validate_review_event(event, item)
        except Exception as exc:
            raise ReviewValidationError(f"invalid event at line {line_no}: {exc}") from exc
        events.append(event)
    return events


def summarize_reviews(queue_by_id, events):
    latest_by_reviewer = {}
    for event in events:
        key = (event["sample_id"], event["reviewer_id"])
        latest_by_reviewer[key] = event
    by_sample = {}
    for (sample_id, _), event in latest_by_reviewer.items():
        by_sample.setdefault(sample_id, []).append(event)
    statuses = {status: 0 for status in REVIEW_STATUSES}
    statuses["pending"] = len(queue_by_id)
    for sample_id, records in by_sample.items():
        status = records[-1]["review_status"]
        statuses[status] += 1
        statuses["pending"] -= 1
    disagreements = []
    overlapping_decisions = []
    for sample_id, records in by_sample.items():
        independent = [record for record in records if record.get("review_status") == "reviewed" and record.get("reviewed_label")]
        if len({record["reviewer_id"] for record in independent}) >= 2:
            overlapping_decisions.append(len({record["reviewed_label"] for record in independent}) == 1)
        decisions = {record.get("reviewed_label") for record in records if record.get("review_status") in {"reviewed", "adjudicated"}}
        if len(decisions) > 1 and not any(record.get("review_status") == "adjudicated" for record in records):
            disagreements.append(sample_id)
    return {"queue_items": len(queue_by_id), "statuses": statuses, "disagreement_sample_ids": disagreements,
            "independent_reviewer_count": len({event["reviewer_id"] for event in events}),
            "inter_annotator_agreement": ({"measure": "observed pairwise agreement", "overlap_samples": len(overlapping_decisions),
                                           "agreement_rate": sum(overlapping_decisions) / len(overlapping_decisions)}
                                          if overlapping_decisions else None)}
