"""Create a new version from explicit adjudication or independent reviewer consensus."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .review_validator import ALLOWED_LABELS, read_review_events


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_version(source_csv, queue_csv, events_jsonl, output_dir, *, version="development_dataset_v4"):
    source_csv, queue_csv, events_jsonl, output_dir = map(Path, (source_csv, queue_csv, events_jsonl, output_dir))
    with source_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
        fieldnames = list(source_rows[0]) if source_rows else []
    with queue_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        queue_rows = list(csv.DictReader(stream))
    queue = {row["sample_id"]: row for row in queue_rows}
    events = read_review_events(events_jsonl, queue)
    applicable = [event for event in events if queue[event["sample_id"]]["source_file"] == source_csv.name]
    latest = {}
    for event in applicable:
        key = (event["sample_id"], event["reviewer_id"])
        latest[key] = event
    by_sample = {}
    for (sample_id, _), event in latest.items():
        by_sample.setdefault(sample_id, []).append(event)
    resolved = {}
    unresolved_disagreements = []
    for sample_id, records in by_sample.items():
        adjudications = [event for event in records if event["review_status"] == "adjudicated"]
        if len(adjudications) > 1 and len({event["reviewed_label"] for event in adjudications}) > 1:
            raise ValueError(f"multiple incompatible adjudications for {sample_id}")
        if adjudications:
            resolved[sample_id] = adjudications[-1]
            continue
        independent_reviews = {event["reviewer_id"]: event for event in records
                               if event["review_status"] == "reviewed" and event.get("reviewed_label")}
        independent_labels = {event["reviewed_label"] for event in independent_reviews.values()}
        if len(independent_reviews) >= 2 and len(independent_labels) == 1:
            resolved[sample_id] = {**next(iter(independent_reviews.values())),
                                   "sample_id": sample_id,
                                   "review_status": "reviewed_by_two_agreement",
                                   "reviewer_ids": sorted(independent_reviews)}
            continue
        if len(independent_reviews) >= 2 and len(independent_labels) > 1:
            unresolved_disagreements.append(sample_id)
        final_reviews = [event for event in records if event["review_status"] in {"rejected", "ambiguous", "needs_context"}]
        reviewers_by_status = {}
        for event in final_reviews:
            reviewers_by_status.setdefault(event["review_status"], set()).add(event["reviewer_id"])
        agreed_exclusion = next((status for status, reviewers in reviewers_by_status.items() if len(reviewers) >= 2), None)
        if agreed_exclusion:
            resolved[sample_id] = {"sample_id": sample_id, "review_status": agreed_exclusion, "reviewer_id": "two_reviewer_agreement",
                                   "reviewed_at": None, "annotation_guideline_version": "eldermeds-emotion-guidelines-1.0"}

    if not resolved:
        raise ValueError("no accepted human resolutions exist for this source; no dataset version was written")

    row_lookup = {}
    for item in queue_rows:
        if item["source_file"] == source_csv.name:
            row_lookup[int(item["source_row_1based"])] = item
    output_rows = []
    decisions_applied, accepted_decisions, exclusions = [], [], []
    for data_index, row in enumerate(source_rows, start=2):
        queue_item = row_lookup.get(data_index)
        event = resolved.get(queue_item["sample_id"]) if queue_item else None
        if event:
            if event["review_status"] in {"rejected", "ambiguous", "needs_context"}:
                exclusions.append({"sample_id": event["sample_id"], "reason": f"human_{event['review_status']}"})
                continue
            if event["reviewed_label"] not in ALLOWED_LABELS:
                raise ValueError(f"unsupported adjudicated label for {event['sample_id']}")
            original = row["label"].strip().casefold()
            corrected = event["reviewed_label"]
            reviewers = sorted({record["reviewer_id"] for record in by_sample.get(queue_item["sample_id"], [])})
            accepted_decisions.append({"sample_id": queue_item["sample_id"], "original_label": original,
                                       "accepted_label": corrected, "changed": original != corrected,
                                       "review_status": event["review_status"], "reviewer_ids": reviewers,
                                       "reviewed_at": event.get("reviewed_at"),
                                       "guideline_version": event["annotation_guideline_version"]})
            if original != corrected:
                decisions_applied.append({"sample_id": event["sample_id"], "original_label": original,
                                          "reviewed_label": corrected,
                                          "reviewer_ids": reviewers,
                                          "reviewed_at": event.get("reviewed_at"),
                                          "guideline_version": event["annotation_guideline_version"]})
            row = dict(row)
            row["label"] = corrected
        output_rows.append(row)
    invalid = sorted({row["label"].strip().casefold() for row in output_rows} - ALLOWED_LABELS)
    if invalid:
        raise ValueError(f"source contains labels outside the seven-class contract: {invalid}")
    counts = Counter(row["label"].strip().casefold() for row in output_rows)
    missing_classes = ALLOWED_LABELS - set(counts)
    if missing_classes:
        raise ValueError(f"dataset version would remove supported classes {sorted(missing_classes)}")
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset_path = output_dir / f"{version}.csv"
    with dataset_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    manifest = {
        "dataset_version": version, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": source_csv.name, "source_sha256": sha256_file(source_csv),
        "review_queue_sha256": sha256_file(queue_csv), "review_event_log_sha256": sha256_file(events_jsonl),
        "output_file": dataset_path.name, "output_sha256": sha256_file(dataset_path),
        "annotation_guideline_version": "eldermeds-emotion-guidelines-1.0",
        "source_rows": len(source_rows), "output_rows": len(output_rows),
        "accepted_review_decisions": accepted_decisions,
        "applied_label_corrections": decisions_applied, "rejected_samples": exclusions,
        "adjudicated_samples": sum(event["review_status"] == "adjudicated" for event in resolved.values()),
        "two_reviewer_agreement_samples": sum(event["review_status"] == "reviewed_by_two_agreement" for event in resolved.values()),
        "unresolved_disagreement_sample_ids": unresolved_disagreements,
        "class_counts": dict(sorted(counts.items())),
        "duplicate_handling": "preserve rows; no automatic duplicate collapse; adjudication only for reviewed sample IDs",
        "split_definition": {
            "status": "not_assigned_until_training_experiment",
            "method": "StratifiedGroupKFold: five outer folds for validation; four inner folds over the remaining records for fit/calibration",
            "grouping": "connected components of exact normalized and character TF-IDF cosine >=0.90 text families; include participant/session keys if present",
            "seed": 42,
            "frozen_evaluation_sets_excluded": True,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
