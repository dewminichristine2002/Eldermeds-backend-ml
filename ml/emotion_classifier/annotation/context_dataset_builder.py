"""Build a new training CSV only from explicit independent human consensus."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from context_dataset_schema import CONTEXT_SCHEMA_VERSION, EMOTION_LABELS, validate_context_dataset
from .context_review_queue import load_collection_jsonl
from .review_validator import read_review_events

OUTPUT_FIELDS = [
    "schema_version", "sample_id", "participant_group_id", "session_id", "turn_index", "question_id",
    "current_question", "current_answer", "previous_turns", "emotion_label", "annotation_source",
    "annotation_status", "annotation_guideline_version", "annotation_reviewers",
    "consent_or_use_authorization_reference", "dataset_split", "data_origin", "source_provenance",
    "question_bank_version",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_contextual_decisions(queue_rows, events):
    latest = {}
    for event in events:
        latest[(event["sample_id"], event["reviewer_id"])] = event
    per_sample = {}
    for (sample_id, _reviewer), event in latest.items():
        per_sample.setdefault(sample_id, []).append(event)
    resolutions = {}
    exclusions = {}
    disagreements = []
    for sample_id, decisions in per_sample.items():
        adjudications = [event for event in decisions if event["review_status"] == "adjudicated"]
        if adjudications:
            final = adjudications[-1]
            initial_reviews = [event for event in decisions
                               if event["review_status"] == "reviewed"
                               and event["annotation_guideline_version"] == "eldermeds-context-emotion-guidelines-1.0"]
            initial_reviewers = {event["reviewer_id"] for event in initial_reviews}
            adjudicator_is_independent = final["reviewer_id"] not in initial_reviewers
            reviewers = sorted(initial_reviewers | {final["reviewer_id"]})
            if (len(initial_reviewers) >= 2 and adjudicator_is_independent
                    and len(reviewers) >= 3
                    and final["annotation_guideline_version"] == "eldermeds-context-emotion-guidelines-1.0"):
                resolutions[sample_id] = {**final, "annotation_source": "human_adjudication",
                                          "annotation_status": "adjudicated", "annotation_reviewers": reviewers}
            continue
        reviews = [event for event in decisions if event["review_status"] == "reviewed"]
        if len(reviews) >= 2:
            labels = {event["reviewed_label"] for event in reviews}
            guidelines = {event["annotation_guideline_version"] for event in reviews}
            if len(labels) == 1 and len(guidelines) == 1:
                resolutions[sample_id] = {**reviews[-1], "annotation_source": "human_double_review",
                                          "annotation_status": "reviewed",
                                          "annotation_reviewers": sorted({event["reviewer_id"] for event in reviews}),
                                          "annotation_guideline_version": next(iter(guidelines))}
            else:
                disagreements.append(sample_id)
                continue
        terminal = [event for event in decisions if event["review_status"] in {"rejected", "ambiguous", "needs_context"}]
        terminal_by_status = {}
        for event in terminal:
            terminal_by_status.setdefault(event["review_status"], set()).add(event["reviewer_id"])
        agreed_exclusions = [(status, reviewers) for status, reviewers in terminal_by_status.items()
                             if len(reviewers) >= 2]
        if agreed_exclusions:
            exclusions[sample_id] = sorted(agreed_exclusions)[-1][0]
    return resolutions, exclusions, disagreements


def build_context_dataset(source_path, queue_path, events_path, output_dir, *, independent_evaluation=False):
    source_path, queue_path, events_path, output_dir = map(Path, (source_path, queue_path, events_path, output_dir))
    records = load_collection_jsonl(source_path)
    with queue_path.open("r", encoding="utf-8-sig", newline="") as stream:
        queue_rows = list(csv.DictReader(stream))
    queue = {row["sample_id"]: row for row in queue_rows}
    if len(queue) != len(queue_rows) or set(queue) != {item["sample_id"] for item in records}:
        raise ValueError("review queue must contain every source sample exactly once")
    events = read_review_events(events_path, queue)
    resolutions, exclusions, disagreements = resolve_contextual_decisions(queue_rows, events)
    if not resolutions:
        raise ValueError("no samples have two-review agreement or valid adjudication; no dataset emitted")

    queue_by_id = queue
    output = []
    for record in records:
        decision = resolutions.get(record["sample_id"])
        if not decision:
            continue
        if decision["reviewed_label"] not in EMOTION_LABELS:
            raise ValueError(f"human decision has an invalid emotion label: {record['sample_id']}")
        queue_item = queue_by_id[record["sample_id"]]
        guideline = decision["annotation_guideline_version"]
        output.append({
            **record,
            "schema_version": CONTEXT_SCHEMA_VERSION,
            "emotion_label": decision["reviewed_label"],
            "annotation_source": decision["annotation_source"],
            "annotation_status": decision["annotation_status"],
            "annotation_guideline_version": guideline,
            "annotation_reviewers": decision["annotation_reviewers"],
            "dataset_split": "independent_test" if independent_evaluation else "unassigned",
            "data_origin": "consented_participant",
            "source_provenance": record["source_provenance"],
        })
    allowed_splits = {"independent_test"} if independent_evaluation else {"unassigned"}
    audit = validate_context_dataset(output, require_training_eligible=True, allowed_training_splits=allowed_splits,
                                     require_previous_turns_present=True)
    counts = Counter(item["emotion_label"] for item in output)
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset_path = output_dir / "context_emotion_training.csv"
    with dataset_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for item in output:
            writer.writerow({**item,
                             "previous_turns": json.dumps(item.get("previous_turns", []), ensure_ascii=False),
                             "annotation_reviewers": json.dumps(item["annotation_reviewers"]),
                             "source_provenance": item["source_provenance"]})
    manifest = {
        "manifest_version": "eldermeds-context-dataset-manifest-1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": source_path.name,
        "source_sha256": sha256_file(source_path),
        "review_queue_sha256": sha256_file(queue_path),
        "review_event_log_sha256": sha256_file(events_path),
        "output_file": dataset_path.name,
        "output_sha256": sha256_file(dataset_path),
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "annotation_guideline_versions": sorted({item["annotation_guideline_version"] for item in output}),
        "records": len(output),
        "class_counts": dict(sorted(counts.items())),
        "missing_classes": sorted(EMOTION_LABELS - set(counts)),
        "dataset_status": "READY_FOR_SPLIT_REVIEW" if set(counts) == EMOTION_LABELS else "WAITING_FOR_CLASS_COVERAGE",
        "dataset_audit": audit,
        "excluded_samples": exclusions,
        "unresolved_disagreement_sample_ids": disagreements,
        "split_status": "frozen independent_test; participant overlap is checked against the training manifest at evaluation time" if independent_evaluation else "unassigned; create immutable participant-disjoint manifests during experiment",
        "independent_evaluation_status": "independently curated and frozen; do not use for tuning" if independent_evaluation else "not an independent benchmark",
        "raw_text_in_manifest": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--independent-evaluation", action="store_true",
                        help="Freeze this separately curated source as independent_test; never use it for training/tuning")
    args = parser.parse_args(argv)
    print(json.dumps(build_context_dataset(args.source, args.queue, args.events, args.output_dir,
                                           independent_evaluation=args.independent_evaluation), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
