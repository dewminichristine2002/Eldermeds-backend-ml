"""Local resumable terminal tool for reviewing Stage 7A queue items."""
from __future__ import annotations

import argparse
import getpass
import json
import sys
from collections import Counter
from pathlib import Path

from .dataset_builder import build_version
from .context_dataset_builder import build_context_dataset
from .review_validator import (
    ALLOWED_LABELS, CONTEXT_GUIDELINE_VERSION, GUIDELINE_VERSION, REVIEW_STATUSES, append_review_event, load_queue,
    read_review_events, reviewer_pseudonym, summarize_reviews,
)

LABELS = ["happiness", "sadness", "loneliness", "anxiety", "anger", "cognitive_fog", "neutral"]
COMMON_CONFUSION_PAIRS = {
    frozenset(("cognitive_fog", "neutral")), frozenset(("happiness", "cognitive_fog")),
    frozenset(("anger", "sadness")), frozenset(("anxiety", "anger")),
}
DEFAULT_QUEUE = Path(__file__).resolve().parents[1] / "stage7a_experiments" / "20261008T_STAGE7A_FINAL" / "annotation_review_queue.csv"
DEFAULT_EVENTS = Path(__file__).resolve().parents[1] / "stage7b_stage8_experiments" / "review_events.jsonl"


def summarize_queue(queue):
    counts = {"emotion_class": Counter({label: 0 for label in ALLOWED_LABELS}), "reason": Counter(), "source": Counter(),
              "duplicate_status": Counter(), "label_conflict": Counter(), "missing_context_data": Counter(),
              "short_context_flag": Counter(), "common_confusion_family": Counter(),
              "priority": Counter()}
    for item in queue.values():
        counts["emotion_class"][item["existing_label"]] += 1
        counts["source"][item["source_file"]] += 1
        reasons = set(filter(None, item.get("reason", "").split(";")))
        counts["reason"].update(reasons)
        counts["duplicate_status"]["near_duplicate" if "near_duplicate_candidate" in reasons else
                                    "exact_duplicate" if "normalized_exact_duplicate" in reasons else "none"] += 1
        counts["label_conflict"]["yes" if "label_conflict" in reasons else "no"] += 1
        counts["missing_context_data"]["not_available_in_source_queue"] += 1
        counts["short_context_flag"]["flagged_by_short_text_heuristic" if "very_short_context_dependent" in reasons else "not_flagged"] += 1
        family_labels = set(filter(None, item.get("family_labels", "").split("|")))
        confusion_family = any(pair <= family_labels for pair in COMMON_CONFUSION_PAIRS)
        counts["common_confusion_family"]["yes" if confusion_family else "no"] += 1
        priority = "high" if "label_conflict" in reasons or confusion_family else "medium" if reasons & {"near_duplicate_candidate", "very_short_context_dependent"} else "low"
        counts["priority"][priority] += 1
    return {key: dict(sorted(value.items())) for key, value in counts.items()}


def review_one(item, pseudonym, events_path, queue_by_id=None):
    print(f"\nSample: {item['sample_id']} | source={item['source_file']} | row={item['source_row_1based']}")
    print(f"Original label: {item.get('existing_label') or '(none; annotate from the conversation)'} | reason: {item['reason']}")
    if item.get("current_question"):
        try:
            previous_turns = json.loads(item.get("previous_turns_json") or "[]")
        except json.JSONDecodeError:
            print("Invalid previous_turns_json; sample skipped.")
            return "invalid"
        for turn in previous_turns:
            print(f"Previous question [{turn.get('turn_index')}]: {turn.get('question', '')}")
            print(f"Previous answer [{turn.get('turn_index')}]: {turn.get('answer', '')}")
        print(f"Current question [{item.get('turn_index', '')}]: {item['current_question']}")
        print(f"Current answer: {item.get('current_answer', item.get('text', ''))}")
    else:
        print(f"Text: {item.get('text', '')}")
    print("Choose: 1 reviewed, 2 ambiguous, 3 needs_context, 4 rejected, 5 adjudicated, s skip, q quit")
    choice = input("> ").strip().casefold()
    if choice in {"q", "s", ""}:
        return choice
    status_map = {"1": "reviewed", "2": "ambiguous", "3": "needs_context", "4": "rejected", "5": "adjudicated"}
    status = status_map.get(choice)
    if status not in REVIEW_STATUSES:
        print("Invalid status; no decision saved.")
        return "invalid"
    label = None
    if status == "adjudicated":
        prior_events = read_review_events(events_path, queue_by_id or {item["sample_id"]: item})
        print("Prior independent decisions (adjudicator view):")
        for prior in prior_events:
            if prior["sample_id"] == item["sample_id"] and prior["review_status"] == "reviewed":
                print(f"- {prior['reviewer_id']}: {prior['reviewed_label']} | rationale: {prior.get('reviewer_notes') or '(none)'}")
    if status in {"reviewed", "adjudicated"}:
        print("Choose label: " + " | ".join(f"{i+1}={label}" for i, label in enumerate(LABELS)))
        selected = input("Label number: ").strip()
        if not selected.isdigit() or not 1 <= int(selected) <= len(LABELS):
            print("Invalid label; no decision saved.")
            return "invalid"
        label = LABELS[int(selected)-1]
    notes = input("Notes (do not enter names, contact details, or private health details): ").strip()
    if len(notes) > 1000:
        print("Notes exceed 1000 characters; no decision saved.")
        return "invalid"
    event = {
        "sample_id": item["sample_id"], "dataset_source": item["source_file"],
        "original_label": item["existing_label"], "reviewed_label": label,
        "review_status": status, "review_reason": item["reason"],
        "reviewer_id": pseudonym, "reviewer_notes": notes,
        "annotation_guideline_version": CONTEXT_GUIDELINE_VERSION if item.get("current_question") else GUIDELINE_VERSION,
    }
    append_review_event(events_path, event, item)
    print("Decision appended. Source data remains unchanged.")
    return "saved"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("review", "summary", "build-dataset", "build-context-dataset"))
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "stage7b_stage8_experiments" / "datasets")
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--source-jsonl", type=Path, help="Authorized contextual session source for build-context-dataset.")
    parser.add_argument("--independent-evaluation", action="store_true",
                        help="Freeze separately curated records as independent_test; never use for training/tuning")
    args = parser.parse_args(argv)
    queue = load_queue(args.queue)
    if args.command == "summary":
        events = read_review_events(args.events, queue)
        report = {"review": summarize_reviews(queue, events), "queue": summarize_queue(queue),
                  "human_decisions_exist": bool(events)}
        rendered = json.dumps(report, indent=2)
        if args.summary_output:
            args.summary_output.parent.mkdir(parents=True, exist_ok=True)
            args.summary_output.write_text(rendered, encoding="utf-8")
        print(rendered)
        return 0
    if args.command == "review" and any("current_question" in item for item in queue.values()):
        for item in queue.values():
            if item.get("current_question"):
                try:
                    previous_turns = json.loads(item.get("previous_turns_json") or "[]")
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid previous_turns_json for {item['sample_id']}") from exc
                if len(previous_turns) > 2:
                    raise ValueError(f"more than two previous turns for {item['sample_id']}")
    if args.command == "build-dataset":
        try:
            result = build_version(
                args.queue.parent.parent.parent / "data" / "development_dataset_v3.csv",
                args.queue, args.events, args.output_dir,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "build-context-dataset":
        if not args.source_jsonl:
            parser.error("--source-jsonl is required for build-context-dataset")
        result = build_context_dataset(args.source_jsonl, args.queue, args.events, args.output_dir,
                                       independent_evaluation=args.independent_evaluation)
        print(json.dumps(result, indent=2))
        return 0
    reviewer_identity = getpass.getpass("Reviewer identity (not stored; pseudonymized locally): ").strip()
    pseudonym = reviewer_pseudonym(reviewer_identity)
    events = read_review_events(args.events, queue)
    reviewed = {event["sample_id"] for event in events if event["reviewer_id"] == pseudonym}
    print(f"Queue has {len(queue)} examples; {len(reviewed)} have an event from this reviewer.")
    priority = lambda item: (0 if "label_conflict" in item.get("reason", "") else
                             1 if "near_duplicate_candidate" in item.get("reason", "") else
                             2 if "very_short_context_dependent" in item.get("reason", "") else 3)
    for item in sorted(queue.values(), key=lambda row: (priority(row), row["source_file"], row["sample_id"])):
        if item["sample_id"] in reviewed:
            continue
        result = review_one(item, pseudonym, args.events, queue)
        if result == "q":
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
