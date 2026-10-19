"""Audit local emotion CSVs without modifying datasets or model artifacts."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def normalize(text):
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def audit_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with path.open("rb") as handle:
        content_sha256 = hashlib.sha256(handle.read()).hexdigest()
    columns = list(rows[0]) if rows else []
    text_key = "text"
    label_key = "label" if "label" in columns else "expected_emotion"
    texts = [normalize(row.get(text_key)) for row in rows]
    labels = [str(row.get(label_key) or "").strip() for row in rows]
    counts = Counter(labels)
    duplicate_counts = Counter(texts)
    normalized_label_sets = {}
    for text, label in zip(texts, labels):
        if text:
            normalized_label_sets.setdefault(text, set()).add(label)
    source_counts = Counter(str(row.get("source") or "") for row in rows) if "source" in columns else {}
    return {
        "path": str(path),
        "columns": columns,
        "content_sha256": content_sha256,
        "rows": len(rows),
        "unique_normalized_texts": len(set(texts)),
        "labels": dict(sorted(counts.items())),
        "unsupported_labels": dict(sorted((counts - Counter({
            "happiness": counts.get("happiness", 0), "sadness": counts.get("sadness", 0),
            "loneliness": counts.get("loneliness", 0), "anxiety": counts.get("anxiety", 0),
            "anger": counts.get("anger", 0), "cognitive_fog": counts.get("cognitive_fog", 0),
            "neutral": counts.get("neutral", 0),
        })).items())),
        "source_counts": dict(sorted(source_counts.items())),
        "missing_text": sum(not text for text in texts),
        "missing_label": sum(not label for label in labels),
        "label_column": label_key,
        "duplicate_normalized_text_rows": sum(count - 1 for count in duplicate_counts.values() if count > 1),
        "conflicting_normalized_text_families": sum(len(value) > 1 for value in normalized_label_sets.values()),
        "empty_text_rows": sum(not text for text in texts),
        "very_short_rows_le_2_tokens": sum(len(text.split()) <= 2 for text in texts if text),
        "long_rows_gt_128_tokens": sum(len(text.split()) > 128 for text in texts),
        "negation_rows": sum(bool(re.search(r"\b(no|not|never|n't|nothing|cannot|can't|dont|don't)\b", text)) for text in texts),
        "has_conversation_identifier": any(key in columns for key in ("session_id", "conversation_id")),
        "has_participant_identifier": any(key in columns for key in ("participant_id", "user_id")),
        "has_question_answer_pairs": any(key in columns for key in ("question", "prompt"))
        and any(key in columns for key in ("answer", "response")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results/stage6_dataset_audit.json"))
    args = parser.parse_args()
    csv_paths = sorted(args.data_dir.glob("*.csv"))
    datasets = [audit_csv(path) for path in csv_paths]
    report = {
        "normalization": "lowercase, whitespace collapse, no punctuation removal",
        "datasets": datasets,
        "cross_dataset_normalized_overlap": {},
        "context_training_ready": any(
            item["has_question_answer_pairs"] and item["has_conversation_identifier"]
            and item["has_participant_identifier"] for item in datasets
        ),
        "context_training_blocker": "No available CSV contains paired question/answer turns and both session and participant identifiers."
        if not any(item["has_question_answer_pairs"] and item["has_conversation_identifier"]
                   and item["has_participant_identifier"] for item in datasets) else None,
    }
    by_path = {}
    for path in csv_paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        by_path[str(path)] = {
            normalize(row.get("text")) for row in rows if normalize(row.get("text"))
        }
    paths = sorted(by_path)
    for index, left in enumerate(paths):
        for right in paths[index + 1:]:
            overlap = len(by_path[left] & by_path[right])
            if overlap:
                report["cross_dataset_normalized_overlap"][f"{left} <> {right}"] = overlap
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
