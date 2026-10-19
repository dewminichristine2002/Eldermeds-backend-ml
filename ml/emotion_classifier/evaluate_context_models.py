"""Evaluate v4 and explicitly supplied candidates on a frozen independent conversation set.

This script refuses development/unreviewed data and checks participant-group hashes
against the candidate training split manifest before calculating metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score)

from context_dataset_schema import EMOTION_LABELS
from model_runtime import V4_VERSION, load_runtime
from train_contextual import CLASSES, expected_calibration_error, load_records

ROOT = Path(__file__).resolve().parent


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def independent_records(path: Path, training_manifest: Path, minimum_class_support: int):
    records, errors = load_records(path, allowed_training_splits={"independent_test"})
    if errors:
        raise ValueError(json.dumps({"invalid_records": len(errors), "examples": errors[:10]}))
    if not records or any(item.get("dataset_split") != "independent_test" for item in records):
        raise ValueError("evaluation data must be explicitly marked independent_test")
    from collections import Counter
    counts = Counter(item["emotion_label"] for item in records)
    missing = EMOTION_LABELS - set(counts)
    under_supported = {label: counts[label] for label in EMOTION_LABELS if counts[label] < minimum_class_support}
    if missing or under_supported:
        raise ValueError(f"independent evaluation lacks class support: missing={sorted(missing)}, under_supported={under_supported}")
    import csv
    with training_manifest.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "participant_group_id_sha256" not in rows[0]:
        raise ValueError("training split manifest must contain participant_group_id_sha256")
    training_people = {row["participant_group_id_sha256"] for row in rows}
    evaluation_people = {sha256(item["participant_group_id"]) for item in records}
    overlap = training_people & evaluation_people
    if overlap:
        raise ValueError(f"participant overlap with training manifest ({len(overlap)} hashed IDs)")
    return records, dict(sorted(counts.items()))


def _probabilities(model, features):
    raw = model.predict_proba(features)
    aligned = np.zeros((len(features), len(CLASSES)), dtype=float)
    for index, label in enumerate(model.classes_):
        aligned[:, CLASSES.index(str(label))] = raw[:, index]
    return aligned


def bootstrap_macro_f1(labels, predictions, groups, *, iterations=1000, seed=42):
    unique_groups = np.asarray(sorted(set(groups)))
    if len(unique_groups) < 2:
        return None
    rng = np.random.default_rng(seed)
    scores = []
    group_indices = {group: np.flatnonzero(np.asarray(groups) == group) for group in unique_groups}
    for _ in range(iterations):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled])
        scores.append(f1_score(labels[indices], predictions[indices], labels=CLASSES,
                               average="macro", zero_division=0))
    return {"method": "participant_group_cluster_bootstrap", "iterations": iterations,
            "confidence_level": 0.95, "lower": float(np.quantile(scores, 0.025)),
            "upper": float(np.quantile(scores, 0.975))}


def metric_report(labels, probabilities, records, *, calibrated=False, latencies_ms=None):
    predictions = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
    details = classification_report(labels, predictions, labels=CLASSES, output_dict=True, zero_division=0)
    margins = np.sort(probabilities, axis=1)[:, -1] - np.sort(probabilities, axis=1)[:, -2]
    short = np.asarray([len(item["current_answer"].split()) <= 2 for item in records])
    clarify = short | ((margins < 0.08) if calibrated else np.zeros(len(records), dtype=bool))
    result = {
        "records": len(records), "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision_score(labels, predictions, labels=CLASSES, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, predictions, labels=CLASSES, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, labels=CLASSES, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, labels=CLASSES, average="weighted", zero_division=0)),
        "per_class": {label: {"precision": details[label]["precision"], "recall": details[label]["recall"],
                              "f1": details[label]["f1-score"], "support": int(details[label]["support"])} for label in CLASSES},
        "confusion_matrix": confusion_matrix(labels, predictions, labels=CLASSES).tolist(),
        "multiclass_log_loss": float(log_loss(labels, probabilities, labels=CLASSES)),
        "multiclass_brier_score": float(np.mean(np.sum((probabilities - np.eye(len(CLASSES))[[CLASSES.index(label) for label in labels]]) ** 2, axis=1))),
        "expected_calibration_error_10_bins": expected_calibration_error(labels, probabilities),
        "clarification_rate": float(clarify.mean()),
        "clarification_rule": "answer has <=2 whitespace tokens, or calibrated top-two margin <0.08",
        "macro_f1_participant_bootstrap_95ci": bootstrap_macro_f1(labels, predictions,
            [item["participant_group_id"] for item in records]),
    }
    if latencies_ms:
        result["model_inference_latency_ms"] = {
            "median": float(np.median(latencies_ms)), "p95": float(np.quantile(latencies_ms, 0.95)),
            "includes_embedding_and_classifier": True, "includes_context_formatting": False,
        }
    return result


def evaluate_v4(records, labels):
    runtime = load_runtime(V4_VERSION)
    if not runtime.ready:
        raise RuntimeError(f"Production v4 is unavailable: {runtime.error}")
    outputs = []
    latencies = []
    for item in records:
        started = time.perf_counter()
        outputs.append(runtime.predict_proba([item["current_answer"]])[0])
        latencies.append((time.perf_counter() - started) * 1000)
    aligned = np.zeros((len(outputs), len(CLASSES)), dtype=float)
    for col, label in enumerate(runtime.classes_):
        aligned[:, CLASSES.index(str(label))] = np.asarray(outputs)[:, col]
    return {"model_version": V4_VERSION, "context_aware": False,
            "artifact_sha256": runtime.metadata.get("artifact_sha256"),
            "metrics": metric_report(labels, aligned, records, calibrated=False, latencies_ms=latencies)}


def evaluate_candidate(path: Path, records, labels, expected_sha256: str | None = None):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"candidate checksum mismatch: {path}")
    bundle = joblib.load(path)
    input_format = bundle.get("input_format")
    if input_format not in {"answer_only", "question_answer", "multi_turn"}:
        raise ValueError(f"unsupported candidate input format in {path}")
    if (bundle.get("context_aware") is True) != (input_format != "answer_only"):
        raise ValueError(f"candidate context_aware flag conflicts with its input format: {path}")
    if bundle.get("supported_classes") != CLASSES or bundle.get("label_order") != CLASSES:
        raise ValueError(f"candidate label order differs from the authoritative project class order: {path}")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(bundle["embedding_model"], device="cpu", local_files_only=True)
    from context_dataset_schema import model_input_from_record
    model_inputs = [model_input_from_record(item, input_format, tokenizer=encoder.tokenizer,
                    max_tokens=max(1, int(encoder.max_seq_length) - 2)) for item in records]
    features = encoder.encode(model_inputs, batch_size=32, show_progress_bar=False,
                              normalize_embeddings=True, convert_to_numpy=True)
    probabilities = _probabilities(bundle["model"], features)
    encoder.encode([model_inputs[0]], show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
    latencies = []
    for text in model_inputs:
        started = time.perf_counter()
        one = encoder.encode([text], show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
        bundle["model"].predict_proba(one)
        latencies.append((time.perf_counter() - started) * 1000)
    version = bundle.get("model_version", path.stem)
    return {"model_version": version, "input_format": input_format,
            "context_aware": bool(bundle.get("context_aware")), "artifact_sha256": digest,
            "metrics": metric_report(labels, probabilities, records,
                calibrated=bool(bundle.get("calibration", {}).get("confidence_calibrated", False)),
                latencies_ms=latencies)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Frozen independent_test JSON-schema CSV")
    parser.add_argument("--training-split-manifest", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[], type=Path,
                        help="Candidate artifact path; may be supplied multiple times")
    parser.add_argument("--candidate-sha256", action="append", default=[],
                        help="Expected SHA-256 for each --candidate in corresponding order")
    parser.add_argument("--minimum-class-support", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    records, class_counts = independent_records(args.data, args.training_split_manifest, args.minimum_class_support)
    labels = np.asarray([item["emotion_label"] for item in records])
    results = [evaluate_v4(records, labels)]
    if args.candidate_sha256 and len(args.candidate_sha256) != len(args.candidate):
        parser.error("supply one --candidate-sha256 for each --candidate")
    expected_hashes = args.candidate_sha256 or [None] * len(args.candidate)
    results.extend(evaluate_candidate(path, records, labels, expected_sha256=digest)
                   for path, digest in zip(args.candidate, expected_hashes))
    output = {"evaluation_status": "independent_participant_disjoint_evaluation",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
              "training_split_manifest_sha256": hashlib.sha256(args.training_split_manifest.read_bytes()).hexdigest(),
              "records": len(records), "participant_groups": len({item["participant_group_id"] for item in records}),
              "class_counts": class_counts, "minimum_class_support": args.minimum_class_support,
              "candidate_results": results,
              "limitations": ["Only appropriate consented human labels qualify.",
                              "Do not treat public or synthetic records as elderly-user clinical validation."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(output, indent=2))
    print(json.dumps({"status": output["evaluation_status"], "records": output["records"],
                      "models_evaluated": len(results), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
