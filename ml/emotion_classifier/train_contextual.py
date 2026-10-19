"""Train bounded question-answer and multi-turn candidates from verified human labels.

Without a valid input file the command emits WAITING_FOR_VALID_LABELED_DATA and
does not create model artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score)
from sklearn.model_selection import StratifiedGroupKFold

from context_dataset_schema import (EMOTION_LABELS, model_input_from_record,
                                    validate_context_dataset, validate_context_record)

ROOT = Path(__file__).resolve().parent
CHECKPOINT = "sentence-transformers/all-MiniLM-L6-v2"
CLASSES = ["happiness", "sadness", "loneliness", "anxiety", "anger", "cognitive_fog", "neutral"]
SEED = 42


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aligned_probabilities(model, features):
    raw = model.predict_proba(features)
    output = np.zeros((len(features), len(CLASSES)), dtype=float)
    for index, label in enumerate(model.classes_):
        output[:, CLASSES.index(str(label))] = raw[:, index]
    return output


def expected_calibration_error(labels, probabilities, bins=10):
    confidence = probabilities.max(axis=1)
    predicted = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
    correct = predicted == labels
    ece = 0.0
    for lower in np.linspace(0, 1, bins + 1)[:-1]:
        upper = lower + 1 / bins
        selected = (confidence >= lower) & ((confidence <= upper) if upper >= 1 else (confidence < upper))
        if selected.any():
            ece += float(selected.mean()) * abs(float(confidence[selected].mean()) - float(correct[selected].mean()))
    return ece


def load_records(path, *, allowed_training_splits=None):
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return [], []
    records = []
    errors = []
    for index, row in frame.iterrows():
        item = row.to_dict()
        try:
            if isinstance(item.get("previous_turns"), str):
                item["previous_turns"] = json.loads(item["previous_turns"] or "[]")
            if isinstance(item.get("annotation_reviewers"), str):
                item["annotation_reviewers"] = json.loads(item["annotation_reviewers"] or "[]")
            if isinstance(item.get("turn_index"), str):
                item["turn_index"] = int(item["turn_index"])
            records.append(validate_context_record(item))
        except Exception as exc:
            errors.append({"row_1based": int(index) + 2, "error": str(exc)})
    if records and not errors:
        try:
            validate_context_dataset(records, require_training_eligible=True,
                                     allowed_training_splits=allowed_training_splits,
                                     require_previous_turns_present=True)
        except Exception as exc:
            errors.append({"row_1based": None, "error": str(exc)})
    return records, errors


def grouped_splits(labels, groups):
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_cal, validation = next(outer.split(np.zeros(len(labels)), labels, groups))
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + 1)
    fit_rel, calibration_rel = next(inner.split(np.zeros(len(train_cal)), labels[train_cal], groups[train_cal]))
    fit, calibration = train_cal[fit_rel], train_cal[calibration_rel]
    for left, right, left_name, right_name in ((fit, calibration, "fit", "calibration"),
                                                (fit, validation, "fit", "validation"),
                                                (calibration, validation, "calibration", "validation")):
        overlap = set(groups[left]) & set(groups[right])
        if overlap:
            raise RuntimeError(f"group leakage across {left_name}/{right_name}")
    return fit, calibration, validation


def build_leakage_groups(records, threshold=0.90):
    if len(records) > 20000:
        raise ValueError("context dataset exceeds the audited 20,000-row duplicate-scan bound")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    by_participant, by_exact_text = {}, {}
    all_text = []
    for index, row in enumerate(records):
        owner = row["participant_group_id"]
        if owner in by_participant:
            union(index, by_participant[owner])
        else:
            by_participant[owner] = index
        text = " ".join(f"{row['current_question']} {row['current_answer']}".casefold().split())
        all_text.append(text)
        if text in by_exact_text:
            union(index, by_exact_text[text])
        else:
            by_exact_text[text] = index
    if len(all_text) > 1:
        matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit_transform(all_text)
        neighbor_index = NearestNeighbors(metric="cosine", algorithm="brute", radius=1.0-threshold, n_jobs=1).fit(matrix)
        distances, neighbors = neighbor_index.radius_neighbors(matrix, return_distance=True)
        near_pairs = 0
        for i, (distances_i, neighbors_i) in enumerate(zip(distances, neighbors)):
            for distance, j in zip(distances_i, neighbors_i):
                j = int(j)
                if j > i and all_text[i] != all_text[j] and 1.0 - float(distance) >= threshold:
                    union(i, j)
                    near_pairs += 1
    else:
        near_pairs = 0
    return np.asarray([find(i) for i in range(len(records))]), {
        "participant_group_count": len(by_participant), "exact_text_families": len(by_exact_text),
        "near_duplicate_pairs_at_or_above_threshold": near_pairs, "threshold": threshold,
        "group_count_after_union": len({find(i) for i in range(len(records))}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "context_emotion_training.csv")
    parser.add_argument("--output-root", type=Path, default=ROOT / "stage9_experiments")
    args = parser.parse_args()
    if not args.data.exists():
        print(json.dumps({"status": "WAITING_FOR_VALID_LABELED_DATA", "data_path": str(args.data),
                          "records": 0, "model_artifacts_created": False}, indent=2))
        return 0
    records, errors = load_records(args.data)
    if errors:
        raise ValueError(json.dumps({"invalid_records": len(errors), "examples": errors[:10]}))
    counts = pd.Series([row["emotion_label"] for row in records]).value_counts().to_dict()
    if not records or set(counts) != EMOTION_LABELS:
        print(json.dumps({"status": "WAITING_FOR_VALID_LABELED_DATA", "records": len(records),
                          "class_counts": counts, "model_artifacts_created": False,
                          "reason": "all seven human-annotated classes are required"}, indent=2))
        return 0
    labels = np.asarray([row["emotion_label"] for row in records])
    groups, leakage_audit = build_leakage_groups(records)
    fit_idx, cal_idx, val_idx = grouped_splits(labels, groups)
    for part in (fit_idx, cal_idx, val_idx):
        missing = EMOTION_LABELS - set(labels[part])
        if missing:
            raise ValueError(f"split missing classes {sorted(missing)}; collect more group-disjoint samples")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(CHECKPOINT, device="cpu", local_files_only=True)
    fit_data = {}
    output_root = args.output_root
    run_dir = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "trained_experimental_candidates", "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": sha256(args.data), "source_records": len(records), "class_counts": counts,
              "checkpoint": CHECKPOINT, "seed": SEED, "split_counts": {"fit": len(fit_idx), "calibration": len(cal_idx), "validation": len(val_idx)},
              "candidate_results": {}, "participants_used_for_splitting": True,
              "duplicate_group_audit": leakage_audit,
              "dependency_versions": {"scikit_learn": sklearn.__version__, "pandas": pd.__version__,
                                       "numpy": np.__version__, "joblib": joblib.__version__},
              "evaluation_status": "participant-disjoint internal validation only; no independent benchmark used"}
    partition_name = {int(i): name for name, indices in (("fit", fit_idx), ("calibration", cal_idx), ("validation", val_idx)) for i in indices}
    split_manifest_path = run_dir / "split_manifest.csv"
    with split_manifest_path.open("x", encoding="utf-8", newline="") as stream:
        import csv
        writer = csv.DictWriter(stream, fieldnames=["sample_id_sha256", "participant_group_id_sha256", "session_id_sha256", "partition", "label"])
        writer.writeheader()
        for i, row in enumerate(records):
            writer.writerow({"sample_id_sha256": hashlib.sha256(str(row["sample_id"]).encode()).hexdigest(),
                             "participant_group_id_sha256": hashlib.sha256(str(row["participant_group_id"]).encode()).hexdigest(),
                             "session_id_sha256": hashlib.sha256(str(row["session_id"]).encode()).hexdigest(),
                             "partition": partition_name[i], "label": labels[i]})
    split_manifest_sha256 = sha256(split_manifest_path)
    report["split_manifest_sha256"] = split_manifest_sha256
    for input_format in ("answer_only", "question_answer", "multi_turn"):
        model_inputs = [model_input_from_record(
            row, input_format, tokenizer=encoder.tokenizer,
            max_tokens=max(1, int(encoder.max_seq_length) - 2),
        ) for row in records]
        embeddings = encoder.encode(model_inputs, batch_size=32, show_progress_bar=False,
                                    normalize_embeddings=True, convert_to_numpy=True)
        candidates = {
            "logistic_c1": LogisticRegression(C=1.0, class_weight=None, max_iter=3000, random_state=SEED),
            "balanced_logistic_c1": LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=SEED),
            "balanced_logistic_c0_3": LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000, random_state=SEED),
        }
        result_rows = []
        for name, estimator in candidates.items():
            estimator.fit(embeddings[fit_idx], labels[fit_idx])
            pred = estimator.predict(embeddings[val_idx])
            per_class = classification_report(labels[val_idx], pred, labels=CLASSES, output_dict=True, zero_division=0)
            result_rows.append({"candidate": name, "accuracy": float(accuracy_score(labels[val_idx], pred)),
                                "macro_f1": float(f1_score(labels[val_idx], pred, labels=CLASSES, average="macro", zero_division=0)),
                                "per_class": {label: {"precision": per_class[label]["precision"],
                                                      "recall": per_class[label]["recall"],
                                                      "f1": per_class[label]["f1-score"],
                                                      "support": int(per_class[label]["support"])} for label in CLASSES}})
            fit_data[name] = estimator
        winner = max(result_rows, key=lambda row: (row["macro_f1"], row["accuracy"]))
        selected = fit_data[winner["candidate"]]
        calibration = CalibratedClassifierCV(FrozenEstimator(selected), method="sigmoid")
        calibration.fit(embeddings[cal_idx], labels[cal_idx])
        prob = aligned_probabilities(calibration, embeddings[val_idx])
        validation_labels = labels[val_idx]
        validation_pred = np.asarray(CLASSES)[prob.argmax(axis=1)]
        one_hot = np.eye(len(CLASSES))[[CLASSES.index(label) for label in validation_labels]]
        selected_embeddings = embeddings[val_idx[:min(32, len(val_idx))]]
        classifier_times = []
        for _ in range(5):
            started = time.perf_counter()
            calibration.predict_proba(selected_embeddings)
            classifier_times.append((time.perf_counter() - started) * 1000 / len(selected_embeddings))
        selected_texts = [model_inputs[i] for i in val_idx[:min(32, len(val_idx))]]
        encoder_times = []
        for _ in range(3):
            started = time.perf_counter()
            encoder.encode(selected_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
            encoder_times.append((time.perf_counter() - started) * 1000 / len(selected_texts))
        per_class_report = classification_report(validation_labels, validation_pred, labels=CLASSES, output_dict=True, zero_division=0)
        report["candidate_results"][input_format] = {
            "candidates": result_rows, "selected_candidate": winner["candidate"],
            "calibration_rows": len(cal_idx), "validation_log_loss": float(log_loss(labels[val_idx], prob, labels=CLASSES)),
            "validation_brier_score": float(np.mean(np.sum((prob - one_hot) ** 2, axis=1))),
            "validation_ece_10_bins": expected_calibration_error(validation_labels, prob),
            "accuracy": float(accuracy_score(validation_labels, validation_pred)),
            "macro_precision": float(precision_score(validation_labels, validation_pred, labels=CLASSES, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(validation_labels, validation_pred, labels=CLASSES, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(validation_labels, validation_pred, labels=CLASSES, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(validation_labels, validation_pred, labels=CLASSES, average="weighted", zero_division=0)),
            "per_class": {label: {"precision": per_class_report[label]["precision"],
                                  "recall": per_class_report[label]["recall"],
                                  "f1": per_class_report[label]["f1-score"],
                                  "support": int(per_class_report[label]["support"])} for label in CLASSES},
            "confusion_matrix": confusion_matrix(validation_labels, validation_pred, labels=CLASSES).tolist(),
            "selected_macro_f1": winner["macro_f1"], "validation_rows": len(val_idx),
            "classifier_latency_ms_per_record_median": float(np.median(classifier_times)),
            "embedding_latency_ms_per_record_median": float(np.median(encoder_times)),
        }
        with (run_dir / f"{input_format}_validation_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
            import csv
            writer = csv.DictWriter(stream, fieldnames=["sample_id_sha256", "label", "prediction"])
            writer.writeheader()
            for i, truth, pred in zip(val_idx, validation_labels, validation_pred):
                writer.writerow({"sample_id_sha256": hashlib.sha256(str(records[i]["sample_id"]).encode()).hexdigest(),
                                 "label": truth, "prediction": pred})
        artifact = run_dir / f"{input_format}_logistic_regression.joblib"
        joblib.dump({"model_version": f"minilm_{input_format}_stage9_{run_dir.name}",
                     "model": calibration, "embedding_model": CHECKPOINT,
                     "input_format": input_format, "context_aware": input_format != "answer_only",
                     "context_window_turns": 2,
                     "supported_classes": CLASSES, "label_order": CLASSES,
                     "training_dataset_sha256": sha256(args.data),
                     "split_manifest_sha256": split_manifest_sha256,
                     "dependency_versions": {"scikit_learn": sklearn.__version__,
                                             "pandas": pd.__version__, "numpy": np.__version__,
                                             "joblib": joblib.__version__},
                     "calibration": {"confidence_calibrated": True, "method": "sigmoid",
                                     "calibration_rows": len(cal_idx),
                                     "partition": "participant-disjoint calibration split"}}, artifact)
        report["candidate_results"][input_format]["artifact"] = artifact.name
        report["candidate_results"][input_format]["artifact_sha256"] = sha256(artifact)
    (run_dir / "experiment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["run_dir"] = str(run_dir)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
