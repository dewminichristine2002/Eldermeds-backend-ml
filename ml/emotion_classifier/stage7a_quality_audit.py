"""Stage 7A data quality, review-queue, split, and artifact replay utility.

This utility is read-only with respect to source datasets and model artifacts.
It never changes labels and never reads evaluation text into the replay run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RUNS = ROOT / "stage7a_experiments"
LABELS = {"happiness", "sadness", "loneliness", "anxiety", "anger", "cognitive_fog", "neutral"}
NEGATION = re.compile(r"\b(?:no|not|never|nothing|cannot|can't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|n't)\b", re.I)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize(value: object) -> str:
    # Preserve punctuation and negation; normalization is only case-fold + whitespace collapse.
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def read_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader), list(reader.fieldnames or [])


def label_key(columns):
    if "label" in columns:
        return "label"
    if "expected_emotion" in columns:
        return "expected_emotion"
    return None


def dataset_audit(path: Path):
    rows, columns = read_rows(path)
    target = label_key(columns)
    text_col = "text" if "text" in columns else None
    labels = [str(row.get(target) or "").strip().casefold() if target else "" for row in rows]
    texts = [normalize(row.get(text_col)) if text_col else "" for row in rows]
    text_counts = Counter(text for text in texts if text)
    families = defaultdict(set)
    for text, label in zip(texts, labels):
        if text:
            families[text].add(label)
    counts = Counter(labels)
    invalid = {k: v for k, v in counts.items() if k and k not in LABELS}
    return {
        "file": path.name,
        "sha256": sha256_bytes(path.read_bytes()),
        "rows": len(rows), "columns": columns, "text_column": text_col, "label_column": target,
        "label_counts": dict(sorted(counts.items())), "unsupported_label_counts": dict(sorted(invalid.items())),
        "missing_text": sum(not value for value in texts), "missing_label": sum(not value for value in labels),
        "unique_normalized_text_families": len(families),
        "duplicate_rows": sum(count - 1 for count in text_counts.values() if count > 1),
        "conflicting_text_families": sum(len(values) > 1 for values in families.values()),
        "short_le_2_tokens": sum(bool(t) and len(t.split()) <= 2 for t in texts),
        "long_gt_128_tokens": sum(len(t.split()) > 128 for t in texts),
        "negation_rows": sum(bool(NEGATION.search(t)) for t in texts),
        "has_participant_or_session_ids": any(c in columns for c in ("participant_id", "user_id", "session_id", "conversation_id")),
        "has_question_answer_pairs": any(c in columns for c in ("question", "prompt")) and any(c in columns for c in ("answer", "response")),
        "rows_data": rows,
    }


def duplicate_review_queue(dataset_audits, limit=500, near_pairs=(), development_rows=()):
    candidates = []
    for audit in dataset_audits:
        # Evaluation partitions remain sealed from annotation-driven selection.
        if audit["file"] in {"locked_test_v2.csv", "project_evaluation_set.csv", "training_dataset_v2.csv"}:
            continue
        rows = audit.pop("rows_data")
        columns = audit["columns"]
        tcol, lcol = audit["text_column"], audit["label_column"]
        by_text = defaultdict(list)
        for index, row in enumerate(rows):
            text = normalize(row.get(tcol)) if tcol else ""
            label = str(row.get(lcol) or "").strip().casefold() if lcol else ""
            if text:
                by_text[text].append((index, row, label))
        for norm, items in by_text.items():
            labels = sorted({item[2] for item in items})
            short = len(norm.split()) <= 2
            if len(items) < 2 and not short:
                continue
            reasons = []
            if len(items) > 1:
                reasons.append("normalized_exact_duplicate")
            if len(labels) > 1:
                reasons.append("label_conflict")
            if short:
                reasons.append("very_short_context_dependent")
            for index, row, label in items[:3]:
                text = str(row.get(tcol) or "") if tcol else ""
                rid = sha256_bytes(f"{audit['file']}\0{index}\0{norm}".encode("utf-8"))
                candidates.append({
                    "sample_id": rid, "source_file": audit["file"], "source_row_1based": index + 2,
                    "text": text, "existing_label": label, "family_labels": "|".join(labels),
                    "reason": ";".join(reasons), "review_status": "pending", "reviewer_decision": "",
                    "reviewer_rationale": "",
                })
    for pair in near_pairs:
        for side in ("left", "right"):
            index = pair[f"{side}_row_1based"] - 2
            row = development_rows[index]
            text = str(row.get("text") or "")
            label = str(row.get("label") or "").strip().casefold()
            rid = sha256_bytes(f"development_dataset_v3.csv\0near\0{normalize(text)}".encode("utf-8"))
            candidates.append({
                "sample_id": rid, "source_file": "development_dataset_v3.csv",
                "source_row_1based": index + 2, "text": text, "existing_label": label,
                "family_labels": f"{pair['left_label']}|{pair['right_label']}",
                "reason": "near_duplicate_candidate", "review_status": "pending",
                "reviewer_decision": "", "reviewer_rationale": "",
            })
    candidates.sort(key=lambda row: (row["reason"], row["source_file"], row["sample_id"]))
    # Deterministic cap keeps the queue tractable; preserve category coverage by round-robin reason.
    groups = defaultdict(list)
    for item in candidates:
        groups[item["reason"]].append(item)
    selected = []
    keys = sorted(groups)
    while len(selected) < limit and any(groups.values()):
        for key in keys:
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop(0))
    return selected, len(candidates)


def near_duplicate_audit(rows, threshold=0.90):
    texts = [normalize(row.get("text")) for row in rows]
    nonempty_idx = [i for i, text in enumerate(texts) if text]
    if len(nonempty_idx) < 2:
        return {"method": "char_wb TF-IDF cosine similarity", "threshold": threshold, "candidate_pairs": []}
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    corpus = [texts[i] for i in nonempty_idx]
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit_transform(corpus)
    neighbors = NearestNeighbors(metric="cosine", algorithm="brute", radius=1.0-threshold, n_jobs=1).fit(matrix)
    distances, indices = neighbors.radius_neighbors(matrix, return_distance=True, sort_results=True)
    found = []
    for local_i, (ds, js) in enumerate(zip(distances, indices)):
        for distance, local_j in zip(ds, js):
            if local_j <= local_i or local_j == local_i:
                continue
            i, j = nonempty_idx[local_i], nonempty_idx[int(local_j)]
            if texts[i] == texts[j]:
                continue
            found.append({"left_row_1based": i+2, "right_row_1based": j+2,
                          "cosine_similarity": round(1.0-float(distance), 6),
                          "left_label": str(rows[i].get("label", "")), "right_label": str(rows[j].get("label", "")),
                          "labels_conflict": str(rows[i].get("label", "")).casefold() != str(rows[j].get("label", "")).casefold(),
                          "left_text_sha256": sha256_bytes(texts[i].encode("utf-8")),
                          "right_text_sha256": sha256_bytes(texts[j].encode("utf-8"))})
    return {"method": "char_wb TF-IDF 3-5 grams, cosine radius neighbors; candidate pairs only",
            "threshold": threshold, "candidate_pairs": found}


def replay_v5(run_dir: Path):
    import joblib
    import numpy as np
    import pandas as pd
    import sklearn
    from sentence_transformers import SentenceTransformer

    source = DATA / "development_dataset_v3.csv"
    bundle_dir = ROOT / "v5_experiments" / "20261008T175725Z"
    artifact = bundle_dir / "minilm_logistic_regression_v5_answer_only.joblib"
    old_report = json.loads((bundle_dir / "experiment_report_reviewed.json").read_text(encoding="utf-8"))
    source_rows, _ = read_rows(source)
    manifest, _ = read_rows(bundle_dir / "split_manifest.csv")
    by_hash = {sha256_bytes(normalize(row.get("text")).encode("utf-8")): row for row in source_rows}
    if sha256_bytes(source.read_bytes()) != old_report["source_sha256"]:
        raise RuntimeError("development source hash differs from recorded v5 source")
    manifest_rows = []
    for row in manifest:
        key = row["record_sha256"]
        if key not in by_hash:
            raise RuntimeError(f"split manifest record missing from source: {key}")
        current = by_hash[key]
        if str(current.get("label", "")).strip().casefold() != row["label"].casefold():
            raise RuntimeError(f"split manifest label mismatch: {key}")
        manifest_rows.append((row, current))
    if len(manifest_rows) != len(source_rows):
        raise RuntimeError("source rows and manifest row counts differ")
    rows = [current for part, current in manifest_rows if part["partition"] == "validation"]
    y = np.asarray([str(row["label"]).strip().casefold() for row in rows])
    texts = [str(row["text"]).strip() for row in rows]
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu", local_files_only=True)
    embeddings = encoder.encode(texts, batch_size=32, show_progress_bar=False,
                                normalize_embeddings=True, convert_to_numpy=True)
    model = joblib.load(artifact)["model"]
    probabilities = model.predict_proba(embeddings)
    classes = [str(value) for value in model.classes_]
    predicted = np.asarray(classes)[probabilities.argmax(axis=1)]
    metrics = {
        "accuracy": float(np.mean(predicted == y)),
        "macro_f1": float(__import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(y, predicted, labels=sorted(LABELS), average="macro", zero_division=0)),
        "validation_rows": len(rows), "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__, "artifact_sha256": sha256_bytes(artifact.read_bytes()),
        "source_sha256": sha256_bytes(source.read_bytes()), "used_locked_or_project_evaluation": False,
        "recorded_validation_metrics": old_report.get("calibrated_selected_validation"),
        "status": "replayed; compare exact predictions against stored validation_predictions.csv",
    }
    old_predictions, _ = read_rows(bundle_dir / "validation_predictions.csv")
    old_map = {row["record_sha256"]: row for row in old_predictions}
    mismatches = 0
    for text, truth, pred in zip(texts, y, predicted):
        key = sha256_bytes(normalize(text).encode("utf-8"))
        prior = old_map.get(key)
        if prior is None or prior["label"].casefold() != truth or prior["v5_prediction"].casefold() != str(pred).casefold():
            mismatches += 1
    metrics["stored_prediction_mismatches"] = mismatches
    metrics["status"] = "exact_replay_match" if mismatches == 0 else "replay_mismatch"
    (run_dir / "v5_replay.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--replay-v5", action="store_true")
    parser.add_argument("--queue-limit", type=int, default=500)
    args = parser.parse_args()
    run_dir = RUNS / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    paths = sorted(DATA.glob("*.csv"))
    audits = [dataset_audit(path) for path in paths]
    dev_rows, _ = read_rows(DATA / "development_dataset_v3.csv")
    near = near_duplicate_audit(dev_rows)
    queue, total_candidates = duplicate_review_queue(
        audits, args.queue_limit, near["candidate_pairs"], dev_rows
    )
    with (run_dir / "annotation_review_queue.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(queue[0]) if queue else ["sample_id", "text", "existing_label", "reason", "review_status", "reviewer_decision", "reviewer_rationale"])
        writer.writeheader()
        writer.writerows(queue)
    (run_dir / "development_near_duplicate_candidates.json").write_text(json.dumps(near, indent=2), encoding="utf-8")
    for audit in audits:
        audit.pop("rows_data", None)
    manifest = {
        "stage": "7A", "created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "datasets": audits,
        "provenance_status": "source and label lineage is incomplete: data README identifies converted public datasets but gives no per-file citations, license evidence, or source-to-label mapping; project-authored subset attribution is recorded in earlier audit only",
        "review_queue": {"rows": len(queue), "candidate_rows_before_cap": total_candidates, "limit": args.queue_limit,
                         "all_decisions": "pending; this file makes no label changes"},
        "near_duplicate_candidates_development": len(near["candidate_pairs"]),
        "split_manifest_sha256": sha256_bytes((ROOT / "v5_experiments" / "20261008T175725Z" / "split_manifest.csv").read_bytes()),
        "known_constraints": ["No participant/session/conversation keys in available datasets", "locked_test_v2 was previously exposed and excluded from this audit's training and replay", "human review decisions are pending"],
    }
    (run_dir / "dataset_quality_inventory.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.replay_v5:
        manifest["v5_replay"] = replay_v5(run_dir)
        (run_dir / "dataset_quality_inventory.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "review_queue_rows": len(queue),
                      "duplicate_candidate_rows": total_candidates, "dev_near_duplicate_pairs": len(near["candidate_pairs"]),
                      "datasets": len(audits), "v5_replay": manifest.get("v5_replay")}, indent=2))


if __name__ == "__main__":
    main()
