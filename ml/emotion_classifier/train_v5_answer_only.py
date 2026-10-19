"""Train and evaluate an experimental MiniLM v5 answer-only classifier.

The development partition is split by deterministic text-similarity groups.
Locked and project evaluation sets are never used for fitting, selection, or
calibration. Artifacts are created once and never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import platform
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
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "v5_experiments"
LABELS = ["happiness", "sadness", "loneliness", "anxiety", "anger", "cognitive_fog", "neutral"]
SEED = 42
CHECKPOINT = "sentence-transformers/all-MiniLM-L6-v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_key(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def near_duplicate_groups(texts: list[str], threshold: float = 0.90) -> tuple[np.ndarray, dict]:
    """Connected components of exact and char-TFIDF near-duplicate records."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    normalized = [text_key(t) for t in texts]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    matrix = vectorizer.fit_transform(normalized)
    similarities = cosine_similarity(matrix, dense_output=True)
    parent = list(range(len(texts)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    exact_pairs = 0
    near_pairs = 0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if normalized[i] == normalized[j]:
                exact_pairs += 1
                union(i, j)
            elif similarities[i, j] >= threshold:
                near_pairs += 1
                union(i, j)
    groups = np.asarray([find(i) for i in range(len(texts))], dtype=np.int64)
    info = {
        "method": "casefolded whitespace-normalized exact matching plus connected components of char_wb TF-IDF cosine similarity",
        "char_ngram_range": [3, 5],
        "near_duplicate_threshold": threshold,
        "exact_pairs": exact_pairs,
        "near_duplicate_pairs": near_pairs,
        "group_count": int(len(np.unique(groups))),
        "records_in_multi_record_groups": int(sum(np.sum(groups == g) for g in np.unique(groups) if np.sum(groups == g) > 1)),
    }
    return groups, info


def split_indices(y: np.ndarray, groups: np.ndarray):
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    trainval_idx, validation_idx = next(outer.split(np.zeros(len(y)), y, groups))
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + 1)
    fit_relative, calibration_relative = next(
        inner.split(np.zeros(len(trainval_idx)), y[trainval_idx], groups[trainval_idx])
    )
    fit_idx = trainval_idx[fit_relative]
    calibration_idx = trainval_idx[calibration_relative]
    return fit_idx, validation_idx, calibration_idx


def model_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "worst_class_f1": float(min(report[label]["f1-score"] for label in LABELS)),
        "per_class": {label: {"precision": report[label]["precision"], "recall": report[label]["recall"],
                               "f1": report[label]["f1-score"], "support": int(report[label]["support"])}
                      for label in LABELS},
    }


def aligned_proba(model, features):
    raw = model.predict_proba(features)
    output = np.zeros((len(features), len(LABELS)), dtype=float)
    for col, label in enumerate(model.classes_):
        output[:, LABELS.index(str(label))] = raw[:, col]
    return output


def reliability_bins(y_true, probabilities, n_bins: int = 10):
    confidence = probabilities.max(axis=1)
    predicted = np.asarray(LABELS)[probabilities.argmax(axis=1)]
    correct = predicted == np.asarray(y_true)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    for index in range(n_bins):
        if index == n_bins - 1:
            selected = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            selected = (confidence >= edges[index]) & (confidence < edges[index + 1])
        count = int(selected.sum())
        mean_confidence = float(confidence[selected].mean()) if count else None
        accuracy = float(correct[selected].mean()) if count else None
        if count:
            ece += count / len(confidence) * abs(mean_confidence - accuracy)
        rows.append({"bin_lower": edges[index], "bin_upper": edges[index + 1],
                     "count": count, "mean_confidence": mean_confidence, "accuracy": accuracy})
    return rows, float(ece)


def main():
    source = DATA / "development_dataset_v3.csv"
    source_hash = sha256(source)
    v4_artifact_path = ROOT / "domain_hierarchical_experiment" / "candidate_model.joblib"
    v4_metadata_path = ROOT / "production_model_metadata_v4.json"
    v4_metadata = json.loads(v4_metadata_path.read_text(encoding="utf-8"))
    if sha256(v4_artifact_path) != v4_metadata.get("artifact_sha256"):
        raise RuntimeError("v4 production artifact checksum does not match its metadata")
    frame = pd.read_csv(source)
    if not {"text", "label"}.issubset(frame.columns):
        raise ValueError("development_dataset_v3.csv must contain text and label columns")
    frame["text"] = frame["text"].fillna("").astype(str).map(lambda t: re.sub(r"\s+", " ", t).strip())
    frame["label"] = frame["label"].fillna("").astype(str).str.strip().str.lower()
    invalid = sorted(set(frame.label) - set(LABELS))
    if frame.text.eq("").any() or invalid:
        raise ValueError(f"Invalid training records: empty={int(frame.text.eq('').sum())}, labels={invalid}")
    if frame.duplicated(["text", "label"]).any():
        frame = frame.drop_duplicates(["text", "label"], keep="first").reset_index(drop=True)
    conflicts = frame.groupby(frame.text.map(text_key)).label.nunique()
    conflict_keys = set(conflicts[conflicts > 1].index)
    if conflict_keys:
        frame = frame[~frame.text.map(text_key).isin(conflict_keys)].reset_index(drop=True)

    groups, duplicate_info = near_duplicate_groups(frame.text.tolist())
    y = frame.label.to_numpy()
    fit_idx, validation_idx, calibration_idx = split_indices(y, groups)

    # Confirm no exact/near-duplicate group crosses any data partition.
    partitions = {"fit": fit_idx, "validation": validation_idx, "calibration": calibration_idx}
    for left_name, left in partitions.items():
        for right_name, right in partitions.items():
            if left_name < right_name and set(groups[left]).intersection(groups[right]):
                raise RuntimeError(f"near-duplicate group leakage: {left_name}/{right_name}")

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(CHECKPOINT, device="cpu", local_files_only=True)
    texts = frame.text.tolist()
    embeddings = encoder.encode(texts, batch_size=32, show_progress_bar=True,
                                 normalize_embeddings=True, convert_to_numpy=True)
    if embeddings.ndim != 2 or embeddings.shape[1] != 384:
        raise RuntimeError(f"Expected 384-dimensional MiniLM embeddings, got {embeddings.shape}")

    candidates = []
    fitted = {}
    configs = [
        ("A_logistic_baseline", LogisticRegression(C=1.0, class_weight=None, max_iter=3000, random_state=SEED)),
        ("B_logistic_balanced", LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=SEED)),
    ]
    for c_value in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        configs.append((f"C_logistic_balanced_C{c_value:g}",
                        LogisticRegression(C=c_value, class_weight="balanced", max_iter=3000, random_state=SEED)))
    for c_value in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        configs.append((f"D_linear_svm_balanced_C{c_value:g}",
                        LinearSVC(C=c_value, class_weight="balanced", random_state=SEED)))

    for name, model in configs:
        model.fit(embeddings[fit_idx], y[fit_idx])
        predicted = model.predict(embeddings[validation_idx])
        metrics = model_metrics(y[validation_idx], predicted)
        candidates.append({"candidate": name, "classifier": type(model).__name__,
                           "parameters": model.get_params(), **metrics})
        fitted[name] = model

    # Same validation fold, freshly loaded v4 artifact, and same embedding pipeline.
    v4_bundle = joblib.load(v4_artifact_path)
    v4_model = v4_bundle["model"]
    v4_pred = v4_model.predict(embeddings[validation_idx])
    v4_metrics = model_metrics(y[validation_idx], v4_pred)

    winner = sorted(candidates, key=lambda row: (row["macro_f1"], row["worst_class_f1"], row["accuracy"]), reverse=True)[0]
    winner_name = winner["candidate"]
    selected_config = next(config for config in configs if config[0] == winner_name)
    # Keep validation fully out of both classifier fitting and calibration.
    final_estimator = selected_config[1]
    final_estimator.fit(embeddings[fit_idx], y[fit_idx])
    calibrator = CalibratedClassifierCV(FrozenEstimator(final_estimator), method="sigmoid")
    calibrator.fit(embeddings[calibration_idx], y[calibration_idx])
    calibrated = aligned_proba(calibrator, embeddings[validation_idx])
    calibrated_pred = np.asarray(LABELS)[calibrated.argmax(axis=1)]
    calibrated_metrics = model_metrics(y[validation_idx], calibrated_pred)
    validation_one_hot = np.eye(len(LABELS))[
        np.asarray([LABELS.index(value) for value in y[validation_idx]])
    ]
    true_probability = calibrated[np.arange(len(validation_idx)),
                                  np.asarray([LABELS.index(value) for value in y[validation_idx]])]
    calibration_metrics = {
        "multiclass_log_loss": float(-np.log(np.clip(true_probability, 1e-15, 1.0)).mean()),
        "multiclass_brier_score": float(np.mean(np.sum((calibrated - validation_one_hot) ** 2, axis=1))),
        "mean_confidence": float(calibrated.max(axis=1).mean()),
        "probability_sums_min": float(calibrated.sum(axis=1).min()),
        "probability_sums_max": float(calibrated.sum(axis=1).max()),
        "calibration_rows": int(len(calibration_idx)),
        "method": "sigmoid/Platt; classifier fitted only on fit partition, calibrator fitted only on grouped calibration partition",
    }
    reliability_rows, expected_calibration_error = reliability_bins(y[validation_idx], calibrated)
    calibration_metrics["top_label_expected_calibration_error_10_bins"] = expected_calibration_error

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = OUT / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    artifact_path = run_dir / "minilm_logistic_regression_v5_answer_only.joblib"
    bundle = {"architecture": "flat", "model": calibrator, "model_version": "minilm_logistic_regression_v5_answer_only",
              "embedding_model": CHECKPOINT, "embedding_normalization": "L2 normalized", "supported_classes": LABELS}
    joblib.dump(bundle, artifact_path)
    artifact_hash = sha256(artifact_path)

    metadata = {
        "model_version": "minilm_logistic_regression_v5_answer_only",
        "model_family": f"MiniLM sentence embeddings with calibrated {type(final_estimator).__name__}",
        "artifact_file": artifact_path.name,
        "artifact_sha256": artifact_hash,
        "embedding_model": CHECKPOINT,
        "embedding_normalization": "L2 normalized",
        "classifier": final_estimator.get_params(),
        "calibration": calibration_metrics,
        "scikit_learn_version": sklearn.__version__,
        "sentence_transformers_version": __import__("sentence_transformers").__version__,
        "confidence_calibrated": True,
        "supported_classes": LABELS,
        "input_format": "answer-only text",
        "context_aware": False,
        "random_seed": SEED,
        "training_partition_records": int(len(fit_idx)),
        "calibration_partition_records": int(len(calibration_idx)),
        "validation_partition_records": int(len(validation_idx)),
        "activation": "experimental; not active by default",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    # Validation-only latency: classifier plus precomputed embedding; encoder cost is measured separately.
    sample = embeddings[validation_idx[:min(32, len(validation_idx))]]
    timings = []
    v4_timings = []
    for _ in range(10):
        start = time.perf_counter()
        calibrator.predict_proba(sample)
        timings.append((time.perf_counter() - start) * 1000 / len(sample))
        start = time.perf_counter()
        v4_model.predict_proba(sample)
        v4_timings.append((time.perf_counter() - start) * 1000 / len(sample))
    encoder_timings = []
    latency_texts = frame.text.iloc[validation_idx[:min(32, len(validation_idx))]].tolist()
    for _ in range(5):
        start = time.perf_counter()
        encoder.encode(latency_texts, batch_size=32, show_progress_bar=False,
                       normalize_embeddings=True, convert_to_numpy=True)
        encoder_timings.append((time.perf_counter() - start) * 1000 / len(latency_texts))
    locked_path = DATA / "locked_test_v2.csv"
    locked = pd.read_csv(locked_path)
    locked["text"] = locked["text"].fillna("").astype(str).map(lambda t: re.sub(r"\s+", " ", t).strip())
    locked["label"] = locked["label"].astype(str).str.strip().str.lower()
    exact_locked_overlap = len(set(map(text_key, frame.text)) & set(map(text_key, locked.text)))
    if exact_locked_overlap:
        raise RuntimeError(f"development/locked-test exact text overlap: {exact_locked_overlap}")
    combined_groups, cross_duplicate_info = near_duplicate_groups(
        frame.text.tolist() + locked.text.tolist()
    )
    development_group_ids = set(combined_groups[:len(frame)])
    test_group_ids = set(combined_groups[len(frame):])
    shared_text_family_groups = development_group_ids & test_group_ids
    cross_duplicate_info["cross_partition_shared_groups"] = len(shared_text_family_groups)
    cross_duplicate_info["development_records_in_shared_groups"] = int(
        sum(group in shared_text_family_groups for group in combined_groups[:len(frame)])
    )
    cross_duplicate_info["locked_test_records_in_shared_groups"] = int(
        sum(group in shared_text_family_groups for group in combined_groups[len(frame):])
    )
    locked_embeddings = encoder.encode(locked.text.tolist(), batch_size=32, show_progress_bar=False,
                                      normalize_embeddings=True, convert_to_numpy=True)
    locked_v4_pred = v4_model.predict(locked_embeddings)
    locked_v5_prob = aligned_proba(calibrator, locked_embeddings)
    locked_v5_pred = np.asarray(LABELS)[locked_v5_prob.argmax(axis=1)]

    report = {
        "run_id": timestamp,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__,
        "sentence_transformers": __import__("sentence_transformers").__version__,
        "seed": SEED, "checkpoint": CHECKPOINT, "embedding_dimension": int(embeddings.shape[1]),
        "embedding_normalization": "L2 normalized", "source_file": source.name, "source_sha256": source_hash,
        "source_records_after_conflict_filter": int(len(frame)),
        "excluded_conflicting_text_families": int(len(conflict_keys)),
        "class_counts": frame.label.value_counts().reindex(LABELS, fill_value=0).to_dict(),
        "split_counts": {name: int(len(indices)) for name, indices in partitions.items()},
        "split_class_counts": {name: frame.iloc[indices].label.value_counts().reindex(LABELS, fill_value=0).to_dict()
                               for name, indices in partitions.items()},
        "duplicate_audit": duplicate_info,
        "leakage_check": "no exact or >=0.90 char-TFIDF duplicate component crosses fit/validation/calibration",
        "participant_or_session_ids": "unavailable; split is text-family grouped only",
        "v4_model_sha256": v4_metadata["artifact_sha256"],
        "candidate_results_validation": candidates,
        "v4_same_split_validation": v4_metrics,
        "selected_candidate": winner_name,
        "calibrated_selected_validation": calibrated_metrics,
        "calibration_metrics_validation": calibration_metrics,
        "frozen_benchmark": {
            "dataset": locked_path.name,
            "sha256": sha256(locked_path),
            "rows": int(len(locked)),
            "exact_overlap_with_development": exact_locked_overlap,
            "near_duplicate_audit": cross_duplicate_info,
            "prior_exposure_note": "Previously exposed Stage 4 benchmark; scored during multiple procedural reruns while correcting the validation methodology and adding reliability outputs. Benchmark values were not used for fitting, calibration, candidate selection, threshold choice, or configuration changes.",
            "v4_metrics": model_metrics(locked.label.to_numpy(), locked_v4_pred),
            "v5_metrics": model_metrics(locked.label.to_numpy(), locked_v5_pred),
        },
        "classifier_latency_ms_per_record_median_excluding_encoder": float(np.median(timings)),
        "v4_classifier_latency_ms_per_record_median_excluding_encoder": float(np.median(v4_timings)),
        "embedding_latency_ms_per_record_median": float(np.median(encoder_timings)),
        "estimated_v5_end_to_end_latency_ms_per_record": float(np.median(timings) + np.median(encoder_timings)),
        "estimated_v4_end_to_end_latency_ms_per_record": float(np.median(v4_timings) + np.median(encoder_timings)),
        "v5_artifact": artifact_path.name,
        "v5_artifact_sha256": artifact_hash,
        "model_activation": "not active; v4 remains the production default",
        "frozen_evaluation_sets_used_for_selection": False,
    }
    validation_output = frame.iloc[validation_idx][["text", "label"]].copy()
    validation_output["record_sha256"] = validation_output.text.map(
        lambda value: hashlib.sha256(text_key(value).encode("utf-8")).hexdigest()
    )
    validation_output = validation_output.drop(columns="text")
    validation_output["v4_prediction"] = v4_pred
    validation_output["v5_prediction"] = calibrated_pred
    validation_output.to_csv(run_dir / "validation_predictions.csv", index=False)
    split_rows = []
    for name, indices in partitions.items():
        for index in indices:
            split_rows.append({
                "record_sha256": hashlib.sha256(text_key(frame.text.iloc[index]).encode("utf-8")).hexdigest(),
                "label": frame.label.iloc[index],
                "partition": name,
            })
    pd.DataFrame(split_rows).to_csv(run_dir / "split_manifest.csv", index=False)
    pd.DataFrame(confusion_matrix(y[validation_idx], calibrated_pred, labels=LABELS),
                 index=LABELS, columns=LABELS).to_csv(run_dir / "v5_validation_confusion_matrix.csv")
    pd.DataFrame(reliability_rows).to_csv(run_dir / "v5_validation_reliability.csv", index=False)
    reliability_plot = "not evaluated: matplotlib unavailable"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_rows = [row for row in reliability_rows if row["count"]]
        figure, axis = plt.subplots(figsize=(6, 6))
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
        axis.plot([row["mean_confidence"] for row in plot_rows],
                  [row["accuracy"] for row in plot_rows], marker="o", label="v5 calibrated")
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted confidence",
                 ylabel="Observed accuracy", title="Validation top-label reliability")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(run_dir / "v5_validation_reliability.png", dpi=150)
        plt.close(figure)
        reliability_plot = "generated from validation predictions"
    except ImportError:
        pass
    report["reliability_plot"] = reliability_plot
    (run_dir / "experiment_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["run_id", "split_counts", "candidate_results_validation",
                                              "v4_same_split_validation", "selected_candidate",
                                              "calibrated_selected_validation", "calibration_metrics_validation",
                                              "frozen_benchmark",
                                              "v5_artifact", "v5_artifact_sha256"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
