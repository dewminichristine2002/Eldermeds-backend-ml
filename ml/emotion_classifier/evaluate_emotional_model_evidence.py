"""Reproduce ElderMeds domain-test evidence for the frozen production v4 model.

This script performs inference only. It verifies the production model and frozen
evaluation-set checksums before loading the normal production runtime.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from model_runtime import V4_VERSION, load_runtime


MODULE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODULE_DIR.parents[2]
OUTPUT_DIR = REPOSITORY_ROOT / "evidence" / "emotional_model"
MODEL_METADATA_PATH = MODULE_DIR / "production_model_metadata_v4.json"
MODEL_ARTIFACT_PATH = MODULE_DIR / "domain_hierarchical_experiment" / "candidate_model.joblib"
TEST_PATH = MODULE_DIR / "advanced_experiment" / "domain_test_frozen.csv"
TEST_METADATA_PATH = MODULE_DIR / "advanced_experiment" / "domain_test_metadata.json"
TRAINING_SPLIT_PATH = MODULE_DIR / "domain_hierarchical_experiment" / "domain_development_split.csv"

LABELS = [
    "happiness",
    "sadness",
    "loneliness",
    "anxiety",
    "anger",
    "cognitive_fog",
    "neutral",
]
DISPLAY_LABELS = [
    "Happiness",
    "Sadness",
    "Loneliness",
    "Anxiety",
    "Anger",
    "Cognitive Fog",
    "Neutral",
]
EVALUATION_SET = "ElderMeds Domain Test"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_frame_hash(frame: pd.DataFrame) -> str:
    payload = frame.sort_values(["text", "label", "source", "domain"]).to_csv(
        index=False, lineterminator="\n"
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_inputs(test: pd.DataFrame, model_metadata: dict, test_metadata: dict) -> dict:
    required = {"text", "label", "source", "domain"}
    if set(test.columns) != required:
        raise RuntimeError(f"Frozen test columns differ from {sorted(required)}")
    if model_metadata.get("model_version") != V4_VERSION:
        raise RuntimeError("Production metadata does not select v4")
    artifact_hash = sha256_file(MODEL_ARTIFACT_PATH)
    if artifact_hash != model_metadata.get("artifact_sha256"):
        raise RuntimeError("Production model artifact checksum mismatch")
    test_hash = frozen_frame_hash(test)
    if test_hash != test_metadata.get("sha256_sorted_csv"):
        raise RuntimeError("Frozen ElderMeds domain test checksum mismatch")
    if len(test) != int(test_metadata.get("rows", -1)):
        raise RuntimeError("Frozen test row count does not match its metadata")
    if test.isna().any().any():
        raise RuntimeError("Frozen test contains missing values")
    if test.duplicated().any() or test["text"].duplicated().any():
        raise RuntimeError("Frozen test contains duplicate samples")
    if set(test["label"]) != set(LABELS):
        raise RuntimeError("Frozen test label set does not contain the seven production classes")
    if set(test["domain"]) != {"eldermeds_conversation"}:
        raise RuntimeError("Frozen test contains a non-ElderMeds domain row")
    if int(test_metadata.get("exact_model_data_overlap", -1)) != 0:
        raise RuntimeError("Test metadata does not attest zero model-data overlap")

    development = pd.read_csv(TRAINING_SPLIT_PATH)
    training = development.loc[development["split"] != "calibration"]
    normalized_test = set(test["text"].astype(str).str.strip().str.casefold())
    normalized_training = set(training["text"].astype(str).str.strip().str.casefold())
    exact_training_overlap = normalized_test.intersection(normalized_training)
    if exact_training_overlap:
        raise RuntimeError(
            f"Frozen test has {len(exact_training_overlap)} exact text overlaps with model training rows"
        )

    return {
        "model_artifact_sha256": artifact_hash,
        "evaluation_set_sha256_sorted_csv": test_hash,
        "duplicate_test_samples": 0,
        "exact_training_text_overlap": 0,
        "training_rows_checked": int(len(training)),
        "model_retrained": False,
    }


def specificity_by_class(matrix: np.ndarray) -> list[float]:
    total = int(matrix.sum())
    values = []
    for index in range(len(LABELS)):
        true_positive = int(matrix[index, index])
        false_negative = int(matrix[index, :].sum() - true_positive)
        false_positive = int(matrix[:, index].sum() - true_positive)
        true_negative = total - true_positive - false_negative - false_positive
        denominator = true_negative + false_positive
        values.append(true_negative / denominator if denominator else 0.0)
    return values


def save_confusion_matrix(matrix: np.ndarray, sample_count: int) -> None:
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )
    fig, axis = plt.subplots(figsize=(10.5, 9))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Sample count")
    axis.set(
        xticks=np.arange(len(DISPLAY_LABELS)),
        yticks=np.arange(len(DISPLAY_LABELS)),
        xticklabels=DISPLAY_LABELS,
        yticklabels=DISPLAY_LABELS,
        xlabel="Predicted",
        ylabel="Actual",
    )
    axis.tick_params(axis="x", rotation=35)
    axis.set_title(
        "Emotion Classification Confusion Matrix\n"
        "MiniLM + Logistic Regression v4\n"
        f"ElderMeds Domain Test | n = {sample_count}",
        fontsize=15,
        pad=18,
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]}\n{normalized[row, column]:.0%}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "#172554",
                fontsize=9,
            )
    axis.set_ylim(len(DISPLAY_LABELS) - 0.5, -0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_per_class_metrics(report: pd.DataFrame) -> None:
    positions = np.arange(len(DISPLAY_LABELS))
    width = 0.24
    fig, axis = plt.subplots(figsize=(12, 7.5))
    colors = ["#2563eb", "#0f766e", "#d97706"]
    for offset, (column, label, color) in enumerate(
        zip(["precision", "recall", "f1_score"], ["Precision", "Recall", "F1-score"], colors)
    ):
        bars = axis.bar(
            positions + (offset - 1) * width,
            report[column],
            width,
            label=label,
            color=color,
        )
        for bar in bars:
            value = bar.get_height()
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.012, 0.985),
                f"{value:.2f}",
                ha="center",
                va="top" if value >= 0.98 else "bottom",
                fontsize=8,
            )
    axis.set_ylim(0, 1.0)
    axis.set_ylabel("Score")
    axis.set_xticks(positions, DISPLAY_LABELS, rotation=25, ha="right")
    axis.set_title(
        "Per-Class Emotion Classification Performance\n"
        "MiniLM + Logistic Regression v4 — ElderMeds Domain Test",
        fontsize=15,
        pad=16,
    )
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.25), frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "per_class_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    model_metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    test_metadata = json.loads(TEST_METADATA_PATH.read_text(encoding="utf-8"))
    test = pd.read_csv(TEST_PATH)
    provenance = validate_inputs(test, model_metadata, test_metadata)

    runtime = load_runtime(V4_VERSION)
    if not runtime.ready:
        raise RuntimeError(f"Production model runtime failed to load: {runtime.error}")
    if runtime.model_version != V4_VERSION:
        raise RuntimeError("Runtime loaded a model other than the requested production v4 model")

    probabilities = runtime.predict_proba(test["text"].tolist())
    predicted = np.asarray(runtime.classes_)[np.asarray(probabilities).argmax(axis=1)]
    truth = test["label"].to_numpy()
    matrix = confusion_matrix(truth, predicted, labels=LABELS)
    specificities = specificity_by_class(matrix)
    raw_report = classification_report(
        truth,
        predicted,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    report = pd.DataFrame(
        [
            {
                "class": label,
                "precision": float(raw_report[label]["precision"]),
                "recall": float(raw_report[label]["recall"]),
                "f1_score": float(raw_report[label]["f1-score"]),
                "support": int(raw_report[label]["support"]),
            }
            for label in LABELS
        ]
    )

    sample_count = int(len(test))
    metrics = {
        "model_version": runtime.model_version,
        "evaluation_set": EVALUATION_SET,
        "evaluated_samples": sample_count,
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_precision": float(
            precision_score(truth, predicted, labels=LABELS, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(truth, predicted, labels=LABELS, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(truth, predicted, labels=LABELS, average="macro", zero_division=0)
        ),
        "macro_specificity": float(np.mean(specificities)),
        "specificity_definition": "macro mean of one-vs-rest TN / (TN + FP)",
        "classes": {
            label: {
                "precision": float(report.loc[index, "precision"]),
                "recall": float(report.loc[index, "recall"]),
                "f1_score": float(report.loc[index, "f1_score"]),
                "support": int(report.loc[index, "support"]),
                "specificity": float(specificities[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "validation": provenance,
    }

    if int(matrix.sum()) != sample_count:
        raise RuntimeError("Confusion matrix total does not equal evaluated sample count")
    if int(report["support"].sum()) != sample_count:
        raise RuntimeError("Classification-report support does not equal evaluated sample count")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "emotion_model_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "model_version": metrics["model_version"],
                "evaluation_set": metrics["evaluation_set"],
                "evaluated_samples": metrics["evaluated_samples"],
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "macro_specificity": metrics["macro_specificity"],
            }
        ]
    ).to_csv(OUTPUT_DIR / "emotion_model_metrics.csv", index=False)
    report.to_csv(OUTPUT_DIR / "classification_report.csv", index=False)
    save_confusion_matrix(matrix, sample_count)
    save_per_class_metrics(report)

    print("ElderMeds Emotion Classification Evaluation")
    print(f"Model: {metrics['model_version']}")
    print(f"Evaluation set: {metrics['evaluation_set']}")
    print(f"Evaluated samples: {sample_count}")
    print()
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1-score: {metrics['macro_f1']:.4f}")
    print(f"Macro Specificity: {metrics['macro_specificity']:.4f}")
    print()
    print("Per-Class:")
    for index, display in enumerate(DISPLAY_LABELS):
        row = report.iloc[index]
        print(
            f"{display:<14} Precision {row['precision']:.4f}  "
            f"Recall {row['recall']:.4f}  F1 {row['f1_score']:.4f}  "
            f"Support {int(row['support'])}"
        )
    print()
    print("Confusion matrix saved:")
    print("evidence/emotional_model/confusion_matrix.png")
    print()
    print("Per-class metrics saved:")
    print("evidence/emotional_model/per_class_metrics.png")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Evaluation failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise
