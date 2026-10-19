from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def load_eval_frame(manifest_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(manifest_path)
    required_columns = {"image_path", "count"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {', '.join(sorted(missing))}")

    frame = frame.copy()
    frame["image_path"] = frame["image_path"].astype(str)
    frame["count"] = frame["count"].astype(int)

    if "split" in frame.columns:
        split = frame["split"].astype(str).str.lower()
        eval_frame = frame[split.isin(["val", "valid", "validation", "test"])].copy()
        if not eval_frame.empty:
            frame = eval_frame

    frame = frame[frame["image_path"].map(lambda value: Path(value).exists())].copy()
    if frame.empty:
        raise ValueError("No evaluation rows remain after checking image paths.")
    return frame


def safe_roc_auc(y_true: list[int], scores: list[float]) -> float | None:
    if len(set(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    # Keep Ultralytics settings inside backend/ml instead of user AppData.
    os.environ.setdefault("YOLO_CONFIG_DIR", str((Path.cwd() / ".ultralytics").resolve()))
    from ultralytics import YOLO

    frame = load_eval_frame(args.manifest)
    model = YOLO(str(args.model))

    actual_counts: list[int] = []
    predicted_counts: list[int] = []
    mismatch_scores: list[float] = []

    for _, row in frame.iterrows():
        image = Image.open(row["image_path"]).convert("RGB")
        result = model.predict(
            np.asarray(image),
            imgsz=args.image_size,
            conf=args.confidence,
            iou=args.iou,
            max_det=args.max_det,
            verbose=False,
            device=args.device,
        )[0]

        boxes = result.boxes
        predicted_count = 0 if boxes is None else len(boxes)
        actual_count = int(row["count"])

        if boxes is not None and len(boxes) > 0:
            average_confidence = float(np.mean(boxes.conf.cpu().numpy()))
        else:
            average_confidence = 0.0

        actual_counts.append(actual_count)
        predicted_counts.append(predicted_count)
        mismatch_scores.append(
            average_confidence if predicted_count != args.expected_count else 1.0 - average_confidence
        )

    y_true = [int(actual != args.expected_count) for actual in actual_counts]
    y_pred = [int(predicted != args.expected_count) for predicted in predicted_counts]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0

    output = {
        "model": str(args.model),
        "manifest": str(args.manifest),
        "expected_count": int(args.expected_count),
        "confidence_threshold": float(args.confidence),
        "evaluated_images": int(len(actual_counts)),
        "positive_class": "incorrect_medication_intake",
        "negative_class": "correct_medication_intake",
        "metrics": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": safe_roc_auc(y_true, mismatch_scores),
            "specificity": specificity,
            "count_accuracy": float(accuracy_score(actual_counts, predicted_counts)),
            "confusion_matrix": {
                "labels": ["correct_intake", "incorrect_intake"],
                "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
            },
        },
    }
    return output


def print_summary(output: dict[str, Any]) -> None:
    metrics = output["metrics"]
    confusion = metrics["confusion_matrix"]
    print("YOLO Medication Intake Metrics")
    print(f"Expected count: {output['expected_count']}")
    print(f"Evaluated images: {output['evaluated_images']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall/Sensitivity: {metrics['recall_sensitivity']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "ROC-AUC: n/a")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Count Accuracy: {metrics['count_accuracy']:.4f}")
    print(
        "Confusion Matrix: "
        f"TP={confusion['tp']} TN={confusion['tn']} FP={confusion['fp']} FN={confusion['fn']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO medication intake metrics.")
    parser.add_argument("--model", type=Path, default=Path("runs/detect/pill_detector/weights/best.pt"))
    parser.add_argument("--manifest", type=Path, default=Path("data/public/pills-detection-count-manifest-dose.csv"))
    parser.add_argument("--expected-count", type=int, default=2)
    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = evaluate(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print_summary(output)


if __name__ == "__main__":
    main()
