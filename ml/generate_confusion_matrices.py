"""
Generate binary and hybrid 3-class confusion matrices for dissertation figures.

Writes: models/confusion_matrices.json
Prints ASCII tables suitable for screenshots.

Run: npm run ml:confusion-matrices
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from compare_models import adr_proba, prepare_features
from data_loaders import load_raw_training_data
from hybrid_eval import HYBRID_ALPHA, HYBRID_BETA, derive_risk_labels, derive_rule_score, hybrid_predictions


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "baseline_model.joblib"
OUTPUT_PATH = MODELS_DIR / "confusion_matrices.json"

BINARY_LABELS = ["no_adr", "severe_adr"]
HYBRID_LABELS = ["Safe", "Warning", "Dangerous"]


def format_binary_table(matrix: list[list[int]], labels: list[str]) -> str:
    header = f"{'':18} | " + " | ".join(f"pred {label:>10}" for label in labels)
    sep = "-" * len(header)
    rows = [header, sep]
    for i, label in enumerate(labels):
        cells = " | ".join(f"{matrix[i][j]:>14}" for j in range(len(labels)))
        rows.append(f"true {label:12} | {cells}")
    return "\n".join(rows)


def format_hybrid_table(matrix: np.ndarray, labels: list[str]) -> str:
    header = f"{'':14} | " + " | ".join(f"pred {label:>9}" for label in labels)
    sep = "-" * len(header)
    rows = [header, sep]
    for i, label in enumerate(labels):
        cells = " | ".join(f"{int(matrix[i, j]):>13}" for j in range(len(labels)))
        rows.append(f"true {label:>9} | {cells}")
    return "\n".join(rows)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Run npm run ml:train first.")

    df, dataset_names = load_raw_training_data()
    X, y = prepare_features(df)

    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, random_state=42, stratify=y
    )
    X_train = X.loc[train_idx].reset_index(drop=True)
    X_test = X.loc[test_idx].reset_index(drop=True)
    y_train = y.loc[train_idx].reset_index(drop=True)
    y_test = y.loc[test_idx].reset_index(drop=True)
    df_test = df.loc[test_idx].reset_index(drop=True)

    model = joblib.load(MODEL_PATH)
    model.fit(X_train, y_train)
    y_pred_binary = model.predict(X_test)
    y_prob = adr_proba(model, X_test)

    binary_cm = confusion_matrix(y_test, y_pred_binary, labels=[0, 1]).tolist()
    binary_f1 = float(f1_score(y_test, y_pred_binary, average="weighted", zero_division=0))

    rule_scores = derive_rule_score(df_test).reset_index(drop=True)
    ml_scores = pd.Series(np.round(y_prob * 100), index=rule_scores.index)
    y_hybrid_true = derive_risk_labels(df_test).reset_index(drop=True)
    y_hybrid_pred = hybrid_predictions(rule_scores, ml_scores, HYBRID_ALPHA, HYBRID_BETA)
    hybrid_cm = confusion_matrix(y_hybrid_true, y_hybrid_pred, labels=HYBRID_LABELS)
    hybrid_f1 = float(
        f1_score(y_hybrid_true, y_hybrid_pred, labels=HYBRID_LABELS, average="weighted", zero_division=0)
    )

    class_distribution = y_hybrid_true.value_counts().to_dict()
    warning_pct = round(100 * class_distribution.get("Warning", 0) / len(y_hybrid_true), 1)

    payload = {
        "section": "dissertation_confusion_matrices",
        "model": "XGBClassifier (production)",
        "datasets": dataset_names,
        "rows_total": int(len(X)),
        "holdout_rows": int(len(X_test)),
        "split": "80/20 stratified, random_state=42",
        "binary_adr_event": {
            "labels": BINARY_LABELS,
            "confusion_matrix": {
                "rows": BINARY_LABELS,
                "columns": BINARY_LABELS,
                "values": binary_cm,
                "interpretation": {
                    "tn": binary_cm[0][0],
                    "fp": binary_cm[0][1],
                    "fn": binary_cm[1][0],
                    "tp": binary_cm[1][1],
                },
            },
            "f1_weighted": binary_f1,
        },
        "hybrid_3class": {
            "formula": f"round({HYBRID_ALPHA} * ruleScore + {HYBRID_BETA} * mlDangerScore)",
            "thresholds": {"warning": 25, "dangerous": 60},
            "ground_truth": "risk_label_eval (DDI-derived Warning proxy)",
            "labels": HYBRID_LABELS,
            "confusion_matrix": {
                "rows": HYBRID_LABELS,
                "columns": HYBRID_LABELS,
                "values": hybrid_cm.tolist(),
            },
            "f1_weighted": hybrid_f1,
            "class_distribution_holdout": class_distribution,
            "warning_pct_holdout": warning_pct,
        },
        "f1_gap_explanation": (
            "Binary F1 measures severe-ADR vs non-severe classification — the ML training target. "
            "Hybrid 3-class F1 measures Safe/Warning/Dangerous after blending rule scores with ML "
            "probabilities and applying clinical thresholds. The drop is expected: (1) Warning is a "
            "post-hoc DDI proxy with only ~8% of rows, (2) rule-score proxy does not fully match live "
            "P1–P16 engine output, (3) three-class boundary errors (Safe↔Warning) do not affect binary "
            "severe-ADR detection."
        ),
    }

    binary_ascii = format_binary_table(binary_cm, BINARY_LABELS)
    hybrid_ascii = format_hybrid_table(hybrid_cm, HYBRID_LABELS)

    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf8")

    print("=" * 60)
    print("BINARY CONFUSION MATRIX (adr_event, 20% hold-out)")
    print(f"F1 (weighted): {binary_f1:.4f}")
    print("=" * 60)
    print(binary_ascii)
    print()
    print("=" * 60)
    print("HYBRID 3-CLASS CONFUSION MATRIX (risk_label_eval, 20% hold-out)")
    print(f"F1 (weighted): {hybrid_f1:.4f}")
    print(f"Hold-out class distribution: {class_distribution}")
    print("=" * 60)
    print(hybrid_ascii)
    print()
    print(f"[confusion] Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
