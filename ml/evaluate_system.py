"""
Objective 6 — full technical evaluation bundle.
Writes: models/system_evaluation.json
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from compare_models import prepare_features
from data_loaders import load_raw_training_data
from validation_utils import calibration_metrics


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "baseline_model.joblib"
OUTPUT_PATH = MODELS_DIR / "system_evaluation.json"
ADR_CLASSES = ["no_adr", "severe_adr"]


def adr_probability(model, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = list(getattr(model.named_steps["classifier"], "classes_", []))
    adr_idx = classes.index(1) if 1 in classes else len(classes) - 1
    return proba[:, adr_idx]


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}. Run npm run ml:train first.")

    df, dataset_names = load_raw_training_data()
    X, y = prepare_features(df)
    model = joblib.load(MODEL_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = adr_probability(model, X_test)
    calibration = calibration_metrics(y_test, y_prob)

    from sklearn.metrics import classification_report

    report = classification_report(
        y_test,
        y_pred,
        target_names=ADR_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    payload = {
        "section": "11.2_validation_protocol",
        "datasets": dataset_names,
        "rows": int(len(X)),
        "target": "adr_event",
        "classes": ADR_CLASSES,
        "validation_protocol": {
            "train_test_split": "80/20 stratified on adr_event",
            "cross_validation": "5-fold StratifiedKFold on training set (see train_baseline / compare_models)",
            "threshold_tuning": "Youden's J on out-of-fold training probabilities (baseline_metrics.json)",
            "calibration_evaluation": "Brier score + reliability diagram bins on 20% hold-out",
        },
        "holdout_accuracy": float(accuracy_score(y_test, y_pred)),
        "holdout_f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "holdout_roc_auc": float(roc_auc_score(y_test, y_prob)) if y.nunique() > 1 else 0.0,
        "holdout_mcc": float(matthews_corrcoef(y_test, y_pred)),
        "holdout_brier_score": calibration["brier_score"],
        "youdens_j": calibration["youdens_j"],
        "reliability_diagram": calibration["reliability_diagram"],
        "calibration_bins": calibration["reliability_diagram"],
        "per_class": report,
    }

    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf8")
    print(f"[evaluate] Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
