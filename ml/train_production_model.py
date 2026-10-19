"""
Train and save the production XGBoost model (SMOTE + preprocessor pipeline).

Writes:
  models/baseline_model.joblib      — loaded by predict.py / mlPredictionService.js
  models/xgboost_production.joblib  — explicit production alias
  models/baseline_model_metadata.json
  models/baseline_metrics.json

Run: npm run ml:train
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from xgboost import XGBClassifier

from data_loaders import load_training_data
from feature_schema import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    FEATURE_COLUMNS,
    ORDINAL_FEATURES,
    TARGET_COLUMN,
)
from train_baseline import (
    DEFAULT_N_JOBS,
    build_preprocessor,
    export_global_mdi_importance,
    get_feature_names,
)
from validation_utils import calibration_metrics, youdens_j_threshold


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "baseline_model.joblib"
XGBOOST_ALIAS_PATH = MODELS_DIR / "xgboost_production.joblib"
METADATA_PATH = MODELS_DIR / "baseline_model_metadata.json"
METRICS_PATH = MODELS_DIR / "baseline_metrics.json"
CV_METRICS_PATH = MODELS_DIR / "baseline_cv_metrics.json"
GLOBAL_IMPORTANCE_PATH = MODELS_DIR / "global_feature_importance.json"
SHAP_BACKGROUND_PATH = MODELS_DIR / "shap_background.joblib"
CALIBRATOR_PATH = MODELS_DIR / "probability_calibrator.joblib"
CALIBRATION_EPS = 1e-6
HYBRID_WARNING_THRESHOLD = 20
HYBRID_DANGEROUS_THRESHOLD = 55


def build_pipeline(n_jobs: int = DEFAULT_N_JOBS) -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("smote", SMOTE(random_state=42)),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )


def calibration_design_matrix(raw_prob: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(raw_prob, dtype=float), CALIBRATION_EPS, 1.0 - CALIBRATION_EPS)
    logits = np.log(clipped / (1.0 - clipped))
    return logits.reshape(-1, 1)


def calibrate_probabilities(raw_prob: np.ndarray, labels: pd.Series) -> tuple[LogisticRegression, np.ndarray]:
    calibrator = LogisticRegression(random_state=42, max_iter=1000)
    X_cal = calibration_design_matrix(raw_prob)
    calibrator.fit(X_cal, np.asarray(labels).astype(int))
    calibrated = calibrator.predict_proba(X_cal)[:, 1]
    return calibrator, calibrated


def main() -> None:
    X, y, dataset_names = load_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base_pipeline = build_pipeline(DEFAULT_N_JOBS)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [4, 6, 8],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__subsample": [0.8, 0.9],
    }

    search = GridSearchCV(
        base_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_weighted",
        refit=True,
        n_jobs=DEFAULT_N_JOBS,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = {
        key.replace("classifier__", ""): value for key, value in search.best_params_.items()
    }

    cv_summary = {
        "best_cv_f1_weighted_mean": float(search.best_score_),
        "best_params": best_params,
        "cv_folds": 5,
        "refit_metric": "f1_weighted",
        "search_method": "GridSearchCV",
        "smote": True,
    }

    raw_test_prob = best_model.predict_proba(X_test)[:, 1]
    raw_oof_proba = cross_val_predict(best_model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    calibrator, calibrated_oof_proba = calibrate_probabilities(raw_oof_proba, y_train)
    calibrated_test_prob = calibrator.predict_proba(calibration_design_matrix(raw_test_prob))[:, 1]

    youden_on_train_cv = youdens_j_threshold(y_train, calibrated_oof_proba)
    calibration_on_test = calibration_metrics(y_test, calibrated_test_prob)
    holdout_threshold_reference = calibration_on_test.pop("youdens_j", None)
    y_pred = (calibrated_test_prob >= float(youden_on_train_cv["optimal_threshold"])).astype(int)

    holdout_report = classification_report(
        y_test,
        y_pred,
        target_names=["no_adr", "severe_adr"],
        output_dict=True,
        zero_division=0,
    )
    holdout_report["accuracy"] = float(accuracy_score(y_test, y_pred))
    holdout_report["f1_weighted"] = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    holdout_report["roc_auc"] = float(roc_auc_score(y_test, calibrated_test_prob)) if y.nunique() > 1 else 0.0
    holdout_confusion_matrix = confusion_matrix(y_test, y_pred).tolist()
    holdout_report["probability_calibration"] = {
        "method": "sigmoid",
        "cv_folds": 5,
        "raw_brier_score": calibration_metrics(y_test, raw_test_prob)["brier_score"],
        "calibrated_brier_score": calibration_on_test["brier_score"],
    }

    final_model = build_pipeline(DEFAULT_N_JOBS)
    final_model.set_params(**search.best_params_)
    final_model.fit(X, y)
    full_oof_proba = cross_val_predict(final_model, X, y, cv=cv, method="predict_proba")[:, 1]
    final_calibrator, _ = calibrate_probabilities(full_oof_proba, y)

    global_importance = export_global_mdi_importance(final_model)
    GLOBAL_IMPORTANCE_PATH.write_text(
        json.dumps(
            {
                "method": "xgboost_gain_importance",
                "description": "XGBoost global feature importance (gain-based MDI fallback)",
                "features": global_importance,
            },
            indent=2,
        ),
        encoding="utf8",
    )

    background_size = min(100, len(X))
    background = X.sample(n=background_size, random_state=42) if len(X) > 0 else X
    try:
        joblib.dump(background, SHAP_BACKGROUND_PATH)
    except PermissionError:
        print(f"[warn] Could not write SHAP background to {SHAP_BACKGROUND_PATH}; keeping existing file.")

    metrics_bundle = {
        "evaluation_note": (
            "Binary severe ADR classifier — production XGBoost with GridSearchCV (F1-weighted) "
            "and SMOTE oversampling, with sigmoid probability calibration."
        ),
        "cross_validation_best": cv_summary,
        "holdout_test_20pct": holdout_report,
        "holdout_confusion_matrix": {
            "labels": ["no_adr", "severe_adr"],
            "matrix": holdout_confusion_matrix,
        },
        "validation_protocol": {
            "train_test_split": "80/20 stratified on adr_event",
            "cross_validation": "5-fold StratifiedKFold on training set (GridSearchCV)",
            "threshold_tuning": "Youden's J statistic on calibrated out-of-fold training probabilities",
            "calibration": "Sigmoid calibration fitted on out-of-fold training probabilities; Brier score and reliability diagram on 20% hold-out test set",
        },
        "youdens_j_threshold": youden_on_train_cv,
        "calibration": calibration_on_test,
        "holdout_threshold_reference": {
            "description": (
                "Reference Youden's J computed on hold-out calibrated probabilities for diagnostics only; "
                "not used for runtime classification."
            ),
            "youdens_j": holdout_threshold_reference,
        },
        "target": TARGET_COLUMN,
        "hybrid_inference": {
            "formula": "hybridScore = round(0.6 * ruleScore + 0.4 * mlDangerScore)",
            "mlDangerScore": "adr_risk_probability * 100",
            "thresholds": {
                "warning": HYBRID_WARNING_THRESHOLD,
                "dangerous": HYBRID_DANGEROUS_THRESHOLD,
            },
        },
    }

    metadata = {
        "model_type": "XGBClassifier",
        "model_role": "production",
        "dataset": " + ".join(dataset_names),
        "dataset_sources": dataset_names,
        "training_data": {
            "primary": "FAERS (faers_adrs.csv)" if any("FAERS" in source for source in dataset_names) else None,
            "augmentation": "SMOTE oversampling for minority severe ADR class",
        },
        "rows": int(len(X)),
        "target": TARGET_COLUMN,
        "n_jobs": DEFAULT_N_JOBS,
        "feature_columns": FEATURE_COLUMNS,
        "continuous_features": CONTINUOUS_FEATURES,
        "binary_features": BINARY_FEATURES,
        "ordinal_features": ORDINAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "tuning": {
            "method": "GridSearchCV",
            "cv_folds": 5,
            "scoring": "f1_weighted",
            "smote": True,
            "param_grid": param_grid,
            "best_params": best_params,
        },
        "probability_calibration": {
            "method": "sigmoid",
            "calibrator_path": str(CALIBRATOR_PATH),
            "fit_protocol": "LogisticRegression (Platt-style) on logit-transformed 5-fold out-of-fold training probabilities",
        },
        "output": {
            "adr_risk_probability_range": [0, 1],
            "shap_feature_importance": True,
            "global_mdi_importance_path": str(GLOBAL_IMPORTANCE_PATH),
            "youdens_j_optimal_threshold": youden_on_train_cv.get("optimal_threshold"),
            "production_alias": str(XGBOOST_ALIAS_PATH),
        },
        "threshold_tuning": {
            "method": "youdens_j",
            "description": "Optimal binary threshold from Youden's J on calibrated 5-fold out-of-fold training probabilities",
            "youdens_j": youden_on_train_cv,
        },
        "comparative_study": {
            "selected_over": "random_forest",
            "rationale": "XGBoost achieved highest binary F1 and AUC-ROC on FAERS hold-out (Section 11.1)",
        },
    }

    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(final_calibrator, CALIBRATOR_PATH)
    shutil.copy2(MODEL_PATH, XGBOOST_ALIAS_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf8")
    METRICS_PATH.write_text(json.dumps(metrics_bundle, indent=2), encoding="utf8")
    CV_METRICS_PATH.write_text(json.dumps(cv_summary, indent=2), encoding="utf8")

    report_path = MODELS_DIR / "baseline_feature_report.txt"
    report_path.write_text(
        "\n".join(
            [
                "Production XGBoost training complete (GridSearchCV + SMOTE + 5-fold CV + sigmoid calibration).",
                f"Dataset: {' + '.join(dataset_names)}",
                f"Rows: {len(X)}",
                f"Target: {TARGET_COLUMN} (binary severe ADR)",
                f"Best CV F1 (weighted): {cv_summary['best_cv_f1_weighted_mean']:.4f}",
                f"Hold-out accuracy (20%): {holdout_report['accuracy']:.4f}",
                f"Hold-out F1 (weighted): {holdout_report['f1_weighted']:.4f}",
                f"Hold-out ROC-AUC: {holdout_report['roc_auc']:.4f}",
                f"Youden's J optimal threshold: {youden_on_train_cv.get('optimal_threshold', 0.5):.4f}",
                f"Brier score (hold-out, calibrated): {calibration_on_test.get('brier_score', 0):.4f}",
                f"Best params: {best_params}",
                "",
                (
                    "Hybrid inference: 0.6 * ruleScore + 0.4 * (P(ADR) * 100); "
                    f"thresholds {HYBRID_WARNING_THRESHOLD}/{HYBRID_DANGEROUS_THRESHOLD}."
                ),
            ]
        ),
        encoding="utf8",
    )

    print(f"[ML] Production model saved -> {MODEL_PATH}")
    print(f"[ML] XGBoost alias -> {XGBOOST_ALIAS_PATH}")
    print(f"[ML] Best CV F1 (weighted): {cv_summary['best_cv_f1_weighted_mean']:.4f}")
    print(f"[ML] Hold-out ROC-AUC: {holdout_report['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
