from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_loaders import load_training_data
from feature_schema import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    FEATURE_COLUMNS,
    ORDINAL_FEATURES,
    TARGET_COLUMN,
)
from validation_utils import calibration_metrics, youdens_j_threshold


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "baseline_model.joblib"
METADATA_PATH = MODELS_DIR / "baseline_model_metadata.json"
METRICS_PATH = MODELS_DIR / "baseline_metrics.json"
CV_METRICS_PATH = MODELS_DIR / "baseline_cv_metrics.json"
GLOBAL_IMPORTANCE_PATH = MODELS_DIR / "global_feature_importance.json"
SHAP_BACKGROUND_PATH = MODELS_DIR / "shap_background.joblib"
DEFAULT_N_JOBS = int(os.getenv("ML_N_JOBS", "1"))


def build_preprocessor() -> ColumnTransformer:
    continuous_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("continuous", continuous_pipeline, CONTINUOUS_FEATURES),
            ("ordinal", ordinal_pipeline, ORDINAL_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(n_jobs: int = DEFAULT_N_JOBS) -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("smote", SMOTE(random_state=42)),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names: list[str] = []
    names.extend(CONTINUOUS_FEATURES)
    names.extend(ORDINAL_FEATURES)
    names.extend(BINARY_FEATURES)
    cat_transformer = preprocessor.named_transformers_["categorical"]
    onehot: OneHotEncoder = cat_transformer.named_steps["onehot"]
    names.extend(onehot.get_feature_names_out(CATEGORICAL_FEATURES).tolist())
    return names


def export_global_mdi_importance(model, top_n: int = 20) -> list[dict]:
    classifier = model.named_steps["classifier"]
    importances = getattr(classifier, "feature_importances_", None)
    if importances is None:
        return []

    preprocessor = model.named_steps["preprocessor"]
    feature_names = get_feature_names(preprocessor)
    ranked = sorted(
        [
            {"feature": name, "mdi_importance": float(value)}
            for name, value in zip(feature_names, importances)
        ],
        key=lambda item: item["mdi_importance"],
        reverse=True,
    )
    return ranked[:top_n]


def main() -> None:
    X, y, dataset_names = load_training_data()

    base_pipeline = build_pipeline(DEFAULT_N_JOBS)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [5, 10, 15, None],
        "classifier__min_samples_split": [2, 5, 10],
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
    search.fit(X, y)

    best_model = search.best_estimator_
    best_params = {
        key.replace("classifier__", ""): (None if value is None or (isinstance(value, float) and np.isnan(value)) else value)
        for key, value in search.best_params_.items()
    }

    cv_summary = {
        "best_cv_f1_weighted_mean": float(search.best_score_),
        "best_params": best_params,
        "cv_folds": 5,
        "refit_metric": "f1_weighted",
        "search_method": "GridSearchCV",
        "class_weight": "balanced",
        "smote": True,
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    oof_proba = cross_val_predict(best_model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    youden_on_train_cv = youdens_j_threshold(y_train, oof_proba)
    calibration_on_test = calibration_metrics(y_test, y_prob)

    holdout_report = classification_report(
        y_test,
        y_pred,
        target_names=["no_adr", "severe_adr"],
        output_dict=True,
        zero_division=0,
    )
    holdout_report["accuracy"] = float(accuracy_score(y_test, y_pred))
    holdout_report["f1_weighted"] = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    holdout_report["roc_auc"] = float(roc_auc_score(y_test, y_prob)) if y.nunique() > 1 else 0.0
    holdout_confusion_matrix = confusion_matrix(y_test, y_pred).tolist()

    best_model.fit(X, y)

    global_importance = export_global_mdi_importance(best_model)
    GLOBAL_IMPORTANCE_PATH.write_text(
        json.dumps(
            {
                "method": "random_forest_mdi",
                "description": "Mean decrease in impurity (MDI) global feature importance",
                "features": global_importance,
            },
            indent=2,
        ),
        encoding="utf8",
    )

    background_size = min(100, len(X))
    background = X.sample(n=background_size, random_state=42) if len(X) > 0 else X
    joblib.dump(background, SHAP_BACKGROUND_PATH)

    metrics_bundle = {
        "evaluation_note": (
            "Binary severe ADR classifier trained with GridSearchCV (F1-weighted), "
            "class_weight=balanced, and SMOTE oversampling for the minority class."
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
            "threshold_tuning": "Youden's J statistic on out-of-fold training probabilities",
            "calibration": "Brier score and reliability diagram on 20% hold-out test set",
        },
        "youdens_j_threshold": youden_on_train_cv,
        "calibration": calibration_on_test,
        "target": TARGET_COLUMN,
    }

    metadata = {
        "model_type": "RandomForestClassifier",
        "dataset": " + ".join(dataset_names),
        "dataset_sources": dataset_names,
        "training_data": {
            "primary": "FAERS (faers_adrs.csv)" if any("FAERS" in source for source in dataset_names) else None,
            "secondary": "OMOP CDM synthetic (omop_synthetic_adrs.csv)"
            if any("OMOP" in source for source in dataset_names)
            else None,
            "fallback": [name for name in dataset_names if name.endswith(".csv") and "FAERS" not in name and "OMOP" not in name],
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
            "class_weight": "balanced",
            "smote": True,
            "param_grid": param_grid,
            "best_params": best_params,
        },
        "output": {
            "adr_risk_probability_range": [0, 1],
            "shap_feature_importance": True,
            "global_mdi_importance_path": str(GLOBAL_IMPORTANCE_PATH),
            "youdens_j_optimal_threshold": youden_on_train_cv.get("optimal_threshold"),
        },
        "threshold_tuning": {
            "method": "youdens_j",
            "description": "Optimal binary threshold from Youden's J = TPR − FPR on 5-fold out-of-fold training probabilities",
            "youdens_j": youden_on_train_cv,
        },
    }

    joblib.dump(best_model, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf8")
    METRICS_PATH.write_text(json.dumps(metrics_bundle, indent=2), encoding="utf8")
    CV_METRICS_PATH.write_text(json.dumps(cv_summary, indent=2), encoding="utf8")

    report_path = MODELS_DIR / "baseline_feature_report.txt"
    report_path.write_text(
        "\n".join(
            [
                "ADR Random Forest training complete (GridSearchCV + SMOTE + 5-fold CV).",
                f"Dataset: {' + '.join(dataset_names)}",
                f"Rows: {len(X)}",
                f"Target: {TARGET_COLUMN} (binary severe ADR)",
                f"Features: {FEATURE_COLUMNS}",
                f"Best CV F1 (weighted): {cv_summary['best_cv_f1_weighted_mean']:.4f}",
                f"Hold-out accuracy (20%): {holdout_report['accuracy']:.4f}",
                f"Hold-out F1 (weighted): {holdout_report['f1_weighted']:.4f}",
                f"Hold-out ROC-AUC: {holdout_report['roc_auc']:.4f}",
                f"Youden's J optimal threshold: {youden_on_train_cv.get('optimal_threshold', 0.5):.4f}",
                f"Brier score (hold-out): {calibration_on_test.get('brier_score', 0):.4f}",
                f"Best params: {best_params}",
                "",
                "Top global MDI feature importances:",
                *[f"  - {row['feature']}: {row['mdi_importance']:.4f}" for row in global_importance[:10]],
            ]
        ),
        encoding="utf8",
    )

    print(f"[ML] Model saved -> {MODEL_PATH}")
    print(f"[ML] SHAP background -> {SHAP_BACKGROUND_PATH}")
    print(f"[ML] Best CV F1 (weighted): {cv_summary['best_cv_f1_weighted_mean']:.4f}")
    print(f"[ML] Hold-out ROC-AUC: {holdout_report['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
