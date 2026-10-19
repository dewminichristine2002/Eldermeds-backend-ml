"""
Section 11.1 — comparative study: Random Forest, XGBoost, Logistic Regression.

All models:
  - Same binary adr_event target
  - Same SMOTE + preprocessor pipeline
  - Same 80/20 stratified hold-out and 5-fold CV
  - Same metrics: Accuracy, Precision, Recall, F1, AUC-ROC
  - Each model's probability fed into the same hybrid formula (0.6 rule + 0.4 ML)

Writes: models/compare_models_results.json
        models/compare_{model_name}.joblib
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

from data_loaders import load_raw_training_data
from feature_schema import FEATURE_COLUMNS, TARGET_COLUMN, dataframe_to_features, derive_adr_target
from hybrid_eval import HYBRID_ALPHA, HYBRID_BETA, derive_risk_labels, derive_rule_score, evaluate_hybrid_holdout
from train_baseline import build_preprocessor


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
OUTPUT_PATH = MODELS_DIR / "compare_models_results.json"

ADR_CLASSES = ["no_adr", "severe_adr"]

VIVA_RATIONALE = (
    "Random Forest as an ensemble tree method, XGBoost as a gradient boosting variant, "
    "and Logistic Regression as an interpretable linear baseline — together they represent "
    "a spectrum from simple to complex, allowing meaningful comparison."
)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = dataframe_to_features(df)
    y = derive_adr_target(df)
    return X, y


def make_pipeline(classifier) -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("smote", SMOTE(random_state=42)),
            ("classifier", classifier),
        ]
    )


def build_model_catalog() -> list[tuple[str, ImbPipeline, str]]:
    return [
        (
            "random_forest",
            make_pipeline(
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=1,
                )
            ),
            "sklearn RandomForestClassifier — primary proposed baseline",
        ),
        (
            "xgboost",
            make_pipeline(
                XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=1,
                )
            ),
            "XGBoost gradient boosting — strong tabular comparator",
        ),
        (
            "logistic_regression",
            make_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
            "sklearn LogisticRegression — interpretable linear baseline",
        ),
    ]


def adr_proba(pipeline: ImbPipeline, X: pd.DataFrame) -> np.ndarray:
    proba = pipeline.predict_proba(X)
    classes = list(getattr(pipeline.named_steps["classifier"], "classes_", [0, 1]))
    adr_idx = classes.index(1) if 1 in classes else 1
    return proba[:, adr_idx]


def evaluate_model(
    name: str,
    pipeline: ImbPipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    df_test: pd.DataFrame,
) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=1)
    acc_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=1)

    fitted = pipeline.fit(X_train, y_train)
    y_pred = fitted.predict(X_test)
    y_prob = adr_proba(fitted, X_test)

    holdout_report = classification_report(
        y_test, y_pred, target_names=ADR_CLASSES, output_dict=True, zero_division=0
    )

    roc_auc = float(roc_auc_score(y_test, y_prob)) if y_test.nunique() > 1 else None

    rule_scores = derive_rule_score(df_test).reset_index(drop=True)
    ml_scores = pd.Series(np.round(y_prob * 100), index=rule_scores.index)
    y_hybrid_labels = derive_risk_labels(df_test).reset_index(drop=True)
    hybrid_metrics = evaluate_hybrid_holdout(rule_scores, ml_scores, y_hybrid_labels)

    model_path = MODELS_DIR / f"compare_{name}.joblib"
    joblib.dump(fitted, model_path)

    return {
        "model": name,
        "model_path": str(model_path),
        "cv_f1_weighted_mean": float(np.mean(f1_scores)),
        "cv_f1_weighted_std": float(np.std(f1_scores)),
        "cv_accuracy_mean": float(np.mean(acc_scores)),
        "holdout": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "roc_auc": roc_auc,
            "per_class": holdout_report,
        },
        "hybrid_3class_holdout": hybrid_metrics,
    }


def main() -> None:
    df, dataset_names = load_raw_training_data()
    X, y = prepare_features(df)

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    X_train = X.loc[train_idx].reset_index(drop=True)
    X_test = X.loc[test_idx].reset_index(drop=True)
    y_train = y.loc[train_idx].reset_index(drop=True)
    y_test = y.loc[test_idx].reset_index(drop=True)
    df_test = df.loc[test_idx].reset_index(drop=True)

    results = []
    implementations = []
    for name, pipeline, impl_note in build_model_catalog():
        print(f"[compare] Training & evaluating {name}...")
        row = evaluate_model(name, pipeline, X_train, X_test, y_train, y_test, df_test)
        results.append(row)
        implementations.append({"model": name, "implementation": impl_note})
        h = row["holdout"]
        print(
            f"  binary holdout: acc={h['accuracy']:.4f} f1={h['f1_weighted']:.4f} "
            f"auc={h['roc_auc']:.4f}"
        )
        print(
            f"  hybrid 3-class: f1_weighted={row['hybrid_3class_holdout']['f1_weighted']:.4f}"
        )

    best_binary = max(results, key=lambda r: r["holdout"]["f1_weighted"])
    best_hybrid = max(results, key=lambda r: r["hybrid_3class_holdout"]["f1_weighted"])

    payload = {
        "section": "11.1_algorithm_comparison",
        "viva_rationale": VIVA_RATIONALE,
        "algorithms": ["random_forest", "xgboost", "logistic_regression"],
        "datasets": dataset_names,
        "rows": int(len(X)),
        "target": TARGET_COLUMN,
        "binary_classes": ADR_CLASSES,
        "hybrid_evaluation": {
            "formula": "hybridScore = round(0.6 * ruleScore + 0.4 * mlDangerScore)",
            "alpha": HYBRID_ALPHA,
            "beta": HYBRID_BETA,
            "ground_truth_column": "risk_label_eval",
            "note": "3-class labels are evaluation-only; all models train on binary adr_event.",
        },
        "feature_columns": FEATURE_COLUMNS,
        "validation_protocol": {
            "train_test_split": "80/20 stratified on adr_event",
            "cross_validation": "5-fold StratifiedKFold on training set only",
            "balancing": "SMOTE inside pipeline for all three models equally",
            "holdout_metrics": "Accuracy, Precision, Recall, F1 (weighted + macro), AUC-ROC",
        },
        "implementations": implementations,
        "models": results,
        "best_by_binary_f1_weighted": best_binary["model"],
        "best_by_hybrid_f1_weighted": best_hybrid["model"],
    }

    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf8")
    print(f"[compare] Wrote {OUTPUT_PATH}")
    print(f"[compare] Best binary F1: {best_binary['model']}")
    print(f"[compare] Best hybrid 3-class F1: {best_hybrid['model']}")


if __name__ == "__main__":
    main()
