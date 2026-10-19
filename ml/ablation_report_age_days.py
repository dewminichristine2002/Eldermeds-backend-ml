from __future__ import annotations

import argparse
import json
from pathlib import Path

import train_fda_serious_models as base_training
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from train_fda_serious_models import (
    CLASS_ORDER,
    DATASET_PATH,
    FEATURE_COLUMNS,
    MODELS_DIR,
    SERIOUS_LABEL,
    TARGET_COLUMN,
    build_models,
    cap_numeric_outliers,
    load_and_validate,
)


ABLATION_OUTPUT_JSON = MODELS_DIR / "ablation_report_age_days.json"
FEATURE_TO_REMOVE = "report_age_days"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a separate feature ablation experiment for report_age_days "
            "without changing the main FDA model training pipeline."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to the cleaned FDA adverse event dataset.",
    )
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "random_forest", "xgboost"],
        default="logistic_regression",
        help="Model family to compare with and without report_age_days.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Optional row cap for faster smoke tests.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ABLATION_OUTPUT_JSON,
        help="Where to write the ablation comparison results.",
    )
    return parser.parse_args()


def summarize_metrics(y_true, predictions, probabilities) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, pos_label=SERIOUS_LABEL, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, pos_label=SERIOUS_LABEL, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, pos_label=SERIOUS_LABEL, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=CLASS_ORDER).tolist(),
    }


def build_models_for_feature_set(feature_columns: list[str]) -> dict:
    removed_from_numeric = set(base_training.NUMERIC_COLUMNS) - set(feature_columns)
    removed_from_categorical = set(base_training.CATEGORICAL_COLUMNS) - set(feature_columns)
    removed_from_text = set(base_training.TEXT_COLUMNS) - set(feature_columns)

    original_numeric = list(base_training.NUMERIC_COLUMNS)
    original_categorical = list(base_training.CATEGORICAL_COLUMNS)
    original_text = list(base_training.TEXT_COLUMNS)

    try:
        base_training.NUMERIC_COLUMNS = [
            column for column in original_numeric if column in feature_columns
        ]
        base_training.CATEGORICAL_COLUMNS = [
            column for column in original_categorical if column in feature_columns
        ]
        base_training.TEXT_COLUMNS = [
            column for column in original_text if column in feature_columns
        ]
        return build_models()
    finally:
        base_training.NUMERIC_COLUMNS = original_numeric
        base_training.CATEGORICAL_COLUMNS = original_categorical
        base_training.TEXT_COLUMNS = original_text


def fit_and_evaluate(model_name: str, feature_columns: list[str], X_train, X_test, y_train, y_test) -> dict:
    models = build_models_for_feature_set(feature_columns)
    if model_name not in models:
        raise ValueError(f"Requested model '{model_name}' is not available in this environment.")
    fitted = models[model_name].fit(X_train[feature_columns], y_train)
    predictions = fitted.predict(X_test[feature_columns])
    probabilities = fitted.predict_proba(X_test[feature_columns])[:, 1]
    return summarize_metrics(y_test, predictions, probabilities)


def compute_metric_deltas(full_metrics: dict, ablated_metrics: dict) -> dict:
    tracked_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    return {
        metric: float(ablated_metrics[metric] - full_metrics[metric])
        for metric in tracked_metrics
    }


def main() -> None:
    args = parse_args()

    df, summary = load_and_validate(args.dataset, args.sample_size)
    y = df[TARGET_COLUMN].astype(int).copy()

    full_feature_columns = list(FEATURE_COLUMNS)
    ablated_feature_columns = [column for column in FEATURE_COLUMNS if column != FEATURE_TO_REMOVE]

    if FEATURE_TO_REMOVE not in full_feature_columns:
        raise ValueError(f"Feature '{FEATURE_TO_REMOVE}' is not present in FEATURE_COLUMNS.")

    X = df[full_feature_columns].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )
    X_train, X_test, full_caps = cap_numeric_outliers(X_train, X_test)

    X_full_train = X_train.copy()
    X_full_test = X_test.copy()

    X_ablated_train = X_train.drop(columns=[FEATURE_TO_REMOVE]).copy()
    X_ablated_test = X_test.drop(columns=[FEATURE_TO_REMOVE]).copy()

    full_metrics = fit_and_evaluate(
        model_name=args.model,
        feature_columns=full_feature_columns,
        X_train=X_full_train,
        X_test=X_full_test,
        y_train=y_train,
        y_test=y_test,
    )
    ablated_metrics = fit_and_evaluate(
        model_name=args.model,
        feature_columns=ablated_feature_columns,
        X_train=X_ablated_train,
        X_test=X_ablated_test,
        y_train=y_train,
        y_test=y_test,
    )

    output = {
        "experiment": "report_age_days_feature_ablation",
        "model": args.model,
        "dataset_path": str(args.dataset),
        "rows_used": int(len(df)),
        "sample_size": int(args.sample_size) if args.sample_size else None,
        "target_column": TARGET_COLUMN,
        "removed_feature": FEATURE_TO_REMOVE,
        "decision_rule": (
            "If removing report_age_days produces nearly identical performance, "
            "prefer the simpler 12-feature model."
        ),
        "shared_split": {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_label_distribution": {
                str(key): int(value)
                for key, value in y_train.value_counts().sort_index().to_dict().items()
            },
            "test_label_distribution": {
                str(key): int(value)
                for key, value in y_test.value_counts().sort_index().to_dict().items()
            },
        },
        "full_model_a": {
            "feature_count": len(full_feature_columns),
            "features": full_feature_columns,
            "numeric_caps": full_caps,
            "metrics": full_metrics,
        },
        "ablated_model_b": {
            "feature_count": len(ablated_feature_columns),
            "features": ablated_feature_columns,
            "numeric_caps": full_caps,
            "metrics": ablated_metrics,
        },
        "metric_delta_b_minus_a": compute_metric_deltas(full_metrics, ablated_metrics),
        "training_summary_reference": summary,
    }

    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
