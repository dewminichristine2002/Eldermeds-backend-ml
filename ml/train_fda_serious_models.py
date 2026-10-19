from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:
    XGBClassifier = None


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "data" / "fda_adverse_events_2015_2026_CLEAN.csv"
MODELS_DIR = ROOT / "models" / "fda_serious_baselines"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "serious"
TEXT_COLUMNS = ["primary_reaction", "suspect_drug", "drug_indication"]
CATEGORICAL_COLUMNS = ["drug_route", "patient_sex", "country", "pharm_class"]
NUMERIC_COLUMNS = ["year", "month", "num_reactions", "num_drugs", "patient_age_years", "report_age_days"]
FEATURE_COLUMNS = [
    "year",
    "month",
    "primary_reaction",
    "num_reactions",
    "suspect_drug",
    "drug_route",
    "drug_indication",
    "pharm_class",
    "num_drugs",
    "patient_age_years",
    "patient_sex",
    "country",
    "report_age_days",
]
LEAKAGE_COLUMNS = [
    "report_id",
    "receive_date",
    "quarter",
    "serious_flags",
    "is_fatal",
    "is_hospitalized",
    "is_life_threat",
    "is_disabling",
    "reaction_outcomes",
    "patient_recovered",
    "patient_weight_kg",
    "brand_name",
    "manufacturer",
    "reactions",
    "age_group",
    "drug_count_category",
]
TEXT_UNKNOWN = "Unknown"
CLASS_ORDER = [0, 1]
SERIOUS_LABEL = 1
SCORING = {
    "recall_serious": "recall",
    "f1_serious": "f1",
    "roc_auc": "roc_auc",
    "precision_serious": "precision",
    "accuracy": "accuracy",
}
THRESHOLD_GRID = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
MIN_SERIOUS_RECALL_FOR_THRESHOLD = 0.975


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and baseline-train the FDA serious/non-serious report classifier."
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--sample-size", type=int, default=0, help="Optional row cap for smoke tests.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run dataset and split checks only, without fitting models.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=MODELS_DIR / "baseline_metrics.json",
        help="Where to write baseline results.",
    )
    parser.add_argument(
        "--run-cv",
        action="store_true",
        help="Run 5-fold cross-validation on the training split.",
    )
    parser.add_argument(
        "--run-tuning",
        action="store_true",
        help="Run RandomizedSearchCV tuning on the training split, then evaluate best models on the held-out test set.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified folds for CV and tuning.",
    )
    parser.add_argument(
        "--search-iter",
        type=int,
        default=12,
        help="Number of parameter settings sampled per model during RandomizedSearchCV.",
    )
    parser.add_argument(
        "--run-threshold-analysis",
        action="store_true",
        help="Run threshold analysis for the tuned Logistic Regression model using CV predictions from the training split.",
    )
    parser.add_argument(
        "--tuning-results-json",
        type=Path,
        default=None,
        help="Optional path to previously saved tuning results JSON. When provided, threshold analysis can reuse saved Logistic Regression best parameters without rerunning tuning.",
    )
    return parser.parse_args()


def normalize_text_value(value: object) -> str:
    if pd.isna(value):
        return TEXT_UNKNOWN
    text = str(value).strip()
    if not text:
        return TEXT_UNKNOWN
    if text.upper() in {"UNK", "N/A", "NA", "NULL", "NONE", "UNKNOWN"}:
        return TEXT_UNKNOWN
    return text


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    df = df[df[TARGET_COLUMN].notna()].copy()

    for column in TEXT_COLUMNS + CATEGORICAL_COLUMNS:
        df[column] = df[column].map(normalize_text_value)

    df["patient_sex"] = df["patient_sex"].replace(
        {"M": "Male", "F": "Female", "UNK": TEXT_UNKNOWN, "Unknown": TEXT_UNKNOWN}
    )

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["patient_age_years"] = df["patient_age_years"].clip(lower=0, upper=100)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    df = df[df[TARGET_COLUMN].isin(CLASS_ORDER)].copy()

    return df.reset_index(drop=True)


def cap_numeric_outliers(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_df = train_df.copy()
    test_df = test_df.copy()
    caps = {}
    for column in ["num_reactions", "num_drugs", "report_age_days"]:
        cap = float(train_df[column].quantile(0.99))
        caps[column] = cap
        train_df[column] = train_df[column].clip(upper=cap)
        test_df[column] = test_df[column].clip(upper=cap)
    return train_df, test_df, caps


def select_text_column(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].fillna(TEXT_UNKNOWN).astype(str)


def array_to_frame(arr, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(arr, columns=columns)


def text_selector(column: str) -> FunctionTransformer:
    return FunctionTransformer(
        select_text_column,
        validate=False,
        kw_args={"column": column},
    )


def build_logistic_preprocessor() -> ColumnTransformer:
    transformers = [
        (
            "numeric",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            NUMERIC_COLUMNS,
        ),
        (
            "categorical",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            CATEGORICAL_COLUMNS,
        ),
    ]

    for column in TEXT_COLUMNS:
        transformers.append(
            (
                f"tfidf_{column}",
                Pipeline(
                    steps=[
                        ("select", text_selector(column)),
                        ("tfidf", TfidfVectorizer(max_features=4000, ngram_range=(1, 2))),
                    ]
                ),
                [column],
            )
        )

    return ColumnTransformer(transformers=transformers, sparse_threshold=0.3)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str], top_n: int = 200) -> None:
        self.columns = columns
        self.top_n = top_n
        self.frequent_values_: dict[str, set[str]] = {}
        self.frequency_maps_: dict[str, dict[str, float]] = {}

    def fit(self, X: pd.DataFrame, y=None):
        frame = X.copy()
        for column in self.columns:
            values = frame[column].fillna(TEXT_UNKNOWN).astype(str)
            counts = values.value_counts(dropna=False)
            top_values = set(counts.head(self.top_n).index.tolist())
            self.frequent_values_[column] = top_values
            freqs = (counts / len(values)).to_dict()
            self.frequency_maps_[column] = {str(key): float(value) for key, value in freqs.items()}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        encoded = {}
        for column in self.columns:
            values = frame[column].fillna(TEXT_UNKNOWN).astype(str)
            grouped = values.where(values.isin(self.frequent_values_[column]), "__OTHER__")
            freq_map = self.frequency_maps_[column]
            encoded[f"{column}_freq"] = grouped.map(lambda value: freq_map.get(value, 0.0)).astype(float)
        return pd.DataFrame(encoded, index=frame.index)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{column}_freq" for column in self.columns], dtype=object)


def build_tree_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), NUMERIC_COLUMNS),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["drug_route", "patient_sex", "country"],
            ),
            (
                "freq",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=TEXT_UNKNOWN)),
                        (
                            "to_frame",
                            FunctionTransformer(
                                array_to_frame,
                                validate=False,
                                kw_args={"columns": TEXT_COLUMNS + ["pharm_class"]},
                            ),
                        ),
                        ("freq", FrequencyEncoder(TEXT_COLUMNS + ["pharm_class"])),
                    ]
                ),
                TEXT_COLUMNS + ["pharm_class"],
            ),
        ],
        sparse_threshold=0.0,
    )


def build_models() -> dict[str, Pipeline]:
    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", build_logistic_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", build_tree_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        class_weight="balanced_subsample",
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }
    if XGBClassifier is not None:
        models["xgboost"] = Pipeline(
            steps=[
                ("preprocessor", build_tree_preprocessor()),
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
                        n_jobs=1,
                    ),
                ),
            ]
        )
    return models


def build_param_distributions() -> dict[str, dict]:
    distributions = {
        "logistic_regression": {
            "classifier__C": loguniform(1e-2, 10),
            "classifier__class_weight": [None, "balanced", {0: 1, 1: 2}, {0: 1, 1: 3}],
        },
        "random_forest": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [10, 20, 30],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2"],
            "classifier__class_weight": ["balanced", "balanced_subsample"],
        },
    }
    if XGBClassifier is not None:
        distributions["xgboost"] = {
            "classifier__n_estimators": randint(150, 501),
            "classifier__max_depth": randint(3, 11),
            "classifier__learning_rate": loguniform(0.01, 0.3),
            "classifier__min_child_weight": randint(1, 9),
            "classifier__subsample": uniform(0.6, 0.4),
            "classifier__colsample_bytree": uniform(0.6, 0.4),
        }
    return distributions


def summarize_holdout(y_true: pd.Series, predictions: np.ndarray, probabilities: np.ndarray) -> dict:
    report = classification_report(y_true, predictions, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_serious": float(precision_score(y_true, predictions, pos_label=SERIOUS_LABEL, zero_division=0)),
        "recall_serious": float(recall_score(y_true, predictions, pos_label=SERIOUS_LABEL, zero_division=0)),
        "f1_serious": float(f1_score(y_true, predictions, pos_label=SERIOUS_LABEL, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=CLASS_ORDER).tolist(),
        "per_class": report,
    }


def summarize_threshold(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=CLASS_ORDER).ravel()
    return {
        "threshold": float(threshold),
        "serious_precision": float(precision_score(y_true, predictions, pos_label=SERIOUS_LABEL, zero_division=0)),
        "serious_recall": float(recall_score(y_true, predictions, pos_label=SERIOUS_LABEL, zero_division=0)),
        "f1_serious": float(f1_score(y_true, predictions, pos_label=SERIOUS_LABEL, zero_division=0)),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "true_positives": int(tp),
    }


def choose_threshold(threshold_rows: list[dict]) -> dict:
    eligible_rows = [
        row for row in threshold_rows if row["serious_recall"] >= MIN_SERIOUS_RECALL_FOR_THRESHOLD
    ]
    candidate_rows = eligible_rows or threshold_rows
    return max(
        candidate_rows,
        key=lambda row: (
            row["f1_serious"],
            row["serious_precision"],
            row["serious_recall"],
            -row["false_positives"],
        ),
    )


def evaluate_model(model: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict:
    fitted = model.fit(X_train, y_train)
    predictions = fitted.predict(X_test)
    probabilities = fitted.predict_proba(X_test)[:, 1]
    holdout_metrics = summarize_holdout(y_test, predictions, probabilities)
    return {
        "accuracy": holdout_metrics["accuracy"],
        "precision": holdout_metrics["precision_serious"],
        "recall": holdout_metrics["recall_serious"],
        "f1": holdout_metrics["f1_serious"],
        "roc_auc": holdout_metrics["roc_auc"],
        "confusion_matrix": holdout_metrics["confusion_matrix"],
        "per_class": holdout_metrics["per_class"],
    }, fitted


def run_cross_validation(models: dict[str, Pipeline], X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int) -> dict:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {}
    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=1)
        results[name] = {
            metric.replace("test_", ""): {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
            for metric, values in scores.items()
            if metric.startswith("test_")
        }
    return results


def run_hyperparameter_tuning(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int,
    search_iter: int,
    output_json_path: Path | None = None,
    base_output: dict | None = None,
) -> dict:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    param_distributions = build_param_distributions()
    tuning_results = {}
    label_map = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }

    for name, model in models.items():
        print(f"Starting {label_map.get(name, name)} tuning...")
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions[name],
            n_iter=search_iter,
            scoring=SCORING,
            refit="recall_serious",
            cv=cv,
            n_jobs=1,
            random_state=42,
            verbose=1,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        probabilities = best_model.predict_proba(X_test)[:, 1]
        predictions = best_model.predict(X_test)
        holdout = summarize_holdout(y_test, predictions, probabilities)

        tuning_results[name] = {
            "best_params": search.best_params_,
            "best_cv_recall_serious": float(search.best_score_),
            "cv_results_summary": {
                "recall_serious_mean": float(search.cv_results_["mean_test_recall_serious"][search.best_index_]),
                "f1_serious_mean": float(search.cv_results_["mean_test_f1_serious"][search.best_index_]),
                "roc_auc_mean": float(search.cv_results_["mean_test_roc_auc"][search.best_index_]),
                "precision_serious_mean": float(search.cv_results_["mean_test_precision_serious"][search.best_index_]),
                "accuracy_mean": float(search.cv_results_["mean_test_accuracy"][search.best_index_]),
            },
            "holdout_test": holdout,
        }
        joblib.dump(best_model, MODELS_DIR / f"{name}_tuned.joblib")
        if output_json_path is not None and base_output is not None:
            partial_output = dict(base_output)
            partial_output["hyperparameter_tuning"] = dict(tuning_results)
            output_json_path.write_text(json.dumps(partial_output, indent=2), encoding="utf-8")

    return tuning_results


def run_threshold_analysis(
    models: dict[str, Pipeline],
    tuning_results: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int,
    output_json_path: Path | None = None,
    base_output: dict | None = None,
) -> dict:
    if "logistic_regression" not in tuning_results:
        raise ValueError("Threshold analysis requires completed Logistic Regression tuning results.")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    lr_model = models["logistic_regression"].set_params(**tuning_results["logistic_regression"]["best_params"])

    print("Starting Logistic Regression threshold analysis...")
    cv_probabilities = cross_val_predict(
        lr_model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    threshold_rows = [summarize_threshold(y_train, cv_probabilities, threshold) for threshold in THRESHOLD_GRID]
    selected = choose_threshold(threshold_rows)

    final_lr_model = lr_model.fit(X_train, y_train)
    holdout_probabilities = final_lr_model.predict_proba(X_test)[:, 1]
    holdout_summary = summarize_threshold(y_test, holdout_probabilities, selected["threshold"])
    holdout_summary["roc_auc"] = float(roc_auc_score(y_test, holdout_probabilities))

    threshold_payload = {
        "thresholds_evaluated": threshold_rows,
        "selected_threshold": selected,
        "selection_rule": (
            f"Require Serious recall >= {MIN_SERIOUS_RECALL_FOR_THRESHOLD:.3f}, "
            "then maximize F1, then precision, then recall, then minimize false positives."
        ),
        "holdout_test_at_selected_threshold": holdout_summary,
    }

    joblib.dump(
        {"model": final_lr_model, "threshold": selected["threshold"]},
        MODELS_DIR / "logistic_regression_tuned_with_threshold.joblib",
    )

    if output_json_path is not None and base_output is not None:
        partial_output = dict(base_output)
        partial_output["threshold_analysis"] = threshold_payload
        output_json_path.write_text(json.dumps(partial_output, indent=2), encoding="utf-8")

    return threshold_payload


def load_saved_tuning_results(tuning_results_json: Path) -> dict:
    if not tuning_results_json.exists():
        raise FileNotFoundError(f"Tuning results JSON not found: {tuning_results_json}")
    payload = json.loads(tuning_results_json.read_text(encoding="utf-8"))
    tuning_results = payload.get("hyperparameter_tuning")
    if not tuning_results or "logistic_regression" not in tuning_results:
        raise ValueError(
            "Saved tuning results JSON does not contain hyperparameter_tuning.logistic_regression."
        )
    lr_params = tuning_results["logistic_regression"].get("best_params", {})
    class_weight = lr_params.get("classifier__class_weight")
    if isinstance(class_weight, dict):
        lr_params["classifier__class_weight"] = {
            int(key): value for key, value in class_weight.items()
        }
    return tuning_results


def load_and_validate(dataset_path: Path, sample_size: int) -> tuple[pd.DataFrame, dict]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    original_rows = len(df)
    missing_features = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in df.columns]
    if missing_features:
        raise ValueError(f"Dataset is missing required columns: {missing_features}")

    cleaned = clean_frame(df[FEATURE_COLUMNS + [TARGET_COLUMN] + [c for c in LEAKAGE_COLUMNS if c in df.columns]])
    if sample_size and sample_size > 0 and sample_size < len(cleaned):
        sample_fraction = sample_size / len(cleaned)
        sampled_index, _ = train_test_split(
            cleaned.index,
            train_size=sample_fraction,
            stratify=cleaned[TARGET_COLUMN],
            random_state=42,
        )
        cleaned = cleaned.loc[sampled_index].sample(frac=1, random_state=42).reset_index(drop=True)

    label_distribution = cleaned[TARGET_COLUMN].value_counts().sort_index().to_dict()

    summary = {
        "dataset_path": str(dataset_path),
        "rows_loaded": int(original_rows),
        "rows_used": int(len(cleaned)),
        "sample_size": int(sample_size) if sample_size else None,
        "label_distribution": {str(key): int(value) for key, value in label_distribution.items()},
        "leakage_columns_present": [column for column in LEAKAGE_COLUMNS if column in df.columns],
        "feature_columns": FEATURE_COLUMNS,
    }
    return cleaned, summary


def main() -> None:
    args = parse_args()
    df, summary = load_and_validate(args.dataset, args.sample_size)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )
    X_train, X_test, caps = cap_numeric_outliers(X_train, X_test)

    split_summary = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_label_distribution": {str(k): int(v) for k, v in y_train.value_counts().sort_index().to_dict().items()},
        "test_label_distribution": {str(k): int(v) for k, v in y_test.value_counts().sort_index().to_dict().items()},
        "numeric_caps": caps,
    }

    output = {
        "task": "Serious vs Non-serious adverse event report classification",
        "target_column": TARGET_COLUMN,
        "summary": summary,
        "split": split_summary,
    }

    if args.check_only:
        args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(json.dumps(output, indent=2))
        return

    models = build_models()

    metrics = {}
    for name, model in models.items():
        model_metrics, fitted = evaluate_model(model, X_train, X_test, y_train, y_test)
        metrics[name] = model_metrics
        joblib.dump(fitted, MODELS_DIR / f"{name}.joblib")

    output["baseline_metrics"] = metrics

    if args.run_cv:
        output["cross_validation"] = run_cross_validation(models, X_train, y_train, args.cv_folds)

    if args.run_tuning:
        output["hyperparameter_tuning"] = run_hyperparameter_tuning(
            models=models,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            cv_folds=args.cv_folds,
            search_iter=args.search_iter,
            output_json_path=args.output_json,
            base_output=output,
        )

    if args.run_threshold_analysis:
        if "hyperparameter_tuning" not in output:
            if args.tuning_results_json is None:
                raise ValueError(
                    "Run threshold analysis with tuning, or provide --tuning-results-json to reuse saved Logistic Regression tuning results."
                )
            output["hyperparameter_tuning"] = load_saved_tuning_results(args.tuning_results_json)
        output["threshold_analysis"] = run_threshold_analysis(
            models=models,
            tuning_results=output["hyperparameter_tuning"],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            cv_folds=args.cv_folds,
            output_json_path=args.output_json,
            base_output=output,
        )

    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
