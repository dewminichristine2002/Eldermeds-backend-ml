from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split

import train_fda_serious_models as training_module
from train_fda_serious_models import (
    CATEGORICAL_COLUMNS,
    DATASET_PATH,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    TEXT_COLUMNS,
    cap_numeric_outliers,
    load_and_validate,
)


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models" / "fda_serious_baselines"
RESEARCH_EVIDENCE_DIR = ROOT.parent / "Research_Evidence"
EXPLAINABILITY_DIR = RESEARCH_EVIDENCE_DIR / "07_Explainability"

MODEL_BUNDLE_PATH = MODELS_DIR / "logistic_regression_tuned_with_threshold.joblib"
FEATURE_IMPORTANCE_PATH = EXPLAINABILITY_DIR / "feature_importance.png"
SHAP_SUMMARY_PATH = EXPLAINABILITY_DIR / "shap_summary.png"
SHAP_BAR_PATH = EXPLAINABILITY_DIR / "shap_bar.png"
EXAMPLE_EXPLANATIONS_PATH = EXPLAINABILITY_DIR / "example_explanations.json"
SUMMARY_OUTPUT_PATH = EXPLAINABILITY_DIR / "explainability_summary.json"

TOP_FEATURES_TO_PLOT = 20
TOP_EXAMPLE_FEATURES = 10
TOP_SCATTER_FEATURES = 12


def ensure_output_dir() -> None:
    EXPLAINABILITY_DIR.mkdir(parents=True, exist_ok=True)


def register_pickled_training_helpers() -> None:
    main_module = sys.modules.get("__main__")
    if main_module is None:
        return

    for name in [
        "select_text_column",
        "array_to_frame",
        "text_selector",
        "FrequencyEncoder",
        "normalize_text_value",
    ]:
        if not hasattr(main_module, name) and hasattr(training_module, name):
            setattr(main_module, name, getattr(training_module, name))


def load_model_bundle() -> tuple[object, float]:
    if not MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError(f"Missing final FDA model bundle: {MODEL_BUNDLE_PATH}")

    bundle = joblib.load(MODEL_BUNDLE_PATH)
    if not isinstance(bundle, dict) or "model" not in bundle or "threshold" not in bundle:
        raise ValueError(
            "Expected logistic_regression_tuned_with_threshold.joblib to contain {'model', 'threshold'}."
        )
    return bundle["model"], float(bundle["threshold"])


def build_feature_names_from_preprocessor(preprocessor) -> list[str]:
    feature_names: list[str] = []

    feature_names.extend([f"numeric__{column}" for column in NUMERIC_COLUMNS])

    categorical_pipeline = preprocessor.named_transformers_["categorical"]
    onehot = categorical_pipeline.named_steps["onehot"]
    categorical_feature_names = onehot.get_feature_names_out(CATEGORICAL_COLUMNS)
    feature_names.extend([f"categorical__{name}" for name in categorical_feature_names])

    for column in TEXT_COLUMNS:
        tfidf_pipeline = preprocessor.named_transformers_[f"tfidf_{column}"]
        tfidf = tfidf_pipeline.named_steps["tfidf"]
        token_names = tfidf.get_feature_names_out()
        feature_names.extend([f"text__{column}::{token}" for token in token_names])

    return feature_names


def to_dense_array(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def classify_feature_group(feature_name: str) -> str:
    if feature_name.startswith("numeric__"):
        return feature_name.split("__", 1)[1]
    if feature_name.startswith("categorical__"):
        raw_name = feature_name.split("__", 1)[1]
        return raw_name.split("_", 1)[0]
    if feature_name.startswith("text__"):
        raw_name = feature_name.split("__", 1)[1]
        return raw_name.split("::", 1)[0]
    return "other"


def to_json_safe(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_global_feature_frame(feature_names: list[str], coefficients: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    )
    frame["feature_group"] = frame["feature"].map(classify_feature_group)
    return frame


def build_contribution_frame(
    transformed_test,
    feature_names: list[str],
    coefficients: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    if sparse.issparse(transformed_test):
        contributions_matrix = transformed_test.multiply(coefficients).tocsr()
        mean_abs_contribution = np.asarray(np.abs(contributions_matrix).mean(axis=0)).ravel()
        mean_signed_contribution = np.asarray(contributions_matrix.mean(axis=0)).ravel()
    else:
        contributions_matrix = np.asarray(transformed_test) * coefficients
        mean_abs_contribution = np.abs(contributions_matrix).mean(axis=0)
        mean_signed_contribution = contributions_matrix.mean(axis=0)

    contribution_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_contribution": mean_abs_contribution,
            "mean_signed_contribution": mean_signed_contribution,
        }
    )
    contribution_frame["feature_group"] = contribution_frame["feature"].map(classify_feature_group)
    return contribution_frame, contributions_matrix


def plot_feature_importance(global_frame: pd.DataFrame) -> None:
    top_positive = global_frame.nlargest(TOP_FEATURES_TO_PLOT, "coefficient").copy()
    top_positive = top_positive.sort_values("coefficient")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_positive["feature"], top_positive["coefficient"], color="#1b9e77")
    ax.set_title("Top Positive Logistic Coefficients for Serious ADR Prediction")
    ax.set_xlabel("Coefficient value")
    ax.set_ylabel("Transformed feature")
    fig.tight_layout()
    fig.savefig(FEATURE_IMPORTANCE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_shap_style_bar(contribution_frame: pd.DataFrame) -> None:
    top_features = contribution_frame.nlargest(TOP_FEATURES_TO_PLOT, "mean_abs_contribution").copy()
    top_features = top_features.sort_values("mean_abs_contribution")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_features["feature"], top_features["mean_abs_contribution"], color="#d95f02")
    ax.set_title("Global Contribution Bar Plot (SHAP-style proxy)")
    ax.set_xlabel("Mean absolute contribution on holdout set")
    ax.set_ylabel("Transformed feature")
    fig.tight_layout()
    fig.savefig(SHAP_BAR_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_shap_style_summary(
    contribution_frame: pd.DataFrame,
    contributions,
    transformed_test,
) -> None:
    top_features = contribution_frame.nlargest(TOP_SCATTER_FEATURES, "mean_abs_contribution").copy()
    feature_to_index = {feature: index for index, feature in enumerate(contribution_frame["feature"])}

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = None
    for row_index, feature in enumerate(top_features["feature"]):
        feature_index = feature_to_index[feature]
        if sparse.issparse(contributions):
            x_values = contributions[:, feature_index].toarray().ravel()
        else:
            x_values = np.asarray(contributions[:, feature_index]).ravel()

        if sparse.issparse(transformed_test):
            color_values = transformed_test[:, feature_index].toarray().ravel()
        else:
            color_values = np.asarray(transformed_test[:, feature_index]).ravel()

        y_values = np.full_like(x_values, fill_value=row_index, dtype=float)
        jitter = np.random.default_rng(42 + row_index).normal(0, 0.08, size=len(x_values))
        scatter = ax.scatter(
            x_values,
            y_values + jitter,
            c=color_values,
            cmap="coolwarm",
            alpha=0.4,
            s=12,
        )

    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["feature"])
    ax.set_title("Per-row Contribution Summary (SHAP-style proxy)")
    ax.set_xlabel("Signed contribution toward serious prediction")
    ax.set_ylabel("Transformed feature")
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Feature value after preprocessing")
    fig.tight_layout()
    fig.savefig(SHAP_SUMMARY_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def select_example_indices(
    y_true: pd.Series,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> list[tuple[str, int]]:
    examples: list[tuple[str, int]] = []

    true_positive_mask = (y_true.to_numpy() == 1) & (predictions == 1)
    if true_positive_mask.any():
        tp_indices = np.where(true_positive_mask)[0]
        chosen = tp_indices[np.argmax(probabilities[tp_indices])]
        examples.append(("high_confidence_true_positive", int(chosen)))

    false_negative_mask = (y_true.to_numpy() == 1) & (predictions == 0)
    if false_negative_mask.any():
        fn_indices = np.where(false_negative_mask)[0]
        chosen = fn_indices[np.argmin(probabilities[fn_indices])]
        examples.append(("false_negative_example", int(chosen)))

    true_negative_mask = (y_true.to_numpy() == 0) & (predictions == 0)
    if true_negative_mask.any():
        tn_indices = np.where(true_negative_mask)[0]
        chosen = tn_indices[np.argmin(probabilities[tn_indices])]
        examples.append(("high_confidence_true_negative", int(chosen)))

    return examples


def explain_single_prediction(
    label: str,
    row_position: int,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    feature_names: list[str],
    contributions,
) -> dict:
    row = X_test.iloc[row_position]
    if sparse.issparse(contributions):
        row_contributions = contributions.getrow(row_position).toarray().ravel()
    else:
        row_contributions = np.asarray(contributions[row_position]).ravel()
    positive_indices = np.argsort(row_contributions)[-TOP_EXAMPLE_FEATURES:][::-1]
    negative_indices = np.argsort(row_contributions)[:TOP_EXAMPLE_FEATURES]

    return {
        "example_type": label,
        "row_position_in_holdout": int(row_position),
        "true_label": int(y_test.iloc[row_position]),
        "predicted_label": int(predictions[row_position]),
        "predicted_probability_serious": float(probabilities[row_position]),
        "selected_input_snapshot": {
            "year": to_json_safe(row["year"]),
            "month": to_json_safe(row["month"]),
            "primary_reaction": to_json_safe(row["primary_reaction"]),
            "suspect_drug": to_json_safe(row["suspect_drug"]),
            "drug_indication": to_json_safe(row["drug_indication"]),
            "patient_age_years": to_json_safe(row["patient_age_years"]),
            "patient_sex": to_json_safe(row["patient_sex"]),
            "country": to_json_safe(row["country"]),
            "report_age_days": to_json_safe(row["report_age_days"]),
        },
        "top_contributors_toward_serious": [
            {
                "feature": feature_names[index],
                "contribution": float(row_contributions[index]),
            }
            for index in positive_indices
            if row_contributions[index] > 0
        ],
        "top_contributors_toward_non_serious": [
            {
                "feature": feature_names[index],
                "contribution": float(row_contributions[index]),
            }
            for index in negative_indices
            if row_contributions[index] < 0
        ],
    }


def main() -> None:
    ensure_output_dir()
    register_pickled_training_helpers()

    df, dataset_summary = load_and_validate(DATASET_PATH, sample_size=0)
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )
    X_train, X_test, numeric_caps = cap_numeric_outliers(X_train, X_test)

    model, selected_threshold = load_model_bundle()
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = build_feature_names_from_preprocessor(preprocessor)
    coefficients = classifier.coef_[0]
    if len(feature_names) != len(coefficients):
        raise ValueError(
            f"Feature-name count mismatch: {len(feature_names)} names vs {len(coefficients)} coefficients."
        )

    transformed_test = preprocessor.transform(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= selected_threshold).astype(int)

    global_frame = build_global_feature_frame(feature_names, coefficients)
    contribution_frame, contributions = build_contribution_frame(
        transformed_test,
        feature_names,
        coefficients,
    )

    plot_feature_importance(global_frame)
    plot_shap_style_bar(contribution_frame)
    plot_shap_style_summary(contribution_frame, contributions, transformed_test)

    example_payload = [
        explain_single_prediction(
            label=label,
            row_position=row_position,
            X_test=X_test.reset_index(drop=True),
            y_test=y_test.reset_index(drop=True),
            probabilities=probabilities,
            predictions=predictions,
            feature_names=feature_names,
            contributions=contributions,
        )
        for label, row_position in select_example_indices(
            y_true=y_test.reset_index(drop=True),
            probabilities=probabilities,
            predictions=predictions,
        )
    ]

    grouped_contribution = (
        contribution_frame.groupby("feature_group", as_index=False)["mean_abs_contribution"]
        .sum()
        .sort_values("mean_abs_contribution", ascending=False)
    )

    summary_payload = {
        "model_artifact": str(MODEL_BUNDLE_PATH),
        "operating_threshold": selected_threshold,
        "dataset_path": str(DATASET_PATH),
        "holdout_rows": int(len(X_test)),
        "numeric_caps": numeric_caps,
        "explanation_method": {
            "primary_method": "fitted logistic regression coefficient and per-row contribution analysis",
            "shap_status": "not used in this script; SHAP-style plots are coefficient-based proxies built from the deployed pipeline",
            "note": (
                "Because the saved FDA model is a sparse text-plus-tabular Logistic Regression pipeline, "
                "this script derives explanations directly from transformed feature values multiplied by "
                "the fitted coefficients. This preserves alignment with the deployed artifact without retraining."
            ),
        },
        "top_positive_features_for_serious": global_frame.nlargest(15, "coefficient")[
            ["feature", "coefficient", "feature_group"]
        ].to_dict(orient="records"),
        "top_negative_features_for_non_serious": global_frame.nsmallest(15, "coefficient")[
            ["feature", "coefficient", "feature_group"]
        ].to_dict(orient="records"),
        "top_feature_groups_by_mean_absolute_contribution": grouped_contribution.head(10).to_dict(orient="records"),
        "dataset_summary_reference": dataset_summary,
    }

    EXAMPLE_EXPLANATIONS_PATH.write_text(json.dumps(example_payload, indent=2), encoding="utf-8")
    SUMMARY_OUTPUT_PATH.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"[explainability] Wrote {FEATURE_IMPORTANCE_PATH}")
    print(f"[explainability] Wrote {SHAP_SUMMARY_PATH}")
    print(f"[explainability] Wrote {SHAP_BAR_PATH}")
    print(f"[explainability] Wrote {EXAMPLE_EXPLANATIONS_PATH}")
    print(f"[explainability] Wrote {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
