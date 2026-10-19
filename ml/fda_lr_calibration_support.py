from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from train_fda_serious_models import (
    FEATURE_COLUMNS,
    build_logistic_preprocessor,
    cap_numeric_outliers,
    normalize_text_value,
)


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
FDA_SERIOUS_MODELS_DIR = MODELS_DIR / "fda_serious_baselines"
FDA_TUNED_LOGISTIC_MODEL_PATH = FDA_SERIOUS_MODELS_DIR / "logistic_regression_tuned.joblib"

FDA_TUNED_LOGISTIC_PARAMS = {
    "classifier__C": 0.014936568554617643,
    "classifier__class_weight": {0: 1, 1: 3},
}

FDA_DEFAULTS = {
    "year": 2021,
    "month": 6,
    "drug_route": "Unknown",
    "drug_indication": "Unknown",
    "country": "Unknown",
    "report_age_days": 0,
}


def build_tuned_fda_logistic_pipeline() -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_logistic_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    return pipeline.set_params(**FDA_TUNED_LOGISTIC_PARAMS)


def _series_or_default(df: pd.DataFrame, column: str, default: object) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def _first_available(row: pd.Series, columns: list[str], fallback: object = "Unknown") -> object:
    for column in columns:
        if column in row.index:
            value = row[column]
            if pd.notna(value) and str(value).strip():
                return value
    return fallback


def project_proxy_frame_to_fda_features(df: pd.DataFrame) -> pd.DataFrame:
    projected = pd.DataFrame(index=df.index)
    projected["year"] = FDA_DEFAULTS["year"]
    projected["month"] = FDA_DEFAULTS["month"]
    projected["primary_reaction"] = df.apply(
        lambda row: _first_available(row, ["reaction_query", "reaction_category"], FDA_DEFAULTS["drug_indication"]),
        axis=1,
    )
    projected["num_reactions"] = pd.to_numeric(
        _series_or_default(df, "reaction_count", 0),
        errors="coerce",
    ).fillna(0)
    projected["suspect_drug"] = df.apply(
        lambda row: _first_available(
            row,
            ["normalized_ingredient", "ingredient_name", "medicine_name", "normalized_drug_name"],
            "Unknown",
        ),
        axis=1,
    )
    projected["drug_route"] = FDA_DEFAULTS["drug_route"]
    projected["drug_indication"] = FDA_DEFAULTS["drug_indication"]
    projected["pharm_class"] = df.apply(
        lambda row: _first_available(row, ["therapeutic_class", "atc_class_encoded"], "Unknown"),
        axis=1,
    )
    projected["num_drugs"] = pd.to_numeric(
        _series_or_default(
            df,
            "num_current_meds",
            _series_or_default(df, "interaction_count", 1),
        ),
        errors="coerce",
    ).fillna(1)
    projected["patient_age_years"] = pd.to_numeric(
        _series_or_default(
            df,
            "patient_age",
            _series_or_default(df, "age", np.nan),
        ),
        errors="coerce",
    )
    projected["patient_sex"] = df.apply(
        lambda row: _first_available(row, ["patient_sex", "gender"], "Unknown"),
        axis=1,
    )
    projected["country"] = FDA_DEFAULTS["country"]
    projected["report_age_days"] = FDA_DEFAULTS["report_age_days"]

    for column in ["primary_reaction", "suspect_drug", "drug_indication", "drug_route", "patient_sex", "country", "pharm_class"]:
        projected[column] = projected[column].map(normalize_text_value)

    projected["patient_sex"] = projected["patient_sex"].replace(
        {"M": "Male", "F": "Female", "male": "Male", "female": "Female"}
    )

    for column in ["year", "month", "num_reactions", "num_drugs", "patient_age_years", "report_age_days"]:
        projected[column] = pd.to_numeric(projected[column], errors="coerce")

    projected["patient_age_years"] = projected["patient_age_years"].clip(lower=0, upper=100)
    return projected[FEATURE_COLUMNS].reset_index(drop=True)


def cap_fda_feature_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    return cap_numeric_outliers(train_df, test_df)


def load_or_fit_tuned_fda_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    if not FDA_TUNED_LOGISTIC_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Existing tuned Logistic Regression model not found: "
            f"{FDA_TUNED_LOGISTIC_MODEL_PATH}"
        )

    print(f"[calibration] Loading existing model: {FDA_TUNED_LOGISTIC_MODEL_PATH}")
    return joblib.load(FDA_TUNED_LOGISTIC_MODEL_PATH)


def ml_scores_from_fda_model(model, X: pd.DataFrame) -> pd.Series:
    probabilities = model.predict_proba(X)
    classes = list(getattr(model, "classes_", getattr(getattr(model, "named_steps", {}).get("classifier", None), "classes_", [0, 1])))
    serious_index = classes.index(1) if 1 in classes else len(classes) - 1
    return pd.Series(np.round(probabilities[:, serious_index] * 100), index=X.index)
