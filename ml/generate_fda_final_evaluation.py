from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import train_fda_serious_models as training_module
from train_fda_serious_models import (
    CLASS_ORDER,
    DATASET_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    cap_numeric_outliers,
    load_and_validate,
)


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models" / "fda_serious_baselines"
RESEARCH_EVIDENCE_DIR = ROOT.parent / "Research_Evidence"
FINAL_EVAL_DIR = RESEARCH_EVIDENCE_DIR / "06_Final_Evaluation"

MODEL_BUNDLE_PATH = MODELS_DIR / "logistic_regression_tuned_with_threshold.joblib"
TUNING_RESULTS_PATH = MODELS_DIR / "full_cv_tuning_results.json"
SUMMARY_OUTPUT_PATH = FINAL_EVAL_DIR / "final_evaluation_summary.json"
FALSE_NEGATIVE_OUTPUT_PATH = FINAL_EVAL_DIR / "false_negative_analysis.json"


def ensure_evidence_dirs() -> None:
    for relative in [
        "01_Dataset_EDA",
        "02_Preprocessing_Leakage",
        "03_Baseline_Models",
        "04_Hyperparameter_Tuning",
        "05_Model_Selection",
        "06_Final_Evaluation",
        "07_Explainability",
        "08_Hybrid_Validation",
        "09_System_Testing",
        "10_Ablation",
    ]:
        (RESEARCH_EVIDENCE_DIR / relative).mkdir(parents=True, exist_ok=True)


def load_model_and_threshold():
    if not MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"Missing final FDA model bundle: {MODEL_BUNDLE_PATH}"
        )

    bundle = joblib.load(MODEL_BUNDLE_PATH)
    if not isinstance(bundle, dict) or "model" not in bundle or "threshold" not in bundle:
        raise ValueError(
            "Expected logistic_regression_tuned_with_threshold.joblib to contain {'model', 'threshold'}."
        )
    return bundle["model"], float(bundle["threshold"])


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


def load_selected_threshold_details() -> dict:
    if not TUNING_RESULTS_PATH.exists():
        return {}
    payload = json.loads(TUNING_RESULTS_PATH.read_text(encoding="utf-8"))
    return payload.get("threshold_analysis", {})


def main() -> None:
    ensure_evidence_dirs()
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

    model, selected_threshold = load_model_and_threshold()
    threshold_details = load_selected_threshold_details()

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= selected_threshold).astype(int)

    cm = confusion_matrix(y_test, predictions, labels=CLASS_ORDER)
    tn, fp, fn, tp = cm.ravel()
    precision = float(precision_score(y_test, predictions, pos_label=1, zero_division=0))
    recall = float(recall_score(y_test, predictions, pos_label=1, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, probabilities))
    pr_auc = float(average_precision_score(y_test, probabilities))

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Greens")
    plt.title("Confusion Matrix - Final FDA Logistic Model")
    plt.colorbar()
    tick_labels = ["Non-serious", "Serious"]
    plt.xticks([0, 1], tick_labels)
    plt.yticks([0, 1], tick_labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            plt.text(
                col,
                row,
                format(cm[row, col], "d"),
                ha="center",
                va="center",
                color="white" if cm[row, col] > cm.max() / 2 else "black",
            )
    plt.tight_layout()
    plt.savefig(FINAL_EVAL_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, probabilities, ax=ax, name="Tuned Logistic Regression")
    ax.set_title("ROC Curve - Final FDA Logistic Model")
    fig.tight_layout()
    fig.savefig(FINAL_EVAL_DIR / "roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(
        y_test,
        probabilities,
        ax=ax,
        name="Tuned Logistic Regression",
    )
    ax.set_title("Precision-Recall Curve - Final FDA Logistic Model")
    fig.tight_layout()
    fig.savefig(FINAL_EVAL_DIR / "precision_recall_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    false_negative_rows = X_test.loc[y_test.index[predictions == 0]].loc[y_test[predictions == 0] == 1]
    false_negative_profile = {
        "count": int(fn),
        "percentage_of_serious_holdout": float(fn / max(int((y_test == 1).sum()), 1)),
        "top_primary_reactions": false_negative_rows["primary_reaction"].value_counts().head(10).to_dict()
        if not false_negative_rows.empty
        else {},
        "top_suspect_drugs": false_negative_rows["suspect_drug"].value_counts().head(10).to_dict()
        if not false_negative_rows.empty
        else {},
        "age_summary": false_negative_rows["patient_age_years"].describe().to_dict()
        if not false_negative_rows.empty
        else {},
    }

    summary_payload = {
        "model_family": "logistic_regression",
        "operating_configuration": "tuned logistic regression with deployed artifact threshold",
        "model_artifact": str(MODEL_BUNDLE_PATH),
        "deployed_operating_threshold": selected_threshold,
        "historical_threshold_experiment_reference": threshold_details.get("selected_threshold", {}),
        "threshold_documentation_note": (
            "Final evaluation metrics in this file are computed using the threshold stored inside "
            "logistic_regression_tuned_with_threshold.joblib, which is treated as the deployed "
            "system artifact. Any different threshold shown under "
            "historical_threshold_experiment_reference is retained only as an earlier threshold "
            "analysis result and should not be reported as the final operating threshold."
        ),
        "dataset_path": str(DATASET_PATH),
        "rows_used": int(len(df)),
        "holdout_rows": int(len(X_test)),
        "numeric_caps": numeric_caps,
        "metrics": {
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        },
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm.tolist(),
        },
        "clinical_interpretation": {
            "false_negatives_are_high_priority": True,
            "false_negative_count": int(fn),
            "false_positive_count": int(fp),
            "false_negative_rate_over_serious_cases": float(fn / max(int((y_test == 1).sum()), 1)),
        },
        "dataset_summary_reference": dataset_summary,
    }

    SUMMARY_OUTPUT_PATH.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    FALSE_NEGATIVE_OUTPUT_PATH.write_text(json.dumps(false_negative_profile, indent=2), encoding="utf-8")
    print(f"[final-eval] Wrote {SUMMARY_OUTPUT_PATH}")
    print(f"[final-eval] Wrote {FALSE_NEGATIVE_OUTPUT_PATH}")
    print(f"[final-eval] Wrote {FINAL_EVAL_DIR / 'confusion_matrix.png'}")
    print(f"[final-eval] Wrote {FINAL_EVAL_DIR / 'roc_curve.png'}")
    print(f"[final-eval] Wrote {FINAL_EVAL_DIR / 'precision_recall_curve.png'}")


if __name__ == "__main__":
    main()
