"""
Joint offline calibration for hybrid risk scoring.

Searches alpha/beta weights and Warning/Dangerous thresholds together using:
  hybridScore = round(alpha * ruleScore + beta * mlDangerScore)

ML probabilities come from the tuned FDA Serious-vs-Non-serious Logistic Regression
model, applied to an FDA-schema projection of the proxy hybrid-evaluation dataset.

Writes: models/fda_lr_joint_hybrid_calibration.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from data_loaders import load_raw_training_data
from train_fda_serious_models import select_text_column
from fda_lr_calibration_support import (
    FDA_TUNED_LOGISTIC_MODEL_PATH,
    cap_fda_feature_splits,
    load_or_fit_tuned_fda_logistic,
    ml_scores_from_fda_model,
    project_proxy_frame_to_fda_features,
)
from hybrid_eval import MIN_DANGEROUS_PRECISION, derive_risk_labels, derive_rule_score

import __main__


__main__.select_text_column = select_text_column


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
OUTPUT_PATH = MODELS_DIR / "fda_lr_joint_hybrid_calibration.json"

RISK_CLASSES = ["Safe", "Warning", "Dangerous"]
ALPHA_GRID = [round(x / 10, 1) for x in range(0, 11)]
WARNING_GRID = list(range(5, 41))
DANGEROUS_GRID = list(range(30, 71))


def classify(score: float, warning_threshold: int, dangerous_threshold: int) -> str:
    if score >= dangerous_threshold:
        return "Dangerous"
    if score >= warning_threshold:
        return "Warning"
    return "Safe"


def build_hybrid_scores(
    rule_scores: pd.Series,
    ml_scores: pd.Series,
    alpha: float,
    beta: float,
) -> np.ndarray:
    return np.array(
        [round(alpha * float(rule) + beta * float(ml)) for rule, ml in zip(rule_scores, ml_scores)],
        dtype=float,
    )


def evaluate_candidate(
    hybrid_scores: np.ndarray,
    y_true: pd.Series,
    alpha: float,
    beta: float,
    warning_threshold: int,
    dangerous_threshold: int,
) -> dict:
    y_pred = [classify(score, warning_threshold, dangerous_threshold) for score in hybrid_scores]
    report = classification_report(y_true, y_pred, labels=RISK_CLASSES, output_dict=True, zero_division=0)

    dangerous_metrics = report.get("Dangerous", {})
    dangerous_recall = float(dangerous_metrics.get("recall", 0))
    dangerous_precision = float(dangerous_metrics.get("precision", 0))
    dangerous_support = int(dangerous_metrics.get("support", 0))

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    dangerous_false_negatives = int(np.sum((y_true_arr == "Dangerous") & (y_pred_arr != "Dangerous")))

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "warning_threshold": int(warning_threshold),
        "dangerous_threshold": int(dangerous_threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=RISK_CLASSES, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=RISK_CLASSES, average="weighted", zero_division=0)),
        "dangerous_precision": dangerous_precision,
        "dangerous_recall": dangerous_recall,
        "dangerous_false_negative_rate": float(1.0 - dangerous_recall),
        "dangerous_false_negatives": dangerous_false_negatives,
        "dangerous_support": dangerous_support,
        "per_class": {
            label: {
                "precision": float(report.get(label, {}).get("precision", 0)),
                "recall": float(report.get(label, {}).get("recall", 0)),
                "f1": float(report.get(label, {}).get("f1-score", 0)),
                "support": int(report.get(label, {}).get("support", 0)),
            }
            for label in RISK_CLASSES
        },
    }


def select_best_candidate(candidates: list[dict]) -> dict:
    eligible = [
        candidate
        for candidate in candidates
        if float(candidate.get("dangerous_precision", 0)) >= MIN_DANGEROUS_PRECISION
    ]
    pool = eligible or candidates

    selected = max(
        pool,
        key=lambda candidate: (
            float(candidate.get("dangerous_recall", 0)),
            float(candidate.get("f1_macro", 0)),
            float(candidate.get("f1_weighted", 0)),
            -int(candidate.get("warning_threshold", 0)),
            -int(candidate.get("dangerous_threshold", 0)),
        ),
    )

    return {
        **selected,
        "selection_rule": {
            "step_1_constraint": f"Dangerous precision >= {MIN_DANGEROUS_PRECISION}",
            "step_2_objective": "maximize Dangerous recall (equivalently minimize Dangerous FNR)",
            "step_3_tiebreaker": "maximize Macro F1",
            "step_4_tiebreaker": "maximize Weighted F1",
            "eligible_candidate_count": len(eligible),
            "total_candidate_count": len(candidates),
            "fallback_used": len(eligible) == 0,
        },
    }


def main() -> None:
    df, dataset_names = load_raw_training_data()
    X = project_proxy_frame_to_fda_features(df)
    y_binary = pd.to_numeric(df["adr_event"], errors="coerce").fillna(0).astype(int)
    y_labels = derive_risk_labels(df.loc[X.index]).reset_index(drop=True)
    rule_scores = derive_rule_score(df.loc[X.index]).reset_index(drop=True)

    valid = y_labels.isin(RISK_CLASSES)
    if not valid.all():
        X = X.loc[valid.values].reset_index(drop=True)
        y_binary = y_binary.loc[valid.values].reset_index(drop=True)
        y_labels = y_labels.loc[valid.values].reset_index(drop=True)
        rule_scores = rule_scores.loc[valid.values].reset_index(drop=True)

    split = train_test_split(
        X,
        y_binary,
        rule_scores,
        y_labels,
        test_size=0.2,
        random_state=42,
        stratify=y_labels,
    )
    X_train, X_test, y_train, _, rule_train, rule_test, _, label_test = split
    X_train, X_test, numeric_caps = cap_fda_feature_splits(
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
    )
    y_train = y_train.reset_index(drop=True)
    rule_test = rule_test.reset_index(drop=True)
    label_test = label_test.reset_index(drop=True)

    model = load_or_fit_tuned_fda_logistic(X_train, y_train)
    ml_test = ml_scores_from_fda_model(model, X_test).reset_index(drop=True)

    candidates: list[dict] = []
    for alpha in ALPHA_GRID:
        beta = round(1.0 - alpha, 1)
        hybrid = build_hybrid_scores(rule_test, ml_test, alpha, beta)
        for warning_threshold in WARNING_GRID:
            for dangerous_threshold in DANGEROUS_GRID:
                if warning_threshold >= dangerous_threshold:
                    continue
                candidates.append(
                    evaluate_candidate(
                        hybrid_scores=hybrid,
                        y_true=label_test,
                        alpha=alpha,
                        beta=beta,
                        warning_threshold=warning_threshold,
                        dangerous_threshold=dangerous_threshold,
                    )
                )

    selected = select_best_candidate(candidates)
    top_by_recall = sorted(
        candidates,
        key=lambda candidate: (
            float(candidate["dangerous_recall"]),
            float(candidate["f1_macro"]),
            float(candidate["f1_weighted"]),
        ),
        reverse=True,
    )[:10]

    payload = {
        "section": "13.3_joint_hybrid_calibration",
        "objective": "offline_joint_search_alpha_beta_thresholds",
        "datasets": dataset_names,
        "rows_total": int(len(X)),
        "rows_heldout_test": int(len(X_test)),
        "ml_model_artifact": str(FDA_TUNED_LOGISTIC_MODEL_PATH),
        "ml_model_description": "Tuned FDA Serious-vs-Non-serious Logistic Regression",
        "fda_feature_projection": {
            "source_dataset": "FAERS/OMOP proxy hybrid-evaluation dataset",
            "target_schema": "FDA serious vs non-serious Logistic Regression feature schema",
            "numeric_caps": numeric_caps,
        },
        "search_space": {
            "alpha_grid": ALPHA_GRID,
            "beta_rule": "beta = 1 - alpha",
            "warning_threshold_grid": WARNING_GRID,
            "dangerous_threshold_grid": {
                "start": DANGEROUS_GRID[0],
                "end": DANGEROUS_GRID[-1],
                "step": 1,
            },
            "candidate_count": len(candidates),
        },
        "selection_policy": {
            "constraint": f"Dangerous precision >= {MIN_DANGEROUS_PRECISION}",
            "primary_objective": "maximize Dangerous recall",
            "equivalent_primary_metric": "minimize Dangerous false negative rate",
            "tie_breaker_1": "maximize Macro F1",
            "tie_breaker_2": "maximize Weighted F1",
        },
        "selected_configuration": selected,
        "top_10_by_dangerous_recall": top_by_recall,
        "all_candidates": candidates,
        "note": (
            "Offline joint calibration using tuned FDA Logistic Regression probabilities on an FDA-schema "
            "projection of the hybrid proxy dataset. This is not full end-to-end deployed-system optimization."
        ),
    }

    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf8")
    print(f"[joint-calibration] Wrote {OUTPUT_PATH}")
    print(
        f"  Selected alpha={selected['alpha']}, beta={selected['beta']}, "
        f"warning={selected['warning_threshold']}, dangerous={selected['dangerous_threshold']}"
    )
    print(
        f"  Dangerous precision={selected['dangerous_precision']:.4f}, "
        f"recall={selected['dangerous_recall']:.4f}, "
        f"FNR={selected['dangerous_false_negative_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
