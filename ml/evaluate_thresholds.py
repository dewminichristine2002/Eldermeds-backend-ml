"""
Section 13.2 — threshold sensitivity for hybrid risk classification.

Evaluates warning/dangerous boundaries at ±5 and ±10 from proposed thresholds (25, 60)
on a held-out test set. Reports precision-recall tradeoff curves and clinical impact
(false negative rate for Dangerous cases).

Writes: models/threshold_sensitivity.json
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split

from data_loaders import load_raw_training_data
from fda_lr_calibration_support import (
    FDA_TUNED_LOGISTIC_MODEL_PATH,
    cap_fda_feature_splits,
    load_or_fit_tuned_fda_logistic,
    ml_scores_from_fda_model,
    project_proxy_frame_to_fda_features,
)
from hybrid_eval import (
    HYBRID_ALPHA,
    HYBRID_BETA,
    MIN_DANGEROUS_PRECISION,
    derive_risk_labels,
    derive_rule_score,
    select_threshold_pair,
)


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
OUTPUT_PATH = MODELS_DIR / "threshold_sensitivity.json"
SELECTED_PATH = MODELS_DIR / "selected_hybrid_thresholds.json"

RISK_CLASSES = ["Safe", "Warning", "Dangerous"]
PROPOSED_WARNING = 25
PROPOSED_DANGEROUS = 60
HYBRID_ALPHA = 0.6
HYBRID_BETA = 0.4

# Section 13.2 — symmetric ±5 / ±10 shifts from proposed (25, 60)
THRESHOLD_SENSITIVITY_CONFIGS = [
    {"configuration": "delta_minus_10", "warning_threshold": 15, "dangerous_threshold": 50, "delta": -10},
    {"configuration": "delta_minus_5", "warning_threshold": 20, "dangerous_threshold": 55, "delta": -5},
    {"configuration": "proposed", "warning_threshold": 25, "dangerous_threshold": 60, "delta": 0},
    {"configuration": "delta_plus_5", "warning_threshold": 30, "dangerous_threshold": 65, "delta": 5},
    {"configuration": "delta_plus_10", "warning_threshold": 35, "dangerous_threshold": 70, "delta": 10},
]


def classify(score: float, warning: int, dangerous: int) -> str:
    if score >= dangerous:
        return "Dangerous"
    if score >= warning:
        return "Warning"
    return "Safe"


def hybrid_scores(rule_scores: pd.Series, ml_scores: pd.Series) -> np.ndarray:
    return np.array(
        [round(HYBRID_ALPHA * float(r) + HYBRID_BETA * float(m)) for r, m in zip(rule_scores, ml_scores)],
        dtype=float,
    )


def evaluate_configuration(
    hybrid: np.ndarray,
    y_true: pd.Series,
    warning: int,
    dangerous: int,
    configuration: str,
    delta: int,
) -> dict:
    y_pred = [classify(float(score), warning, dangerous) for score in hybrid]
    report = classification_report(y_true, y_pred, labels=RISK_CLASSES, output_dict=True, zero_division=0)
    dangerous_metrics = report.get("Dangerous", {})
    recall = float(dangerous_metrics.get("recall", 0))
    precision = float(dangerous_metrics.get("precision", 0))
    support = int(dangerous_metrics.get("support", 0))

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    false_negatives = int(np.sum((y_true_arr == "Dangerous") & (y_pred_arr != "Dangerous")))

    return {
        "configuration": configuration,
        "delta_from_proposed": delta,
        "warning_threshold": warning,
        "dangerous_threshold": dangerous,
        "f1_weighted": float(f1_score(y_true, y_pred, labels=RISK_CLASSES, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=RISK_CLASSES, average="macro", zero_division=0)),
        "dangerous_recall": recall,
        "dangerous_precision": precision,
        "dangerous_false_negative_rate": float(1.0 - recall),
        "dangerous_false_negatives": false_negatives,
        "dangerous_support": support,
        "clinical_impact": {
            "primary_metric": "dangerous_false_negative_rate",
            "description": "Fraction of true Dangerous cases misclassified as Safe or Warning (safety-critical misses).",
            "false_negatives": false_negatives,
            "false_negative_rate": float(1.0 - recall),
        },
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


def build_dangerous_precision_recall_curve(hybrid: np.ndarray, y_true: pd.Series) -> dict:
    """
    Binary Dangerous vs not-Dangerous PR curve using continuous hybrid score.
    Threshold sweep on score axis (0–100) maps to precision/recall tradeoff.
    """
    y_binary = (np.array(y_true) == "Dangerous").astype(int)
    precision, recall, thresholds = precision_recall_curve(y_binary, hybrid)

    points = []
    for idx, threshold in enumerate(thresholds):
        points.append(
            {
                "score_threshold": float(threshold),
                "precision": float(precision[idx + 1]) if idx + 1 < len(precision) else float(precision[-1]),
                "recall": float(recall[idx + 1]) if idx + 1 < len(recall) else float(recall[-1]),
            }
        )

    # Mark proposed dangerous boundary (60) on the curve
    proposed_point = next(
        (p for p in points if abs(p["score_threshold"] - PROPOSED_DANGEROUS) < 0.51),
        None,
    )
    if proposed_point is None and len(points) > 0:
        nearest = min(points, key=lambda p: abs(p["score_threshold"] - PROPOSED_DANGEROUS))
        proposed_point = {**nearest, "note": "nearest score threshold to proposed dangerous=60"}

    return {
        "task": "binary_dangerous_vs_not",
        "score_axis": "hybrid_score_0_100",
        "proposed_dangerous_threshold": PROPOSED_DANGEROUS,
        "operating_point_at_proposed": proposed_point,
        "curve_points": points,
    }


def build_multiclass_threshold_pr_summary(configurations: list[dict]) -> list[dict]:
    """PR-style summary at each Section 13.2 boundary configuration."""
    return [
        {
            "configuration": row["configuration"],
            "delta": row["delta_from_proposed"],
            "dangerous_precision": row["dangerous_precision"],
            "dangerous_recall": row["dangerous_recall"],
            "dangerous_false_negative_rate": row["dangerous_false_negative_rate"],
        }
        for row in configurations
    ]
def build_grid_configurations(hybrid: np.ndarray, y_true: pd.Series) -> list[dict]:
    """Exhaustive grid for automatic T_D* selection (Option C)."""
    rows: list[dict] = []
    for warning in range(15, 36, 5):
        for dangerous in range(50, 71, 1):
            if warning >= dangerous:
                continue
            rows.append(
                evaluate_configuration(
                    hybrid,
                    y_true,
                    warning,
                    dangerous,
                    configuration=f"grid_{warning}_{dangerous}",
                    delta=0,
                )
            )
    return rows


def main() -> None:
    df, dataset_names = load_raw_training_data()
    X = project_proxy_frame_to_fda_features(df)
    y_adr = pd.to_numeric(df["adr_event"], errors="coerce").fillna(0).astype(int)
    y_labels = derive_risk_labels(df.loc[X.index]).reset_index(drop=True)
    rule_scores = derive_rule_score(df.loc[X.index]).reset_index(drop=True)

    valid = y_labels.isin(RISK_CLASSES)
    if not valid.all():
        X = X.loc[valid.values].reset_index(drop=True)
        y_adr = y_adr.loc[valid.values].reset_index(drop=True)
        y_labels = y_labels.loc[valid.values].reset_index(drop=True)
        rule_scores = rule_scores.loc[valid.values].reset_index(drop=True)

    split = train_test_split(
        X,
        y_adr,
        rule_scores,
        y_labels,
        test_size=0.2,
        random_state=42,
        stratify=y_labels,
    )
    X_train, X_test, y_train, _, rule_train, rule_test, _, label_test = split
    X_train, X_test, numeric_caps = cap_fda_feature_splits(X_train.reset_index(drop=True), X_test.reset_index(drop=True))
    y_train = y_train.reset_index(drop=True)
    rule_test = rule_test.reset_index(drop=True)
    label_test = label_test.reset_index(drop=True)

    model = load_or_fit_tuned_fda_logistic(X_train, y_train)
    ml_test = ml_scores_from_fda_model(model, X_test).reset_index(drop=True)

    hybrid = hybrid_scores(rule_test, ml_test)
    configurations = [
        evaluate_configuration(
            hybrid,
            label_test,
            cfg["warning_threshold"],
            cfg["dangerous_threshold"],
            cfg["configuration"],
            cfg["delta"],
        )
        for cfg in THRESHOLD_SENSITIVITY_CONFIGS
    ]
    grid_configurations = build_grid_configurations(hybrid, label_test)
    all_candidates = configurations + grid_configurations
    selected = select_threshold_pair(all_candidates, MIN_DANGEROUS_PRECISION)

    pr_curve = build_dangerous_precision_recall_curve(hybrid, label_test)
    pr_at_boundaries = build_multiclass_threshold_pr_summary(configurations)

    proposed = next(row for row in configurations if row["configuration"] == "proposed")
    legacy = next(row for row in configurations if row["configuration"] == "delta_minus_5")

    payload = {
        "section": "13.2_threshold_sensitivity",
        "objective": "O5_threshold_sensitivity",
        "datasets": dataset_names,
        "rows_total": int(len(X)),
        "rows_heldout_test": int(len(X_test)),
        "validation": {
            "method": "20% stratified held-out test set",
            "hybrid_weights": {"alpha": HYBRID_ALPHA, "beta": HYBRID_BETA},
            "proposed_thresholds": {
                "warning": PROPOSED_WARNING,
                "dangerous": PROPOSED_DANGEROUS,
            },
            "sensitivity_deltas": [-10, -5, 0, 5, 10],
        },
        "evaluation_scope": {
            "includes": [
                "Tuned Logistic Regression Serious-class probability × 100 as mlDangerScore",
                "Hybrid score: round(0.6 × ruleScore + 0.4 × mlDangerScore)",
                "Threshold banding into Safe / Warning / Dangerous",
            ],
            "excludes_not_replayed_offline": [
                "Live P1–P16 clinical rule engine (P1 allergy short-circuit, DDI overrides, etc.)",
                "allergyController.js score floors and ML downgrade blocks",
                "Patient history / questionnaire context from the mobile app",
            ],
            "ground_truth": "risk_label_eval (FAERS proxy: adr_event=1 → Dangerous; DDI-only → Warning)",
            "rule_score_source": "risk_score column if present, else FAERS feature-derived proxy (not live rule engine output)",
            "interpretation": (
                "Dangerous recall here measures offline hybrid+threshold alignment with proxy labels — "
                "NOT the same as binary severe-ADR detection (see baseline_metrics.json) and NOT full "
                "deployed-system recall with clinical overrides."
            ),
        },
        "ml_model_artifact": str(FDA_TUNED_LOGISTIC_MODEL_PATH),
        "fda_feature_projection": {
            "source_dataset": "FAERS/OMOP proxy hybrid-evaluation dataset",
            "target_schema": "FDA serious vs non-serious Logistic Regression feature schema",
            "numeric_caps": numeric_caps,
        },
        "configurations": configurations,
        "grid_search": {
            "warning_range": [15, 20, 25, 30, 35],
            "dangerous_range": "50–70 (step 1)",
            "pairs_evaluated": len(grid_configurations),
            "selection_constraint": f"Precision_D >= {MIN_DANGEROUS_PRECISION}",
            "selection_objective": "minimize Dangerous FNR (equivalently maximize Dangerous recall)",
        },
        "selected_operating_point": {
            "warning_threshold": selected["warning_threshold"],
            "dangerous_threshold": selected["dangerous_threshold"],
            "configuration": selected.get("configuration"),
            "dangerous_false_negative_rate": selected["dangerous_false_negative_rate"],
            "dangerous_false_negatives": selected["dangerous_false_negatives"],
            "dangerous_recall": selected["dangerous_recall"],
            "dangerous_precision": selected["dangerous_precision"],
            "f1_weighted": selected["f1_weighted"],
            "selection_rule": selected["selection_rule"],
            "production_defaults_updated": True,
        },
        "comparison_legacy_proposed": {
            "legacy_candidate_25_60": {
                "warning_threshold": proposed["warning_threshold"],
                "dangerous_threshold": proposed["dangerous_threshold"],
                "dangerous_recall": proposed["dangerous_recall"],
                "dangerous_precision": proposed["dangerous_precision"],
                "dangerous_false_negative_rate": proposed["dangerous_false_negative_rate"],
            },
            "sensitivity_delta_minus_5_20_55": {
                "warning_threshold": legacy["warning_threshold"],
                "dangerous_threshold": legacy["dangerous_threshold"],
                "dangerous_recall": legacy["dangerous_recall"],
                "dangerous_precision": legacy["dangerous_precision"],
                "dangerous_false_negative_rate": legacy["dangerous_false_negative_rate"],
            },
        },
        "precision_recall_tradeoff": {
            "dangerous_binary_curve": pr_curve,
            "operating_points_at_boundary_configs": pr_at_boundaries,
            "clinical_primary_metric": "dangerous_false_negative_rate",
        },
        "proposed_operating_point": {
            "warning_threshold": PROPOSED_WARNING,
            "dangerous_threshold": PROPOSED_DANGEROUS,
            "dangerous_false_negative_rate": proposed["dangerous_false_negative_rate"],
            "dangerous_false_negatives": proposed["dangerous_false_negatives"],
            "dangerous_recall": proposed["dangerous_recall"],
            "dangerous_precision": proposed["dangerous_precision"],
        },
        "note": (
            "Hybrid score = round(0.6 * ruleScore + 0.4 * mlDangerScore). "
            "Ground truth: risk_label_eval (FAERS) or risk_level (legacy). "
            "Rule scores: risk_score column or FAERS feature-derived proxy. "
            "mlDangerScore is the tuned Logistic Regression probability of the Serious class × 100 "
            "on a projected FDA-style feature frame. "
            "FNR = 1 − Dangerous recall (true Dangerous cases predicted Safe or Warning)."
        ),
        "ground_truth_column": "risk_label_eval" if "risk_label_eval" in df.columns else "risk_level",
        "rule_score_source": "risk_score" if "risk_score" in df.columns else "faers_feature_proxy",
    }

    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf8")
    SELECTED_PATH.write_text(
        json.dumps(
            {
                "warningMin": selected["warning_threshold"],
                "dangerousMin": selected["dangerous_threshold"],
                "selected_operating_point": payload["selected_operating_point"],
                "source": str(OUTPUT_PATH),
            },
            indent=2,
        ),
        encoding="utf8",
    )
    print(f"[thresholds] Wrote {OUTPUT_PATH}")
    print(f"[thresholds] Selected production thresholds -> {SELECTED_PATH}")
    print(
        f"  Selected ({selected['warning_threshold']}/{selected['dangerous_threshold']}): "
        f"FNR={selected['dangerous_false_negative_rate']:.4f}, "
        f"recall={selected['dangerous_recall']:.4f}, precision={selected['dangerous_precision']:.4f}"
    )
    print(
        f"  Legacy (25/60): FNR={proposed['dangerous_false_negative_rate']:.4f}, "
        f"recall={proposed['dangerous_recall']:.4f}, precision={proposed['dangerous_precision']:.4f}"
    )
    worst = max(configurations, key=lambda row: row["dangerous_false_negative_rate"])
    print(
        f"  Highest FNR config: {worst['configuration']} ({worst['warning_threshold']}/{worst['dangerous_threshold']}) "
        f"FNR={worst['dangerous_false_negative_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
