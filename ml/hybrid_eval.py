"""Shared hybrid scoring helpers for ML evaluation scripts."""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import classification_report, f1_score

RISK_CLASSES = ["Safe", "Warning", "Dangerous"]
THRESHOLDS = {"warning": 20, "dangerous": 55}
HYBRID_ALPHA = 0.6
HYBRID_BETA = 0.4
MIN_DANGEROUS_PRECISION = 0.99


def derive_risk_labels(df: pd.DataFrame) -> pd.Series:
    if "risk_label_eval" in df.columns:
        return df["risk_label_eval"].astype(str)
    if "risk_level" in df.columns:
        return df["risk_level"].astype(str)
    raise ValueError("Dataset needs risk_label_eval or risk_level for hybrid evaluation.")


def derive_rule_score(df: pd.DataFrame) -> pd.Series:
    """Rule-engine proxy score (0–100) from FAERS features when risk_score is absent."""
    if "risk_score" in df.columns:
        return pd.to_numeric(df["risk_score"], errors="coerce").fillna(0).clip(0, 100)

    score = (
        df.get("allergy_severity_max", 0).fillna(0) * 15
        + df.get("ddi_severity_max", 0).fillna(0) * 12
        + df.get("ddi_pair_count", 0).fillna(0).clip(upper=3) * 8
        + df.get("has_renal_disease", 0).fillna(0) * 8
        + df.get("has_hepatic_disease", 0).fillna(0) * 8
        + df.get("has_diabetes", 0).fillna(0) * 5
        + df.get("has_cardiovascular", 0).fillna(0) * 5
        + df.get("has_epilepsy", 0).fillna(0) * 5
        + df.get("nti_drug_flag", 0).fillna(0) * 10
    )
    return score.clip(0, 100).round()


def classify_hybrid_score(score: float, warning: int = THRESHOLDS["warning"], dangerous: int = THRESHOLDS["dangerous"]) -> str:
    if score >= dangerous:
        return "Dangerous"
    if score >= warning:
        return "Warning"
    return "Safe"


def hybrid_predictions(
    rule_scores: pd.Series,
    ml_scores: pd.Series,
    alpha: float = HYBRID_ALPHA,
    beta: float = HYBRID_BETA,
) -> list[str]:
    hybrid_values = [
        round(alpha * float(rule_scores.iloc[i]) + beta * float(ml_scores.iloc[i]))
        for i in range(len(rule_scores))
    ]
    return [classify_hybrid_score(score) for score in hybrid_values]


def evaluate_hybrid_holdout(
    rule_scores: pd.Series,
    ml_scores: pd.Series,
    y_true_labels: pd.Series,
    alpha: float = HYBRID_ALPHA,
    beta: float = HYBRID_BETA,
) -> dict:
    y_pred = hybrid_predictions(rule_scores, ml_scores, alpha, beta)
    report = classification_report(
        y_true_labels,
        y_pred,
        labels=RISK_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "alpha": alpha,
        "beta": beta,
        "thresholds": THRESHOLDS,
        "f1_weighted": float(f1_score(y_true_labels, y_pred, labels=RISK_CLASSES, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true_labels, y_pred, labels=RISK_CLASSES, average="macro", zero_division=0)),
        "per_class": report,
    }


def select_threshold_pair(
    candidates: list[dict],
    min_precision: float = MIN_DANGEROUS_PRECISION,
    preferred_warning: int = 20,
) -> dict:
    """
    Step 1 — T_D* = argmin FNR_D(T)  subject to  Precision_D(T) >= min_precision.
    (Dangerous recall/FNR depends only on T_D, not T_W, for T_W < T_D.)

    Step 2 — T_W = preferred_warning (default 20) when T_W < T_D*, preserving the
    intermediate Warning band; otherwise the highest interpretable T_W below T_D*.
    """
    eligible = [
        row
        for row in candidates
        if float(row.get("dangerous_precision", 0)) >= min_precision
    ]
    if not eligible:
        eligible = sorted(candidates, key=lambda r: float(r.get("dangerous_precision", 0)), reverse=True)

    best_by_dangerous: dict[int, dict] = {}
    for row in eligible:
        t_d = int(row["dangerous_threshold"])
        current = best_by_dangerous.get(t_d)
        if current is None or float(row["dangerous_false_negative_rate"]) < float(
            current["dangerous_false_negative_rate"]
        ):
            best_by_dangerous[t_d] = row

    best_dangerous = min(
        best_by_dangerous.values(),
        key=lambda row: (
            float(row["dangerous_false_negative_rate"]),
            -float(row["dangerous_recall"]),
            int(row["dangerous_threshold"]),
        ),
    )
    t_d = int(best_dangerous["dangerous_threshold"])
    warning_rows = [row for row in eligible if int(row["dangerous_threshold"]) == t_d]
    selected = next(
        (row for row in warning_rows if int(row["warning_threshold"]) == preferred_warning),
        None,
    )
    if selected is None:
        target_w = preferred_warning if preferred_warning < t_d else max(15, t_d - 5)
        selected = min(
            warning_rows,
            key=lambda row: (abs(int(row["warning_threshold"]) - target_w), int(row["warning_threshold"])),
        )

    return {
        **selected,
        "selection_rule": {
            "objective": "minimize Dangerous false-negative rate (FNR_D)",
            "constraint": f"Precision_D >= {min_precision}",
            "latex": r"T_D^* = \arg\min_T FNR_D(T) \text{ s.t. } Precision_D(T) \geq 0.99",
            "warning_rationale": (
                f"After selecting T_D*={t_d}, Warning threshold T_W={selected['warning_threshold']} "
                f"was set to preserve an interpretable intermediate-risk band "
                f"(preferred T_W={preferred_warning} when feasible)."
            ),
        },
    }
