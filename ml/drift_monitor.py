"""
Phase 9 — drift monitoring and retraining triggers.

- Population Stability Index (PSI) on risk_score and interaction_count
- Manual review when F1 drops >2% from baseline CV
- Quarterly retraining schedule on accumulated feedback corpus

Writes: models/drift_monitor_report.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from compare_models import make_pipeline, prepare_features
from data_loaders import load_raw_training_data
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
BASELINE_METRICS_PATH = MODELS_DIR / "baseline_cv_metrics.json"
BASELINE_SNAPSHOT_PATH = MODELS_DIR / "baseline_distribution_snapshot.json"
RETRAIN_STATE_PATH = MODELS_DIR / "retrain_state.json"
DRIFT_REPORT_PATH = MODELS_DIR / "drift_monitor_report.json"
FEEDBACK_PATH = ROOT / "data" / "feedback_anonymized.json"

PSI_THRESHOLD = 0.2
ACCURACY_DROP_THRESHOLD = 0.02
QUARTERLY_DAYS = 90
ADR_CLASSES = ["no_adr", "severe_adr"]


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = pd.to_numeric(expected, errors="coerce").fillna(0)
    actual = pd.to_numeric(actual, errors="coerce").fillna(0)
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1,
    )
    breakpoints[0] -= 1e-6
    breakpoints[-1] += 1e-6
    expected_counts = pd.cut(expected, breakpoints).value_counts(normalize=True, sort=False)
    actual_counts = pd.cut(actual, breakpoints).value_counts(normalize=True, sort=False)
    psi = 0.0
    for bucket in expected_counts.index:
        e = max(float(expected_counts.get(bucket, 0.0)), 1e-6)
        a = max(float(actual_counts.get(bucket, 0.0)), 1e-6)
        psi += (a - e) * np.log(a / e)
    return float(max(0.0, psi))


def load_baseline_f1() -> float:
    if not BASELINE_METRICS_PATH.exists():
        return 0.0
    payload = json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf8"))
    return float(payload.get("best_cv_f1_weighted_mean", 0.0))


def evaluate_current_f1(df: pd.DataFrame) -> float:
    if len(df) < 20:
        return load_baseline_f1()

    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]
    X_train, y_train, _, _ = prepare_features(train_df)
    X_test, y_test, _, _ = prepare_features(test_df)
    pipeline = make_pipeline(
        RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=1),
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_true = y_test.map({idx: label for idx, label in enumerate(ADR_CLASSES)})
    y_pred_labels = [ADR_CLASSES[int(v)] for v in y_pred]
    return float(f1_score(y_true, y_pred_labels, labels=ADR_CLASSES, average="weighted", zero_division=0))


def days_since_last_retrain() -> float | None:
    if not RETRAIN_STATE_PATH.exists():
        return None
    state = json.loads(RETRAIN_STATE_PATH.read_text(encoding="utf8"))
    last = state.get("lastRetrainAt")
    if not last:
        return None
    last_dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    return (now - last_dt).total_seconds() / 86400.0


def save_baseline_snapshot(df: pd.DataFrame) -> None:
    snapshot = {
        "savedAt": datetime.now(timezone.utc).isoformat(),
        "riskScoreMean": float(pd.to_numeric(df.get("risk_score", 0), errors="coerce").fillna(0).mean()),
        "interactionCountMean": float(pd.to_numeric(df.get("interaction_count", 0), errors="coerce").fillna(0).mean()),
        "rowCount": int(len(df)),
    }
    BASELINE_SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2), encoding="utf8")


def compute_psi_vs_baseline(current_df: pd.DataFrame) -> dict:
    if not BASELINE_SNAPSHOT_PATH.exists() or len(current_df) < 10:
        save_baseline_snapshot(current_df)
        return {
            "psi_risk_score": 0.0,
            "psi_interaction_count": 0.0,
            "drift_detected": False,
            "note": "Baseline snapshot initialized",
        }

    baseline_df, _ = load_raw_training_data()
    if len(baseline_df) < 10:
        baseline_df = current_df.iloc[: max(10, len(current_df) // 2)]

    psi_risk = population_stability_index(
        baseline_df.get("risk_score", pd.Series(0, index=baseline_df.index)),
        current_df.get("risk_score", pd.Series(0, index=current_df.index)),
    )
    psi_interactions = population_stability_index(
        baseline_df.get("interaction_count", baseline_df.get("ddi_pair_count", pd.Series(0, index=baseline_df.index))),
        current_df.get("interaction_count", current_df.get("ddi_pair_count", pd.Series(0, index=current_df.index))),
    )

    return {
        "psi_risk_score": round(psi_risk, 4),
        "psi_interaction_count": round(psi_interactions, 4),
        "drift_detected": psi_risk > PSI_THRESHOLD or psi_interactions > PSI_THRESHOLD,
        "psi_threshold": PSI_THRESHOLD,
    }


def main() -> None:
    df, dataset_names = load_raw_training_data()
    baseline_f1 = load_baseline_f1()
    current_f1 = evaluate_current_f1(df)
    f1_drop = round(baseline_f1 - current_f1, 4)
    accuracy_review_required = f1_drop > ACCURACY_DROP_THRESHOLD

    psi_metrics = compute_psi_vs_baseline(df)
    days_since = days_since_last_retrain()
    quarterly_due = days_since is None or days_since >= QUARTERLY_DAYS

    feedback_rows = 0
    if FEEDBACK_PATH.exists():
        feedback_payload = json.loads(FEEDBACK_PATH.read_text(encoding="utf8"))
        feedback_rows = int(feedback_payload.get("counts", {}).get("reactions", 0))

    triggers = {
        "psi_drift_retrain": bool(psi_metrics.get("drift_detected")),
        "accuracy_drop_manual_review": accuracy_review_required,
        "quarterly_scheduled_retrain": quarterly_due,
    }
    should_retrain = any(triggers.values())

    report = {
        "objective": "phase_9_drift_monitoring",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "datasets": dataset_names,
        "metrics": {
            "baseline_cv_f1_weighted": baseline_f1,
            "current_holdout_f1_weighted": current_f1,
            "f1_drop_from_baseline": f1_drop,
            "accuracy_drop_threshold": ACCURACY_DROP_THRESHOLD,
        },
        "psi": psi_metrics,
        "schedule": {
            "quarterly_retrain_days": QUARTERLY_DAYS,
            "days_since_last_retrain": days_since,
            "quarterly_retrain_due": quarterly_due,
        },
        "feedback_corpus": {
            "anonymized_reaction_rows": feedback_rows,
            "feedback_export_path": str(FEEDBACK_PATH),
        },
        "triggers": triggers,
        "recommended_action": "retrain" if should_retrain else "monitor",
        "retrain_command": "npm run ml:retrain",
    }

    MODELS_DIR.mkdir(exist_ok=True)
    DRIFT_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(f"[drift-monitor] Wrote {DRIFT_REPORT_PATH}")
    print(f"  baseline F1={baseline_f1:.4f} current F1={current_f1:.4f} drop={f1_drop:.4f}")
    print(f"  PSI drift={psi_metrics.get('drift_detected')} quarterly_due={quarterly_due}")
    print(f"  recommended_action={report['recommended_action']}")

    if "--force-retrain" in __import__("sys").argv and should_retrain:
        print("[drift-monitor] --force-retrain flag set; invoke retrain_pipeline separately.")


if __name__ == "__main__":
    main()
