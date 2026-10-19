"""
Objective 7 — simulated deployment cycles with retraining and drift metrics.
Uses existing CSV (no DB schema changes). Writes: models/continuous_learning_simulation.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

from compare_models import make_pipeline, prepare_features
from data_loaders import load_raw_training_data


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
OUTPUT_PATH = MODELS_DIR / "continuous_learning_simulation.json"
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


def train_and_eval(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    X_train, y_train, _, _ = prepare_features(train_df)
    X_test, y_test, _, _ = prepare_features(test_df)
    pipeline = make_pipeline(
        RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=1),
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_true = y_test.map({idx: label for idx, label in enumerate(ADR_CLASSES)})
    y_pred_labels = [ADR_CLASSES[int(v)] for v in y_pred]
    report = classification_report(
        y_true, y_pred_labels, labels=ADR_CLASSES, output_dict=True, zero_division=0
    )
    return {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "f1_weighted": float(f1_score(y_true, y_pred_labels, labels=ADR_CLASSES, average="weighted", zero_division=0)),
        "severe_adr_recall": float(report.get("severe_adr", {}).get("recall", 0)),
    }


def main() -> None:
    df, dataset_names = load_raw_training_data()
    df = df.reset_index(drop=True)

    if "medicine_check_created_at" in df.columns:
        df["_sort_ts"] = pd.to_datetime(df["medicine_check_created_at"], errors="coerce")
        df = df.sort_values("_sort_ts")
    else:
        df = df.sample(frac=1, random_state=42)

    n = len(df)
    cut1 = int(n * 0.4)
    cut2 = int(n * 0.7)
    cycle1 = df.iloc[:cut1]
    cycle2 = df.iloc[cut1:cut2]
    cycle3 = df.iloc[cut2:]

    stages = []

    stage_a = train_and_eval(cycle1, cycle2)
    stage_a["stage"] = "initial_deploy"
    stage_a["description"] = "Train on first 40% of checks; evaluate on next 30%."
    stages.append(stage_a)

    stage_b = train_and_eval(pd.concat([cycle1, cycle2]), cycle3)
    stage_b["stage"] = "after_feedback_retrain"
    stage_b["description"] = "Retrain on first 70% (simulated feedback corpus); evaluate on final 30%."
    stage_b["f1_improvement_vs_initial"] = round(stage_b["f1_weighted"] - stage_a["f1_weighted"], 4)
    stages.append(stage_b)

    psi_interactions = population_stability_index(
        cycle1.get("interaction_count", cycle1.get("ddi_pair_count", pd.Series(0, index=cycle1.index))),
        cycle3.get("interaction_count", cycle3.get("ddi_pair_count", pd.Series(0, index=cycle3.index))),
    )
    psi_risk_score = population_stability_index(
        cycle1.get("risk_score", pd.Series(0, index=cycle1.index)),
        cycle3.get("risk_score", pd.Series(0, index=cycle3.index)),
    )

    payload = {
        "objective": "O7_continuous_learning",
        "datasets": dataset_names,
        "total_rows": n,
        "target": "adr_event",
        "simulation": {
            "cycles": [
                {"name": "cycle_1", "rows": len(cycle1)},
                {"name": "cycle_2", "rows": len(cycle2)},
                {"name": "cycle_3", "rows": len(cycle3)},
            ],
            "stages": stages,
            "drift_metrics": {
                "psi_interaction_count_cycle1_vs_cycle3": psi_interactions,
                "psi_risk_score_cycle1_vs_cycle3": psi_risk_score,
                "drift_flag_psi_gt_0_2": psi_interactions > 0.2 or psi_risk_score > 0.2,
            },
            "retraining_trigger_note": "In production, npm run ml:retrain exports reaction_logs + check history then retrains.",
        },
    }

    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf8")
    print(f"[continuous-learning] Wrote {OUTPUT_PATH}")
    for stage in stages:
        print(f"  {stage['stage']}: F1={stage['f1_weighted']:.4f}")


if __name__ == "__main__":
    main()
