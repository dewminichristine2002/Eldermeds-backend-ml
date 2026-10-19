"""
Phase 9 — differential privacy noise on aggregated feedback statistics.

Applied before writing aggregate training-update summaries (not on individual rows).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_EPSILON = 1.0


def laplace_noise(value: float, sensitivity: float, epsilon: float = DEFAULT_EPSILON) -> float:
    if epsilon <= 0:
        return float(value)
    scale = sensitivity / epsilon
    return float(value + np.random.laplace(0.0, scale))


def privatize_severity_counts(counts: dict[str, int], epsilon: float = DEFAULT_EPSILON) -> dict[str, float]:
    sensitivity = 1.0
    return {key: round(laplace_noise(float(value), sensitivity, epsilon), 4) for key, value in counts.items()}


def privatize_aggregate_summary(feedback_path: Path, epsilon: float = DEFAULT_EPSILON) -> dict:
    payload = json.loads(feedback_path.read_text(encoding="utf8"))
    reactions = payload.get("reactionFeedback", [])

    severity_counts: dict[str, int] = {}
    pharmacist_confirmed = 0
    override_count = 0

    for row in reactions:
        severity = str(row.get("severity", "unknown")).lower()
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        if row.get("pharmacistConfirmed"):
            pharmacist_confirmed += 1
        if row.get("recordType") == "clinical_override":
            override_count += 1

    private_counts = privatize_severity_counts(severity_counts, epsilon)
    private_pharmacist = round(laplace_noise(float(pharmacist_confirmed), 1.0, epsilon), 4)
    private_overrides = round(laplace_noise(float(override_count), 1.0, epsilon), 4)

    return {
        "source": str(feedback_path),
        "epsilon": epsilon,
        "mechanism": "laplace",
        "rawSeverityCounts": severity_counts,
        "privateSeverityCounts": private_counts,
        "rawPharmacistConfirmedCount": pharmacist_confirmed,
        "privatePharmacistConfirmedCount": max(0.0, private_pharmacist),
        "rawClinicalOverrideCount": override_count,
        "privateClinicalOverrideCount": max(0.0, private_overrides),
        "totalReactions": len(reactions),
    }


def main() -> None:
    feedback_path = ROOT / "data" / "feedback_anonymized.json"
    output_path = ROOT / "models" / "feedback_privacy_summary.json"

    if not feedback_path.exists():
        print("[feedback-privacy] No anonymized feedback file; skipping.")
        return

    epsilon = float(__import__("os").environ.get("FEEDBACK_DP_EPSILON", "1.0"))
    summary = privatize_aggregate_summary(feedback_path, epsilon)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(f"[feedback-privacy] Wrote {output_path}")


if __name__ == "__main__":
    main()
