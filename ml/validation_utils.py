"""Section 11.2 — shared validation helpers (Youden's J, reliability bins)."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_curve


def youdens_j_threshold(y_true, y_prob) -> dict:
    """Optimal probability threshold via Youden's J = TPR − FPR on validation probabilities."""
    y = np.asarray(y_true).astype(int)
    prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y)) < 2:
        return {"optimal_threshold": 0.5, "youdens_j": 0.0, "tpr": 0.0, "fpr": 0.0}

    fpr, tpr, thresholds = roc_curve(y, prob)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    return {
        "optimal_threshold": threshold,
        "youdens_j": float(j_scores[best_idx]),
        "tpr": float(tpr[best_idx]),
        "fpr": float(fpr[best_idx]),
    }


def reliability_diagram_bins(y_true, y_prob, n_bins: int = 10) -> list[dict]:
    y = np.asarray(y_true).astype(int)
    prob = np.asarray(y_prob, dtype=float)
    try:
        prob_true, prob_pred = calibration_curve(y, prob, n_bins=n_bins, strategy="quantile")
        return [
            {"mean_predicted_probability": float(p), "observed_positive_fraction": float(t)}
            for p, t in zip(prob_pred, prob_true)
        ]
    except ValueError:
        return []


def calibration_metrics(y_true, y_prob) -> dict:
    y = np.asarray(y_true).astype(int)
    prob = np.asarray(y_prob, dtype=float)
    return {
        "brier_score": float(brier_score_loss(y, prob)),
        "reliability_diagram": reliability_diagram_bins(y, prob),
        "youdens_j": youdens_j_threshold(y, prob),
    }
