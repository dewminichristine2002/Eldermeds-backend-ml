from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from data_loaders import load_raw_training_data
from fda_lr_calibration_support import (
    cap_fda_feature_splits,
    load_or_fit_tuned_fda_logistic,
    project_proxy_frame_to_fda_features,
)


ROOT = Path(__file__).resolve().parent


FDA_FEATURE_COLUMNS = [
    "year",
    "month",
    "primary_reaction",
    "num_reactions",
    "suspect_drug",
    "drug_route",
    "drug_indication",
    "pharm_class",
    "num_drugs",
    "patient_age_years",
    "patient_sex",
    "country",
    "report_age_days",
]


def load_model():
    df, _ = load_raw_training_data()
    y = pd.to_numeric(df["adr_event"], errors="coerce").fillna(0).astype(int)
    feature_frame = project_proxy_frame_to_fda_features(df).reindex(columns=FDA_FEATURE_COLUMNS)
    feature_frame, _, _ = cap_fda_feature_splits(feature_frame, feature_frame.copy())
    # The helper will usually load the saved artifact; if that fails, it can rebuild
    # from the projected proxy dataset with both target classes present.
    return load_or_fit_tuned_fda_logistic(feature_frame, y.loc[feature_frame.index].reset_index(drop=True))


def main() -> None:
    payload = json.load(sys.stdin)
    rows = payload if isinstance(payload, list) else [payload]

    feature_frame = pd.DataFrame(rows).reindex(columns=FDA_FEATURE_COLUMNS)
    model = load_model()
    proba = model.predict_proba(feature_frame)
    classes = list(model.classes_)
    serious_index = classes.index(1)
    scores = [round(float(prob[serious_index]) * 100, 1) for prob in proba]
    json.dump(scores, sys.stdout)


if __name__ == "__main__":
    main()
