"""Reproducible Stage 4 research hardening protocol.

The final test partition is reconstructed once from the v2 seed-42 split,
persisted, hashed, and never used by CV, tuning, calibration, or threshold
selection. New project-authored records are development-only.
"""

import hashlib
import json
import re
import string
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
LABELS = ["happiness", "sadness", "loneliness", "anxiety", "anger", "cognitive_fog", "neutral"]
SEED = 42
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50]
SOURCE_MAP = {
    "goemotions_unambiguous": "goemotions",
    "goemotions_explicit_loneliness_relabel": "domain_relabel",
    "project_authored": "project_authored",
}


def normalize(text):
    value = "" if pd.isna(text) else str(text).lower()
    value = value.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", value).strip()


def sha256_frame(frame):
    payload = frame.sort_values(["text", "label", "source"]).to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
    }


def validate(frame, name):
    required = {"text", "label", "source"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{name} lacks {sorted(required - set(frame.columns))}")
    if frame.text.eq("").any() or not set(frame.label).issubset(LABELS):
        raise ValueError(f"{name} has empty text or unsupported labels")
    if frame.text.duplicated().any():
        raise ValueError(f"{name} has duplicate normalized texts")
    if not set(frame.source).issubset({"goemotions", "project_authored", "domain_relabel", "other_verified_source"}):
        raise ValueError(f"{name} has noncanonical provenance")


def prepare_data():
    baseline = pd.read_csv(DATA / "training_dataset_v2.csv")
    baseline["text"] = baseline.text.map(normalize)
    baseline["source"] = baseline.source.map(SOURCE_MAP)
    if baseline.source.isna().any():
        raise ValueError("Unknown baseline source provenance")

    development_base, locked_test = train_test_split(
        baseline, test_size=0.20, random_state=SEED, stratify=baseline.label
    )
    additions = pd.read_csv(DATA / "project_authored_hardening.csv")
    additions["text"] = additions.text.map(normalize)
    additions["label"] = additions.label.str.strip().str.lower()
    additions["source"] = "project_authored"

    combined = pd.concat([development_base, additions], ignore_index=True)
    conflicts = combined.groupby("text").label.nunique()
    conflicting = set(conflicts[conflicts > 1].index)
    combined = combined[~combined.text.isin(conflicting)].copy()
    combined = combined.drop_duplicates("text", keep="first").reset_index(drop=True)
    overlap = set(combined.text) & set(locked_test.text)
    if overlap:
        raise ValueError(f"Locked-test leakage detected for {len(overlap)} texts")
    validate(combined, "development")
    validate(locked_test, "locked test")

    locked_test.to_csv(DATA / "locked_test_v2.csv", index=False)
    combined.to_csv(DATA / "development_dataset_v3.csv", index=False)
    frozen = {
        "created_by": "Stage 4 hardening protocol",
        "random_seed": SEED,
        "origin": "unchanged 20% held-out partition from training_dataset_v2.csv",
        "rows": len(locked_test),
        "sha256_sorted_csv": sha256_frame(locked_test),
        "class_support": locked_test.label.value_counts().sort_index().to_dict(),
        "prohibited_uses": ["model_selection", "hyperparameter_tuning", "feature_selection", "calibration", "threshold_selection"],
    }
    (DATA / "locked_test_metadata.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    return baseline, combined, locked_test, additions, conflicting


def features(config):
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=config["word_ngram"], min_df=config["word_min_df"],
                                 sublinear_tf=config["sublinear_tf"], max_features=9000)),
        ("character", TfidfVectorizer(analyzer="char_wb", ngram_range=config["char_ngram"],
                                      min_df=config["char_min_df"], sublinear_tf=True, max_features=8000)),
    ])


def pipeline_for(config, calibration_folds=3):
    if config["model"] == "logistic_regression":
        estimator = LogisticRegression(C=config["C"], class_weight="balanced", max_iter=2000, random_state=SEED)
    elif config["model"] == "linear_svm_calibrated":
        estimator = CalibratedClassifierCV(
            LinearSVC(C=config["C"], class_weight="balanced", random_state=SEED),
            method="sigmoid", cv=calibration_folds,
        )
    else:
        estimator = MultinomialNB(alpha=config["alpha"])
    return Pipeline([("tfidf", features(config)), ("model", estimator)])


def configs():
    shared = [
        ((1, 1), 1, False, (3, 5), 1),
        ((1, 2), 1, True, (3, 6), 1),
        ((1, 2), 2, True, (4, 6), 2),
        ((1, 2), 3, False, (3, 5), 2),
    ]
    rows = []
    for model in ["logistic_regression", "linear_svm_calibrated"]:
        for index, c_value in enumerate([0.5, 1.0, 2.0]):
            word_ngram, word_min_df, sublinear, char_ngram, char_min_df = shared[index + 1 if index < 2 else 2]
            rows.append({"model": model, "C": c_value, "word_ngram": word_ngram,
                         "word_min_df": word_min_df, "sublinear_tf": sublinear,
                         "char_ngram": char_ngram, "char_min_df": char_min_df})
    for index, alpha in enumerate([0.25, 0.5, 1.0]):
        word_ngram, word_min_df, sublinear, char_ngram, char_min_df = shared[index]
        rows.append({"model": "multinomial_naive_bayes", "alpha": alpha,
                     "word_ngram": word_ngram, "word_min_df": word_min_df,
                     "sublinear_tf": sublinear, "char_ngram": char_ngram,
                     "char_min_df": char_min_df})
    return rows


def cross_validate_config(config, development, folds):
    probabilities = np.zeros((len(development), len(LABELS)))
    predictions = np.empty(len(development), dtype=object)
    fold_rows = []
    for fold_number, (train_idx, valid_idx) in enumerate(folds.split(development.text, development.label), 1):
        model = pipeline_for(config, calibration_folds=3)
        train, valid = development.iloc[train_idx], development.iloc[valid_idx]
        model.fit(train.text, train.label)
        raw_prob = model.predict_proba(valid.text)
        aligned = np.zeros((len(valid), len(LABELS)))
        for column, label in enumerate(model.classes_):
            aligned[:, LABELS.index(label)] = raw_prob[:, column]
        probabilities[valid_idx] = aligned
        predictions[valid_idx] = np.array(LABELS)[aligned.argmax(axis=1)]
        fold_rows.append(metrics(valid.label, predictions[valid_idx]))
    summary = {f"cv_{key}_mean": float(np.mean([r[key] for r in fold_rows])) for key in fold_rows[0]}
    summary["cv_macro_f1_std"] = float(np.std([r["macro_f1"] for r in fold_rows], ddof=1))
    return summary, predictions, probabilities


RULES = {
    "loneliness": r"\b(lonely|alone|no one|nobody|quiet house|company|isolated)\b",
    "cognitive_fog": r"\b(concentrat|lost track|foggy|cloudy|unclear|forget|muddled|focus)\w*\b",
    "anger": r"\b(angry|furious|frustrat|annoy|irritat|temper)\w*\b",
    "anxiety": r"\b(worried|worry|anxious|uneasy|cannot relax|nervous|tense)\b",
    "sadness": r"\b(sad|down|tearful|unhappy|heavy heart|cry)\w*\b",
    "happiness": r"\b(happy|happier|cheerful|felt good|enjoyed|delight|pleased|smile)\w*\b",
    "neutral": r"\b(ordinary|normal|usual|routine|nothing unusual|uneventful)\b",
}


def rule_fallback(text):
    normalized = normalize(text)
    matches = [label for label, pattern in RULES.items() if re.search(pattern, normalized)]
    return matches[0] if len(matches) == 1 else "neutral"


def threshold_rows(labels, texts, probabilities):
    ml_predictions = np.array(LABELS)[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    truth = np.asarray(labels)
    fallback_predictions = np.array([rule_fallback(text) for text in texts])
    rows = []
    for threshold in THRESHOLDS:
        accepted = confidence >= threshold
        hybrid = np.where(accepted, ml_predictions, fallback_predictions)
        rows.append({
            "threshold": threshold,
            "ml_coverage": float(accepted.mean()),
            "ml_accepted_accuracy": float((ml_predictions[accepted] == truth[accepted]).mean()) if accepted.any() else 0.0,
            "fallback_rate": float((~accepted).mean()),
            "fallback_correctness": float((fallback_predictions[~accepted] == truth[~accepted]).mean()) if (~accepted).any() else 0.0,
            "overall_hybrid_correctness": float((hybrid == truth).mean()),
        })
    return rows


def report_frame(report):
    return pd.DataFrame([{"label": label, "precision": report[label]["precision"],
                          "recall": report[label]["recall"], "f1": report[label]["f1-score"],
                          "support": int(report[label]["support"])} for label in LABELS])


def distribution(frame, stage):
    return pd.DataFrame([{"stage": stage, "label": label, "count": int(count)}
                         for label, count in frame.label.value_counts().sort_index().items()])


def source_distribution(frame, stage):
    return frame.groupby(["label", "source"]).size().rename("count").reset_index().assign(stage=stage)


def near_duplicate_count(development, locked_test, threshold=0.85):
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    development_vectors = vectorizer.fit_transform(development.text)
    test_vectors = vectorizer.transform(locked_test.text)
    distances, _ = NearestNeighbors(n_neighbors=1, metric="cosine").fit(development_vectors).kneighbors(test_vectors)
    return int(((1.0 - distances[:, 0]) >= threshold).sum())


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    baseline, development, locked_test, additions, conflicts = prepare_data()
    distribution(baseline, "before").to_csv(RESULTS / "dataset_distribution.csv", index=False)
    pd.concat([distribution(baseline, "before"), distribution(pd.concat([development, locked_test]), "after")]).to_csv(
        RESULTS / "dataset_distribution.csv", index=False)
    pd.concat([source_distribution(baseline, "before"),
               source_distribution(pd.concat([development, locked_test]), "after")]).to_csv(
        RESULTS / "dataset_source_distribution.csv", index=False)

    short = baseline[baseline.text.str.split().map(len).le(2)][["text", "label", "source"]]
    short.assign(review_status="flagged_extremely_short_ambiguous_review").to_csv(RESULTS / "very_short_record_audit.csv", index=False)
    audit = {"baseline_rows": len(baseline), "development_rows_after_additions": len(development),
             "locked_test_rows": len(locked_test), "added_project_authored": len(additions),
             "empty": int(baseline.text.eq("").sum()), "duplicate_texts": int(baseline.text.duplicated().sum()),
             "conflicting_texts": int(len(conflicts)), "very_short_records": len(short),
             "exact_development_test_overlap": 0, "near_duplicate_similarity_threshold": 0.85,
             "near_duplicate_development_test_pairs": near_duplicate_count(development, locked_test)}
    (RESULTS / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    all_results = []
    oof_by_index = {}
    for index, config in enumerate(configs()):
        summary, predictions, probabilities = cross_validate_config(config, development, folds)
        row = {"config_id": index, **config, **summary}
        all_results.append(row)
        oof_by_index[index] = (predictions, probabilities)
    hyper = pd.DataFrame(all_results)
    for column in ["word_ngram", "char_ngram"]:
        hyper[column] = hyper[column].map(str)
    hyper.to_csv(RESULTS / "hyperparameter_results.csv", index=False)
    best_rows = hyper.sort_values(["cv_macro_f1_mean", "cv_macro_recall_mean", "cv_macro_f1_std"],
                                  ascending=[False, False, True]).groupby("model", as_index=False).first()
    best_rows.to_csv(RESULTS / "cv_model_comparison.csv", index=False)
    winner_id = int(hyper.sort_values(["cv_macro_f1_mean", "cv_macro_recall_mean", "cv_macro_f1_std"],
                                     ascending=[False, False, True]).iloc[0].config_id)
    winner_config = configs()[winner_id]
    winner_predictions, winner_oof_probabilities = oof_by_index[winner_id]

    thresholds = threshold_rows(development.label.to_numpy(), development.text.to_numpy(), winner_oof_probabilities)
    pd.DataFrame(thresholds).to_csv(RESULTS / "threshold_analysis.csv", index=False)
    eligible = [row for row in thresholds if row["ml_coverage"] >= 0.70]
    threshold_choice = max(eligible or thresholds, key=lambda row: (row["overall_hybrid_correctness"], row["ml_accepted_accuracy"], row["ml_coverage"]))

    candidate = pipeline_for(winner_config, calibration_folds=5)
    candidate.fit(development.text, development.label)
    test_probabilities = candidate.predict_proba(locked_test.text)
    test_predictions = candidate.classes_[test_probabilities.argmax(axis=1)]
    final_metrics = metrics(locked_test.label, test_predictions)
    final_report = classification_report(locked_test.label, test_predictions, labels=LABELS, output_dict=True, zero_division=0)
    final_report_frame = report_frame(final_report)
    final_report_frame.to_csv(RESULTS / "final_classification_report.csv", index=False)
    pd.DataFrame(confusion_matrix(locked_test.label, test_predictions, labels=LABELS), index=LABELS, columns=LABELS).to_csv(
        RESULTS / "confusion_matrix.csv")

    old_pipeline = joblib.load(ROOT / "emotion_pipeline.pkl")
    old_predictions = old_pipeline.predict(locked_test.text)
    old_metrics = metrics(locked_test.label, old_predictions)
    old_report = classification_report(locked_test.label, old_predictions, labels=LABELS, output_dict=True, zero_division=0)
    per_class = pd.DataFrame([{"label": label, "v2_f1": old_report[label]["f1-score"],
                               "hardened_f1": final_report[label]["f1-score"],
                               "difference": final_report[label]["f1-score"] - old_report[label]["f1-score"],
                               "test_support": int(final_report[label]["support"])} for label in LABELS])
    per_class.to_csv(RESULTS / "per_class_old_vs_new.csv", index=False)

    accepted = (final_metrics["macro_f1"] > old_metrics["macro_f1"] and
                final_report["cognitive_fog"]["f1-score"] >= old_report["cognitive_fog"]["f1-score"] and
                final_report["neutral"]["f1-score"] >= old_report["neutral"]["f1-score"] and
                all(final_report[label]["f1-score"] >= old_report[label]["f1-score"] - 0.05 for label in LABELS))
    version = f"tfidf_{winner_config['model']}_v3" if accepted else "tfidf_linear_svm_calibrated_v2"
    if accepted:
        joblib.dump(candidate, ROOT / "emotion_pipeline.pkl")

    evaluation = pd.read_csv(DATA / "project_evaluation_set.csv")
    evaluation["text_normalized"] = evaluation.text.map(normalize)
    if set(evaluation.text_normalized) & (set(development.text) | set(locked_test.text)):
        raise ValueError("Project evaluation set overlaps model data")
    deployed = candidate if accepted else old_pipeline
    project_probabilities = deployed.predict_proba(evaluation.text_normalized)
    project_predictions = deployed.classes_[project_probabilities.argmax(axis=1)]
    confidence = project_probabilities.max(axis=1)
    accepted_ml = confidence >= threshold_choice["threshold"]
    fallback = np.array([rule_fallback(text) for text in evaluation.text_normalized])
    hybrid = np.where(accepted_ml, project_predictions, fallback)
    evaluation["predicted_emotion"] = project_predictions
    evaluation["confidence"] = confidence
    evaluation["pure_ml_correct"] = evaluation.expected_emotion == project_predictions
    evaluation["routing_source"] = np.where(accepted_ml, "ml_model", "rule_fallback")
    evaluation["hybrid_emotion"] = hybrid
    evaluation["hybrid_correct"] = evaluation.expected_emotion == hybrid
    evaluation.drop(columns=["text_normalized"]).to_csv(RESULTS / "project_utterance_evaluation.csv", index=False)
    fog = evaluation[evaluation.expected_emotion.eq("cognitive_fog")].copy()
    fog["observed_pattern"] = np.where(fog.pure_ml_correct, "correct", np.where(fog.predicted_emotion.eq("neutral"), "misclassified_as_neutral",
        np.where(fog.predicted_emotion.eq("anger"), "misclassified_as_anger", np.where(fog.predicted_emotion.eq("anxiety"), "misclassified_as_anxiety", "other_or_ambiguous"))))
    fog[["text", "expected_emotion", "predicted_emotion", "confidence", "observed_pattern"]].to_csv(
        RESULTS / "cognitive_fog_error_analysis.csv", index=False)

    hybrid_metrics = {"overall_correctness": float(evaluation.hybrid_correct.mean()),
                      "ml_coverage": float(accepted_ml.mean()), "fallback_rate": float((~accepted_ml).mean()),
                      "ml_accepted_correctness": float(evaluation.loc[accepted_ml, "pure_ml_correct"].mean()) if accepted_ml.any() else 0.0,
                      "fallback_correctness": float(evaluation.loc[~accepted_ml, "hybrid_correct"].mean()) if (~accepted_ml).any() else 0.0}
    metadata = {
        "model_version": version, "selected_model": winner_config["model"], "accepted": bool(accepted),
        "acceptance_rule": "macro_f1 improves; cognitive_fog and neutral do not decline; no class declines >0.05",
        "selection_basis": "development-only 5-fold stratified CV macro F1",
        "selected_config": {k: list(v) if isinstance(v, tuple) else v for k, v in winner_config.items()},
        "calibration": {"method": "sigmoid" if winner_config["model"] == "linear_svm_calibrated" else "native_predict_proba",
                        "final_folds": 5 if winner_config["model"] == "linear_svm_calibrated" else None,
                        "data": "development only; final held-out test excluded"},
        "confidence_threshold": threshold_choice["threshold"], "threshold_role": "software confidence-routing threshold; not clinical",
        "threshold_selection_data": "development-only out-of-fold predictions", "supported_classes": LABELS,
        "random_seed": SEED, "locked_test_sha256": sha256_frame(locked_test),
        "split": {"development": len(development), "test": len(locked_test)},
        "cv_winner": all_results[winner_id], "v2_test_metrics": old_metrics, "hardened_test_metrics": final_metrics,
        "pure_ml_project_correctness": float(evaluation.pure_ml_correct.mean()), "hybrid_project_metrics": hybrid_metrics,
        "loneliness_support": {"development": int((development.label == "loneliness").sum()),
                               "test": int((locked_test.label == "loneliness").sum())},
    }
    (ROOT / "selected_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (RESULTS / "selected_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame([{"metric": key, "v2": old_metrics[key], "hardened_model": final_metrics[key],
                   "difference": final_metrics[key] - old_metrics[key]} for key in old_metrics]).to_csv(
        RESULTS / "old_vs_new_metrics.csv", index=False)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
