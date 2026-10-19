"""Finalize the advanced experiment after development-only model selection."""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from harden_emotion_model import LABELS, rule_fallback  # noqa: E402
from prepare_advanced_data import frame_hash  # noqa: E402

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def metric_dict(truth, predicted):
    return {"accuracy": accuracy_score(truth, predicted),
            "macro_precision": precision_score(truth, predicted, labels=LABELS, average="macro", zero_division=0),
            "macro_recall": recall_score(truth, predicted, labels=LABELS, average="macro", zero_division=0),
            "macro_f1": f1_score(truth, predicted, labels=LABELS, average="macro", zero_division=0),
            "weighted_f1": f1_score(truth, predicted, labels=LABELS, average="weighted", zero_division=0)}


def report_csv(truth, predicted, path):
    report = classification_report(truth, predicted, labels=LABELS, output_dict=True, zero_division=0)
    pd.DataFrame([{"label":label,"precision":report[label]["precision"],"recall":report[label]["recall"],
                   "f1":report[label]["f1-score"],"support":int(report[label]["support"])} for label in LABELS]).to_csv(path,index=False)
    return report


def main():
    selection = json.loads((HERE / "development_selection.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(HERE / "transformer_model_comparison.csv")
    svm_row = comparison[comparison.architecture == "v3_svm_reference"].iloc[0]
    if selection["result"]["macro_f1"] >= svm_row.macro_f1:
        raise RuntimeError("Unexpected selection state: transformer would require finalization review")

    baseline_meta = json.loads((HERE / "baseline_v3" / "selected_model_metadata_v3.json").read_text(encoding="utf-8"))
    locked = pd.read_csv(HERE / "baseline_v3" / "data" / "locked_test_v2.csv")
    if frame_hash(locked.assign(domain="locked_original")[["text","label","source","domain"]]) == "":
        raise RuntimeError("Unreachable checksum guard")
    expected_locked_hash = baseline_meta["locked_test_sha256"]
    canonical_locked_payload = locked.sort_values(["text","label","source"]).to_csv(index=False,lineterminator="\n")
    import hashlib
    if hashlib.sha256(canonical_locked_payload.encode()).hexdigest() != expected_locked_hash:
        raise RuntimeError("Original locked-test checksum changed")

    domain = pd.read_csv(HERE / "domain_test_frozen.csv")
    domain_meta = json.loads((HERE / "domain_test_metadata.json").read_text(encoding="utf-8"))
    if frame_hash(domain) != domain_meta["sha256_sorted_csv"]:
        raise RuntimeError("Domain-test checksum changed")

    model = joblib.load(HERE / "baseline_v3" / "emotion_pipeline_v3.pkl")
    domain_prob = model.predict_proba(domain.text)
    domain_pred = model.classes_[domain_prob.argmax(axis=1)]
    domain_conf = domain_prob.max(axis=1)
    domain_metrics = metric_dict(domain.label, domain_pred)
    domain_report = report_csv(domain.label, domain_pred, HERE / "domain_test_classification_report.csv")
    pd.DataFrame(confusion_matrix(domain.label, domain_pred, labels=LABELS),index=LABELS,columns=LABELS).to_csv(HERE / "domain_test_confusion_matrix.csv")
    errors = domain[domain.label.to_numpy() != domain_pred].copy()
    errors["predicted_label"] = domain_pred[domain.label.to_numpy() != domain_pred]
    errors["confidence"] = domain_conf[domain.label.to_numpy() != domain_pred]
    errors["error_category"] = errors.label + "_as_" + errors.predicted_label
    errors.to_csv(HERE / "domain_error_analysis.csv",index=False)

    baseline_results = HERE / "baseline_v3" / "results"
    pd.read_csv(baseline_results / "final_classification_report.csv").to_csv(HERE / "original_test_classification_report.csv",index=False)
    pd.read_csv(baseline_results / "confusion_matrix.csv",index_col=0).to_csv(HERE / "original_test_confusion_matrix.csv")

    calibration = pd.read_csv(HERE / "advanced_development_split.csv")
    calibration = calibration[calibration.split == "calibration"]
    probabilities = model.predict_proba(calibration.text)
    predictions = model.classes_[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    fallback = np.array([rule_fallback(text) for text in calibration.text])
    threshold_rows=[]
    for threshold in THRESHOLDS:
        accepted = confidence >= threshold; hybrid=np.where(accepted,predictions,fallback)
        threshold_rows.append({"model":"v3_retained_no_v4_deployed","threshold":threshold,
            "ml_coverage":float(accepted.mean()),"accepted_prediction_accuracy":float((predictions[accepted]==calibration.label.to_numpy()[accepted]).mean()) if accepted.any() else 0.0,
            "fallback_rate":float((~accepted).mean()),"hybrid_overall_correctness":float((hybrid==calibration.label.to_numpy()).mean())})
    pd.DataFrame(threshold_rows).to_csv(HERE / "threshold_analysis_v4.csv",index=False)
    chosen=max([row for row in threshold_rows if row["ml_coverage"]>=.70],key=lambda row:(row["hybrid_overall_correctness"],row["accepted_prediction_accuracy"]))

    domain_accepted=domain_conf>=chosen["threshold"]; domain_fallback=np.array([rule_fallback(text) for text in domain.text]); domain_hybrid=np.where(domain_accepted,domain_pred,domain_fallback)
    hybrid={"correctness":float((domain_hybrid==domain.label.to_numpy()).mean()),"ml_coverage":float(domain_accepted.mean()),
            "fallback_rate":float((~domain_accepted).mean()),"accepted_correctness":float((domain_pred[domain_accepted]==domain.label.to_numpy()[domain_accepted]).mean()) if domain_accepted.any() else 0.0,
            "fallback_correctness":float((domain_fallback[~domain_accepted]==domain.label.to_numpy()[~domain_accepted]).mean()) if (~domain_accepted).any() else 0.0}

    samples=domain.text.iloc[:100].tolist(); started=time.perf_counter()
    for text in samples: model.predict_proba([text])
    latency_ms=(time.perf_counter()-started)*1000/len(samples)
    artifact_size=(HERE / "baseline_v3" / "emotion_pipeline_v3.pkl").stat().st_size
    original=baseline_meta["hardened_test_metrics"]
    pd.DataFrame([{"metric":metric,"v3_original_test":original[metric],"v4_original_test":None,
                   "difference":None,"note":"No v4 final-test evaluation: transformer rejected on development"} for metric in original]).to_csv(HERE / "v3_vs_v4_comparison.csv",index=False)
    metadata={"experiment":"advanced_stage4_english","selected_production_model":"tfidf_linear_svm_calibrated_v3",
              "v4_deployed":False,"reason":"Both transformers underperformed fixed v3 on development Macro F1; final tests were not used to choose or rescue them.",
              "transformer_development_winner":selection,"v3_development_reference":{"accuracy":float(svm_row.accuracy),"macro_f1":float(svm_row.macro_f1),"macro_recall":float(svm_row.macro_recall)},
              "original_test_metrics_preserved":original,"original_test_reopened":False,"original_test_checksum_verified":True,
              "domain_test_metrics":domain_metrics,"domain_test_checksum":domain_meta["sha256_sorted_csv"],"pure_model_domain_per_class":{label:{"f1":domain_report[label]["f1-score"],"support":int(domain_report[label]["support"])} for label in LABELS},
              "selected_routing_threshold":chosen["threshold"],"threshold_selection_data":"development calibration partition only","hybrid_domain_metrics":hybrid,
              "calibration_note":"Rejected transformer softmax was not operationally calibrated; retained v3 already uses five-fold development-only sigmoid calibration.",
              "multiple_seed_note":"Not run: CPU cost was disproportionate after transformers clearly lost development selection.",
              "deployment":{"v3_artifact_bytes":artifact_size,"v3_mean_cpu_inference_ms":latency_ms,"transformer_tradeoff":"Hundreds of MB and multi-minute CPU training epochs; rejected before deployment."},
              "targets":{"accuracy_0_80_genuinely_achieved_on_original_test":False,"macro_f1_0_75_genuinely_achieved_on_original_test":False}}
    (HERE / "advanced_selected_model_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(json.dumps(metadata,indent=2))


if __name__ == "__main__": main()
