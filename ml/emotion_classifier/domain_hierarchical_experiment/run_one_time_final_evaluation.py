"""One-time pure-model evaluation of the already-frozen MiniLM candidate."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
ADVANCED=ROOT/"advanced_experiment"
MARKER=HERE/"final_evaluation_completed.json"
LABELS=["happiness","sadness","loneliness","anxiety","anger","cognitive_fog","neutral"]


def frame_hash(frame):
    payload=frame.sort_values(["text","label","source","domain"]).to_csv(index=False,lineterminator="\n")
    return hashlib.sha256(payload.encode()).hexdigest()


def locked_hash(frame):
    payload=frame.sort_values(["text","label","source"]).to_csv(index=False,lineterminator="\n")
    return hashlib.sha256(payload.encode()).hexdigest()


def evaluate(name, frame, predicted, confidence):
    truth=frame.label.to_numpy(); report=classification_report(truth,predicted,labels=LABELS,output_dict=True,zero_division=0)
    summary={"test":name,"accuracy":accuracy_score(truth,predicted),
             "macro_precision":precision_score(truth,predicted,labels=LABELS,average="macro",zero_division=0),
             "macro_recall":recall_score(truth,predicted,labels=LABELS,average="macro",zero_division=0),
             "macro_f1":f1_score(truth,predicted,labels=LABELS,average="macro",zero_division=0),
             "weighted_f1":f1_score(truth,predicted,labels=LABELS,average="weighted",zero_division=0)}
    per_class=pd.DataFrame([{"test":name,"label":label,"precision":report[label]["precision"],"recall":report[label]["recall"],
                             "f1":report[label]["f1-score"],"support":int(report[label]["support"])} for label in LABELS])
    matrix=pd.DataFrame(confusion_matrix(truth,predicted,labels=LABELS),index=LABELS,columns=LABELS)
    predictions=frame.copy(); predictions["predicted_label"]=predicted; predictions["confidence"]=confidence; predictions["correct"]=truth==predicted
    return summary,per_class,matrix,predictions


def main():
    if MARKER.exists(): raise RuntimeError("One-time final evaluation already completed; rerun prohibited.")
    metadata=json.loads((HERE/"candidate_metadata.json").read_text(encoding="utf-8")); artifact=HERE/"candidate_model.joblib"
    artifact_hash=hashlib.sha256(artifact.read_bytes()).hexdigest()
    if artifact_hash!=metadata["artifact_sha256"]: raise RuntimeError("Candidate artifact checksum mismatch")
    bundle=joblib.load(artifact); classifier=bundle["model"]
    if bundle.get("architecture")!="flat" or classifier.random_state!=42 or classifier.C!=1.0 or classifier.class_weight!="balanced":
        raise RuntimeError("Frozen classifier configuration mismatch")
    if classifier.classes_.tolist()!=sorted(LABELS): raise RuntimeError("Frozen label mapping mismatch")

    original=pd.read_csv(ADVANCED/"baseline_v3"/"data"/"locked_test_v2.csv")
    original_meta=json.loads((ADVANCED/"baseline_v3"/"selected_model_metadata_v3.json").read_text(encoding="utf-8"))
    if locked_hash(original)!=original_meta["locked_test_sha256"]: raise RuntimeError("Original test checksum mismatch")
    domain=pd.read_csv(ADVANCED/"domain_test_frozen.csv")
    domain_meta=json.loads((ADVANCED/"domain_test_metadata.json").read_text(encoding="utf-8"))
    if frame_hash(domain)!=domain_meta["sha256_sorted_csv"]: raise RuntimeError("Domain test checksum mismatch")

    embedding_model=SentenceTransformer(metadata["minilm_checkpoint"],device="cpu",local_files_only=True)
    original_embeddings=embedding_model.encode(original.text.tolist(),batch_size=64,show_progress_bar=True,normalize_embeddings=True)
    domain_embeddings=embedding_model.encode(domain.text.tolist(),batch_size=64,show_progress_bar=True,normalize_embeddings=True)
    original_probability=classifier.predict_proba(original_embeddings); domain_probability=classifier.predict_proba(domain_embeddings)
    original_prediction=classifier.classes_[original_probability.argmax(axis=1)]; domain_prediction=classifier.classes_[domain_probability.argmax(axis=1)]
    original_result=evaluate("original_frozen_381",original,original_prediction,original_probability.max(axis=1))
    domain_result=evaluate("eldermeds_domain_140",domain,domain_prediction,domain_probability.max(axis=1))
    summaries=pd.DataFrame([original_result[0],domain_result[0]]); summaries.to_csv(HERE/"minilm_final_summary.csv",index=False)
    pd.concat([original_result[1],domain_result[1]],ignore_index=True).to_csv(HERE/"minilm_final_per_class.csv",index=False)
    original_result[2].to_csv(HERE/"minilm_original_confusion_matrix.csv"); domain_result[2].to_csv(HERE/"minilm_domain_confusion_matrix.csv")
    original_result[3].to_csv(HERE/"minilm_original_predictions.csv",index=False); domain_result[3].to_csv(HERE/"minilm_domain_predictions.csv",index=False)

    v3_meta=json.loads((ADVANCED/"advanced_selected_model_metadata.json").read_text(encoding="utf-8"))
    v3_original=v3_meta["original_test_metrics_preserved"]; v3_domain=v3_meta["domain_test_metrics"]
    comparisons=[]
    for test,v3,candidate in [("original",v3_original,original_result[0]),("domain",v3_domain,domain_result[0])]:
        for metric in ["accuracy","macro_f1","macro_recall","weighted_f1"]:
            comparisons.append({"test":test,"metric":metric,"v3":v3[metric],"minilm_candidate":candidate[metric],"difference":candidate[metric]-v3[metric]})
    pd.DataFrame(comparisons).to_csv(HERE/"v3_vs_minilm_final_metrics.csv",index=False)
    v3_original_class=pd.read_csv(ADVANCED/"original_test_classification_report.csv").set_index("label")
    v3_domain_class=pd.read_csv(ADVANCED/"domain_test_classification_report.csv").set_index("label")
    class_compare=[]
    for test,v3frame,candidateframe in [("original",v3_original_class,original_result[1].set_index("label")),("domain",v3_domain_class,domain_result[1].set_index("label"))]:
        for label in LABELS:
            class_compare.append({"test":test,"label":label,"v3_f1":v3frame.loc[label,"f1"],"minilm_f1":candidateframe.loc[label,"f1"],
                                  "difference":candidateframe.loc[label,"f1"]-v3frame.loc[label,"f1"],"support":int(candidateframe.loc[label,"support"])})
    pd.DataFrame(class_compare).to_csv(HERE/"v3_vs_minilm_per_class.csv",index=False)
    snapshots=list((Path.home()/".cache"/"huggingface"/"hub"/"models--sentence-transformers--all-MiniLM-L6-v2"/"snapshots").glob("*"))
    marker={"completed_at_utc":datetime.now(timezone.utc).isoformat(),"candidate_sha256":artifact_hash,
            "minilm_identifier":metadata["minilm_checkpoint"],"minilm_cached_snapshot":snapshots[0].name if snapshots else None,
            "original_test_sha256":original_meta["locked_test_sha256"],"domain_test_sha256":domain_meta["sha256_sorted_csv"],
            "original_rows":len(original),"domain_rows":len(domain),"pure_model_only":True,
            "results":{"original":original_result[0],"domain":domain_result[0]},"rerun_prohibited":True}
    MARKER.write_text(json.dumps(marker,indent=2),encoding="utf-8")
    print(json.dumps(marker,indent=2))


if __name__=="__main__": main()
