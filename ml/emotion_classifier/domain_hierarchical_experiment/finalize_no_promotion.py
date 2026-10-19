"""Finalize development artifacts when no candidate passes promotion."""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
ADVANCED=ROOT/"advanced_experiment"
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(HERE))
from harden_emotion_model import rule_fallback  # noqa: E402
from run_development_models import fit_hierarchy,predict_hierarchy,estimator,MINILM  # noqa: E402

THRESHOLDS=[0.30,0.35,0.40,0.45,0.50,0.55,0.60]


def main():
    metadata=json.loads((HERE/"candidate_metadata.json").read_text(encoding="utf-8"))
    if metadata["promotion_gate_passed"]: raise RuntimeError("Use promoted-candidate finalization instead")
    split=pd.read_csv(HERE/"domain_development_split.csv")
    calibration=split[split.split=="calibration"].reset_index(drop=True)
    train=split[split.split!="calibration"].reset_index(drop=True)
    calibration_embeddings=np.load(HERE/"minilm_calibration_embeddings.npy")
    train_embeddings=np.load(HERE/"minilm_train_embeddings.npy")
    classifier=joblib.load(HERE/"candidate_model.joblib")["model"]
    probability=classifier.predict_proba(calibration_embeddings); prediction=classifier.classes_[probability.argmax(axis=1)]; confidence=probability.max(axis=1)
    fallback=np.array([rule_fallback(text) for text in calibration.text]); truth=calibration.label.to_numpy(); rows=[]
    for threshold in THRESHOLDS:
        accepted=confidence>=threshold; hybrid=np.where(accepted,prediction,fallback)
        rows.append({"threshold":threshold,"coverage":float(accepted.mean()),
                     "accepted_accuracy":float((prediction[accepted]==truth[accepted]).mean()) if accepted.any() else 0.0,
                     "fallback_rate":float((~accepted).mean()),"fallback_accuracy":float((fallback[~accepted]==truth[~accepted]).mean()) if (~accepted).any() else 0.0,
                     "hybrid_correctness":float((hybrid==truth).mean())})
    pd.DataFrame(rows).to_csv(HERE/"threshold_analysis_hierarchical.csv",index=False)
    eligible=[row for row in rows if row["coverage"]>=.70]; chosen=max(eligible,key=lambda row:(row["hybrid_correctness"],row["accepted_accuracy"]))

    domain=pd.read_csv(HERE/"domain_development_evaluation_frozen.csv"); domain_embeddings=np.load(HERE/"minilm_domain_dev_embeddings.npy")
    hierarchy=fit_hierarchy("hierarchy_a",train_embeddings,train.label.to_numpy(),lambda:estimator("linear_svm",42))
    hierarchy_prediction,_=predict_hierarchy(hierarchy,domain_embeddings)
    hierarchy_errors=domain[domain.label.to_numpy()!=hierarchy_prediction].copy()
    hierarchy_errors["predicted_label"]=hierarchy_prediction[domain.label.to_numpy()!=hierarchy_prediction]
    hierarchy_errors["error_stage"]="end_to_end_path"
    hierarchy_errors["error_category"]=hierarchy_errors.label+"_as_"+hierarchy_errors.predicted_label
    hierarchy_errors.to_csv(HERE/"hierarchical_error_analysis.csv",index=False)
    pd.DataFrame(columns=["text","true_label","predicted_label","confidence","source","error_category","note"]).assign(
        note=pd.Series(dtype=str)).to_csv(HERE/"domain_error_analysis.csv",index=False)

    original=pd.read_csv(ADVANCED/"original_test_classification_report.csv"); original.insert(0,"model","v3_production_preserved")
    original["note"]="No new candidate evaluated; promotion gate failed"; original.to_csv(HERE/"original_test_results.csv",index=False)
    domain_result=pd.read_csv(ADVANCED/"domain_test_classification_report.csv"); domain_result.insert(0,"model","v3_production_preserved")
    domain_result["note"]="No new candidate evaluated; promotion gate failed"; domain_result.to_csv(HERE/"domain_test_results.csv",index=False)
    pd.read_csv(ADVANCED/"original_test_confusion_matrix.csv",index_col=0).to_csv(HERE/"original_test_confusion_matrix.csv")
    pd.read_csv(ADVANCED/"domain_test_confusion_matrix.csv",index_col=0).to_csv(HERE/"domain_test_confusion_matrix.csv")
    results=pd.read_csv(HERE/"flat_vs_hierarchical_dev_results.csv")
    best_hierarchy=results[results.architecture.str.startswith("hierarchy")].sort_values("macro_f1",ascending=False).iloc[0]
    comparison=pd.DataFrame([
        {"model":"v3_production","dev_accuracy":metadata["v3_domain_development_metrics"]["accuracy"],"dev_macro_f1":metadata["v3_domain_development_metrics"]["macro_f1"],"original_test_accuracy":0.6141732283,"original_test_macro_f1":0.6422345849,"domain_test_accuracy":0.6,"domain_test_macro_f1":0.5869150969,"final_evaluation_status":"preserved"},
        {"model":"best_new_flat_minilm_logistic","dev_accuracy":metadata["domain_development_metrics"]["accuracy"],"dev_macro_f1":metadata["domain_development_metrics"]["macro_f1"],"original_test_accuracy":np.nan,"original_test_macro_f1":np.nan,"domain_test_accuracy":np.nan,"domain_test_macro_f1":np.nan,"final_evaluation_status":"not evaluated; gate failed"},
        {"model":f"best_{best_hierarchy.architecture}_{best_hierarchy.representation}_{best_hierarchy.family}","dev_accuracy":best_hierarchy.accuracy,"dev_macro_f1":best_hierarchy.macro_f1,"original_test_accuracy":np.nan,"original_test_macro_f1":np.nan,"domain_test_accuracy":np.nan,"domain_test_macro_f1":np.nan,"final_evaluation_status":"not evaluated; gate failed"},
    ])
    comparison.to_csv(HERE/"v3_vs_hierarchical_comparison.csv",index=False)

    started=time.perf_counter(); embedding_model=SentenceTransformer(MINILM,device="cpu",local_files_only=True); startup=time.perf_counter()-started
    samples=calibration.text.iloc[:100].tolist(); started=time.perf_counter(); embeds=embedding_model.encode(samples,batch_size=64,show_progress_bar=False,normalize_embeddings=True); classifier.predict_proba(embeds); latency=(time.perf_counter()-started)*1000/len(samples)
    cache=Path.home()/".cache"/"huggingface"/"hub"/"models--sentence-transformers--all-MiniLM-L6-v2"
    embedding_bytes=sum(path.stat().st_size for path in cache.rglob("*") if path.is_file()) if cache.exists() else None
    metadata.update({"selected_development_threshold":chosen["threshold"],"threshold_data":"development calibration partition only",
                     "candidate_startup_seconds":startup,"candidate_mean_cpu_inference_ms":latency,"candidate_classifier_bytes":(HERE/"candidate_model.joblib").stat().st_size,
                     "minilm_cache_bytes":embedding_bytes,"final_tests_opened":False,"v4_deployed":False,
                     "recommendation":"retain_v3","final_test_reason":"Candidate failed promotion because cognitive_fog F1 declined from 0.7778 to 0.7500 despite strong overall development gains."})
    (HERE/"candidate_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(json.dumps(metadata,indent=2))


if __name__=="__main__": main()
