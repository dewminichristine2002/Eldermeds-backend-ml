"""Development-only flat, hierarchical, and MiniLM comparison."""

import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ADVANCED = ROOT / "advanced_experiment"
LABELS = ["happiness", "sadness", "loneliness", "anxiety", "anger", "cognitive_fog", "neutral"]
EMOTIONS = ["happiness", "sadness", "loneliness", "anxiety", "anger"]
MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def metrics(truth, predicted, labels=LABELS):
    result={"accuracy":accuracy_score(truth,predicted),
            "macro_precision":precision_score(truth,predicted,labels=labels,average="macro",zero_division=0),
            "macro_recall":recall_score(truth,predicted,labels=labels,average="macro",zero_division=0),
            "macro_f1":f1_score(truth,predicted,labels=labels,average="macro",zero_division=0),
            "weighted_f1":f1_score(truth,predicted,labels=labels,average="weighted",zero_division=0)}
    for label in LABELS:
        if label in labels: result[f"f1_{label}"]=f1_score(truth,predicted,labels=[label],average="macro",zero_division=0)
    return result


def estimator(family, seed):
    if family == "logistic_regression": return LogisticRegression(C=1.0,class_weight="balanced",max_iter=2000,random_state=seed)
    if family == "linear_svm": return LinearSVC(C=1.0,class_weight="balanced",random_state=seed)
    return CalibratedClassifierCV(LinearSVC(C=1.0,class_weight="balanced",random_state=seed),method="sigmoid",cv=3)


def tfidf_model(family, seed):
    features=FeatureUnion([("word",TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True,max_features=9000)),
                           ("char",TfidfVectorizer(analyzer="char_wb",ngram_range=(3,6),min_df=1,sublinear_tf=True,max_features=8000))])
    return Pipeline([("features",features),("model",estimator(family,seed))])


def model_factory(representation, family, seed):
    return (lambda:tfidf_model(family,seed)) if representation=="tfidf" else (lambda:estimator(family,seed))


def probabilities(model, values):
    return model.predict_proba(values) if hasattr(model,"predict_proba") else None


def fit_hierarchy(architecture, values, labels, factory):
    labels=np.asarray(labels); models={}
    if architecture=="hierarchy_a":
        level1=np.where(labels=="neutral","neutral",np.where(labels=="cognitive_fog","cognitive_fog","emotional"))
        models["level1"]=factory().fit(values,level1)
        mask=np.isin(labels,EMOTIONS); models["level2_emotion"]=factory().fit(values[mask] if not isinstance(values,pd.Series) else values.iloc[np.where(mask)[0]],labels[mask])
    else:
        level1=np.where(labels=="neutral","neutral","non_neutral"); models["level1"]=factory().fit(values,level1)
        mask=labels!="neutral"; level2=np.where(labels[mask]=="cognitive_fog","cognitive_fog","emotional")
        subset=values[mask] if not isinstance(values,pd.Series) else values.iloc[np.where(mask)[0]]; models["level2_non_neutral"]=factory().fit(subset,level2)
        emotional=np.isin(labels,EMOTIONS); subset=values[emotional] if not isinstance(values,pd.Series) else values.iloc[np.where(emotional)[0]]
        models["level3_emotion"]=factory().fit(subset,labels[emotional])
    return {"architecture":architecture,"models":models}


def class_probability(model, values, predicted):
    prob=probabilities(model,values)
    if prob is None: return np.full(len(predicted),np.nan)
    indexes={label:i for i,label in enumerate(model.classes_)}
    return np.array([prob[row,indexes[label]] for row,label in enumerate(predicted)])


def predict_hierarchy(bundle, values):
    models=bundle["models"]; level1=models["level1"].predict(values); final=[]; confidence=[]
    level1_prob=class_probability(models["level1"],values,level1)
    for index, first in enumerate(level1):
        single=values[index:index+1] if not isinstance(values,pd.Series) else values.iloc[index:index+1]
        if bundle["architecture"]=="hierarchy_a":
            if first!="emotional": final.append(first); confidence.append(level1_prob[index]); continue
            second=models["level2_emotion"].predict(single)[0]; second_prob=class_probability(models["level2_emotion"],single,[second])[0]
            final.append(second); confidence.append(level1_prob[index]*second_prob)
        else:
            if first=="neutral": final.append("neutral"); confidence.append(level1_prob[index]); continue
            second=models["level2_non_neutral"].predict(single)[0]; second_prob=class_probability(models["level2_non_neutral"],single,[second])[0]
            if second=="cognitive_fog": final.append(second); confidence.append(level1_prob[index]*second_prob); continue
            third=models["level3_emotion"].predict(single)[0]; third_prob=class_probability(models["level3_emotion"],single,[third])[0]
            final.append(third); confidence.append(level1_prob[index]*second_prob*third_prob)
    return np.asarray(final),np.asarray(confidence)


def node_metrics(bundle, values, labels, representation, family):
    labels=np.asarray(labels); rows=[]; models=bundle["models"]
    if bundle["architecture"]=="hierarchy_a":
        truth=np.where(labels=="neutral","neutral",np.where(labels=="cognitive_fog","cognitive_fog","emotional")); pred=models["level1"].predict(values)
        rows.append({"architecture":"hierarchy_a","representation":representation,"family":family,"node":"level1","support":len(labels),**metrics(truth,pred,["neutral","cognitive_fog","emotional"])})
        mask=np.isin(labels,EMOTIONS); subset=values[mask] if not isinstance(values,pd.Series) else values.iloc[np.where(mask)[0]]; pred=models["level2_emotion"].predict(subset)
        rows.append({"architecture":"hierarchy_a","representation":representation,"family":family,"node":"level2_emotion","support":int(mask.sum()),**metrics(labels[mask],pred,EMOTIONS)})
    else:
        truth=np.where(labels=="neutral","neutral","non_neutral"); pred=models["level1"].predict(values)
        rows.append({"architecture":"hierarchy_b","representation":representation,"family":family,"node":"level1","support":len(labels),**metrics(truth,pred,["neutral","non_neutral"])})
        mask=labels!="neutral"; subset=values[mask] if not isinstance(values,pd.Series) else values.iloc[np.where(mask)[0]]; truth2=np.where(labels[mask]=="cognitive_fog","cognitive_fog","emotional"); pred=models["level2_non_neutral"].predict(subset)
        rows.append({"architecture":"hierarchy_b","representation":representation,"family":family,"node":"level2_non_neutral","support":int(mask.sum()),**metrics(truth2,pred,["cognitive_fog","emotional"])})
        emotional=np.isin(labels,EMOTIONS); subset=values[emotional] if not isinstance(values,pd.Series) else values.iloc[np.where(emotional)[0]]; pred=models["level3_emotion"].predict(subset)
        rows.append({"architecture":"hierarchy_b","representation":representation,"family":family,"node":"level3_emotion","support":int(emotional.sum()),**metrics(labels[emotional],pred,EMOTIONS)})
    return rows


def evaluate_candidate(architecture,representation,family,train_values,train_labels,eval_values,eval_labels,seed):
    factory=model_factory(representation,family,seed); started=time.perf_counter()
    if architecture=="flat":
        model=factory().fit(train_values,train_labels); predicted=model.predict(eval_values); confidence=class_probability(model,eval_values,predicted); nodes=[]
        bundle={"architecture":"flat","model":model}
    else:
        bundle=fit_hierarchy(architecture,train_values,train_labels,factory); predicted,confidence=predict_hierarchy(bundle,eval_values); nodes=node_metrics(bundle,eval_values,eval_labels,representation,family)
    result={"architecture":architecture,"representation":representation,"family":family,"seed":seed,"fit_and_eval_seconds":time.perf_counter()-started,
            "has_probability_confidence":bool(np.isfinite(confidence).all()),**metrics(eval_labels,predicted)}
    return bundle,result,nodes


def main():
    split=pd.read_csv(HERE/"domain_development_split.csv"); domain=pd.read_csv(HERE/"domain_development_evaluation_frozen.csv")
    train=split[split.split!="calibration"].reset_index(drop=True); calibration=split[split.split=="calibration"].reset_index(drop=True)
    rows=[]; nodes=[]; bundles={}
    v3=joblib.load(ADVANCED/"baseline_v3"/"emotion_pipeline_v3.pkl"); v3_pred=v3.predict(domain.text)
    rows.append({"architecture":"flat_v3_reference","representation":"tfidf","family":"calibrated_svm_v3","seed":42,"fit_and_eval_seconds":0,"has_probability_confidence":True,**metrics(domain.label,v3_pred)})
    for architecture in ["flat","hierarchy_a","hierarchy_b"]:
        for family in ["logistic_regression","linear_svm","calibrated_svm"]:
            bundle,result,node=evaluate_candidate(architecture,"tfidf",family,train.text,train.label,domain.text,domain.label,42)
            rows.append(result); nodes.extend(node); bundles[(architecture,"tfidf",family)]=bundle

    embedding_model=SentenceTransformer(MINILM,device="cpu",local_files_only=True)
    train_embed=embedding_model.encode(train.text.tolist(),batch_size=64,show_progress_bar=True,normalize_embeddings=True)
    domain_embed=embedding_model.encode(domain.text.tolist(),batch_size=64,show_progress_bar=False,normalize_embeddings=True)
    calibration_embed=embedding_model.encode(calibration.text.tolist(),batch_size=64,show_progress_bar=False,normalize_embeddings=True)
    np.save(HERE/"minilm_train_embeddings.npy",train_embed); np.save(HERE/"minilm_domain_dev_embeddings.npy",domain_embed); np.save(HERE/"minilm_calibration_embeddings.npy",calibration_embed)
    for architecture in ["flat","hierarchy_a","hierarchy_b"]:
        for family in ["logistic_regression","linear_svm"]:
            bundle,result,node=evaluate_candidate(architecture,"minilm",family,train_embed,train.label.to_numpy(),domain_embed,domain.label.to_numpy(),42)
            rows.append(result); nodes.extend(node); bundles[(architecture,"minilm",family)]=bundle

    results=pd.DataFrame(rows).sort_values(["macro_f1","accuracy"],ascending=False); results.to_csv(HERE/"flat_vs_hierarchical_dev_results.csv",index=False)
    pd.DataFrame(nodes).to_csv(HERE/"hierarchy_node_metrics.csv",index=False)
    results[results.representation=="minilm"].to_csv(HERE/"minilm_results.csv",index=False)
    v3_row=results[results.architecture=="flat_v3_reference"].iloc[0]; candidates=results[results.architecture!="flat_v3_reference"]
    best=candidates.iloc[0]; meaningful=bool(best.macro_f1>=v3_row.macro_f1+0.02 and best.f1_cognitive_fog>=v3_row.f1_cognitive_fog and best.f1_neutral>=v3_row.f1_neutral)

    stability=[]; key=(best.architecture,best.representation,best.family)
    for seed in [42,123,2026]:
        if seed==42: result=best.to_dict(); bundle=bundles[key]
        else:
            values=train.text if best.representation=="tfidf" else train_embed; eval_values=domain.text if best.representation=="tfidf" else domain_embed
            bundle,result,_=evaluate_candidate(best.architecture,best.representation,best.family,values,train.label.to_numpy(),eval_values,domain.label.to_numpy(),seed)
        stability.append({"seed":seed,"accuracy":float(result["accuracy"]),"macro_f1":float(result["macro_f1"])})
    stability_frame=pd.DataFrame(stability); stability_frame.to_csv(HERE/"candidate_seed_stability.csv",index=False)
    final_bundle=bundles[key]
    artifact=HERE/"candidate_model.joblib"; joblib.dump(final_bundle,artifact)
    metadata={"candidate":{"architecture":best.architecture,"representation":best.representation,"family":best.family,"seed_policy":[42,123,2026]},
              "domain_development_metrics":{k:float(best[k]) for k in ["accuracy","macro_precision","macro_recall","macro_f1","weighted_f1","f1_cognitive_fog","f1_neutral","f1_anger","f1_loneliness"]},
              "v3_domain_development_metrics":{k:float(v3_row[k]) for k in ["accuracy","macro_f1","f1_cognitive_fog","f1_neutral","f1_anger","f1_loneliness"]},
              "seed_stability":{"accuracy_mean":float(stability_frame.accuracy.mean()),"accuracy_std":float(stability_frame.accuracy.std(ddof=1)),"macro_f1_mean":float(stability_frame.macro_f1.mean()),"macro_f1_std":float(stability_frame.macro_f1.std(ddof=1))},
              "promotion_gate_passed":meaningful,"promotion_rule":"macro F1 >= v3 + 0.02 and cognitive_fog/neutral do not decline",
              "final_tests_opened":False,"artifact_sha256":hashlib.sha256(artifact.read_bytes()).hexdigest(),"minilm_checkpoint":MINILM if best.representation=="minilm" else None}
    (HERE/"candidate_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(results.head(12).to_string(index=False)); print(json.dumps(metadata,indent=2))


if __name__=="__main__": main()
