"""Freeze and audit the domain/hierarchical experiment inputs before tuning."""

import hashlib
import json
import re
import string
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ADVANCED = ROOT / "advanced_experiment"
LABELS = ["happiness", "sadness", "loneliness", "anxiety", "anger", "cognitive_fog", "neutral"]


def normalize(text):
    value = str(text or "").lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", value).strip()


def canonical_hash(frame, columns):
    payload = frame.sort_values(columns).to_csv(index=False, columns=columns, lineterminator="\n")
    return hashlib.sha256(payload.encode()).hexdigest()


def near_count(reference, query, threshold=0.85):
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    reference_vectors = vectorizer.fit_transform(reference)
    query_vectors = vectorizer.transform(query)
    distances, _ = NearestNeighbors(n_neighbors=1, metric="cosine").fit(reference_vectors).kneighbors(query_vectors)
    return int(((1.0 - distances[:, 0]) >= threshold).sum())


def main():
    domain_dev = pd.read_csv(HERE / "domain_development_evaluation.csv")
    domain_dev["text"] = domain_dev.text.map(normalize)
    if domain_dev.text.duplicated().any() or set(domain_dev.label) != set(LABELS):
        raise ValueError("Invalid domain-development set")
    if not (domain_dev.groupby("label").size() == 8).all():
        raise ValueError("Expected eight domain-development examples per class")

    training = pd.read_csv(ADVANCED / "advanced_development_split.csv")
    original_test = pd.read_csv(ADVANCED / "baseline_v3" / "data" / "locked_test_v2.csv")
    final_domain = pd.read_csv(ADVANCED / "domain_test_frozen.csv")
    for frame in [training, original_test, final_domain]: frame["text"] = frame.text.map(normalize)
    exact_training = set(domain_dev.text) & set(training.text)
    exact_original = set(domain_dev.text) & set(original_test.text)
    exact_domain = set(domain_dev.text) & set(final_domain.text)
    near_training = near_count(training.text, domain_dev.text)
    near_original = near_count(original_test.text, domain_dev.text)
    near_domain = near_count(final_domain.text, domain_dev.text)
    if exact_training or exact_original or exact_domain or near_training or near_original or near_domain:
        raise ValueError("Domain-development leakage detected")

    conflict_count = int((training.groupby("text").label.nunique() > 1).sum())
    audit_rows = [
        {"check":"domain_development_rows","value":len(domain_dev)},
        {"check":"training_rows","value":len(training)},
        {"check":"training_exact_duplicates","value":int(training.text.duplicated().sum())},
        {"check":"training_conflicting_labels","value":conflict_count},
        {"check":"domain_dev_exact_training_overlap","value":len(exact_training)},
        {"check":"domain_dev_exact_original_test_overlap","value":len(exact_original)},
        {"check":"domain_dev_exact_domain_test_overlap","value":len(exact_domain)},
        {"check":"domain_dev_near_training_overlap_ge_0_85","value":near_training},
        {"check":"domain_dev_near_original_test_overlap_ge_0_85","value":near_original},
        {"check":"domain_dev_near_domain_test_overlap_ge_0_85","value":near_domain},
        {"check":"short_training_records_le_2_tokens","value":int(training.text.str.split().map(len).le(2).sum())},
    ]
    pd.DataFrame(audit_rows).to_csv(HERE / "domain_dataset_audit.csv", index=False)
    training.groupby(["label","source","domain"]).size().rename("count").reset_index().to_csv(HERE / "domain_source_distribution.csv",index=False)
    training.groupby("label").size().rename("count").reset_index().to_csv(HERE / "domain_class_distribution.csv",index=False)
    training.to_csv(HERE / "domain_development_split.csv",index=False)
    domain_dev.to_csv(HERE / "domain_development_evaluation_frozen.csv",index=False)
    review = domain_dev[["text","label"]].rename(columns={"label":"original_label"})
    review["reviewer_1_label"]=""; review["reviewer_2_label"]=""; review["agreement"]=""
    review.to_csv(HERE / "label_ambiguity_review.csv",index=False)
    metadata={"domain_development_rows":len(domain_dev),"support":domain_dev.label.value_counts().sort_index().to_dict(),
              "sha256":canonical_hash(domain_dev,["text","label","source","domain"]),"near_duplicate_threshold":0.85,
              "known_exact_final_test_leakage":0,"known_near_final_test_leakage":0,
              "review_note":"Reviewer columns intentionally blank; independent human reviewers were unavailable."}
    (HERE / "domain_development_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(json.dumps(metadata,indent=2))


if __name__ == "__main__": main()
