"""Prepare and freeze advanced Stage 4 data before transformer tuning."""

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
LABELS = ["happiness", "sadness", "loneliness", "anxiety", "anger", "cognitive_fog", "neutral"]
SEED = 42


def normalize(text):
    value = str(text or "").lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", value).strip()


def frame_hash(frame):
    payload = frame.sort_values(["text", "label", "source", "domain"]).to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode()).hexdigest()


def nearest_pairs(left, right, threshold=0.85):
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    left_vectors = vectorizer.fit_transform(left.text)
    right_vectors = vectorizer.transform(right.text)
    distances, indexes = NearestNeighbors(n_neighbors=1, metric="cosine").fit(left_vectors).kneighbors(right_vectors)
    return [{"similarity": float(1 - distances[i, 0]), "left": left.iloc[indexes[i, 0]].text,
             "right": right.iloc[i].text} for i in range(len(right)) if 1 - distances[i, 0] >= threshold]


def freeze_domain_test():
    original = pd.read_csv(ROOT / "data" / "project_evaluation_set.csv").rename(columns={"expected_emotion": "label"})
    original["source"] = "project_authored_independent_v3"
    original["domain"] = "eldermeds_conversation"
    additions = pd.read_csv(HERE / "domain_test_additions.csv")
    domain = pd.concat([original, additions], ignore_index=True)
    domain["text"] = domain.text.map(normalize)
    if domain.text.duplicated().any() or set(domain.label) != set(LABELS):
        raise ValueError("Invalid domain-test labels or duplicate text")
    if not (domain.groupby("label").size() == 20).all():
        raise ValueError("Domain test must contain exactly 20 examples per class")

    development = pd.read_csv(ROOT / "data" / "development_dataset_v3.csv")
    locked = pd.read_csv(ROOT / "data" / "locked_test_v2.csv")
    model_data = pd.concat([development, locked], ignore_index=True)
    model_data["text"] = model_data.text.map(normalize)
    exact = set(domain.text) & set(model_data.text)
    near = nearest_pairs(model_data, domain)
    if exact or near:
        raise ValueError(f"Domain-test leakage: exact={len(exact)}, near={len(near)}")
    domain.to_csv(HERE / "domain_test_frozen.csv", index=False)
    metadata = {"rows": len(domain), "class_support": domain.label.value_counts().sort_index().to_dict(),
                "source_support": domain.source.value_counts().sort_index().to_dict(),
                "domain": "eldermeds_conversation", "sha256_sorted_csv": frame_hash(domain),
                "exact_model_data_overlap": 0, "near_duplicate_threshold": 0.85,
                "near_duplicate_model_data_pairs": 0,
                "prohibited_uses": ["training", "model_selection", "hyperparameter_tuning", "calibration", "threshold_selection"]}
    (HERE / "domain_test_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    print(json.dumps(freeze_domain_test(), indent=2))
