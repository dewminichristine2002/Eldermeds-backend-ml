"""Build provenance-aware advanced English development data without test reuse."""

import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from prepare_advanced_data import LABELS, ROOT, HERE, normalize

SEED = 42
TARGETS = {"happiness": 500, "sadness": 500, "anxiety": 500, "anger": 400,
           "neutral": 500, "cognitive_fog": 400, "loneliness": 400}
SAFE_EXTERNAL_LABELS = {"happiness", "sadness", "anxiety", "anger", "neutral"}


def remove_near_test_candidates(candidates, final_tests, threshold=0.85):
    if candidates.empty:
        return candidates, 0
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    test_vectors = vectorizer.fit_transform(final_tests.text)
    candidate_vectors = vectorizer.transform(candidates.text)
    distances, _ = NearestNeighbors(n_neighbors=1, metric="cosine").fit(test_vectors).kneighbors(candidate_vectors)
    keep = (1.0 - distances[:, 0]) < threshold
    return candidates.loc[keep].copy(), int((~keep).sum())


def main():
    base = pd.read_csv(ROOT / "data" / "development_dataset_v3.csv")
    base["text"] = base.text.map(normalize)
    base["domain"] = base.source.map({"goemotions": "general_english_social_media",
                                      "domain_relabel": "general_english_social_media",
                                      "project_authored": "eldermeds_conversation"})
    hard = pd.read_csv(HERE / "hard_negative_examples.csv")
    hard["text"] = hard.text.map(normalize)
    hard = hard[["text", "label", "source", "domain"]]

    pool = pd.read_csv(ROOT / "data" / "goemotions_prepared_dataset.csv")
    pool["text"] = pool.text.map(normalize)
    pool = pool[pool.label.isin(SAFE_EXTERNAL_LABELS) & pool.text.ne("")]
    pool = pool[pool.text.str.split().map(len).between(4, 60)]
    unambiguous = pool.groupby("text").label.nunique()
    pool = pool[pool.text.isin(unambiguous[unambiguous == 1].index)].drop_duplicates("text")

    locked = pd.read_csv(ROOT / "data" / "locked_test_v2.csv")
    domain_test = pd.read_csv(HERE / "domain_test_frozen.csv")
    final_tests = pd.concat([locked[["text"]], domain_test[["text"]]], ignore_index=True)
    final_tests["text"] = final_tests.text.map(normalize)
    forbidden = set(base.text) | set(hard.text) | set(final_tests.text)
    pool = pool[~pool.text.isin(forbidden)].copy()
    pool, removed_near = remove_near_test_candidates(pool, final_tests)

    pieces = [base, hard]
    interim = pd.concat(pieces, ignore_index=True)
    for label in sorted(SAFE_EXTERNAL_LABELS):
        needed = max(0, TARGETS[label] - int((interim.label == label).sum()))
        available = pool[pool.label == label]
        sampled = available.sample(n=min(needed, len(available)), random_state=SEED).copy()
        sampled["source"] = "goemotions"
        sampled["domain"] = "general_english_social_media"
        pieces.append(sampled[["text", "label", "source", "domain"]])

    advanced = pd.concat(pieces, ignore_index=True)
    conflicts = advanced.groupby("text").label.nunique()
    conflicting = set(conflicts[conflicts > 1].index)
    advanced = advanced[~advanced.text.isin(conflicting)].drop_duplicates("text").reset_index(drop=True)
    if set(advanced.text) & set(final_tests.text):
        raise ValueError("Exact final-test leakage")
    advanced.to_csv(HERE / "advanced_development.csv", index=False)
    distribution = advanced.groupby("label").size().rename("count").reset_index()
    distribution["target"] = distribution.label.map(TARGETS)
    distribution["target_met"] = distribution["count"] >= distribution["target"]
    distribution.to_csv(HERE / "advanced_dataset_distribution.csv", index=False)
    advanced.groupby(["label", "source", "domain"]).size().rename("count").reset_index().to_csv(
        HERE / "advanced_source_distribution.csv", index=False)
    audit = {"rows": len(advanced), "exact_duplicates": int(advanced.text.duplicated().sum()),
             "conflicting_texts_removed": len(conflicting), "near_final_test_candidates_removed": removed_near,
             "short_le_2_tokens": int(advanced.text.str.split().map(len).le(2).sum()),
             "note": "Cognitive-fog and loneliness targets were not padded from semantically unsafe external mappings."}
    (HERE / "advanced_data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(distribution.to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
