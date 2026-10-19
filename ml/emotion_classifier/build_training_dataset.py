import argparse
import re
import string
from pathlib import Path

import pandas as pd


SEED = 42
PROJECT_LABELS = {
    "happiness", "sadness", "loneliness", "anxiety", "anger",
    "cognitive_fog", "neutral",
}
CUSTOM_LABEL_MAPPING = {
    "happy": "happiness",
    "sad": "sadness",
    "lonely": "loneliness",
    "anxious": "anxiety",
    "angry": "anger",
    "confused": "cognitive_fog",
    "neutral": "neutral",
}
EXPLICIT_LONELINESS_PATTERN = re.compile(
    r"\b(lonely|loneliness|feel alone|felt alone|i am alone|i'm alone|im alone|"
    r"isolated|no one to talk|nobody to talk)\b",
    re.IGNORECASE,
)


def normalize_text(text):
    normalized = str(text or "").lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", normalized).strip()


def build_dataset(goemotions_path, custom_path, output_path, max_external_per_class=300):
    external = pd.read_csv(goemotions_path, usecols=["text", "label"])
    external["text"] = external["text"].map(normalize_text)
    external = external[external["label"].isin(PROJECT_LABELS - {"loneliness"})]
    external = external[external["text"] != ""]

    explicit_loneliness = external[
        external["text"].str.contains(EXPLICIT_LONELINESS_PATTERN, na=False)
    ][["text"]].drop_duplicates()
    explicit_loneliness["label"] = "loneliness"
    explicit_loneliness["source"] = "domain_relabel"

    non_loneliness = external[~external["text"].isin(explicit_loneliness["text"])]
    label_counts_per_text = non_loneliness.groupby("text")["label"].nunique()
    unambiguous_texts = label_counts_per_text[label_counts_per_text == 1].index
    unambiguous = non_loneliness[non_loneliness["text"].isin(unambiguous_texts)].drop_duplicates("text")
    sampled_parts = []
    for label, rows in unambiguous.groupby("label", sort=True):
        count = min(len(rows), max_external_per_class)
        sampled = rows.sample(n=count, random_state=SEED).copy()
        sampled["source"] = "goemotions"
        sampled_parts.append(sampled[["text", "label", "source"]])

    custom = pd.read_csv(custom_path, usecols=["text", "label"])
    custom["text"] = custom["text"].map(normalize_text)
    custom["label"] = custom["label"].astype(str).str.lower().map(CUSTOM_LABEL_MAPPING)
    custom = custom.dropna(subset=["label"])
    custom = custom[custom["text"] != ""]
    custom["source"] = "project_authored"

    combined = pd.concat(sampled_parts + [explicit_loneliness, custom], ignore_index=True)
    conflicts = combined.groupby("text")["label"].nunique()
    conflicting_texts = set(conflicts[conflicts > 1].index)
    combined = combined[~combined["text"].isin(conflicting_texts)]
    source_order = {
        "project_authored": 0,
        "domain_relabel": 1,
        "goemotions": 2,
    }
    combined["_source_order"] = combined["source"].map(source_order)
    combined = combined.sort_values(["_source_order", "text"]).drop_duplicates("text")
    combined = combined.drop(columns=["_source_order"]).sort_values(["label", "text"]).reset_index(drop=True)

    missing = PROJECT_LABELS.difference(combined["label"].unique())
    if missing:
        raise ValueError(f"Prepared dataset is missing labels: {sorted(missing)}")
    if combined["text"].duplicated().any():
        raise ValueError("Exact duplicate texts remain after preparation.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Saved {len(combined)} rows to {output_path}")
    print(combined.groupby(["label", "source"]).size().to_string())
    print(f"Removed conflicting combined texts: {len(conflicting_texts)}")


def parse_args():
    module_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build the seven-label ElderMeds emotion dataset.")
    parser.add_argument("--goemotions", type=Path, default=module_dir / "data" / "goemotions_prepared_dataset.csv")
    parser.add_argument("--custom", type=Path, default=module_dir.parents[2] / "ml-service" / "data" / "raw" / "custom_text.csv")
    parser.add_argument("--output", type=Path, default=module_dir / "data" / "training_dataset_v2.csv")
    parser.add_argument("--max-external-per-class", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(args.goemotions, args.custom, args.output, args.max_external_per_class)
