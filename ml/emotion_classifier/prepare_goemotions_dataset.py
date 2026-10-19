import argparse
import re
from pathlib import Path

import pandas as pd


GOEMOTIONS_FILES = (
    "goemotions_1.csv",
    "goemotions_2.csv",
    "goemotions_3.csv",
)

GOEMOTIONS_LABEL_MAPPING = {
    "joy": "happiness",
    "amusement": "happiness",
    "excitement": "happiness",
    "gratitude": "happiness",
    "love": "happiness",
    "optimism": "happiness",
    "pride": "happiness",
    "relief": "happiness",
    "sadness": "sadness",
    "grief": "sadness",
    "disappointment": "sadness",
    "remorse": "sadness",
    "fear": "anxiety",
    "nervousness": "anxiety",
    "embarrassment": "anxiety",
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    "disgust": "anger",
    "confusion": "cognitive_fog",
    "curiosity": "cognitive_fog",
    "realization": "cognitive_fog",
    "neutral": "neutral",
}


def clean_text(text):
    text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_goemotions_files(raw_dir):
    frames = []

    for filename in GOEMOTIONS_FILES:
        input_path = raw_dir / filename
        if not input_path.exists():
            raise FileNotFoundError(f"Missing GoEmotions file: {input_path}")
        frames.append(pd.read_csv(input_path))

    return pd.concat(frames, ignore_index=True)


def prepare_goemotions_dataset(raw_dir, output_path):
    data = load_goemotions_files(raw_dir)
    raw_rows_loaded = len(data)

    if "text" not in data.columns:
        raise ValueError("GoEmotions input is missing required text column: text")

    missing_label_columns = [
        label for label in GOEMOTIONS_LABEL_MAPPING if label not in data.columns
    ]
    if missing_label_columns:
        missing = ", ".join(missing_label_columns)
        raise ValueError(f"GoEmotions input is missing expected label column(s): {missing}")

    prepared_rows = []
    skipped_rows = 0

    for _, row in data.iterrows():
        text = clean_text(row["text"])
        if not text:
            skipped_rows += 1
            continue

        mapped_labels = {
            project_label
            for raw_label, project_label in GOEMOTIONS_LABEL_MAPPING.items()
            if int(row.get(raw_label, 0) or 0) == 1
        }

        if not mapped_labels:
            skipped_rows += 1
            continue

        for label in sorted(mapped_labels):
            prepared_rows.append({"text": text, "label": label})

    prepared = pd.DataFrame(prepared_rows, columns=["text", "label"])
    rows_before_deduplication = len(prepared)
    prepared = prepared.drop_duplicates(subset=["text", "label"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)

    print(f"Raw rows loaded: {raw_rows_loaded}")
    print(f"Prepared rows before duplicate removal: {rows_before_deduplication}")
    print(f"Prepared rows saved: {len(prepared)}")
    print(f"Rows skipped: {skipped_rows}")
    print("Label distribution after mapping:")
    print(prepared["label"].value_counts().to_string())
    print(f"Output file: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare GoEmotions multi-label CSV files for Component 4 emotion classifier training."
    )
    parser.add_argument(
        "--raw-dir",
        default=Path("data") / "raw",
        type=Path,
        help="Directory containing goemotions_1.csv, goemotions_2.csv, and goemotions_3.csv.",
    )
    parser.add_argument(
        "--output",
        default=Path("data") / "goemotions_prepared_dataset.csv",
        type=Path,
        help="Path where the prepared text,label CSV should be saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    prepare_goemotions_dataset(raw_dir=args.raw_dir, output_path=args.output)


if __name__ == "__main__":
    main()
