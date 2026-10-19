import argparse
import re
from pathlib import Path

import pandas as pd


def clean_text(text):
    text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_dataset(input_path, output_path, text_column, label_column):
    data = pd.read_csv(input_path)
    total_rows_before = len(data)

    missing_columns = [column for column in (text_column, label_column) if column not in data.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Input CSV is missing required column(s): {missing}")

    prepared = data[[text_column, label_column]].copy()
    prepared.columns = ["text", "label"]
    prepared = prepared.dropna(subset=["text", "label"])
    prepared["text"] = prepared["text"].apply(clean_text)
    prepared["label"] = prepared["label"].astype(str).str.strip()
    prepared = prepared[(prepared["text"] != "") & (prepared["label"] != "")]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)

    print(f"Total rows before cleaning: {total_rows_before}")
    print(f"Total rows after cleaning: {len(prepared)}")
    print("Label distribution:")
    print(prepared["label"].value_counts().to_string())
    print(f"Output file: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a public emotion dataset for Component 4 emotion classifier training."
    )
    parser.add_argument("--input", required=True, help="Path to the raw public dataset CSV.")
    parser.add_argument("--output", required=True, help="Path where the prepared CSV should be saved.")
    parser.add_argument("--text-column", required=True, help="Name of the input CSV text column.")
    parser.add_argument("--label-column", required=True, help="Name of the input CSV label column.")
    return parser.parse_args()


def main():
    args = parse_args()
    prepare_dataset(
        input_path=Path(args.input),
        output_path=Path(args.output),
        text_column=args.text_column,
        label_column=args.label_column,
    )


if __name__ == "__main__":
    main()
