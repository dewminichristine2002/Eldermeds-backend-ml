import argparse
import json
import re
import string
from pathlib import Path

import joblib


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_PIPELINE_PATH = MODULE_DIR / "emotion_pipeline.pkl"
DEFAULT_METADATA_PATH = MODULE_DIR / "selected_model_metadata.json"


def clean_text(text):
    normalized = "" if text is None else str(text).lower()
    normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", normalized).strip()


def predict_emotion(text, pipeline_path=DEFAULT_PIPELINE_PATH, metadata_path=DEFAULT_METADATA_PATH):
    pipeline = joblib.load(pipeline_path)
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    cleaned_text = clean_text(text)
    probabilities = pipeline.predict_proba([cleaned_text])[0]
    index = int(probabilities.argmax())
    return {
        "emotion": str(pipeline.classes_[index]),
        "confidence": float(probabilities[index]),
        "source": "ml_model",
        "model_version": metadata["model_version"],
        "supported_classes": metadata["supported_classes"],
    }


def main():
    parser = argparse.ArgumentParser(description="Predict an ElderMeds emotional-support class.")
    parser.add_argument("text")
    parser.add_argument("--pipeline-path", type=Path, default=DEFAULT_PIPELINE_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    args = parser.parse_args()
    print(json.dumps(predict_emotion(args.text, args.pipeline_path, args.metadata_path), indent=2))


if __name__ == "__main__":
    main()
