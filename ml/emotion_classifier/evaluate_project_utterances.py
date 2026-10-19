from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent


def normalize(text):
    return " ".join(str(text).lower().split())


def main():
    evaluation = pd.read_csv(ROOT / "data" / "project_evaluation_set.csv")
    training = pd.read_csv(ROOT / "data" / "training_dataset_v2.csv")
    overlap = set(evaluation.text.map(normalize)) & set(training.text.map(normalize))
    if overlap:
        raise ValueError(f"Project evaluation leakage: {sorted(overlap)}")

    pipeline = joblib.load(ROOT / "emotion_pipeline.pkl")
    probabilities = pipeline.predict_proba(evaluation.text)
    predictions = pipeline.classes_[probabilities.argmax(axis=1)]
    evaluation["predicted_emotion"] = predictions
    evaluation["confidence"] = probabilities.max(axis=1)
    evaluation["source"] = "ml_model"
    evaluation["correct"] = evaluation.expected_emotion == evaluation.predicted_emotion
    output = ROOT / "results" / "project_utterance_evaluation.csv"
    evaluation.to_csv(output, index=False)
    print(evaluation.to_string(index=False))
    print(f"\nAccuracy: {evaluation.correct.mean():.4f}; exact training overlap: 0")


if __name__ == "__main__":
    main()
