"""Local demo for experimental DailyDialog artifacts; never used by production."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from .dailydialog import CLASS_ORDER, INPUT_FORMATS, LICENSE, format_public_input


def load_public_artifact(path: Path):
    path = Path(path)
    artifact = joblib.load(path)
    if (artifact.get("artifact_type") != "public_research_experiment_only"
            or artifact.get("dataset_origin") != "public_research"
            or artifact.get("source_license") != LICENSE
            or not artifact.get("source_citation")
            or artifact.get("population_status") != "public_nonclinical_not_eldermeds_data"
            or artifact.get("input_format") not in INPUT_FORMATS
            or artifact.get("class_order") != CLASS_ORDER):
        raise ValueError("artifact is not a supported isolated public-data experiment")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_name(path.stem + ".metadata.json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("artifact_sha256") != digest:
        raise ValueError("public experiment artifact checksum mismatch")
    if not hasattr(artifact.get("model"), "predict_proba"):
        raise ValueError("public experiment artifact has no probability classifier")
    return artifact


def predict(artifact, encoder, utterance, *, speaker="user", previous_turns=None):
    record = {"sample_id": "local-demo", "dialogue_id": "local-demo-dialogue", "turn_index": len(previous_turns or []),
              "speaker": speaker, "current_utterance": utterance, "previous_turns": previous_turns or []}
    text = format_public_input(record, artifact["input_format"], tokenizer=encoder.tokenizer,
                               max_tokens=int(artifact["max_seq_length"]))
    features = encoder.encode([text], batch_size=1, show_progress_bar=False,
                              normalize_embeddings=True, convert_to_numpy=True)
    probabilities = artifact["model"].predict_proba(np.asarray(features, dtype=np.float32))[0]
    classes = list(artifact["model"].classes_)
    ordered = {label: float(probabilities[classes.index(label)]) for label in CLASS_ORDER}
    return {"dataset_origin": "public_research", "population_status": artifact["population_status"],
            "input_format": artifact["input_format"], "prediction": max(ordered, key=ordered.get),
            "probability_calibration": "sigmoid_calibrated_on_separate_public_training_dialogues",
            "class_probabilities": ordered,
            "warning": "Research demo only; not clinically validated and not for ElderMeds production."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--speaker", choices=("user", "system"), default="user")
    parser.add_argument("--previous-turn", action="append", default=[], metavar="SPEAKER|TEXT",
                        help="Repeat at most twice, in chronological order.")
    args = parser.parse_args(argv)
    if len(args.previous_turn) > 2:
        parser.error("at most two previous turns are supported")
    previous = []
    for index, value in enumerate(args.previous_turn):
        if "|" not in value:
            parser.error("--previous-turn must use SPEAKER|TEXT")
        speaker, text = value.split("|", 1)
        if speaker not in {"user", "system"} or not text.strip():
            parser.error("previous turn needs speaker user/system and nonempty text")
        previous.append({"turn_index": index, "speaker": speaker, "utterance": text.strip()})
    artifact = load_public_artifact(args.artifact)
    if artifact["input_format"] == "answer_only" and previous:
        parser.error("selected artifact is answer_only and does not accept previous turns")
    if artifact["input_format"] == "one_previous" and len(previous) > 1:
        parser.error("selected artifact accepts at most one previous turn")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(artifact["checkpoint"], device="cpu", local_files_only=True)
    print(json.dumps(predict(artifact, encoder, args.text, speaker=args.speaker,
                             previous_turns=previous), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
