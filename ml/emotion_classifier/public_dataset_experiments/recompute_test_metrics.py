"""Recompute final test metrics from frozen public experiment artifacts.

This post-hoc utility does not retrain or retune candidates. It is useful when
reporting code is corrected while preserving the original dialogue split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from sentence_transformers import SentenceTransformer

import joblib

from .dailydialog import CITATION, INPUT_FORMATS, LICENSE, SOURCE_URL, sha256_file
from .predict_dailydialog import load_public_artifact, predict as predict_public_demo
from .train_dailydialog import (_aligned_probabilities, _encode, _metrics,
                                create_partitions, load_records)


def repair_missing_citation_metadata(run_dir: Path):
    """Add citation provenance only to artifacts whose source identity is verified."""
    repaired = []
    for input_format in INPUT_FORMATS:
        artifact_path = Path(run_dir) / "artifacts" / f"dailydialog_{input_format}.joblib"
        artifact = joblib.load(artifact_path)
        if artifact.get("source_citation"):
            continue
        if (artifact.get("dataset_id") != "dailydialog_public_emotion_context_v1"
                or artifact.get("dataset_origin") != "public_research"
                or artifact.get("source_url") != SOURCE_URL
                or artifact.get("source_license") != LICENSE
                or artifact.get("population_status") != "public_nonclinical_not_eldermeds_data"):
            raise ValueError(f"cannot establish source provenance for {artifact_path}")
        artifact["source_citation"] = CITATION
        joblib.dump(artifact, artifact_path, compress=3)
        sidecar_path = artifact_path.with_name(artifact_path.stem + ".metadata.json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar["source_citation"] = CITATION
        sidecar["artifact_sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        repaired.append(input_format)
    return repaired


def recompute(run_dir: Path, data_path: Path, batch_size: int = 64, *, repair_citation=False):
    run_dir, data_path = Path(run_dir), Path(data_path)
    repaired = repair_missing_citation_metadata(run_dir) if repair_citation else []
    report_path = run_dir / "results" / "experiment_results.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    split = json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8"))
    caps = split["deterministic_source_subset"]
    records = load_records(data_path)
    partitions, manifest, records = create_partitions(
        records,
        max_dialogues_per_split={
            key: value["selected_dialogues"] for key, value in caps.items()
        },
    )
    if manifest["dialogue_counts"] != split["dialogue_counts"]:
        raise ValueError("reconstructed dialogue partition does not match the saved run")

    encoder = SentenceTransformer(report["checkpoint"], device="cpu", local_files_only=True)
    encoder.max_seq_length = report["max_seq_length"]
    labels = [records[int(i)]["mapped_label"] for i in partitions["test"]]
    corrected = {}
    citations_verified = True
    for input_format in INPUT_FORMATS:
        model_info = report["models"][input_format]
        artifact_path = run_dir / "artifacts" / f"dailydialog_{input_format}.joblib"
        artifact = load_public_artifact(artifact_path)
        citations_verified = citations_verified and artifact.get("source_citation") == CITATION
        if not citations_verified:
            raise ValueError(f"source citation mismatch in {artifact_path}")
        features, _, _ = _encode(encoder, records, partitions["test"], input_format, batch_size)
        probabilities = _aligned_probabilities(artifact["model"], features)
        predictions = [artifact["class_order"][index] for index in probabilities.argmax(axis=1)]
        metrics = _metrics(labels, predictions, probabilities)
        model_info["test_metrics"] = metrics
        model_info["artifact_sha256"] = sha256_file(artifact_path)
        smoke_record = next((row for row in records if len(row["previous_turns"]) >= 2
                             and row["native_split"] == "test"), records[int(partitions["test"][0])])
        smoke = predict_public_demo(artifact, encoder, smoke_record["current_utterance"],
                                   speaker=smoke_record["speaker"],
                                   previous_turns=smoke_record["previous_turns"])
        model_info["artifact_inference_smoke"] = {
            "status": "passed", "prediction_in_reduced_public_taxonomy": smoke["prediction"],
            "probability_sum": float(sum(smoke["class_probabilities"].values())),
            "artifact_sha256_verified": True,
        }
        corrected[input_format] = {
            "artifact_sha256_verified": True,
            "log_loss": metrics["log_loss"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        }

    report["metric_recalculation"] = {
        "status": "completed",
        "reason": "probability-column-safe negative log-likelihood computation",
        "method": "predictions recalculated using the frozen checksum-verified artifacts on the original test dialogue groups; no model fitting or tuning",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "citation_metadata_repaired": repaired,
        "citation_metadata_verified": citations_verified,
        "models": corrected,
    }
    _write_comparison(run_dir / "results" / "context_model_comparison.svg", report)
    temporary = report_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(report_path)
    return report["metric_recalculation"]


def _write_comparison(path: Path, report: dict):
    from .train_dailydialog import _draw_comparison
    formats = ("answer_only", "one_previous", "two_previous")
    _draw_comparison(
        path,
        [report["models"][fmt]["test_metrics"]["accuracy"] for fmt in formats],
        [report["models"][fmt]["test_metrics"]["macro_f1"] for fmt in formats],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data" / "processed" / "dailydialog_public_v1" / "dailydialog_mapped_turns.jsonl")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repair-missing-source-citation", action="store_true",
                        help="Add the verified DailyDialog citation to legacy artifacts missing only that provenance field and refresh checksums.")
    args = parser.parse_args(argv)
    print(json.dumps(recompute(args.run_dir, args.data, args.batch_size,
                               repair_citation=args.repair_missing_source_citation), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
