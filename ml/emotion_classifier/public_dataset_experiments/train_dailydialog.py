"""Train isolated public-data DailyDialog classifiers; never registers models."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GroupShuffleSplit

from .dailydialog import (CITATION, CLASS_ORDER, INPUT_FORMATS, LICENSE, SOURCE_URL,
                          format_public_input, sha256_file)
from .predict_dailydialog import load_public_artifact, predict as predict_public_demo

CHECKPOINT = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42
HYPERPARAMETERS = [(c, weight) for c in (0.1, 1.0) for weight in (None, "balanced")]


def load_records(path: Path):
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("prepared public dataset is empty")
    required = {"sample_id", "dialogue_id", "native_split", "turn_index", "speaker",
                "current_utterance", "previous_turns", "original_label", "mapped_label",
                "dataset_origin", "source_license", "source_citation"}
    if any(not required.issubset(item) for item in records):
        raise ValueError("prepared public records are missing provenance or context fields")
    if any(item["dataset_origin"] != "public_research" or item["source_license"] != LICENSE for item in records):
        raise ValueError("only explicitly licensed public_research rows can enter this experiment")
    if any(item["mapped_label"] not in CLASS_ORDER for item in records):
        raise ValueError("record contains a label outside the declared reduced public taxonomy")
    if len({item["sample_id"] for item in records}) != len(records):
        raise ValueError("sample_id values must be unique")
    dialogue_splits = {}
    for item in records:
        prior = dialogue_splits.setdefault(item["dialogue_id"], item["native_split"])
        if prior != item["native_split"]:
            raise ValueError("dialogue-level split leakage detected")
        for turn in item["previous_turns"]:
            if turn["turn_index"] >= item["turn_index"]:
                raise ValueError("future-turn leakage detected")
    return records


def create_partitions(records, max_dialogues_per_split=None):
    selected_records = records
    selected_indices = []
    subset_manifest = {}
    if max_dialogues_per_split:
        for split in ("train", "validation", "test"):
            split_indices = [i for i, row in enumerate(records) if row["native_split"] == split]
            dialogue_ids = sorted({records[i]["dialogue_id"] for i in split_indices},
                                  key=lambda value: hashlib.sha256(f"{SEED}:{split}:{value}".encode()).hexdigest())
            limit = max_dialogues_per_split.get(split)
            selected_dialogues = set(dialogue_ids[:limit]) if limit else set(dialogue_ids)
            kept = [i for i in split_indices if records[i]["dialogue_id"] in selected_dialogues]
            selected_indices.extend(kept)
            subset_manifest[split] = {
                "available_dialogues": len(dialogue_ids), "selected_dialogues": len(selected_dialogues),
                "selected_dialogue_ids_sha256": hashlib.sha256("\n".join(sorted(selected_dialogues)).encode()).hexdigest(),
                "selected_samples": len(kept), "selection": "stable SHA-256 ordering by seed and split; labels were not used for subset selection",
            }
        selected_records = [records[i] for i in selected_indices]
    records = selected_records
    train_indices = np.asarray([i for i, row in enumerate(records) if row["native_split"] == "train"])
    validation = np.asarray([i for i, row in enumerate(records) if row["native_split"] == "validation"])
    test = np.asarray([i for i, row in enumerate(records) if row["native_split"] == "test"])
    y = np.asarray([row["mapped_label"] for row in records])
    groups = np.asarray([row["dialogue_id"] for row in records])
    if not len(train_indices) or not len(validation) or not len(test):
        raise ValueError("official train, validation and test splits must all be present")
    expected = set(CLASS_ORDER)
    if set(y[validation]) != expected or set(y[test]) != expected:
        raise ValueError("official validation and test splits must support every reduced public class")
    fit = calibration = None
    split_seed = None
    for candidate_seed in range(SEED, SEED + 200):
        split = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=candidate_seed)
        fit_rel, cal_rel = next(split.split(train_indices, y[train_indices], groups[train_indices]))
        maybe_fit, maybe_cal = train_indices[fit_rel], train_indices[cal_rel]
        if set(y[maybe_fit]) == expected and set(y[maybe_cal]) == expected:
            fit, calibration, split_seed = maybe_fit, maybe_cal, candidate_seed
            break
    if fit is None:
        raise ValueError("could not create a dialogue-disjoint calibration split with all reduced classes")
    partitions = {"fit": fit, "calibration": calibration, "validation": validation, "test": test}
    for left_name, right_name in (("fit", "calibration"), ("fit", "validation"), ("fit", "test"),
                                  ("calibration", "validation"), ("calibration", "test"), ("validation", "test")):
        left = set(groups[partitions[left_name]])
        right = set(groups[partitions[right_name]])
        if left & right:
            raise ValueError(f"dialogue overlap between {left_name} and {right_name}")
    return partitions, {"seed": split_seed, "group_key": "dialogue_id",
                        "source_split_policy": "dialogue-disjoint subsets of official DailyDialog splits; 10% of selected official train dialogues reserved for calibration",
                        "dialogue_counts": {name: len(set(groups[index])) for name, index in partitions.items()},
                        "sample_counts": {name: int(len(index)) for name, index in partitions.items()},
                        "deterministic_source_subset": subset_manifest or "all mapped rows from official source splits",
                        "dialogue_overlap": False}, selected_records


def _encode(encoder, records, indices, input_format, batch_size=64):
    inputs = [format_public_input(records[int(index)], input_format,
                                  tokenizer=encoder.tokenizer, max_tokens=int(encoder.max_seq_length))
              for index in indices]
    requested = 0 if input_format == "answer_only" else 1 if input_format == "one_previous" else 2
    retained = [min(requested, len(records[int(index)]["previous_turns"]),
                    sum(line.startswith("Previous turn by ") for line in value.splitlines()))
                for index, value in zip(indices, inputs)]
    features = encoder.encode(inputs, batch_size=batch_size, show_progress_bar=True,
                              normalize_embeddings=True, convert_to_numpy=True)
    return np.asarray(features, dtype=np.float32), inputs, {
        "records": len(inputs), "context_turns_requested": int(requested * len(inputs)),
        "context_turns_retained": int(sum(retained)),
        "context_turns_removed_for_token_budget": int(requested * len(inputs) - sum(retained)),
    }


def _metrics(labels, predictions, probabilities):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    report = classification_report(labels, predictions, labels=CLASS_ORDER, output_dict=True, zero_division=0)
    one_hot = np.eye(len(CLASS_ORDER))[[CLASS_ORDER.index(label) for label in labels]]
    confidence = probabilities.max(axis=1)
    class_indices = np.asarray([CLASS_ORDER.index(label) for label in labels])
    nll = -np.log(np.clip(probabilities[np.arange(len(labels)), class_indices], 1e-15, 1.0)).mean()
    correct = predictions == labels
    ece = 0.0
    for lower in np.linspace(0, 1, 11)[:-1]:
        upper = lower + 0.1
        mask = (confidence >= lower) & ((confidence <= upper) if upper >= 1 else (confidence < upper))
        if mask.any():
            ece += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(labels, predictions, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, predictions, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "per_class": {label: {"precision": float(report[label]["precision"]),
                               "recall": float(report[label]["recall"]),
                               "f1": float(report[label]["f1-score"]),
                               "support": int(report[label]["support"])} for label in CLASS_ORDER},
        "confusion_matrix": confusion_matrix(labels, predictions, labels=CLASS_ORDER).tolist(),
        # Compute NLL in declared CLASS_ORDER directly. sklearn.log_loss expects
        # lexicographically ordered labels and would misread these columns.
        "log_loss": float(nll),
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece_10_bins": ece,
    }


def _aligned_probabilities(model, features):
    raw = model.predict_proba(features)
    result = np.zeros((len(features), len(CLASS_ORDER)), dtype=np.float64)
    for index, label in enumerate(model.classes_):
        result[:, CLASS_ORDER.index(str(label))] = raw[:, index]
    return result


def _draw_confusion(path, matrix, title):
    from html import escape
    matrix = np.asarray(matrix)
    max_value = max(int(matrix.max()), 1)
    cell, left, top = 76, 150, 110
    width, height = left + cell * (len(CLASS_ORDER) + 1) + 35, top + cell * (len(CLASS_ORDER) + 1) + 35
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{left}" y="32" font-family="Arial" font-size="18" font-weight="bold">{escape(title)}</text>',
             f'<text x="{left + 115}" y="{top - 35}" font-family="Arial" font-size="14">Predicted label →</text>',
             f'<text x="20" y="{top + 130}" transform="rotate(-90 20 {top + 130})" font-family="Arial" font-size="14">True label →</text>']
    for index, label in enumerate(CLASS_ORDER):
        parts.append(f'<text x="{left + (index + 1) * cell + cell / 2}" y="{top + 20}" text-anchor="middle" font-family="Arial" font-size="11" transform="rotate(35 {left + (index + 1) * cell + cell / 2} {top + 20})">{escape(label)}</text>')
        parts.append(f'<text x="{left - 8}" y="{top + (index + 1) * cell + cell / 2 + 4}" text-anchor="end" font-family="Arial" font-size="12">{escape(label)}</text>')
    for row in range(len(CLASS_ORDER)):
        for col in range(len(CLASS_ORDER)):
            value = int(matrix[row, col])
            shade = int(247 - (value / max_value) * 150)
            x, y = left + (col + 1) * cell, top + (row + 1) * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="rgb({shade},{shade + 3},{min(255, shade + 12)})" stroke="#888"/>')
            parts.append(f'<text x="{x + cell / 2}" y="{y + cell / 2 + 5}" text-anchor="middle" font-family="Arial" font-size="15">{value}</text>')
    parts.append('</svg>')
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def _draw_comparison(path, accuracies, macro_f1s):
    from html import escape
    names = ["Current utterance", "+ one previous", "+ two previous"]
    width, height, chart_left, chart_top, chart_width, chart_height = 820, 470, 90, 80, 680, 290
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{chart_left}" y="34" font-family="Arial" font-size="19" font-weight="bold">DailyDialog public four-label test comparison</text>']
    for tick in range(0, 11, 2):
        y = chart_top + chart_height - chart_height * tick / 10
        parts.extend([f'<line x1="{chart_left}" y1="{y}" x2="{chart_left + chart_width}" y2="{y}" stroke="#ddd"/>',
                      f'<text x="{chart_left - 12}" y="{y + 4}" text-anchor="end" font-family="Arial" font-size="12">{tick / 10:.1f}</text>'])
    for index, name in enumerate(names):
        base_x = chart_left + 75 + index * 205
        for offset, value, color, label in ((0, accuracies[index], "#3976b8", "Accuracy"),
                                             (52, macro_f1s[index], "#e08b35", "Macro F1")):
            bar_height = chart_height * value
            y = chart_top + chart_height - bar_height
            parts.extend([f'<rect x="{base_x + offset}" y="{y}" width="40" height="{bar_height}" fill="{color}"/>',
                          f'<text x="{base_x + offset + 20}" y="{y - 7}" text-anchor="middle" font-family="Arial" font-size="11">{value:.3f}</text>'])
        parts.append(f'<text x="{base_x + 46}" y="{chart_top + chart_height + 25}" text-anchor="middle" font-family="Arial" font-size="12">{escape(name)}</text>')
    parts.extend([f'<rect x="{chart_left + 210}" y="{height - 35}" width="14" height="14" fill="#3976b8"/>',
                  f'<text x="{chart_left + 230}" y="{height - 23}" font-family="Arial" font-size="12">Accuracy</text>',
                  f'<rect x="{chart_left + 320}" y="{height - 35}" width="14" height="14" fill="#e08b35"/>',
                  f'<text x="{chart_left + 340}" y="{height - 23}" font-family="Arial" font-size="12">Macro F1</text>', '</svg>'])
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def _measure_latency(encoder, model, input_texts, count=40):
    selected = input_texts[:min(count, len(input_texts))]
    if not selected:
        return {"records": 0, "p50_ms": None, "p95_ms": None}
    encoder.encode([selected[0]], batch_size=1, show_progress_bar=False, normalize_embeddings=True,
                   convert_to_numpy=True)
    model.predict_proba(np.zeros((1, encoder.get_sentence_embedding_dimension()), dtype=np.float32))
    times = []
    for text in selected:
        started = time.perf_counter()
        vector = encoder.encode([text], batch_size=1, show_progress_bar=False, normalize_embeddings=True,
                                convert_to_numpy=True)
        model.predict_proba(vector)
        times.append((time.perf_counter() - started) * 1000)
    return {"records": len(times), "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)), "measurement": "single-record CPU embedding plus calibrated classifier"}


def train(prepared_path: Path, output_root: Path, *, batch_size=64,
          max_dialogues_train=2000, max_dialogues_validation=500, max_dialogues_test=500):
    prepared_path, output_root = Path(prepared_path), Path(output_root)
    records = load_records(prepared_path)
    max_dialogues = {"train": max_dialogues_train, "validation": max_dialogues_validation,
                     "test": max_dialogues_test}
    partitions, split_manifest, records = create_partitions(records, max_dialogues_per_split=max_dialogues)
    labels = np.asarray([row["mapped_label"] for row in records])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"dailydialog_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir()
    results_dir = run_dir / "results"
    results_dir.mkdir()

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(CHECKPOINT, device="cpu", local_files_only=True)
    model_position_limit = int(encoder[0].auto_model.config.max_position_embeddings)
    max_seq_length = min(512, model_position_limit)
    encoder.max_seq_length = max_seq_length
    split_manifest.update({"source_dataset_sha256": sha256_file(prepared_path),
                           "sample_id_sha256_by_partition": {
                               name: hashlib.sha256("\n".join(sorted(records[int(i)]["sample_id"] for i in indices)).encode()).hexdigest()
                               for name, indices in partitions.items()}})
    with (run_dir / "split_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(split_manifest, stream, indent=2)

    run_report = {
        "status": "public_research_experiment",
        "dataset_origin": "public_research",
        "dataset_id": "dailydialog_public_emotion_context_v1",
        "source_url": SOURCE_URL,
        "source_license": LICENSE,
        "source_citation": CITATION,
        "population_status": "public_nonclinical_not_eldermeds_data",
        "eldermeds_seven_class_claim": False,
        "class_order": CLASS_ORDER,
        "models": {},
        "split_manifest": "split_manifest.json",
        "checkpoint": CHECKPOINT,
        "checkpoint_local_files_only": True,
        "max_seq_length": max_seq_length,
        "normalization": "L2",
        "seed": SEED,
        "hyperparameter_search": [{"C": c, "class_weight": weight} for c, weight in HYPERPARAMETERS],
        "calibration": "sigmoid calibration fit only on held-out dialogue groups from official training split",
        "test_policy": "official test split accessed after each format's hyperparameters and calibration were frozen",
        "dependency_versions": {"scikit_learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
    }
    y_fit, y_cal, y_val, y_test = (labels[partitions[name]] for name in ("fit", "calibration", "validation", "test"))
    test_predictions = {}
    test_rows = [records[int(i)] for i in partitions["test"]]
    for input_format in INPUT_FORMATS:
        x_fit, _, fit_input_audit = _encode(encoder, records, partitions["fit"], input_format, batch_size)
        x_val, val_texts, val_input_audit = _encode(encoder, records, partitions["validation"], input_format, batch_size)
        candidates = []
        for c, class_weight in HYPERPARAMETERS:
            estimator = LogisticRegression(C=c, class_weight=class_weight, max_iter=2000,
                                           solver="lbfgs", random_state=SEED)
            estimator.fit(x_fit, y_fit)
            pred = estimator.predict(x_val)
            candidate = {"C": c, "class_weight": class_weight,
                         "accuracy": float(accuracy_score(y_val, pred)),
                         "macro_f1": float(f1_score(y_val, pred, labels=CLASS_ORDER,
                                                     average="macro", zero_division=0))}
            candidates.append(candidate)
        # Resolve by lexicographic score with the declared candidate order as the stable tie break.
        best_candidate = max(candidates, key=lambda row: (row["macro_f1"], row["accuracy"], -row["C"], row["class_weight"] is None))
        selected_estimator = LogisticRegression(C=best_candidate["C"], class_weight=best_candidate["class_weight"],
                                                max_iter=2000, solver="lbfgs", random_state=SEED)
        selected_estimator.fit(x_fit, y_fit)
        val_pred = selected_estimator.predict(x_val)
        val_prob = _aligned_probabilities(selected_estimator, x_val)
        validation_metrics = _metrics(y_val, val_pred, val_prob)

        x_cal, _, cal_input_audit = _encode(encoder, records, partitions["calibration"], input_format, batch_size)
        calibrated = CalibratedClassifierCV(FrozenEstimator(selected_estimator), method="sigmoid")
        calibrated.fit(x_cal, y_cal)

        artifact_path = artifact_dir / f"dailydialog_{input_format}.joblib"
        artifact = {
            "artifact_type": "public_research_experiment_only",
            "dataset_origin": "public_research", "dataset_id": run_report["dataset_id"],
            "source_url": SOURCE_URL, "source_license": LICENSE, "source_citation": CITATION,
            "population_status": "public_nonclinical_not_eldermeds_data",
            "input_format": input_format, "class_order": CLASS_ORDER,
            "checkpoint": CHECKPOINT, "max_seq_length": max_seq_length,
            "embedding_normalization": "L2", "model": calibrated,
            "hyperparameters": best_candidate, "random_seed": SEED,
            "split_manifest_sha256": sha256_file(run_dir / "split_manifest.json"),
            "training_data_sha256": sha256_file(prepared_path),
        }
        joblib.dump(artifact, artifact_path, compress=3)
        artifact_sha = sha256_file(artifact_path)
        with (artifact_dir / f"dailydialog_{input_format}.metadata.json").open("x", encoding="utf-8") as stream:
            json.dump({key: value for key, value in artifact.items() if key != "model"} |
                      {"artifact_sha256": artifact_sha}, stream, indent=2)

        # Test features and labels are first evaluated here, after model choice and
        # calibration have been fixed. No test outcome feeds hyperparameter choice.
        x_test, test_texts, test_input_audit = _encode(encoder, records, partitions["test"], input_format, batch_size)
        probabilities = _aligned_probabilities(calibrated, x_test)
        predictions = np.asarray(CLASS_ORDER)[probabilities.argmax(axis=1)]
        test_predictions[input_format] = predictions
        test_metrics = _metrics(y_test, predictions, probabilities)
        matrix = np.asarray(test_metrics["confusion_matrix"])
        _draw_confusion(results_dir / f"confusion_{input_format}.svg", matrix,
                        f"DailyDialog public test — {input_format.replace('_', ' ')}")
        context_slices = {}
        depths = np.asarray([len(row["previous_turns"]) for row in test_rows])
        for depth in (0, 1, 2):
            mask = depths == depth
            context_slices[str(depth)] = {
                "records": int(mask.sum()),
                "accuracy": float(accuracy_score(y_test[mask], predictions[mask])) if mask.any() else None,
                "macro_f1": float(f1_score(y_test[mask], predictions[mask], labels=CLASS_ORDER,
                                            average="macro", zero_division=0)) if mask.any() else None,
            }
        latency = _measure_latency(encoder, calibrated, test_texts)
        run_report["models"][input_format] = {
            "selected_hyperparameters": best_candidate,
            "validation_tuning_results": candidates,
            "validation_metrics_before_calibration": validation_metrics,
            "test_metrics": test_metrics,
            "context_depth_analysis": context_slices,
            "context_token_budget_audit": {"fit": fit_input_audit, "calibration": cal_input_audit,
                                            "validation": val_input_audit, "test": test_input_audit},
            "inference_latency": latency,
            "artifact": str(artifact_path.relative_to(run_dir)),
            "artifact_sha256": artifact_sha,
            "artifact_status": "experimental_not_registered_for_production",
        }
        smoke_artifact = load_public_artifact(artifact_path)
        smoke_record = next((row for row in test_rows if len(row["previous_turns"]) >= 2), test_rows[0])
        smoke_result = predict_public_demo(smoke_artifact, encoder, smoke_record["current_utterance"],
                                           speaker=smoke_record["speaker"],
                                           previous_turns=smoke_record["previous_turns"])
        run_report["models"][input_format]["artifact_inference_smoke"] = {
            "status": "passed", "prediction_in_reduced_public_taxonomy": smoke_result["prediction"],
            "probability_sum": float(sum(smoke_result["class_probabilities"].values())),
            "artifact_sha256_verified": True,
        }
        del x_fit, x_val, x_cal, x_test, probabilities, calibrated, selected_estimator

    baseline = test_predictions["answer_only"]
    multi = test_predictions["two_previous"]
    correctness_baseline = baseline == y_test
    correctness_context = multi == y_test
    run_report["contextual_error_analysis"] = {
        "answer_only_to_two_previous_corrected_errors": int((~correctness_baseline & correctness_context).sum()),
        "answer_only_to_two_previous_new_errors": int((correctness_baseline & ~correctness_context).sum()),
        "agreements": int((baseline == multi).sum()),
        "test_records": int(len(y_test)),
        "per_context_depth": {},
    }
    depths = np.asarray([len(row["previous_turns"]) for row in test_rows])
    for depth in (0, 1, 2):
        mask = depths == depth
        run_report["contextual_error_analysis"]["per_context_depth"][str(depth)] = {
            "records": int(mask.sum()),
            "corrected_errors": int((~correctness_baseline[mask] & correctness_context[mask]).sum()),
            "new_errors": int((correctness_baseline[mask] & ~correctness_context[mask]).sum()),
        }

    accuracies = [run_report["models"][fmt]["test_metrics"]["accuracy"] for fmt in INPUT_FORMATS]
    macro_f1s = [run_report["models"][fmt]["test_metrics"]["macro_f1"] for fmt in INPUT_FORMATS]
    _draw_comparison(results_dir / "context_model_comparison.svg", accuracies, macro_f1s)

    run_report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    report_path = results_dir / "experiment_results.json"
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(run_report, stream, indent=2)
    return {"run_dir": str(run_dir), "report": run_report}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data" / "processed" / "dailydialog_public_v1" / "dailydialog_mapped_turns.jsonl")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-dialogues-train", type=int, default=2000)
    parser.add_argument("--max-dialogues-validation", type=int, default=500)
    parser.add_argument("--max-dialogues-test", type=int, default=500)
    args = parser.parse_args(argv)
    result = train(args.data, args.output_root, batch_size=args.batch_size,
                   max_dialogues_train=args.max_dialogues_train,
                   max_dialogues_validation=args.max_dialogues_validation,
                   max_dialogues_test=args.max_dialogues_test)
    print(json.dumps({"run_dir": result["run_dir"], "report": result["report"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
