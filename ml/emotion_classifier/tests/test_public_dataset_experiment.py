import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from public_dataset_experiments.dailydialog import (CLASS_ORDER, format_public_input,
                                                    prepare_dialogues)
from public_dataset_experiments.predict_dailydialog import load_public_artifact
from public_dataset_experiments.train_dailydialog import create_partitions, load_records
from public_dataset_experiments.train_dailydialog import _metrics


def raw_dialogue(dialogue_id, split, label):
    return {"dialogue_id": dialogue_id, "data_split": split, "turns": [
        {"utt_idx": 0, "speaker": "user", "utterance": "A previous thought.", "emotion": "no emotion"},
        {"utt_idx": 1, "speaker": "system", "utterance": "A previous reply.", "emotion": "happiness"},
        {"utt_idx": 2, "speaker": "user", "utterance": f"Current {label} utterance.", "emotion": label},
    ]}


def experiment_row(dialogue_id, split, label):
    return {"sample_id": f"sample-{dialogue_id}", "dialogue_id": dialogue_id,
            "native_split": split, "turn_index": 0, "speaker": "user",
            "current_utterance": "Current public utterance.", "previous_turns": [],
            "original_label": label, "mapped_label": label, "dataset_origin": "public_research",
            "source_license": "CC BY-NC-SA 4.0", "source_citation": "paper"}


class PublicDatasetExperimentTests(unittest.TestCase):
    def test_mapping_preserves_native_label_and_excludes_non_equivalent_labels(self):
        dialogues = [raw_dialogue("d-train", "train", "anger"),
                     raw_dialogue("d-val", "validation", "fear"),
                     raw_dialogue("d-test", "test", "sadness")]
        records, audit = prepare_dialogues(dialogues)
        mapped = {(row["dialogue_id"], row["turn_index"]): row for row in records}
        self.assertNotIn(("d-val", 2), mapped)
        self.assertEqual(mapped[("d-train", 2)]["original_label"], "anger")
        self.assertEqual(mapped[("d-train", 2)]["mapped_label"], "anger")
        self.assertEqual(mapped[("d-train", 2)]["dataset_origin"], "public_research")
        self.assertEqual(audit["excluded_native_label_counts"], {"fear": 1})

    def test_answer_one_previous_and_two_previous_formats_use_only_past_turns(self):
        record = {"sample_id": "s", "dialogue_id": "d", "turn_index": 3, "speaker": "user",
                  "current_utterance": "Current utterance.", "previous_turns": [
                      {"turn_index": 0, "speaker": "user", "utterance": "Oldest."},
                      {"turn_index": 1, "speaker": "system", "utterance": "Previous."}]}
        answer = format_public_input(record, "answer_only")
        one = format_public_input(record, "one_previous")
        two = format_public_input(record, "two_previous")
        self.assertNotIn("Previous turn", answer)
        self.assertIn("Previous turn by system: Previous.", one)
        self.assertNotIn("Oldest.", one)
        self.assertIn("Oldest.", two)
        self.assertLess(two.index("Oldest."), two.index("Previous."))
        record["previous_turns"][1]["turn_index"] = 4
        with self.assertRaisesRegex(ValueError, "future-turn"):
            format_public_input(record, "two_previous")

    def test_rejects_cross_dialogue_context(self):
        record = {"sample_id": "s", "dialogue_id": "d", "turn_index": 3, "speaker": "user",
                  "current_utterance": "Current.", "previous_turns": [
                      {"turn_index": 1, "dialogue_id": "other-dialogue", "speaker": "user", "utterance": "Leaked."}]}
        with self.assertRaisesRegex(ValueError, "cross-dialogue"):
            format_public_input(record, "one_previous")

    def test_partitions_are_dialogue_disjoint_and_keep_official_validation_test(self):
        rows = []
        for label in CLASS_ORDER:
            rows.extend(experiment_row(f"train-{label}-{i}", "train", label) for i in range(20))
            rows.append(experiment_row(f"validation-{label}", "validation", label))
            rows.append(experiment_row(f"test-{label}", "test", label))
        parts, manifest, selected_rows = create_partitions(rows)
        self.assertTrue(manifest["dialogue_overlap"] is False)
        for left, right in (("fit", "calibration"), ("fit", "validation"), ("fit", "test"),
                            ("calibration", "validation"), ("calibration", "test"), ("validation", "test")):
            self.assertFalse({rows[int(i)]["dialogue_id"] for i in parts[left]} &
                             {rows[int(i)]["dialogue_id"] for i in parts[right]})
        self.assertTrue(all(rows[int(i)]["native_split"] == "validation" for i in parts["validation"]))
        self.assertTrue(all(rows[int(i)]["native_split"] == "test" for i in parts["test"]))

    def test_compute_bounded_group_subsets_are_stable_and_source_split_disjoint(self):
        rows = []
        for split, count in (("train", 20), ("validation", 10), ("test", 10)):
            for dialogue_index in range(count):
                for label in CLASS_ORDER:
                    row = experiment_row(f"{split}-{dialogue_index}", split, label)
                    row["sample_id"] = f"{split}-{dialogue_index}-{label}"
                    rows.append(row)
        caps = {"train": 8, "validation": 4, "test": 4}
        first, manifest, selected = create_partitions(rows, max_dialogues_per_split=caps)
        second, second_manifest, second_selected = create_partitions(rows, max_dialogues_per_split=caps)
        self.assertEqual(manifest["deterministic_source_subset"], second_manifest["deterministic_source_subset"])
        self.assertEqual([row["sample_id"] for row in selected], [row["sample_id"] for row in second_selected])
        self.assertEqual(len({row["dialogue_id"] for row in selected if row["native_split"] == "validation"}), 4)
        self.assertEqual(len({row["dialogue_id"] for row in selected if row["native_split"] == "test"}), 4)

    def test_trainer_refuses_dialogue_overlap_and_nonpublic_provenance(self):
        row = experiment_row("d1", "train", "neutral")
        row["native_split"] = "test"
        row2 = dict(row, sample_id="sample-2", native_split="train")
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "rows.jsonl"
            data.write_text("\n".join(json.dumps(x) for x in (row, row2)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dialogue-level split leakage"):
                load_records(data)
            row2["native_split"] = "test"
            row2["dataset_origin"] = "consented_participant"
            data.write_text("\n".join(json.dumps(x) for x in (row, row2)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "explicitly licensed public_research"):
                load_records(data)

    def test_artifact_integrity_and_isolation_metadata_are_checked(self):
        import hashlib
        import joblib
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        from public_dataset_experiments.dailydialog import LICENSE
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "demo.joblib"
            model = LogisticRegression().fit(np.asarray([[0], [1], [2], [3]]), np.asarray(CLASS_ORDER))
            artifact = {"artifact_type": "public_research_experiment_only", "dataset_origin": "public_research",
                        "source_license": LICENSE, "population_status": "public_nonclinical_not_eldermeds_data",
                        "source_citation": "DailyDialog citation",
                        "input_format": "answer_only", "class_order": CLASS_ORDER, "model": model}
            joblib.dump(artifact, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_name("demo.metadata.json").write_text(json.dumps({"artifact_sha256": digest}), encoding="utf-8")
            self.assertEqual(load_public_artifact(path)["dataset_origin"], "public_research")
            path.with_name("demo.metadata.json").write_text(json.dumps({"artifact_sha256": "wrong"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_public_artifact(path)

    def test_log_loss_respects_declared_probability_column_order(self):
        import math
        labels = ["anger", "neutral"]
        predictions = ["anger", "neutral"]
        probabilities = [[0.1, 0.2, 0.1, 0.6], [0.7, 0.1, 0.1, 0.1]]
        metrics = _metrics(labels, predictions, probabilities)
        self.assertAlmostEqual(metrics["log_loss"], -((math.log(0.6) + math.log(0.7)) / 2))


if __name__ == "__main__":
    unittest.main()
