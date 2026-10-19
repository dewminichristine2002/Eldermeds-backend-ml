import sys
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_contextual import build_leakage_groups, grouped_splits, main as train_main


class ContextTrainingUtilityTests(unittest.TestCase):
    def test_participant_and_duplicate_families_are_unified(self):
        rows = [
            {"participant_group_id": "group-a", "current_question": "Question A", "current_answer": "Answer alpha"},
            {"participant_group_id": "group-a", "current_question": "Question B", "current_answer": "Answer beta"},
            {"participant_group_id": "group-b", "current_question": "Question A", "current_answer": "Answer alpha"},
        ]
        groups, audit = build_leakage_groups(rows)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[0], groups[2])
        self.assertEqual(audit["near_duplicate_pairs_at_or_above_threshold"], 0)

    def test_participant_grouped_splits_are_disjoint(self):
        labels = []
        groups = []
        for label in ("happiness", "sadness", "loneliness", "anxiety", "anger", "cognitive_fog", "neutral"):
            for index in range(10):
                labels.append(label)
                groups.append(f"{label}-participant-{index}")
        import numpy as np
        fit, calibration, validation = grouped_splits(np.asarray(labels), np.asarray(groups))
        for left, right in ((fit, calibration), (fit, validation), (calibration, validation)):
            self.assertFalse(set(np.asarray(groups)[left]) & set(np.asarray(groups)[right]))
        self.assertTrue(set(labels) == set(np.asarray(labels)[fit]))
        self.assertTrue(set(labels) == set(np.asarray(labels)[calibration]))
        self.assertTrue(set(labels) == set(np.asarray(labels)[validation]))

    def test_missing_contextual_training_data_reports_waiting_without_artifacts(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "missing.csv"
            output = Path(directory) / "stage9-runs"
            buffer = io.StringIO()
            with patch("sys.argv", ["train_contextual.py", "--data", str(data), "--output-root", str(output)]), \
                 contextlib.redirect_stdout(buffer):
                code = train_main()
            status = json.loads(buffer.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(status["status"], "WAITING_FOR_VALID_LABELED_DATA")
            self.assertFalse(status["model_artifacts_created"])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
