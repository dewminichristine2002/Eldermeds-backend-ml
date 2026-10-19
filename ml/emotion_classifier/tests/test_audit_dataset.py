import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audit_dataset import audit_csv


class DatasetAuditTests(unittest.TestCase):
    def test_reports_normalized_duplicates_and_negation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["text", "label"])
                writer.writeheader()
                writer.writerows([
                    {"text": "Not today", "label": "sadness"},
                    {"text": "  not   today ", "label": "sadness"},
                ])
            report = audit_csv(path)
        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["unique_normalized_texts"], 1)
        self.assertEqual(report["duplicate_normalized_text_rows"], 1)
        self.assertEqual(report["negation_rows"], 2)

    def test_supports_project_evaluation_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.csv"
            path.write_text("expected_emotion,text\nhappiness,I feel good\n", encoding="utf-8")
            report = audit_csv(path)
        self.assertEqual(report["label_column"], "expected_emotion")
        self.assertEqual(report["labels"], {"happiness": 1})


if __name__ == "__main__":
    unittest.main()
