import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stage10a_collection import (CSV_FIELDS, export_csv, load_records,
                                 main as collection_main, validate_synthetic_fixture)

FIXTURE = ROOT / "stage10a" / "synthetic_five_turn_fixture.jsonl"


def fixture_rows():
    rows = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["data_origin"] = "consented_participant"
        row["consent_or_use_authorization_reference"] = "synthetic-approval-ref"
    return rows


def write_source(folder, rows):
    source = folder / "source.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return source


def write_ledger(folder, **overrides):
    ledger = folder / "ledger.csv"
    row = {"participant_group_id": "synthetic-participant-only", "session_id": "synthetic-session-only",
           "consent_status": "consented", "research_scope": "emotion_model_research",
           "withdrawn": "false", "authorization_reference": "synthetic-approval-ref", **overrides}
    with ledger.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return ledger


def write_question_bank(folder, *, wrong_version=False, wrong_text=False):
    version = "sha256:" + "b" * 64 if wrong_version else "sha256:" + "0" * 64
    questions = []
    for turn in range(1, 6):
        questions.append({"question_id": f"synthetic-q{turn}",
                          "question_text": f"Wrong synthetic text {turn}" if wrong_text and turn == 1 else
                                          f"Synthetic question {turn}: " + [
                                              "What has been on your mind today?", "What part of today stood out?",
                                              "How did you feel during that moment?", "Was there anything that made the day harder?",
                                              "What might help you feel supported now?"][turn - 1],
                          "is_active": True, "is_assessment": True})
    path = folder / "question_bank.json"
    path.write_text(json.dumps({"question_bank_version": version, "questions": questions}), encoding="utf-8")
    return path


class Stage10ACollectionTests(unittest.TestCase):
    def test_checked_in_example_is_explicitly_synthetic_and_five_turn(self):
        result = validate_synthetic_fixture(FIXTURE)
        self.assertEqual(result["status"], "SYNTHETIC_FORMAT_FIXTURE_VALID")
        self.assertFalse(result["research_data"])
        self.assertFalse(result["training_eligible"])

    def test_cli_requires_explicit_authorization_assertion(self):
        with self.assertRaises(SystemExit) as caught:
            collection_main(["export", "--source", "missing.jsonl", "--consent-ledger", "missing.csv",
                             "--question-bank-export", "missing-bank.json", "--output", "never-created.csv"])
        self.assertEqual(caught.exception.code, 2)

    def test_export_accepts_matching_research_consent_and_only_allowlisted_columns(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source = write_source(folder, fixture_rows())
            ledger = write_ledger(folder)
            bank = write_question_bank(folder)
            output = folder / "export.csv"
            result = export_csv(source, ledger, bank, output)
            self.assertEqual(result["records"], 5)
            self.assertFalse(result["database_accessed"])
            with output.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
            self.assertEqual(reader.fieldnames, CSV_FIELDS)
            self.assertEqual([int(row["turn_number"]) for row in rows], [1, 2, 3, 4, 5])
            self.assertNotIn("user_id", reader.fieldnames)

    def test_rejects_nonconsented_withdrawn_or_wrong_scope(self):
        for change in ({"consent_status": "pending"}, {"withdrawn": "true"},
                       {"research_scope": "service_improvement"}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                source = write_source(folder, fixture_rows())
                ledger = write_ledger(folder, **change)
                bank = write_question_bank(folder)
                with self.assertRaisesRegex(ValueError, "consent"):
                    export_csv(source, ledger, bank, folder / "denied.csv")
                self.assertFalse((folder / "denied.csv").exists())

    def test_rejects_nonchronological_turns(self):
        rows = fixture_rows()
        rows[1], rows[2] = rows[2], rows[1]
        with tempfile.TemporaryDirectory() as temp:
            source = write_source(Path(temp), rows)
            with self.assertRaisesRegex(ValueError, "chronological"):
                load_records(source)

    def test_rejects_repeated_adaptive_question_id_in_a_session(self):
        rows = fixture_rows()
        rows[1]["question_id"] = rows[0]["question_id"]
        with tempfile.TemporaryDirectory() as temp:
            source = write_source(Path(temp), rows)
            with self.assertRaisesRegex(ValueError, "question IDs must not repeat"):
                load_records(source)

    def test_rejects_future_turn_leakage(self):
        rows = fixture_rows()
        rows[1]["previous_turns"] = [{"participant_group_id": rows[0]["participant_group_id"],
                                      "session_id": rows[0]["session_id"], "turn_index": 3,
                                      "question": rows[2]["current_question"], "answer": rows[2]["current_answer"]}]
        with tempfile.TemporaryDirectory() as temp:
            source = write_source(Path(temp), rows)
            with self.assertRaisesRegex(ValueError, "previous turns|turn_index"):
                load_records(source)

    def test_rejects_cross_session_history(self):
        rows = fixture_rows()
        rows[1]["previous_turns"][0]["session_id"] = "another-session"
        with tempfile.TemporaryDirectory() as temp:
            source = write_source(Path(temp), rows)
            with self.assertRaisesRegex(ValueError, "cross-session"):
                load_records(source)

    def test_rejects_authorization_reference_mismatch_and_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source = write_source(folder, fixture_rows())
            ledger = write_ledger(folder, authorization_reference="different-ref")
            bank = write_question_bank(folder)
            with self.assertRaisesRegex(ValueError, "does not match"):
                export_csv(source, ledger, bank, folder / "denied.csv")
            output = folder / "existing.csv"
            output.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                export_csv(source, write_ledger(folder), bank, output)
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")

    def test_rejects_question_id_text_or_bank_version_mismatch(self):
        for wrong_version, wrong_text in ((True, False), (False, True)):
            with self.subTest(wrong_version=wrong_version, wrong_text=wrong_text), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                source = write_source(folder, fixture_rows())
                ledger = write_ledger(folder)
                bank = write_question_bank(folder, wrong_version=wrong_version, wrong_text=wrong_text)
                with self.assertRaisesRegex(ValueError, "question-bank|question does not match"):
                    export_csv(source, ledger, bank, folder / "denied.csv")


if __name__ == "__main__":
    unittest.main()
