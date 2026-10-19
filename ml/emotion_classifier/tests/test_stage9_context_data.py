import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEST_QUESTION_BANK_VERSION = "sha256:" + "a" * 64

from context_dataset_schema import ContextRecordValidationError, validate_context_dataset, validate_context_record
from annotation.context_review_queue import build_review_queue
from annotation.context_review_queue import load_collection_jsonl, main as queue_main
from annotation.context_dataset_builder import resolve_contextual_decisions


def raw_records():
    return [
        {"schema_version": "eldermeds-context-emotion-1.1", "sample_id": "sample-1",
         "participant_group_id": "participant-a", "session_id": "session-a", "turn_index": 1,
         "question_id": "q1", "question_bank_version": TEST_QUESTION_BANK_VERSION, "current_question": "How has your day been?",
         "current_answer": "I feel okay.", "previous_turns": [],
         "consent_or_use_authorization_reference": "approval-ref-a", "dataset_split": "unassigned",
         "data_origin": "consented_participant", "source_provenance": "protocol-a"},
        {"schema_version": "eldermeds-context-emotion-1.1", "sample_id": "sample-2",
         "participant_group_id": "participant-a", "session_id": "session-a", "turn_index": 2,
         "question_id": "q2", "question_bank_version": TEST_QUESTION_BANK_VERSION, "current_question": "Who did you speak with?",
         "current_answer": "I called my sister.", "previous_turns": [
             {"participant_group_id": "participant-a", "session_id": "session-a", "turn_index": 1,
              "question": "How has your day been?", "answer": "I feel okay."}],
         "consent_or_use_authorization_reference": "approval-ref-a", "dataset_split": "unassigned",
         "data_origin": "consented_participant", "source_provenance": "protocol-a"},
    ]


class Stage9ContextDataTests(unittest.TestCase):
    def test_collection_records_validate_without_changing_source(self):
        records = raw_records()
        snapshot = copy.deepcopy(records)
        audit = validate_context_dataset(records)
        self.assertEqual(audit["records"], 2)
        self.assertEqual(audit["sessions"], 1)
        self.assertEqual(records, snapshot)

    def test_rejects_duplicate_samples_turns_and_wrong_history_pairing(self):
        records = raw_records()
        duplicate = copy.deepcopy(records)
        duplicate[1]["sample_id"] = duplicate[0]["sample_id"]
        with self.assertRaisesRegex(ContextRecordValidationError, "sample_id"):
            validate_context_dataset(duplicate)
        duplicate = raw_records()
        duplicate[1]["turn_index"] = 1
        with self.assertRaisesRegex(ContextRecordValidationError, "turn_index"):
            validate_context_dataset(duplicate)
        mismatch = raw_records()
        mismatch[1]["previous_turns"][0]["answer"] = "Different answer"
        with self.assertRaisesRegex(ContextRecordValidationError, "does not match"):
            validate_context_dataset(mismatch)

    def test_rejects_cross_participant_session_ownership(self):
        records = raw_records()
        records[1]["participant_group_id"] = "participant-b"
        records[1]["previous_turns"] = []
        with self.assertRaisesRegex(ContextRecordValidationError, "multiple participant"):
            validate_context_dataset(records)

    def test_training_requires_consent_and_independent_human_review(self):
        records = raw_records()
        for item in records:
            item.update({"emotion_label": "neutral", "annotation_source": "human_single_review",
                         "annotation_status": "reviewed", "annotation_guideline_version": "stage9-v1",
                         "annotation_reviewers": ["reviewer-a"], "source_provenance": "collection.csv:1"})
        with self.assertRaisesRegex(ContextRecordValidationError, "consented participant"):
            invalid = copy.deepcopy(records)
            invalid[0]["data_origin"] = "synthetic_test"
            validate_context_dataset(invalid, require_training_eligible=True)
        with self.assertRaisesRegex(ContextRecordValidationError, "double review or adjudication"):
            validate_context_dataset(records, require_training_eligible=True)
        for item in records:
            item.update({"annotation_source": "human_double_review", "annotation_reviewers": ["reviewer-a", "reviewer-b"]})
        audit = validate_context_dataset(records, require_training_eligible=True)
        self.assertTrue(audit["training_eligible"])

    def test_context_review_queue_contains_pairs_but_no_auto_annotations(self):
        queue = build_review_queue(raw_records(), "access_controlled_sessions.jsonl")
        self.assertEqual(queue[1]["current_question"], "Who did you speak with?")
        self.assertIn("How has your day been?", queue[1]["previous_turns_json"])
        self.assertEqual(queue[1]["existing_label"], "")

    def test_contact_identifiers_are_rejected_before_annotation(self):
        records = raw_records()
        records[0]["current_answer"] = "Please email me at participant@example.org"
        with self.assertRaisesRegex(ContextRecordValidationError, "contact identifiers"):
            validate_context_dataset(records)

    def test_queue_creation_requires_consent_assertion_and_recorded_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sessions.jsonl"
            output = Path(directory) / "queue.csv"
            with self.assertRaises(SystemExit) as caught:
                queue_main(["--source", str(source), "--output", str(output)])
            self.assertEqual(caught.exception.code, 2)
            records = raw_records()
            records[0].pop("consent_or_use_authorization_reference")
            source.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "authorization references"):
                load_collection_jsonl(source)
            self.assertFalse(output.exists())

    def test_stage9_json_schema_declares_research_provenance(self):
        schema = json.loads((ROOT / "context_dataset_schema_stage9.json").read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["schema_version"]["const"], "eldermeds-context-emotion-1.1")
        for field in ("consent_or_use_authorization_reference", "question_bank_version", "data_origin"):
            self.assertIn(field, schema["required"])

    def test_two_independent_reviewers_are_required_to_accept_label(self):
        events = [
            {"sample_id": "sample-1", "reviewer_id": "reviewer-a", "review_status": "reviewed", "reviewed_label": "neutral", "annotation_guideline_version": "eldermeds-context-emotion-guidelines-1.0"},
            {"sample_id": "sample-1", "reviewer_id": "reviewer-b", "review_status": "reviewed", "reviewed_label": "neutral", "annotation_guideline_version": "eldermeds-context-emotion-guidelines-1.0"},
            {"sample_id": "sample-2", "reviewer_id": "reviewer-a", "review_status": "reviewed", "reviewed_label": "neutral", "annotation_guideline_version": "eldermeds-context-emotion-guidelines-1.0"},
        ]
        accepted, _excluded, _disagreements = resolve_contextual_decisions([], events)
        self.assertIn("sample-1", accepted)
        self.assertNotIn("sample-2", accepted)

    def test_single_reviewer_cannot_exclude_sample(self):
        events = [
            {"sample_id": "sample-1", "reviewer_id": "reviewer-a", "review_status": "rejected",
             "reviewed_label": "", "annotation_guideline_version": "eldermeds-context-emotion-guidelines-1.0"},
        ]
        accepted, excluded, _disagreements = resolve_contextual_decisions([], events)
        self.assertFalse(accepted)
        self.assertFalse(excluded)

    def test_adjudicator_must_be_independent_of_two_initial_reviewers(self):
        events = [
            {"sample_id": "sample-1", "reviewer_id": "reviewer-a", "review_status": "reviewed",
             "reviewed_label": "neutral", "annotation_guideline_version": "eldermeds-context-emotion-guidelines-1.0"},
            {"sample_id": "sample-1", "reviewer_id": "reviewer-b", "review_status": "reviewed",
             "reviewed_label": "sadness", "annotation_guideline_version": "eldermeds-context-emotion-guidelines-1.0"},
            {"sample_id": "sample-1", "reviewer_id": "reviewer-a", "review_status": "adjudicated",
             "reviewed_label": "neutral", "annotation_guideline_version": "eldermeds-context-emotion-guidelines-1.0"},
        ]
        accepted, _excluded, _disagreements = resolve_contextual_decisions([], events)
        self.assertNotIn("sample-1", accepted)


if __name__ == "__main__":
    unittest.main()
