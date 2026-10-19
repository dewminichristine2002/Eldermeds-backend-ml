import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from annotation.dataset_builder import build_version
from annotation.review_validator import (
    GUIDELINE_VERSION, ReviewValidationError, append_review_event, load_queue,
    read_review_events, summarize_reviews, validate_review_event,
)


class AnnotationReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.queue_path = self.root / "queue.csv"
        self.source_path = self.root / "source.csv"
        self.events_path = self.root / "events.jsonl"
        with self.source_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["text", "label", "source"])
            writer.writeheader()
            writer.writerows([
                {"text": "I feel blue", "label": "neutral", "source": "fixture"},
                {"text": "[name]", "label": "neutral", "source": "fixture"},
                {"text": "A pleasant day", "label": "happiness", "source": "fixture"},
                {"text": "I miss my friend", "label": "sadness", "source": "fixture"},
                {"text": "I feel alone", "label": "loneliness", "source": "fixture"},
                {"text": "I am worried", "label": "anxiety", "source": "fixture"},
                {"text": "That was frustrating", "label": "anger", "source": "fixture"},
                {"text": "I cannot focus", "label": "cognitive_fog", "source": "fixture"},
                {"text": "No preference", "label": "neutral", "source": "fixture"},
            ])
        with self.queue_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["sample_id", "source_file", "source_row_1based", "text", "existing_label", "reason"])
            writer.writeheader()
            writer.writerows([
                {"sample_id": "sample-a", "source_file": "source.csv", "source_row_1based": 2, "text": "I feel blue", "existing_label": "neutral", "reason": "label_conflict"},
                {"sample_id": "sample-b", "source_file": "source.csv", "source_row_1based": 3, "text": "[name]", "existing_label": "neutral", "reason": "very_short_context_dependent"},
            ])
        self.queue = load_queue(self.queue_path)

    def tearDown(self):
        self.temp.cleanup()

    def event(self, sample="sample-a", reviewer="reviewer_1234567890abcdef", status="reviewed", label="sadness", notes="reason"):
        item = self.queue.get(sample, {"source_file": "source.csv", "existing_label": "neutral", "reason": "label_conflict"})
        return {"sample_id": sample, "dataset_source": item["source_file"], "original_label": item["existing_label"],
                "reviewed_label": label, "review_status": status, "review_reason": item["reason"],
                "reviewer_id": reviewer, "reviewer_notes": notes,
                "annotation_guideline_version": GUIDELINE_VERSION}

    def test_invalid_label_and_changed_original_are_rejected(self):
        with self.assertRaises(ReviewValidationError):
            validate_review_event(self.event(label="fear"), self.queue["sample-a"])
        event = self.event()
        event["original_label"] = "sadness"
        with self.assertRaises(ReviewValidationError):
            validate_review_event(event, self.queue["sample-a"])

    def test_pending_or_ambiguous_does_not_replace_label(self):
        for status, label in (("ambiguous", None), ("needs_context", None), ("rejected", None)):
            validate_review_event(self.event(status=status, label=label), self.queue["sample-a"])
        with self.assertRaises(ReviewValidationError):
            validate_review_event(self.event(status="pending", label=None), self.queue["sample-a"])

    def test_terminal_exclusion_requires_distinct_second_reviewer(self):
        item = self.queue["sample-a"]
        append_review_event(self.events_path, self.event(status="rejected", label=None), item)
        with self.assertRaisesRegex(ReviewValidationError, "distinct independent reviewer"):
            append_review_event(self.events_path, self.event(status="rejected", label=None), item)

    def test_context_guideline_version_is_accepted_without_changing_stage7_version(self):
        from annotation.review_validator import CONTEXT_GUIDELINE_VERSION
        event = self.event()
        event["annotation_guideline_version"] = CONTEXT_GUIDELINE_VERSION
        contextual_item = {**self.queue["sample-a"], "current_question": "How are you?"}
        validate_review_event(event, contextual_item)

    def test_reviewer_notes_reject_direct_contact_identifiers(self):
        with self.assertRaisesRegex(ReviewValidationError, "must not contain email"):
            validate_review_event(self.event(notes="contact example@example.org"), self.queue["sample-a"])

    def test_audit_log_contains_no_source_text_and_summary_has_no_fake_agreement(self):
        first = append_review_event(self.events_path, self.event(), self.queue["sample-a"])
        second = append_review_event(self.events_path, self.event(reviewer="reviewer_fedcba0987654321"), self.queue["sample-a"])
        self.assertNotIn("text", first)
        self.assertEqual(len(read_review_events(self.events_path, self.queue)), 2)
        summary = summarize_reviews(self.queue, [first, second])
        self.assertEqual(summary["inter_annotator_agreement"]["agreement_rate"], 1.0)
        self.assertEqual(summary["inter_annotator_agreement"]["overlap_samples"], 1)
        self.assertEqual(summarize_reviews(self.queue, [first])["inter_annotator_agreement"], None)

    def test_dataset_builder_refuses_to_emit_without_human_resolution(self):
        output = self.root / "version-empty"
        with self.assertRaisesRegex(ValueError, "no accepted human resolutions"):
            build_version(self.source_path, self.queue_path, self.events_path, output)
        self.assertFalse(output.exists())

    def test_two_independent_agreeing_reviewers_can_accept_a_correction(self):
        append_review_event(self.events_path, self.event(reviewer="reviewer_aaaaaaaaaaaaaaaa", status="reviewed", label="sadness"), self.queue["sample-a"])
        append_review_event(self.events_path, self.event(reviewer="reviewer_bbbbbbbbbbbbbbbb", status="reviewed", label="sadness"), self.queue["sample-a"])
        output = self.root / "version-consensus"
        manifest = build_version(self.source_path, self.queue_path, self.events_path, output)
        self.assertEqual(manifest["two_reviewer_agreement_samples"], 1)
        with (output / "development_dataset_v4.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["label"], "sadness")

    def test_adjudication_requires_a_distinct_third_reviewer(self):
        first = self.event(reviewer="reviewer_aaaaaaaaaaaaaaaa", status="reviewed", label="sadness")
        second = self.event(reviewer="reviewer_bbbbbbbbbbbbbbbb", status="reviewed", label="anxiety")
        append_review_event(self.events_path, first, self.queue["sample-a"])
        append_review_event(self.events_path, second, self.queue["sample-a"])
        same_reviewer = self.event(reviewer="reviewer_aaaaaaaaaaaaaaaa", status="adjudicated", label="sadness")
        with self.assertRaisesRegex(ReviewValidationError, "third independent reviewer"):
            append_review_event(self.events_path, same_reviewer, self.queue["sample-a"])

    def test_dataset_builder_refuses_to_remove_a_supported_class(self):
        source = self.root / "minimal.csv"
        queue_path = self.root / "minimal_queue.csv"
        event_path = self.root / "minimal_events.jsonl"
        with source.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["text", "label", "source"])
            writer.writeheader()
            writer.writerows([{"text": "Target", "label": "neutral", "source": "fixture"}] +
                             [{"text": f"example-{label}", "label": label, "source": "fixture"}
                              for label in ("sadness", "loneliness", "anxiety", "anger", "cognitive_fog")])
        item = {"sample_id": "minimal-a", "source_file": "minimal.csv", "source_row_1based": "2",
                "text": "Target", "existing_label": "neutral", "reason": "label_conflict"}
        with queue_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(item))
            writer.writeheader()
            writer.writerow(item)
        item["source_row_1based"] = 2
        reviewed1 = self.event(sample="minimal-a", reviewer="reviewer_1111111111111111", status="reviewed", label="happiness")
        reviewed2 = self.event(sample="minimal-a", reviewer="reviewer_2222222222222222", status="reviewed", label="happiness")
        adjudicated = self.event(sample="minimal-a", status="adjudicated", label="happiness", notes="Adjudicated with two reviewers.")
        for event in (reviewed1, reviewed2, adjudicated):
            event["dataset_source"] = "minimal.csv"
        append_review_event(event_path, reviewed1, item)
        append_review_event(event_path, reviewed2, item)
        append_review_event(event_path, adjudicated, item)
        output = self.root / "version-missing-class"
        with self.assertRaisesRegex(ValueError, "would remove supported classes"):
            build_version(source, queue_path, event_path, output)
        self.assertFalse(output.exists())

    def test_dataset_builder_applies_only_adjudication_and_preserves_change_log(self):
        append_review_event(self.events_path, self.event(reviewer="reviewer_1111111111111111", status="reviewed", label="sadness"), self.queue["sample-a"])
        append_review_event(self.events_path, self.event(reviewer="reviewer_2222222222222222", status="reviewed", label="anxiety"), self.queue["sample-a"])
        correction = self.event(status="adjudicated", label="sadness", notes="Two reviewers adjudicated explicit sadness.")
        append_review_event(self.events_path, correction, self.queue["sample-a"])
        append_review_event(self.events_path, self.event(sample="sample-b", reviewer="reviewer_aaaaaaaaaaaaaaaa", status="rejected", label=None), self.queue["sample-b"])
        append_review_event(self.events_path, self.event(sample="sample-b", reviewer="reviewer_bbbbbbbbbbbbbbbb", status="rejected", label=None), self.queue["sample-b"])
        output = self.root / "version-v1"
        manifest = build_version(self.source_path, self.queue_path, self.events_path, output)
        with (output / "development_dataset_v4.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0]["label"], "sadness")
        self.assertEqual(manifest["applied_label_corrections"][0]["original_label"], "neutral")
        self.assertEqual(manifest["rejected_samples"][0]["reason"], "human_rejected")
        self.assertEqual(manifest["output_rows"], 8)
        self.assertTrue((output / "manifest.json").exists())
        with self.source_path.open(encoding="utf-8", newline="") as stream:
            original = list(csv.DictReader(stream))
        self.assertEqual(original[0]["label"], "neutral")


if __name__ == "__main__":
    unittest.main()
