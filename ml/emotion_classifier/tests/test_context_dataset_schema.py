import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from context_dataset_schema import ContextRecordValidationError, model_input_from_record, validate_context_record
from context_builder import ContextValidationError, build_context_text


def record():
    return {
        "sample_id": "synthetic-sample-1", "schema_version": "eldermeds-context-emotion-1.0",
        "participant_group_id": "synthetic-group-a", "session_id": "synthetic-session-a", "turn_index": 3,
        "current_question": "Have you felt lonely recently?", "current_answer": "Not anymore.",
        "previous_turns": [
            {"participant_group_id": "synthetic-group-a", "session_id": "synthetic-session-a", "turn_index": 1,
             "question": "Who did you speak with?", "answer": "I have not spoken with anyone."},
            {"participant_group_id": "synthetic-group-a", "session_id": "synthetic-session-a", "turn_index": 2,
             "question": "Would a visit help?", "answer": "Sometimes."},
        ],
        "emotion_label": "neutral", "annotation_source": "human_adjudication", "annotation_status": "adjudicated",
    }


class ContextDatasetSchemaTests(unittest.TestCase):
    def test_valid_record_and_three_input_formats(self):
        item = validate_context_record(record())
        answer = model_input_from_record(item, "answer_only")
        qa = model_input_from_record(item, "question_answer")
        multi = model_input_from_record(item, "multi_turn")
        self.assertEqual(answer, "Not anymore.")
        self.assertIn("Question: Have you felt lonely recently?", qa)
        self.assertIn("Previous answer: I have not spoken with anyone.", multi)
        self.assertIn("Answer: Not anymore.", multi)
        self.assertNotIn("neutral", multi)

    def test_negation_and_chronological_order_are_preserved(self):
        text = model_input_from_record(record(), "multi_turn")
        self.assertLess(text.index("I have not spoken"), text.index("Sometimes."))
        self.assertIn("not spoken", text)

    def test_cross_user_or_session_history_is_rejected(self):
        item = record()
        item["previous_turns"][0]["participant_group_id"] = "synthetic-group-b"
        with self.assertRaises(ContextRecordValidationError):
            validate_context_record(item)
        item = record()
        item["previous_turns"][0]["session_id"] = "other-session"
        with self.assertRaises(ContextRecordValidationError):
            validate_context_record(item)

    def test_future_duplicate_and_unsorted_turns_are_rejected(self):
        item = record()
        item["previous_turns"][1]["turn_index"] = 4
        with self.assertRaises(ContextRecordValidationError):
            validate_context_record(item)
        item = record()
        item["previous_turns"].reverse()
        with self.assertRaises(ContextRecordValidationError):
            validate_context_record(item)

    def test_invalid_label_status_and_oversized_fields_are_rejected(self):
        for key, value in (("emotion_label", "fear"), ("annotation_status", "pending"), ("current_answer", "x" * 2001)):
            item = record()
            item[key] = value
            with self.assertRaises(ContextRecordValidationError):
                validate_context_record(item)

    def test_question_answer_and_multi_turn_require_question(self):
        with self.assertRaises(ContextValidationError):
            build_context_text(answer="Yes", session_id="s", input_format="question_answer")

    def test_tokenizer_budget_prunes_history_and_preserves_current_negation(self):
        class WhitespaceTokenizer:
            def encode(self, text, add_special_tokens=False):
                return text.split()

            def decode(self, token_ids, skip_special_tokens=True):
                return " ".join(token_ids)

        text = build_context_text(answer="I have not felt lonely lately", question="How are you?",
                                  session_id="s", participant_group_id="p", current_turn_index=2,
                                  previous_turns=[{"participant_group_id": "p", "session_id": "s", "turn_index": 1,
                                                   "question": "A long prior question", "answer": "A long prior answer"}],
                                  input_format="multi_turn", max_tokens=16, tokenizer=WhitespaceTokenizer())
        self.assertLessEqual(len(text.split()), 16)
        self.assertIn("not", text)
        self.assertNotIn("prior question", text)

    def test_oversized_current_answer_fails_without_silently_dropping_negation(self):
        class WhitespaceTokenizer:
            def encode(self, text, add_special_tokens=False):
                return text.split()

        with self.assertRaisesRegex(ContextValidationError, "current question and answer"):
            build_context_text(answer="I have not felt lonely lately", question="How are you?",
                               session_id="s", input_format="question_answer", max_tokens=5,
                               tokenizer=WhitespaceTokenizer())


if __name__ == "__main__":
    unittest.main()
