import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from context_builder import ContextValidationError, build_context_from_payload, build_context_text


class ContextBuilderTests(unittest.TestCase):
    def test_preserves_chronological_order_and_negation(self):
        result = build_context_text(
            answer="Not anymore.",
            session_id="session-a",
            question="Have you felt lonely recently?",
            previous_turns=["I have not spoken to anyone.", "I miss my family."],
        )
        self.assertEqual(
            result,
            "Previous: I have not spoken to anyone.\n"
            "Previous: I miss my family.\n"
            "Question: Have you felt lonely recently?\n"
            "Answer: Not anymore.",
        )

    def test_limits_to_two_previous_turns_and_does_not_use_session_as_text(self):
        result = build_context_text(
            answer="Yes",
            session_id="private-session-a",
            previous_turns=["first", "second", "third"],
        )
        self.assertNotIn("first", result)
        self.assertIn("second", result)
        self.assertIn("third", result)
        self.assertNotIn("private-session-a", result)

    def test_missing_context_is_rejected(self):
        with self.assertRaises(ContextValidationError):
            build_context_from_payload({"answer": "Yes"})

    def test_malformed_context_is_rejected(self):
        with self.assertRaises(ContextValidationError):
            build_context_from_payload({
                "answer": "Yes",
                "session_id": "session-a",
                "previous_turns": ["ok", 4],
            })

    def test_long_context_is_bounded(self):
        result = build_context_text(
            answer="short answer",
            session_id="session-a",
            previous_turns=["old context " * 100],
            max_chars=80,
        )
        self.assertLessEqual(len(result), 80)
        self.assertNotIn("old context", result)


if __name__ == "__main__":
    unittest.main()
