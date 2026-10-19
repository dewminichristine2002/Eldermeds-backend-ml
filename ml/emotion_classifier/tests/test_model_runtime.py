import sys
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import model_runtime


class ModelRuntimeTests(unittest.TestCase):
    def test_explicit_v3_rollback_loads_preserved_artifact(self):
        runtime = model_runtime.load_runtime(model_runtime.V3_VERSION)
        self.assertTrue(runtime.ready, runtime.error)
        self.assertEqual(runtime.model_version, model_runtime.V3_VERSION)
        self.assertEqual(set(runtime.classes_), set(model_runtime.PROJECT_CLASSES))

    def test_unknown_version_fails_closed(self):
        runtime = model_runtime.load_runtime('unknown_model')
        self.assertFalse(runtime.ready)
        self.assertIn('Unsupported EMOTION_MODEL_VERSION', runtime.error)

    def test_v4_checksum_mismatch_fails_closed(self):
        with patch.object(model_runtime, '_sha256', return_value='invalid'):
            runtime = model_runtime.load_runtime(model_runtime.V4_VERSION)
        self.assertFalse(runtime.ready)
        self.assertIn('checksum mismatch', runtime.error)

    def test_v3_checksum_mismatch_fails_closed(self):
        with patch.object(model_runtime, '_sha256', return_value='invalid'):
            runtime = model_runtime.load_runtime(model_runtime.V3_VERSION)
        self.assertFalse(runtime.ready)
        self.assertIn('v3 classifier checksum mismatch', runtime.error)

    def test_registered_but_unavailable_context_model_fails_closed(self):
        runtime = model_runtime.load_runtime(model_runtime.CONTEXTUAL_VERSION)
        self.assertFalse(runtime.ready)
        self.assertIn('no validated artifact', runtime.error)

    def test_v5_dependency_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = Path(temp_dir) / 'test-only-v5-artifact.joblib'
            artifact.write_bytes(b'test fixture, not a model')
            metadata = {
                'model_version': model_runtime.V5_VERSION,
                'artifact_file': artifact.name,
                'artifact_sha256': 'test-fixture-checksum',
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'embedding_normalization': 'L2 normalized',
                'scikit_learn_version': 'intentionally-mismatched-test-version',
                'sentence_transformers_version': 'test-fixture-version',
                'input_format': 'answer-only text',
                'context_aware': False,
                'supported_classes': model_runtime.PROJECT_CLASSES,
            }
            artifact.with_name('metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
            with patch.dict('os.environ', {'EMOTION_V5_ARTIFACT_PATH': str(artifact)}), \
                    patch.object(model_runtime, '_sha256', return_value='test-fixture-checksum'):
                runtime = model_runtime.load_runtime(model_runtime.V5_VERSION)
        self.assertFalse(runtime.ready)
        self.assertIn('scikit-learn version differs', runtime.error)

    def test_context_is_not_retained_or_used_by_current_runtime(self):
        runtime = model_runtime.load_runtime(model_runtime.V4_VERSION)
        self.assertTrue(runtime.ready, runtime.error)
        first = runtime.predict_proba(['I feel worried today'], context={
            'session_id': 'session-a',
            'previous_turns': ['I slept poorly.'],
        })
        second = runtime.predict_proba(['I feel worried today'], context={
            'session_id': 'session-b',
            'previous_turns': ['I had a lovely lunch.'],
        })
        self.assertEqual(first.tolist(), second.tolist())

    def test_declared_context_formats_are_version_gated(self):
        metadata = {"input_format": "answer-only text", "context_aware": False}
        plain = model_runtime.format_model_inputs(["Not anymore."], {"question": "Were you lonely?"}, metadata)
        self.assertEqual(plain, ["Not anymore."])
        contextual = {"input_format": "question_answer", "context_aware": True}
        formatted = model_runtime.format_model_inputs(["Not anymore."], {
            "session_id": "s", "participant_group_id": "group-a", "turn_index": 2,
            "question": "Were you lonely?", "previous_turns": []
        }, contextual)
        self.assertIn("Question: Were you lonely?", formatted[0])
        with self.assertRaises(ValueError):
            model_runtime.format_model_inputs(["Yes"], None, contextual)
        with self.assertRaisesRegex(ValueError, "different session"):
            model_runtime.format_model_inputs(["Yes"], {
                "session_id": "session-a", "participant_group_id": "group-a", "turn_index": 3,
                "question": "Current?", "previous_turns": [{
                    "session_id": "session-b", "participant_group_id": "group-a", "turn_index": 1,
                    "question": "Previous?", "answer": "No",
                }],
            }, {"input_format": "multi_turn", "context_aware": True})


if __name__ == '__main__':
    unittest.main()
