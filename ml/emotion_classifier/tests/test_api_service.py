import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import api_service

class ApiServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(api_service.app)

    def test_health_and_model_artifact_loading(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['ready'])
        self.assertEqual(response.json()['service'], 'emotion_classifier')
        self.assertEqual(response.json()['model_version'], 'minilm_logistic_regression_v4')
        self.assertEqual(response.json()['model_status'], 'ready')
        self.assertEqual(response.json()['artifact_integrity'], 'verified')
        self.assertEqual(set(response.json()['supported_classes']), set(api_service.PROJECT_CLASSES))

    def test_liveness_is_separate_from_model_readiness(self):
        response = self.client.get('/live')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['service'], 'emotion_classifier')

    def test_valid_prediction_contract(self):
        response = self.client.post('/predict-emotion', json={'text': 'My daughter called and I felt much happier after talking with her.'})
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertIn(body['emotion'], api_service.PROJECT_CLASSES)
        self.assertLessEqual(body['confidence'], 1)
        self.assertGreaterEqual(body['confidence'], 0)
        self.assertEqual(body['source'], 'ml_model')
        self.assertEqual(body['model_version'], 'minilm_logistic_regression_v4')
        self.assertEqual(body['confidence_type'], 'model_probability')
        self.assertFalse(body['confidence_calibrated'])
        self.assertFalse(body['context_aware'])

    def test_context_is_accepted_without_changing_answer_only_model_contract(self):
        response = self.client.post('/predict-emotion', json={
            'text': 'I feel worried today',
            'context': {
                'session_id': 'synthetic-session-a',
                'question': 'Have you felt calm today?',
                'previous_turns': ['I slept poorly.'],
            },
        })
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()['context_aware'])

    def test_short_answer_suggests_clarification_without_changing_emotion_contract(self):
        response = self.client.post('/predict-emotion', json={'text': 'Not anymore.'})
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertIn(body['emotion'], api_service.PROJECT_CLASSES)
        self.assertTrue(body['clarification_suggested'])
        self.assertTrue(body['uncertainty']['needs_clarification'])
        self.assertIn('very_short_answer', body['uncertainty']['reasons'])
        self.assertFalse(body['confidence_calibrated'])

    def test_context_limits_and_structured_history_validation(self):
        long_question = self.client.post('/predict-emotion', json={
            'text': 'Sometimes', 'context': {'session_id': 's', 'question': 'q' * 2001}
        })
        too_many = self.client.post('/predict-emotion', json={
            'text': 'Sometimes', 'context': {'session_id': 's', 'question': 'Question?',
                'previous_turns': ['1', '2', '3']}
        })
        self.assertEqual(long_question.status_code, 422)
        self.assertEqual(too_many.status_code, 422)

    def test_context_requires_a_session_id(self):
        response = self.client.post('/predict-emotion', json={
            'text': 'I feel worried today',
            'context': {'question': 'Have you felt calm today?'},
        })
        self.assertEqual(response.status_code, 422)

    def test_empty_and_unsupported_payloads(self):
        self.assertEqual(self.client.post('/predict-emotion', json={'text': '   '}).status_code, 422)
        self.assertEqual(self.client.post('/predict-emotion', json={'text': 'hello', 'extra': True}).status_code, 422)

    def test_missing_artifact_returns_service_error(self):
        with patch.object(api_service.runtime, 'classifier', None):
            response = self.client.post('/predict-emotion', json={'text': 'hello'})
        self.assertEqual(response.status_code, 503)

    def test_runtime_failure_is_visible_and_returns_service_error(self):
        with patch.object(api_service.runtime, 'error', 'MiniLM unavailable'), \
             patch.object(api_service.runtime, 'classifier', None):
            health = self.client.get('/health')
            prediction = self.client.post('/predict-emotion', json={'text': 'hello'})
        self.assertFalse(health.json()['ready'])
        self.assertEqual(health.json()['error'], 'MiniLM unavailable')
        self.assertEqual(prediction.status_code, 503)


if __name__ == '__main__':
    unittest.main()
