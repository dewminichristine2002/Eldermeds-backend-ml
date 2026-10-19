import hashlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import model_runtime
from model_registry import MODEL_REGISTRY, get_registration


class ModelRegistryTests(unittest.TestCase):
    def test_v3_and_v4_registry_checksums_match_artifacts(self):
        for version, registration in MODEL_REGISTRY.items():
            if version == model_runtime.V5_VERSION or registration["artifact"] is None:
                continue
            path = ROOT / registration["artifact"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, registration["artifact_sha256"])

    def test_contextual_registration_is_disabled_without_artifact(self):
        registration = get_registration("minilm_logistic_regression_v6_contextual")
        self.assertIsNone(registration["artifact"])
        runtime = model_runtime.load_runtime("minilm_logistic_regression_v6_contextual")
        self.assertFalse(runtime.ready)
        self.assertIn("no validated artifact", runtime.error)

    def test_unknown_versions_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "Unsupported EMOTION_MODEL_VERSION"):
            get_registration("arbitrary-model")


if __name__ == "__main__":
    unittest.main()
