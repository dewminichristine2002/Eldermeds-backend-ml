# ML Model Selection and Rollback Guide

## Current selection

The Python runtime defaults to `minilm_logistic_regression_v4`. v3 remains the explicit rollback `tfidf_linear_svm_calibrated_v3`. v5 is experimental and requires an explicit model identifier plus `EMOTION_V5_ARTIFACT_PATH`; its recorded dependency versions do not match this venv, so it fails closed here. The contextual v6 registry entry has no artifact and always fails closed. No model activation is automatic.

## Verify artifacts

v4 path: `domain_hierarchical_experiment/candidate_model.joblib`; expected SHA-256: `8c93ccc4a87c975b5a58d3ef8a42be3efc41b71e52653f05879cf45e50d82ee7`.

v3 path: `advanced_experiment/baseline_v3/emotion_pipeline_v3.pkl`; expected SHA-256: `b239e1ee1a257c4693e7400282d1154806e8eaf9851656eceed36d89cb0a033b`.

The runtime verifies hashes before deserialization. `backend/ml/emotion_classifier/model_registry.py` records identifiers, expected hashes, input formats, class mapping, and status. Unknown IDs or unavailable contextual IDs fail closed.

## Select rollback

Set `EMOTION_MODEL_VERSION=tfidf_linear_svm_calibrated_v3` in the service process environment, restart the Python service, and verify `/health` reports that exact version and seven classes. To return to v4, unset the variable (or set it explicitly to `minilm_logistic_regression_v4`) and restart. Preserve the prior environment configuration so rollback does not depend on modifying artifacts. Run Python model-runtime tests and the Node-to-Python smoke test before accepting the changed process. This guide changes no production environment itself.
