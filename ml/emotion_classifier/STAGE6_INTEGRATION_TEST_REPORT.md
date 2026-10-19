# Stage 6 Integration and Test Report

## Python classifier and runtime

- `python -m unittest discover -s tests -p 'test_*.py'`: **21 passed**, including dataset audit, context formatting, v3 rollback, v4 runtime, and three new v5 training utility tests.
- `python -m py_compile train_v5_answer_only.py model_runtime.py audit_dataset.py`: passed.
- Explicit v5 artifact load and `predict_proba`: passed. Checksum, class set, embedding checkpoint/normalization, input contract, and exact training library versions are checked. Default model remains v4.
- Python FastAPI `TestClient` smoke with explicit v5 selection: `GET /health` returned 200/ready and `POST /predict-emotion` returned 200 with the existing response fields, seven supported classes, calibrated confidence metadata, and `context_aware: false`. The probability vector summed to 1 within floating point tolerance.
- v3 rollback load is covered by the Python suite. Existing v4 context-ignored behavior is also covered.

## Node backend suites

Command: `npm run test:all` from `backend/`.

| Suite | Result |
|---|---:|
| Unit | 231 passed, 24 suites |
| Integration | 19 passed, 2 suites |
| NFR | 11 passed, 2 suites |
| E2E | 2 passed, 2 suites |
| Total | **263 passed, 30 suites** |

The Node integration/E2E suites mock the HTTP ML dependency (and test databases/services where applicable). In addition, a live local loopback smoke started the actual FastAPI service with the experimental v5 artifact and called it through the existing Node `requestEmotionPrediction` client. Node received a normalized `anxiety` result, confidence 0.7463, model version `minilm_logistic_regression_v5_answer_only`, and the unchanged seven supported classes. The Python API TestClient was also exercised in-process. This verifies local API compatibility, not deployment hosting. No production service, database, or migration was touched.

## Runtime compatibility and known limits

The v5 artifact was built with scikit-learn 1.8.0 and Sentence Transformers 6.0.0. The checked-in requirement excludes Sentence Transformers 6.0.0, so v5 fails closed unless the runtime matches the artifact's recorded versions. Fixing dependency packaging and validating in the deployment environment remain promotion blockers. A live Node process calling a live Python server was not run.
