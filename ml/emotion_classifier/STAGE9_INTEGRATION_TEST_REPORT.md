# Stage 9 Integration Test Report

## Outcome

Stage 9 data tooling and the unchanged Node-to-Python emotion runtime were verified locally. Tests used synthetic fixtures only for software validation. No live participant conversations or database rows were accessed.

## Test results

| Check | Result |
|---|---:|
| Python emotion-classifier suite | 63 passed |
| Node unit suite | 26 suites, 241 passed |
| Node integration suite | 2 suites, 19 passed |
| Node NFR suite | 2 suites, 11 passed |
| Node E2E suite | 2 suites, 2 passed |
| Node total | 32 suites, 273 passed |
| Real Node-to-Python runtime integration | Passed |
| Contextual training with absent dataset | `WAITING_FOR_VALID_LABELED_DATA`; no artifact created |

The runtime integration launched the Python service through the Node service manager, rejected an unrelated process on the configured port, checked concurrent ensures and readiness, made a prediction through Node using v4, checked unavailable fallback behavior, and stopped the service it owned. The active model remained `minilm_logistic_regression_v4`.

The current v4/v3 serialized artifacts were trained with scikit-learn 1.8.0 while this environment uses 1.9.0. Runtime and integration tests passed, with upstream `InconsistentVersionWarning` and NumPy/joblib deprecation warnings visible. The experimental v5 runtime remains version-gated; no artifact was promoted or altered.

## Scope and limits

These tests establish software behavior only. They do not establish consent, annotation quality, clinical validity, contextual model quality, or performance on older adult conversations. The question-bank export script was not run against PostgreSQL. A future authorized collection must be independently reviewed and separately evaluated before model promotion can be considered.
