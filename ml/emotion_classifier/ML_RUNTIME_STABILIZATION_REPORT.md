# ElderMeds ML Runtime Stabilization Report

## Scope and outcome

Stabilized the local runtime for the active emotion prediction path without changing training, model selection, inference features, or prediction behavior. Node still receives the same `/predict-emotion` result and still uses the existing rule fallback when the model service cannot be used.

The production default remains `minilm_logistic_regression_v4`; explicit rollback remains `tfidf_linear_svm_calibrated_v3`. v5 remains gated by its recorded dependency versions, and v6 remains unavailable without a validated artifact.

## Verified request and process path

The adaptive emotion flow is `adaptiveChatController` → `narrativeAnalysisService` → `emotionPredictionService` → FastAPI `backend/ml/emotion_classifier/api_service.py`. The Node client defaults to `http://localhost:8001/predict-emotion` and accepts `EMOTION_ML_SERVICE_URL` as an override. FastAPI loads its model in `model_runtime.py` at process import/startup. The correct server command is `python -m uvicorn api_service:app` with the classifier directory as its working directory; `start_ml_service.ps1` derives that directory from its own path.

There is no Flask emotion endpoint in this checkout. The separate generic `mlServiceManager.js` serves medication and risk endpoints. Its expected generic service directory, `ml-service` at the repository root, is absent. It can be set with `ML_SERVICE_DIRECTORY`. The generic manager now refuses to mistake an `emotion_classifier` health response for its own service and skips background spawning when its configured directory is absent. Keep generic and emotion services on separate ports if both are installed.

## Changes made

- Added `emotionMlServiceManager.js`, a dedicated lazy manager invoked for real emotion API requests. It uses repository-relative paths, selects `EMOTION_ML_PYTHON`, then `ML_SERVICE_PYTHON`, then the classifier venv, then the platform Python command.
- Concurrent first requests share one startup promise. Local auto-start is bounded (120 seconds by default), polls readiness, and cleans up only its own child. An unrelated port occupant is never stopped. Auto-start can be disabled with `EMOTION_ML_SERVICE_AUTO_START=false`.
- Before sending a prediction, the manager validates service identity, ready state, model version, artifact integrity, and the exact seven project classes. A wrong service or wrong model fails clearly into the pre-existing Node rule fallback. A model load error ends startup and cleans up the owned process.
- Added `/live` for process liveness. `/health` remains the readiness endpoint and reports service version, model version/status, artifact checksum verification, class set, and load error.
- Updated the PowerShell launcher to resolve its script directory, accept `EMOTION_ML_PYTHON`, and run with argument arrays. The generic manager accepts an explicit service directory and does not accept emotion service health as generic readiness.
- Added a 52-line `requirements-v4-runtime-lock.txt` snapshot from the active Python 3.12.1 venv. Existing direct pins remain for v4 and v5 experiment environments.

## Runtime and model integrity audit

Audited the active `backend/ml/emotion_classifier/.venv` on 2026-10-09: CPython 3.12.1, scikit-learn 1.9.0, sentence-transformers 5.7.0, transformers 5.15.1, torch 2.13.0, NumPy 2.5.2, SciPy 1.18.1, joblib 1.5.3, pandas 3.0.5, FastAPI 0.141.1, Uvicorn 0.52.4, and httpx 0.28.1. Flask is not installed.

- v4 artifact SHA-256: `8c93ccc4a87c975b5a58d3ef8a42be3efc41b71e52653f05879cf45e50d82ee7`. Runtime verifies this against the registry and metadata before deserialization. It loaded and produced predictions locally. Its metadata omits training dependency versions; scikit-learn emitted `InconsistentVersionWarning` because the pickle records version 1.8.0 while the runtime is 1.9.0. Successful loading is not an exact-version compatibility guarantee.
- v3 rollback artifact SHA-256: `b239e1ee1a257c4693e7400282d1154806e8eaf9851656eceed36d89cb0a033b`. Runtime verifies the registry checksum before loading. It loaded in the test environment with the same scikit-learn 1.8.0-to-1.9.0 warning.
- v5 artifact SHA-256 from its metadata: `46af71869bb39902742a42a66e37a57e9166858c6bb5b0f28a04fae8da37d115`. Its metadata requires scikit-learn 1.8.0 and sentence-transformers 6.0.0; the active venv has 1.9.0 and 5.7.0, so the runtime rejects it before `joblib.load`.
- v6 has no validated artifact and remains fail-closed.

The v4 lock is a `pip freeze` snapshot of the observed Windows environment, pinned by version but without package hashes; it is not a cross-platform or supply-chain-verifiable lock. A full v5 lock could not be derived from this venv because its installed stack differs from the experiment metadata and all transitive/platform wheel versions were not captured. No dependency installation was attempted.

## Verification performed

- Python: `python -m unittest discover -s tests -p "test_*.py" -v` — 48 tests passed. Coverage includes API health/liveness, real v4 artifact loading and prediction, v3 rollback loading, checksum mismatch rejection, v5 dependency mismatch rejection, and v6 fail-closed status.
- Node: `npm run test:all` — unit, integration, NFR, and E2E suites passed (240 unit tests, 19 integration tests, 11 NFR tests, 2 E2E tests; 272 total).
- Real process: `npm run test:ml-runtime` — passed using an ephemeral local port. Confirmed wrong-port identity rejection without killing the unrelated process, one actual manager-started Uvicorn v4 process shared by concurrent ensures, readiness/model/class/integrity validation, a real Node-to-Python prediction, rule fallback with auto-start disabled after shutdown, and shutdown of only the owned child.

Warnings remain visible by design. Observed warnings include Starlette's `httpx` deprecation, scikit-learn pickle version mismatch, and NumPy/joblib deprecation warnings. They are recorded here and not suppressed.

## Local operation

From the repository root:

```powershell
npm --prefix backend run test:ml-runtime
```

The Node backend starts the emotion service on first emotion inference when it is unavailable and local auto-start is enabled. To launch manually, use `backend/ml/emotion_classifier/start_ml_service.ps1`; it starts on `127.0.0.1:8001` by default. Set `EMOTION_ML_SERVICE_URL`, `EMOTION_ML_PYTHON`, `EMOTION_ML_STARTUP_TIMEOUT_MS`, and `EMOTION_ML_SERVICE_AUTO_START` as needed. `EMOTION_MODEL_VERSION=tfidf_linear_svm_calibrated_v3` explicitly selects rollback; there is no automatic v3 switch.

No deployment, database, model artifact, model activation, or unrelated user changes were made.
