# ML Runtime Integration Report

The Node emotion caller (`backend/src/services/emotionPredictionService.js`) posts `{text}` to `${EMOTION_ML_SERVICE_URL || http://localhost:8001}/predict-emotion`. Its consumer is `narrativeAnalysisService`, called by `adaptiveChatController`; rule fallback remains in Node. The service on that route is FastAPI `api_service.py`, which loads v4 by default from `model_runtime.py`. No Flask emotion route exists. FastAPI emits model version, seven supported classes, confidence calibration status, `context_aware: false`, and an optional clarification suggestion. The five-turn flow and caregiver thresholds are unchanged.

The Node emotion client now owns a separate, lazy local lifecycle in `backend/src/services/emotionMlServiceManager.js`. On the first real emotion request it checks `/health` for the exact service identity, readiness, model version, checksum verification, and seven-class set, then starts the classifier if needed. Concurrent requests share one start promise; startup is bounded by `EMOTION_ML_STARTUP_TIMEOUT_MS` (120 seconds by default). It only stops a child process that it spawned. Set `EMOTION_ML_SERVICE_AUTO_START=false` to require a manually started service, `EMOTION_ML_PYTHON` to select Python, and `EMOTION_ML_SERVICE_URL` to change the URL/port.

Manual startup from any working directory is supported by the script:

```powershell
& .\backend\ml\emotion_classifier\start_ml_service.ps1
```


The generic prediction manager is separate. It uses `ML_SERVICE_DIRECTORY` when set and otherwise expects `ml-service` at the repository root. That service is absent in this checkout. Its manager will not treat the emotion API on the shared default port as a healthy generic predictor, and background startup reports the missing directory instead of launching a wrong service. Generic model endpoints need their own service checkout and a port distinct from the emotion API.

`GET /live` confirms only that the process is alive. `GET /health` is readiness and exposes service/model identity, artifact checksum verification, and model load errors. Node falls back through the existing rule path if startup or inference fails; no model output or selection logic was changed.
