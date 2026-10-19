# Stage 4 v4 Controlled Deployment Review

Review date: 2026-08-24  
Deployment decision: deploy the frozen MiniLM candidate as the default Python API model, with explicit v3 rollback.

## A. Frozen candidate verification

- Artifact: `domain_hierarchical_experiment/candidate_model.joblib` (22,656 bytes)
- SHA-256: `8c93ccc4a87c975b5a58d3ef8a42be3efc41b71e52653f05879cf45e50d82ee7`
- SHA-256 matches both candidate metadata and the final-evaluation completion marker.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`, CPU, local-files-only, normalized embeddings.
- Cached snapshot recorded by the completed evaluation: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Classifier: frozen balanced Logistic Regression, C=1.0, max_iter=2000, random_state=42, lbfgs.
- Actual classifier classes match all seven supported classes.
- `final_evaluation_completed.json` exists and records `rerun_prohibited=true`.
- No training, dataset mutation, hyperparameter tuning, or final-test rerun occurred in this review.

## B. v3 preservation proof

The active pre-review v3 artifact and rollback snapshot were byte-for-byte identical:

- `emotion_pipeline.pkl`: SHA-256 `b239e1ee1a257c4693e7400282d1154806e8eaf9851656eceed36d89cb0a033b`
- `advanced_experiment/baseline_v3/emotion_pipeline_v3.pkl`: same SHA-256
- v3 metadata and all v3 research results remain in `advanced_experiment/baseline_v3/`.
- Neither the v3 artifact nor its snapshot metadata was overwritten.

## C. v4 model version and research metrics

Production version: `minilm_logistic_regression_v4`.

- Domain test accuracy = **0.8143**
- Domain Macro F1 = **0.8166**
- Broader frozen-test accuracy = **0.5459**
- Broader Macro F1 = **0.5649**

The 81.43% figure is ElderMeds English-domain accuracy, not universal accuracy. This is a domain-optimized model: performance is higher on ElderMeds conversational English than on the broader external-domain test.

## D. Python API and loading strategy

`model_runtime.py` provides one probability interface for v4 and v3. The API constructs one runtime during module/service startup; MiniLM and Logistic Regression are not reloaded per request. The default is v4. The Node-facing response keys and supported classes are unchanged.

At startup the runtime verifies the v4 classifier checksum, bundle architecture, metadata version, and label mapping. MiniLM is loaded with `local_files_only=True`; production therefore cannot silently download or substitute another checkpoint.

If MiniLM, metadata, or the classifier is unavailable/corrupt, `/health` reports `ready=false` with an error and prediction returns HTTP 503. The unchanged Node analysis layer catches that unavailability and uses deterministic `rule_fallback`, whose confidence and model version remain null. There is no automatic silent switch to v3.

## E. Confidence and threshold behavior

Confidence is the winning class value returned directly by Logistic Regression `predict_proba`; it is not fabricated. The operational threshold remains 0.30 because the completed v4 development calibration selected 0.30. The threshold was not selected or adjusted from final-test labels. Node still accepts sufficiently confident ML output and otherwise uses its deterministic fallback.

## F. CPU behavior on the development machine

Measured in fresh Windows Python processes on CPU:

| Measurement | v4 MiniLM | v3 TF-IDF |
|---|---:|---:|
| Startup/load | 9.40–15.02 s | 0.182 s |
| First single request | 30.7–32.2 ms | 19.8 ms |
| Warm single request mean | 15.89 ms | 4.47 ms |
| Classifier artifact | 22,656 bytes | 5,394,483 bytes |
| MiniLM local cache | about 91.6 MB | n/a |
| Approximate Python RSS increase after v4 load | 440 MB | not separately measured |
| Approximate Python RSS after first v4 inference | 503 MB | not separately measured |

Startup varies with OS file cache and library import state. Warm CPU latency is suitable for the existing request path, but v4 has a materially larger startup and memory cost than v3.

## G. New endpoint smoke and linguistic tests

These phrases were newly authored for deployment smoke testing and were not copied from final evaluation records.

| Intended conversational variant | Predicted label | Confidence | Accepted at 0.30 |
|---|---|---:|---|
| Positive daily event | happiness | 0.6641 | yes |
| Low mood | sadness | 0.5828 | yes |
| Lack of social contact | loneliness | 0.7069 | yes |
| Worry | anxiety | 0.8655 | yes |
| Frustration | anger | 0.4024 | yes |
| Difficulty concentrating | cognitive_fog | 0.6492 | yes |
| Ordinary neutral routine | neutral | 0.3435 | yes |

Every endpoint response had a valid supported label, numeric confidence, `source=ml_model`, and `model_version=minilm_logistic_regression_v4`. No smoke phrase was added to a rule or used to tune the model. An older API test sentence changed from happiness to loneliness under v4; the test now verifies the stable API/label contract instead of forcing a model-specific label.

## H. Adaptive five-turn and Node integration trace

The controlled no-history service trace selected exactly five unique assessment questions:

1. `open_general`
2. `neutral_clarify`
3. `neutral_energy`
4. `neutral_engagement`
5. `neutral_positive`

A Q6 request returned null. The Node prediction normalizer accepts the unchanged v4 response, narrative analysis retains ML source/model confidence/version for accepted responses, repository mappings retain `detection_source`, `confidence_score`, and `model_version`, and five-turn aggregation remains bounded and deterministic. Activity routing and execution/difficulty behavior were unchanged.

## I. Caregiver safety regression

The policy remains: three completed sessions with the same concern within seven days produce high risk and a caregiver alert. One or two remain medium without an alert. Emotion confidence, model accuracy, and cognitive performance are not caregiver-alert inputs. Duplicate-alert suppression remains active.

## J. Regression results

- Python Stage 4 API/runtime: **8/8 passed**.
- Component 4 Node regression: **13/13 suites passed, 104/104 tests passed**.
- Covered selector, Python/Node contract, narrative analysis, five-turn aggregation, activity routing, execution/difficulty, trends, risk/alerts, repositories, and integration behavior.
- Only warning: FastAPI TestClient reports an upstream Starlette/httpx deprecation warning; no test failed.

## K. Rollback procedure

Rollback requires no retraining and does not alter artifacts:

1. Stop the Python emotion API service.
2. Set `EMOTION_MODEL_VERSION=tfidf_linear_svm_calibrated_v3` in the service environment.
3. Restart the Python emotion API service.
4. Confirm `/health` returns `ready=true` and `model_version=tfidf_linear_svm_calibrated_v3`.
5. Run the Python API smoke tests and the Node emotion/narrative contract tests.

To return to v4, set `EMOTION_MODEL_VERSION=minilm_logistic_regression_v4` (or remove the variable, because v4 is the default) and restart. The explicit v3 load path was regression-tested successfully.

## L. Deployment-review files

New:

- `model_runtime.py`
- `production_model_metadata_v4.json`
- `tests/test_model_runtime.py`
- `STAGE4_V4_DEPLOYMENT_REVIEW.md`

Modified for deployment:

- `api_service.py`
- `requirements.txt`
- `tests/test_api_service.py`

No Stage 3 or Stage 5–10 implementation file, Node public behavior, frontend file, dataset, frozen evaluation record, or frozen model artifact was changed by the v4 deployment review.
