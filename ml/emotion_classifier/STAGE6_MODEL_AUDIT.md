# Stage 6 Model and Dataset Audit

## Scope

This audit was performed against the committed ElderMeds emotion-classifier
artifacts and local CSV datasets. It does not modify the active v4 model, the v3
rollback artifact, datasets, or production configuration.

## Active runtime

The Node-facing endpoint is the legacy Flask service in `api_service.py`, not the
separate `ml-service` FastAPI text scaffold. `model_runtime.py` loads one runtime
at process startup based on `EMOTION_MODEL_VERSION`.

| Item | Verified value |
|---|---|
| Active default | `minilm_logistic_regression_v4` |
| Rollback | `tfidf_linear_svm_calibrated_v3` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Pooling | SentenceTransformer default sentence embedding |
| Normalization | L2-normalized embeddings |
| Classifier | Balanced Logistic Regression, `C=1.0`, `max_iter=2000`, `random_state=42`, `lbfgs` |
| Classes | happiness, sadness, loneliness, anxiety, anger, cognitive_fog, neutral |
| v4 artifact | `domain_hierarchical_experiment/candidate_model.joblib` |
| v4 SHA-256 | `8c93ccc4a87c975b5a58d3ef8a42be3efc41b71e52653f05879cf45e50d82ee7` |
| Runtime confidence | Winning `predict_proba` value |
| Calibration flag | False; no operational v4 calibration artifact is declared |
| Routing threshold | 0.30, selected on development calibration data |

The runtime verifies the v4 metadata version, artifact checksum, bundle
architecture, and class set. It loads the embedding checkpoint locally only.
Unknown model versions and invalid artifacts fail closed and are reported by
health checks.

## Dataset findings

The generated audit is `results/stage6_dataset_audit.json`.

- `training_dataset_v2.csv`: 1,904 rows; 1,904 unique normalized texts; no empty
  rows or duplicates; 44 records with at most two tokens.
- `development_dataset_v3.csv`: 1,582 rows; 1,582 unique normalized texts; no
  empty rows or duplicates; 34 records with at most two tokens.
- `locked_test_v2.csv`: 381 rows; 381 unique normalized texts; no empty rows or
  duplicates; 10 records with at most two tokens.
- `project_authored_hardening.csv`: 59 rows; 59 unique normalized texts.
- `project_evaluation_set.csv`: 70 independently authored, one-turn examples
  using `expected_emotion,text`; it is not a conversation dataset.
- `goemotions_prepared_dataset.csv`: 104,395 rows but only 55,855 unique
  normalized texts, with 48,540 duplicate rows. It was not used as a new
  contextual training set.
- `prepared_emotion_dataset.csv`: 5,937 rows with legacy `anger`, `fear`, and
  `joy` labels, so it is incompatible with the seven-class production mapping.

No available CSV contains `session_id`, `conversation_id`, `participant_id`, or
`user_id`. No genuine question-answer pairs or chronological conversation turns
are available. Therefore multi-turn contextual training cannot be scientifically
validated from the current repository.

## Leakage and generalization assessment

The Stage 4 locked test is previously exposed to model-selection history and is
not a new blind test. The existing Stage 4 audit reported zero exact and
near-duplicate development/test overlap for its frozen splits. The new audit
finds no usable participant/session grouping fields, so participant-level leakage
cannot be ruled out for public source data.

The v4 domain-versus-broader gap is consistent with domain shift and label/source
differences, not evidence that context modeling has improved performance. The
domain set has 20 examples per class, while the broader frozen test has highly
uneven support, including only 11 loneliness examples.

## Decision

The v4 runtime remains active. No v5 artifact is present, and no v5 model was
trained or promoted. A contextual candidate requires an independently labeled
question-answer or multi-turn dataset with session-disjoint evaluation.

## Continuation update — 2026-10-08

The statement above records the earlier audit state. The Stage 6 continuation
trained an experimental answer-only v5 artifact; it remains unpromoted and v4
remains the default. See `STAGE6A_BASELINE_VERIFICATION.md`,
`STAGE6A_DATASET_AUDIT.md`, `STAGE6A_MODEL_TRAINING_REPORT.md`,
`STAGE6A_V4_V5_COMPARISON.md`, and `STAGE6_FINAL_TECHNICAL_REPORT.md` for the
actual continuation results and the v4 validation-overlap correction.
