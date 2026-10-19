# Stage 6 Technical Report

## Status

- **Stage 6A: Partial.** Dataset audit, answer-only candidate training, bounded validation selection, separate calibration, artifacts, regression tests, and promotion assessment are complete. A leakage-free v4 comparison on the new validation partition is unavailable; v5 improvement is not established, so it is not eligible for activation.
- **Stage 6B: Prepared / Awaiting Data.** No labeled multi-turn conversation data with participant/session identifiers exists in the audited files; no contextual model was trained or evaluated.

## Phase A findings

The only training/model-selection source was `development_dataset_v3.csv` (1,582 records). It was grouped by exact/near-duplicate text families and split into 949 fit, 316 validation, and 317 calibration records. The seven-label mapping was preserved. The 104,395-row prepared GoEmotions dataset was excluded because it has 48,540 duplicate rows, 33,595 conflicting text families, and no loneliness label. Other mismatched or frozen datasets were not merged.

The selected candidate was MiniLM embeddings plus unweighted Logistic Regression (`C=1`, `max_iter=3000`, seed 42), with a sigmoid calibrator fitted on the independent calibration partition. The v5 model-selection validation score was **0.6266 accuracy / 0.6599 macro F1**, with worst-class F1 0.4950. A leakage-free v4 comparison on those rows is unavailable: all 316 v5 validation texts are present in v4's 2,620-row training corpus, so v4's 0.7152 / 0.7295 score there is resubstitution performance and is not used as a comparator. The 56-row frozen domain-development partition reproduced v4's historical 0.8393 / 0.8355 result exactly.

The comparison on the previously exposed 381-row locked benchmark was v4 **0.5459 / 0.5649**, v5 **0.5538 / 0.5745** (accuracy / macro F1). It was scored during procedural reruns while correcting validation handling and adding reliability outputs; benchmark values did not drive candidate, calibration, threshold, or configuration choices. Training/test exact and near-duplicate checks found no overlap at the stated threshold, but the benchmark is not newly blind. v5 improved four of seven class F1 scores but declined on happiness, loneliness, and anger. The small benchmark change cannot establish generalization.

Calibration on validation yielded log loss 1.1458, multiclass Brier 0.5496, and 10-bin top-label ECE 0.1400. A reliability CSV and plot are saved. Measured batch-CPU estimated end-to-end latency was 17.38 ms/record for v5 and 17.28 ms/record for v4 on this machine; no deployment latency target was established.

## Phase B and integration

`context_builder.py` remains as bounded formatting functionality. The current v3/v4/v5 paths are answer-only. Contextual classification, ownership checks for a contextual production endpoint, and contextual performance are **Not Evaluated** because genuine labeled context data is absent.

Python classifier tests: **21 passed**. Backend tests: **263 passed** across unit, integration, NFR, and E2E. Node ML calls in Jest integration/E2E are mocked. A separate local loopback smoke passed from the existing Node prediction client to the actual Python API with v5 selected; the Python API also passed an in-process TestClient smoke.

## Promotion decision

Keep `minilm_logistic_regression_v4` active by default. Keep `tfidf_linear_svm_calibrated_v3` as explicit rollback. Preserve v5 as experimental research output only. Promotion is blocked because v5 improvement over v4 on a leakage-free validation split is not established, no fresh blind benchmark is available, and the experiment used Sentence Transformers 6.0.0 while the checked-in requirement is `<6.0.0`. The runtime requires exact library versions and an explicit artifact path for v5.

To select v5 for an isolated compatible local experiment, set `EMOTION_MODEL_VERSION=minilm_logistic_regression_v5_answer_only` and `EMOTION_V5_ARTIFACT_PATH` to the run artifact path shown in `v5_experiments/20261008T175725Z/`. This does not constitute production authorization. Roll back by setting `EMOTION_MODEL_VERSION=tfidf_linear_svm_calibrated_v3`; unset model version to use the existing v4 default.

## Reproduction

From `backend/ml/emotion_classifier/`:

```powershell
python audit_dataset.py --data-dir data --output results/stage6a_dataset_audit.json
python train_v5_answer_only.py
python -m unittest discover -s tests -p 'test_*.py'
```

From `backend/`: `npm run test:all`.
