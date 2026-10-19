# Stage 6A MiniLM v5 Training Report

## Experiment

Authoritative run: `20261008T175725Z` (UTC). Reproduce with `python train_v5_answer_only.py` from `backend/ml/emotion_classifier/`. The run creates a unique timestamped directory and refuses to reuse an existing run directory. The source is `data/development_dataset_v3.csv` (SHA-256 `3ce56deba3e0b87c1e76abe368ba116c0a9c529c3c0c1c7b3d4ef0ccb45dd728`). It retains the seven existing labels and source rows; no evaluation dataset is used for fitting or selection. Other earlier experimental run dispositions are in `v5_experiments/README.md`.

Encoder: `sentence-transformers/all-MiniLM-L6-v2`, CPU, local checkpoint, 384 dimensions, L2 normalization. Text handling collapses whitespace only. Seed: 42. Fit/validation/calibration sizes: 949/316/317. Validation is the model-selection partition; calibration is held out from classifier fitting. The winner classifier is trained only on fit data, then its sigmoid calibrator is fitted on calibration data. The validation rows are not used to fit either component.

## Candidate comparison on validation

| Candidate | Accuracy | Macro F1 | Worst-class F1 |
|---|---:|---:|---:|
| Logistic Regression, C=1, unweighted | 0.6171 | **0.6473** | 0.4762 |
| Logistic Regression, C=1, balanced | 0.6139 | 0.6359 | 0.4902 |
| Balanced Logistic Regression, C=0.03 | 0.5443 | 0.5368 | 0.3721 |
| Balanced Logistic Regression, C=0.1 | 0.5633 | 0.5665 | 0.4176 |
| Balanced Logistic Regression, C=0.3 | 0.5918 | 0.6068 | 0.4466 |
| Balanced Logistic Regression, C=1 | 0.6139 | 0.6359 | 0.4902 |
| Balanced Logistic Regression, C=3 | 0.6266 | 0.6471 | 0.4646 |
| Balanced Logistic Regression, C=10 | 0.5981 | 0.6344 | 0.4571 |
| Balanced Linear SVM, C=0.03 | 0.5791 | 0.5851 | 0.4222 |
| Balanced Linear SVM, C=0.1 | 0.6013 | 0.6154 | 0.4681 |
| Balanced Linear SVM, C=0.3 | 0.6171 | 0.6369 | 0.4646 |
| Balanced Linear SVM, C=1 | 0.5886 | 0.6204 | 0.4615 |
| Balanced Linear SVM, C=3 | 0.5728 | 0.6069 | 0.4314 |
| Balanced Linear SVM, C=10 | 0.5095 | 0.5342 | 0.3396 |

The selected uncalibrated candidate was unweighted Logistic Regression C=1. Its margin over balanced Logistic Regression C=3 is only 0.0002 macro F1, so the exact winner is fragile. The calibrated selected model scored 0.6266 accuracy / 0.6599 macro F1 on the candidate-selection validation partition. This is not an independent confirmation after selecting the classifier on that partition. A v4 prediction was also computed on those rows, but all 316 texts occur in v4's 2,620-row training corpus, so that number is resubstitution performance and is excluded from comparative conclusions.

## Calibration and error evidence

Sigmoid (Platt) calibration was trained on the 317-record calibration partition. On the 316-record model-selection validation set, calibrated macro F1 was 0.6599, multiclass log loss 1.1458, multiclass Brier score 0.5496, 10-bin top-label expected calibration error 0.1400, mean top-label confidence 0.4866, and probability sums ranged from 0.9999999999999997 to 1.0000000000000004. These values do not establish strong calibration by themselves; the generated reliability plot is in `v5_experiments/20261008T175725Z/v5_validation_reliability.png`. The seven-class confusion matrix and per-class report are in that run directory.

The validation confusion matrix shows the largest error flows were cognitive_fog→neutral (13), neutral→cognitive_fog (8), sadness→anger (7), sadness→anxiety (6), and anxiety→anger (6). These are descriptive errors, not label corrections. Negation and short-answer subsets were not independently large enough for reliable subgroup estimates.

Measured CPU batch latency was about 17.38 ms/record for v5 and 17.28 ms/record for v4, with shared encoder cost included. The measurement used batches of up to 32 on this machine; it is not a deployment service latency guarantee.

## Artifacts

- `v5_experiments/20261008T175725Z/minilm_logistic_regression_v5_answer_only.joblib` (SHA-256 `46af71869bb39902742a42a66e37a57e9166858c6bb5b0f28a04fae8da37d115`)
- `metadata.json` (includes class order, checkpoint, preprocessing, calibration, and library-version checks)
- `experiment_report_final.json`, `split_manifest.csv`, `validation_predictions.csv` (text omitted), validation confusion CSV/PNG, reliability CSV/PNG

The artifact is experimental and is not the production default. The recorded environment is Python 3.12.1, scikit-learn 1.8.0, Sentence Transformers 6.0.0, pandas 3.0.2, NumPy 2.0.0, joblib 1.5.3. The repository requirement currently excludes Sentence Transformers 6.0.0; runtime requires exact training versions and fails closed on mismatch.
