# Stage 4 Research Hardening Report

## A. Original baseline

The deployed baseline was `tfidf_linear_svm_calibrated_v2`: word and character TF-IDF, balanced `LinearSVC`, and sigmoid calibration. Its existing seed-42 held-out results were accuracy 0.5827, macro precision 0.6146, macro recall 0.6212, macro F1 0.6164, and weighted F1 0.5772.

## B. Dataset audit

The baseline `training_dataset_v2.csv` contained 1,904 rows. There were no empty normalized texts, exact duplicates, conflicting labels, or exact train/validation/test overlaps. Forty-four records of at most two tokens were flagged in `very_short_record_audit.csv`; they were not automatically deleted because brevity alone does not prove incorrect labeling. Character 3–5 gram nearest-neighbor analysis found no development/test pair with cosine similarity at least 0.85.

The former training code generated, but did not persist, a 1,218/305/381 train/validation/test split. The source data had no separate validation or test file. The reconstructed test class support was anger 62, anxiety 59, cognitive_fog 61, happiness 62, loneliness 11, neutral 62, and sadness 64.

## C–D. Dataset changes and exact counts

Provenance was canonicalized to `goemotions`, `project_authored`, and `domain_relabel`. No source is inferred silently. Fifty-nine independently written, older-adult conversational examples were added to development only: anger 12, cognitive_fog 15, loneliness 20, and neutral 12. No held-out example was changed or moved.

| Label | Before | After | GoEmotions | Project-authored | Domain relabel |
|---|---:|---:|---:|---:|---:|
| anger | 310 | 322 | 300 | 22 | 0 |
| anxiety | 293 | 293 | 283 | 10 | 0 |
| cognitive_fog | 305 | 320 | 300 | 20 | 0 |
| happiness | 310 | 310 | 300 | 10 | 0 |
| loneliness | 58 | 78 | 0 | 30 | 48 |
| neutral | 310 | 322 | 300 | 22 | 0 |
| sadness | 318 | 318 | 300 | 18 | 0 |

The final development/training corpus has 1,582 rows. Its class support is anger 260, anxiety 234, cognitive_fog 259, happiness 248, loneliness 67, neutral 260, and sadness 254. Exact label-by-source counts are in `dataset_source_distribution.csv`.

## E. Locked-test methodology

The original seed-42 20% test partition was preserved because the audit found no exact or obvious near-duplicate leakage. It is stored in `data/locked_test_v2.csv`. Its canonical sorted-CSV SHA-256 is `81c673cd2fdd896e8c813d4f13f09f024d6ab695ec1f71c07f30bb778d8dd413`; construction and prohibited uses are recorded in `data/locked_test_metadata.json`. It was excluded from feature, model, hyperparameter, calibration, and threshold selection.

## F–H. Cross-validation, tuning, and selection

Five-fold shuffled stratified CV with seed 42 was run on development data. Macro F1 was primary; ties used macro recall, lower variance, and then stability/simplicity.

| Model | Mean macro F1 | Std | Accuracy | Macro precision | Macro recall | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| Calibrated Linear SVM | 0.6306 | 0.0298 | 0.5992 | 0.6417 | 0.6251 | 0.6019 |
| Logistic regression | 0.6199 | 0.0279 | 0.5916 | 0.6342 | 0.6147 | 0.5960 |
| Multinomial NB | 0.5626 | 0.0394 | 0.5581 | 0.6198 | 0.5477 | 0.5553 |

Nine predefined configurations covered word unigram/bigram choices, word `min_df` 1–3, sublinear TF on/off, character `char_wb` ranges 3–5/3–6/4–6, character `min_df` 1–2, SVM/logistic C 0.5/1/2, and NB alpha 0.25/0.5/1. Full results are in `hyperparameter_results.csv`.

The frozen winner was balanced Linear SVM with C 0.5, word (1,2) TF-IDF (`min_df=1`, sublinear TF), and character (3,6) TF-IDF (`min_df=1`). Sigmoid calibration used three inner development folds during outer CV and five development-only folds for the final candidate. No final-test data entered calibration.

## I–K. One-time held-out evaluation

| Metric | Hardened candidate |
|---|---:|
| Accuracy | 0.6142 |
| Macro precision | 0.6366 |
| Macro recall | 0.6490 |
| Macro F1 | 0.6422 |
| Weighted F1 | 0.6118 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| happiness | 0.7273 | 0.7742 | 0.7500 | 62 |
| sadness | 0.6875 | 0.6875 | 0.6875 | 64 |
| loneliness | 0.8333 | 0.9091 | 0.8696 | 11 |
| anxiety | 0.7333 | 0.7458 | 0.7395 | 59 |
| anger | 0.5323 | 0.5323 | 0.5323 | 62 |
| cognitive_fog | 0.4516 | 0.4590 | 0.4553 | 61 |
| neutral | 0.4909 | 0.4355 | 0.4615 | 62 |

The confusion matrix is in `confusion_matrix.csv`. The largest relevant errors were cognitive_fog→anger (13), neutral→cognitive_fog (12), neutral→anger (9), cognitive_fog→neutral (9), anger→neutral (9), cognitive_fog→anxiety (5), sadness→loneliness (1), and loneliness→sadness (0). Cognitive fog and neutral remain the main unresolved boundary.

## L. Cognitive-fog analysis

On the independent project set, 4/10 cognitive-fog examples were correct under pure ML and 7/10 under hybrid routing. Errors included three predictions as neutral, one as anxiety, one as happiness, and one as sadness. The complete text-level, prediction, confidence, and observed-pattern audit is in `cognitive_fog_error_analysis.csv`; ambiguous failures are retained rather than hidden.

## M. Loneliness support

Final training/development support is 67; each five-fold validation fold contains approximately 13–14 and its corresponding training fold approximately 53–54. Locked-test support remains 11. The independent project evaluation has 10. Loneliness is therefore not claimed to be solved despite strong held-out F1.

## N. v2 versus hardened candidate

| Metric | v2 | Hardened | Difference |
|---|---:|---:|---:|
| Macro F1 | 0.6164 | 0.6422 | +0.0259 |
| Macro recall | 0.6212 | 0.6490 | +0.0278 |
| Accuracy | 0.5827 | 0.6142 | +0.0315 |
| Weighted F1 | 0.5772 | 0.6118 | +0.0345 |

Per-class F1 differences were happiness −0.0038, sadness +0.0208, loneliness −0.0395, anxiety +0.0288, anger +0.0617, cognitive_fog +0.0334, and neutral +0.0797.

## O–Q. Pure ML, hybrid evaluation, and threshold analysis

The independent project set contains 70 texts, exactly 10 per class, with zero normalized overlap against development or locked test. Pure classifier correctness was 39/70 = 0.5571. This is reported separately from operational routing.

At the development-selected threshold, hybrid emotion-analysis pipeline correctness was 43/70 = 0.6143; ML coverage was 0.8714, fallback rate 0.1286, correctness on ML-accepted examples 0.5902, and correctness on fallback examples 0.7778. These are hybrid pipeline results, not ML accuracy.

| Threshold | ML coverage | Accepted accuracy | Fallback rate | Hybrid correctness |
|---|---:|---:|---:|---:|
| 0.30 | 0.9020 | 0.6405 | 0.0980 | 0.6081 |
| 0.35 | 0.7895 | 0.6853 | 0.2105 | 0.6062 |
| 0.40 | 0.6460 | 0.7417 | 0.3540 | 0.5885 |
| 0.45 | 0.5221 | 0.8220 | 0.4779 | 0.5740 |
| 0.50 | 0.4362 | 0.8652 | 0.5638 | 0.5442 |

These figures are development out-of-fold results. Threshold 0.30 was selected because it produced the highest overall hybrid correctness among options maintaining at least 70% ML coverage. It is a software confidence-routing threshold, not a clinical threshold.

## R–T. Acceptance and deployed version

The predeclared acceptance rule required higher overall macro F1, no cognitive-fog or neutral decline, and no per-class F1 decline greater than 0.05. The candidate passed. The modest loneliness decline is documented and inside the bound. The deployed version is `tfidf_linear_svm_calibrated_v3`.

## U. Regression results

- Component 4 Jest: 13 suites passed; 104/104 tests passed.
- Python API: 4/4 `unittest` tests passed.
- Python modules compile successfully.
- Live prediction returned the unchanged endpoint fields and correctly classified a concentration/lost-track utterance as cognitive fog.

## V–W. Files and Git status

New reproducibility inputs include `project_authored_hardening.csv`, `development_dataset_v3.csv`, `locked_test_v2.csv`, and `locked_test_metadata.json`. The protocol is `harden_emotion_model.py`; the legacy `train_emotion_model.py` delegates to it. Required research outputs are under `results/`, and the deployed pipeline/metadata were updated only after acceptance. The endpoint contract is unchanged.

At completion, all work is confined to `backend/ml/emotion_classifier`. Git remains on `Sandali2`; changes are intentionally uncommitted for review.
