# Stage 4 Domain + Hierarchical Emotion Classification Report

## A–B. v3 baseline and research motivation

Production remains immutable `tfidf_linear_svm_calibrated_v3`. Genuine final performance is original-test accuracy 0.6142 / macro F1 0.6422 and independent-domain accuracy 0.6000 / macro F1 0.5869. The previously reported 0.8130 development accuracy is not a final metric. The experiment tested whether domain-focused data, semantic embeddings, or explicit routing could reduce neutral/cognitive-fog, anxiety/cognitive-fog, sadness/loneliness, and anger/anxiety confusion.

## C–F. Dataset audit, provenance, and leakage control

The provenance-controlled 2,620-row advanced development corpus was reused without inspecting final-test errors. It contains 500 happiness, 500 sadness, 72 loneliness, 384 anxiety, 400 anger, 264 cognitive-fog, and 500 neutral texts. Exact label/source/domain counts are in `domain_source_distribution.csv`.

A new 56-row ElderMeds domain-development evaluation was authored and frozen before model fitting, with eight examples per class and checksum `2f21886ce8efb98cfc5d1ad95c127955be9ed013568dc365b524b164621a4814`. It has zero exact and zero character-ngram near-duplicate overlap at 0.85 with training, the original 381-row test, or the 140-row final domain test. Training has zero exact duplicates and zero conflicting labels; 34 records of at most two tokens remain flagged.

The existing group-aware development split was retained: 2,096 rows for fitting and 524 for calibration/threshold analysis. Neither final test was used for feature design, fitting, selection, calibration, thresholding, or error-driven augmentation.

## G. New flat baselines

| Representation/model | Accuracy | Macro F1 | Cog-fog F1 | Neutral F1 |
|---|---:|---:|---:|---:|
| Preserved v3 reference | 0.7143 | 0.7047 | 0.7778 | 0.7059 |
| TF-IDF Linear SVM | 0.7143 | 0.7158 | 0.8750 | 0.6087 |
| TF-IDF calibrated SVM | 0.7143 | 0.7080 | 0.8750 | 0.6364 |
| MiniLM + Linear SVM | 0.8214 | 0.8201 | 0.6667 | 0.8000 |
| MiniLM + Logistic Regression | 0.8393 | 0.8355 | 0.7500 | 0.7500 |

The strongest gain came from semantic embeddings and improved development data, not hierarchy.

## H–I. Hierarchies A and B

Architecture A predicts `neutral` versus `cognitive_fog` versus `emotional`, then resolves five emotional classes. Architecture B predicts `neutral` versus `non-neutral`, then cognitive fog versus emotional, then the five emotions. Every node was independently fitted only on relevant training rows.

Best end-to-end Architecture A was MiniLM + Linear SVM: accuracy 0.7857, macro F1 0.7893. Best Architecture B was MiniLM + Logistic Regression: accuracy 0.7679, macro F1 0.7757. Both were below flat MiniLM + Logistic Regression.

## J. MiniLM experiment

`sentence-transformers/all-MiniLM-L6-v2` embeddings were evaluated with Logistic Regression and Linear SVM for flat and both hierarchical structures. Embeddings were normalized and generated on CPU. Results are in `minilm_results.csv`; no full transformer was fine-tuned.

## K–M. Development and node-level comparison

For best Hierarchy A, Level 1 accuracy was 0.8393 and macro F1 0.7482; conditional Level 2 emotional accuracy was 0.9000 and macro F1 0.8994. End-to-end macro F1 fell to 0.7893 because Level 1 errors prevented the correct downstream node from being reached. The hierarchical error artifact records 12 end-to-end errors, including cognitive fog routed as sadness/anxiety/neutral and neutral routed as cognitive fog/anxiety.

For best Hierarchy B, node results likewise show stronger conditional emotion classification than end-to-end routing. Full metrics for every family/node are in `hierarchy_node_metrics.csv`. Node scores are not reported as final system performance.

## N. Multiple-seed stability

The best lightweight candidate, flat MiniLM + Logistic Regression, was repeated with seeds 42, 123, and 2026. Accuracy mean ± standard deviation was 0.8393 ± 0.0000; macro F1 was 0.8355 ± 0.0000. The convex solver produced the same solution for all seeds.

## O–P. Candidate selection and freeze

The best candidate beat v3 domain-development macro F1 by 0.1309 and accuracy by 0.1250. However, cognitive-fog F1 declined from 0.7778 to 0.7500 and loneliness from 0.8421 to 0.8000. The predeclared promotion rule required macro F1 at least v3 + 0.02 with no cognitive-fog or neutral decline. The candidate therefore failed promotion.

The research classifier was frozen as `candidate_model.joblib` with checksum `8c93ccc4a87c975b5a58d3ef8a42be3efc41b71e52653f05879cf45e50d82ee7`, but is not a v4 and was not evaluated on either final test.

## Q–R. Final tests

No new final evaluation occurred because the development promotion gate failed. Required result files contain only the preserved v3 reports and explicitly state that the candidate was not evaluated. Rejected-candidate cells in `v3_vs_hierarchical_comparison.csv` are blank rather than fabricated.

## S–V. Weak classes on domain development

For the best candidate: cognitive-fog F1 0.7500, neutral 0.7500, anger 1.0000, and loneliness 0.8000 with support eight each. These are small development supports and must not be presented as final application performance. Cognitive fog improved over the production model's original-test 0.4553 but did not improve over v3 under this experiment's matched development protocol.

## W. Confusion and error propagation

Best Hierarchy A errors included happiness→neutral (1), sadness→loneliness (2), loneliness→neutral (1), anxiety→neutral (1), anger→anxiety/cognitive-fog (2), cognitive-fog→sadness/anxiety/neutral (3), and neutral→cognitive-fog/anxiety (2). Several are routing failures at the upper hierarchy level; the strong conditional emotional node cannot repair them.

## X. Label ambiguity and annotation review

`label_ambiguity_review.csv` contains the frozen domain-development texts and original labels with blank reviewer columns. No independent reviewers were available, so reviewer labels or agreement were not fabricated. Mixed or short statements may reasonably admit secondary interpretations, but no labels were changed after evaluation.

## Y–Z. Pure model and hybrid pipeline

Pure best-candidate domain-development accuracy is 0.8393 and macro F1 0.8355. These are development metrics only. Hybrid threshold results use the separate 524-row calibration partition. At selected threshold 0.30, ML coverage is 0.8340, accepted accuracy 0.6407, fallback rate 0.1660, fallback accuracy 0.2874, and hybrid correctness 0.5821. Hybrid correctness is not model accuracy and did not justify promotion.

## AA. Threshold analysis

| Threshold | Coverage | Accepted accuracy | Fallback rate | Hybrid correctness |
|---|---:|---:|---:|---:|
| 0.30 | 0.8340 | 0.6407 | 0.1660 | 0.5821 |
| 0.35 | 0.6813 | 0.6919 | 0.3187 | 0.5744 |
| 0.40 | 0.5420 | 0.7394 | 0.4580 | 0.5363 |
| 0.45 | 0.4218 | 0.8054 | 0.5782 | 0.5210 |
| 0.50 | 0.3397 | 0.8764 | 0.6603 | 0.4981 |
| 0.55 | 0.2729 | 0.9161 | 0.7271 | 0.4695 |
| 0.60 | 0.2137 | 0.9107 | 0.7863 | 0.4179 |

Threshold 0.30 was the only tested option meeting the 70% coverage floor and was selected on development data. It is a software-routing threshold, not a clinical threshold.

## AB. Deployment characteristics

The research classifier artifact is 22.7 KB plus the approximately 91.6 MB MiniLM cache. CPU startup was approximately 0.59 seconds and combined embedding/classification inference averaged 8.41 ms per text over 100 development texts. Preserved v3 is approximately 5.39 MB and previously measured near 5.26 ms per text. MiniLM remains CPU-practical but adds a substantial model dependency.

## AC–AE. Success criteria and recommendation

Independent domain accuracy 0.80 and macro F1 0.75 were not evaluated or claimed because the candidate failed promotion before final tests. Hierarchical classification was not superior in this experiment. Recommendation: retain v3. The flat MiniLM result is promising enough for human review or a future independently annotated domain-development study, but not deployment.

## AF–AH. Regression, files, and Git status

No production model, API contract, frontend behavior, or Stages 3/5–10 code was changed. All 13 Component 4 Jest suites passed (104/104 tests), and all four Python API tests passed. Post-test SHA-256 checks confirmed that production v3 model and metadata still match the pre-experiment snapshots byte-for-byte. All new artifacts are confined to `domain_hierarchical_experiment/`. Git remains on `Sandali2`; the research work is uncommitted pending human review.
