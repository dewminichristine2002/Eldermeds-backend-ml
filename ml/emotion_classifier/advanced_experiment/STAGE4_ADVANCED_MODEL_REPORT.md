# Stage 4 Advanced English Model Experiment

## A. v3 baseline

The preserved production baseline is `tfidf_linear_svm_calibrated_v3`. Its original frozen-test metrics remain accuracy 0.6142, macro precision 0.6366, macro recall 0.6490, macro F1 0.6422, and weighted F1 0.6118. The production binary, metadata, locked-test files, hardening report, and results were copied byte-for-byte to `baseline_v3/` before this experiment. SHA-256 equality was verified for the model, metadata, and test file.

## B–C. Expanded data design and counts

The advanced development corpus contains 2,620 normalized, deduplicated English texts with explicit label, source, and domain. Quality-controlled unambiguous GoEmotions rows were used for labels with defensible mappings. Forty independently authored hard negatives cover neutral/cognitive-fog, anxiety/cognitive-fog, anger/anxiety, sadness/loneliness, and neutral/sadness boundaries.

| Label | Count | Aspirational target | Target met |
|---|---:|---:|---|
| happiness | 500 | 500 | yes |
| sadness | 500 | 500 | yes |
| loneliness | 72 | 400 | no |
| anxiety | 384 | 500 | no |
| anger | 400 | 400 | yes |
| cognitive_fog | 264 | 400 | no |
| neutral | 500 | 500 | yes |

Counts were not padded. In particular, GoEmotions `curiosity` and `realization` had previously been mapped to cognitive fog; that pool was not sampled for expansion. The prepared external pool contained 33,595 normalized texts with conflicting project labels, so only texts having one unambiguous project label were eligible. Exact label/source/domain counts are in `advanced_source_distribution.csv`.

## D. Leakage controls

Exact duplicates and label conflicts were removed before partitioning. Four external candidates at character 3–5 gram cosine similarity at least 0.85 to a final-test record were excluded. Obvious near-duplicate development families at similarity at least 0.88 were grouped before deterministic stratified splitting. The final development partitions contain 1,572 train, 524 selection, and 524 calibration records. Model selection used only `selection`; threshold analysis used only `calibration`.

## E. Frozen original-test proof

The original 381-row test remained untouched and was never loaded by transformer training. Its canonical checksum `81c673cd2fdd896e8c813d4f13f09f024d6ab695ec1f71c07f30bb778d8dd413` was verified during finalization. Because v3 was selected on development, its already preserved original-test report was reused; rejected transformers were never evaluated on this test.

## F. Independent domain test

Before tuning, a 140-item ElderMeds English conversational test was frozen with 20 independently written examples per class. It has provenance and domain metadata, zero exact overlap, and zero character-ngram near-duplicate overlap at 0.85 against model data. Its canonical checksum is `08cdb59dc101e0dfae1b94927d0a587218fdabbbede25d39c73263e125e3674d`.

## G–H. Transformer development results

CPU feasibility required maximum length 48, batch size 32, and fine-tuning the classification head plus upper two encoder layers while freezing lower pretrained layers. All models used class-weighted cross entropy, weight decay 0.01, seed 42, and early stopping after a non-improving epoch.

| Architecture/configuration | Accuracy | Macro recall | Macro F1 | Best epoch |
|---|---:|---:|---:|---:|
| DistilRoBERTa, 1e-5 / 3 | 0.1317 | 0.1657 | 0.0651 | 3 |
| DistilRoBERTa, 2e-5 / 4 | 0.6107 | 0.6291 | 0.5851 | 4 |
| DistilRoBERTa, 3e-5 / 5 | 0.6851 | 0.7082 | 0.6764 | 5 |
| DeBERTa-v3-small, 1e-5 / 3 | 0.1527 | 0.1429 | 0.0378 | 1 |
| DeBERTa-v3-small, 2e-5 / 4 | 0.3836 | 0.3491 | 0.3214 | 4 |
| DeBERTa-v3-small, 3e-5 / 5 | 0.5477 | 0.5676 | 0.5116 | 5 |
| Preserved v3 SVM reference | 0.8130 | 0.8482 | 0.8329 | fixed |

The long DeBERTa 3e-5 wall time includes a machine/session suspension and should not be treated as pure compute time. Even excluding that interruption, DeBERTa CPU epochs were materially slower than DistilRoBERTa.

## I. Optional MiniLM

MiniLM was not run. It was optional, required another dependency/model family, and would have delayed the two primary transformer comparisons on a CPU-only machine.

## J. Development-only selection

DistilRoBERTa 3e-5 / 5 epochs was the transformer winner. It was rejected because its 0.6764 development Macro F1 was far below the fixed v3 SVM's 0.8329. Production selection therefore remained v3 before either final test was opened.

## K. Multiple-seed stability

Seeds 123 and 2026 were not run. This was judged computationally disproportionate after both architectures clearly lost the development comparison and no transformer was eligible for deployment. This limitation is explicit; no stability claim is made from seed 42 alone.

## L. Original frozen test

No new original-test run occurred. Preserved v3 metrics are accuracy 0.6142, macro precision 0.6366, macro recall 0.6490, macro F1 0.6422, and weighted F1 0.6118. No v4 values exist because no v4 candidate passed development selection.

## M–N. Independent domain-test metrics

The selected v3 model was evaluated once on the new domain test.

| Metric | Value |
|---|---:|
| Accuracy | 0.6000 |
| Macro precision | 0.5990 |
| Macro recall | 0.6000 |
| Macro F1 | 0.5869 |
| Weighted F1 | 0.5869 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| happiness | 0.5385 | 0.3500 | 0.4242 | 20 |
| sadness | 0.5294 | 0.4500 | 0.4865 | 20 |
| loneliness | 0.6364 | 0.7000 | 0.6667 | 20 |
| anxiety | 0.6923 | 0.4500 | 0.5455 | 20 |
| anger | 0.6800 | 0.8500 | 0.7556 | 20 |
| cognitive_fog | 0.5500 | 0.5500 | 0.5500 | 20 |
| neutral | 0.5667 | 0.8500 | 0.6800 | 20 |

## O. Confusion analysis

Full matrices are stored in `original_test_confusion_matrix.csv` and `domain_test_confusion_matrix.csv`. On the domain test, the notable boundaries were cognitive_fog→neutral (4), anxiety→cognitive_fog (4), sadness→loneliness (4), anxiety→anger (3), and cognitive_fog→anxiety (1). Anger→anxiety was 0 and loneliness→sadness was 1. Happiness→neutral (5) was an additional domain weakness.

## P–R. Weak-class findings

Cognitive-fog domain F1 was 0.55, below the 0.60 secondary target; its largest confusion was neutral. Neutral F1 was 0.68, exceeding its 0.65 target on this independent set, but not on the original frozen test. Loneliness F1 was 0.6667 with credible domain support 20, while development support remained only 72; no claim that loneliness is solved is made.

## S–T. Pure model and hybrid pipeline

Pure selected-model domain correctness was 0.6000. At the development-selected threshold 0.50, hybrid domain correctness was 0.6429, ML coverage 0.3857, fallback rate 0.6143, accepted-case correctness 0.8704, and fallback correctness 0.5000. Hybrid correctness is not reported as transformer or ML accuracy.

Rejected transformer softmax was not operationally calibrated. The retained v3 uses its existing development-only five-fold sigmoid calibration. Calibrating a rejected transformer would not alter model selection and would add overinterpretation.

## U. Threshold analysis

| Threshold | ML coverage | Accepted accuracy | Fallback rate | Hybrid correctness |
|---|---:|---:|---:|---:|
| 0.30 | 0.9637 | 0.8139 | 0.0363 | 0.7977 |
| 0.35 | 0.9198 | 0.8361 | 0.0802 | 0.7977 |
| 0.40 | 0.8740 | 0.8603 | 0.1260 | 0.8015 |
| 0.45 | 0.8168 | 0.8995 | 0.1832 | 0.7996 |
| 0.50 | 0.7805 | 0.9291 | 0.2195 | 0.8034 |
| 0.55 | 0.7309 | 0.9478 | 0.2691 | 0.7863 |
| 0.60 | 0.6527 | 0.9737 | 0.3473 | 0.7405 |

Threshold 0.50 maximized development-calibration hybrid correctness among thresholds with at least 70% ML coverage. It is a software routing threshold, not a clinical threshold.

## V. CPU/deployment trade-offs

The v3 artifact is 5.39 MB and averaged approximately 5.26 ms per single-text CPU inference over 100 texts. The saved DeBERTa development checkpoint is approximately 292 MB; downloaded DistilRoBERTa and DeBERTa caches occupied approximately 334 MB and 575 MB respectively without Windows symlink deduplication. Transformer training epochs took minutes on four CPU threads. Because transformers were rejected, production startup and inference latency were not burdened with those artifacts.

## W–Y. v3 versus v4, targets, and deployment decision

There is no defensible v4 final-test row: both transformer candidates were rejected on development. The original-test 0.80 accuracy and 0.75 Macro F1 targets were not genuinely achieved. No v4 was created or deployed. `tfidf_linear_svm_calibrated_v3` remains production, and its artifacts were not overwritten.

## Z. Regression and Git status

No production architecture or API contract changed in this experiment. All 13 Component 4 Jest suites passed (104/104 tests), and all four Python API tests passed. Post-test SHA-256 checks confirmed that the production v3 model and metadata still match their pre-experiment snapshots byte-for-byte. All experiment files are isolated under `advanced_experiment/`; no frontend or Stages 3/5–10 files were modified. Git remains on `Sandali2`, with the preceding uncommitted v3 hardening work plus this isolated experiment available for review.
