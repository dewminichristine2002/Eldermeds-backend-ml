# Stage 4 MiniLM Final One-Time Evaluation

## A. Candidate freeze proof

The evaluated research candidate was frozen before either final test was opened.

- Artifact: `candidate_model.joblib`
- Artifact SHA-256: `8c93ccc4a87c975b5a58d3ef8a42be3efc41b71e52653f05879cf45e50d82ee7`
- Architecture: flat seven-class classifier
- Embeddings: normalized `sentence-transformers/all-MiniLM-L6-v2`
- Cached MiniLM snapshot: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- Classifier: Logistic Regression, C 1.0, balanced class weights, `lbfgs`, L2/default regularization, tolerance 0.0001, maximum 2,000 iterations
- Random seed: 42; stability checks also used 123 and 2026
- Label mapping from the frozen classifier: anger, anxiety, cognitive_fog, happiness, loneliness, neutral, sadness
- Fitting corpus: 2,096 development rows; final tests excluded
- Development accuracy: 0.8393
- Development macro F1: 0.8355
- Development cognitive-fog F1: 0.7500

The actual artifact hash matched frozen metadata before evaluation. No model, embedding, data, label, hyperparameter, or class weight was changed. `final_evaluation_completed.json` records completion and prohibits rerunning.

## B. Final-test integrity proof

| Final test | Rows | Expected checksum | Verified |
|---|---:|---|---|
| Original frozen held-out | 381 | `81c673cd2fdd896e8c813d4f13f09f024d6ab695ec1f71c07f30bb778d8dd413` | yes |
| Independent ElderMeds domain | 140 | `08cdb59dc101e0dfae1b94927d0a587218fdabbbede25d39c73263e125e3674d` | yes |

The existing leakage audit records zero exact or detected near-duplicate overlap between model-development data and the independently frozen domain-development set. The original and domain final tests were excluded from training, selection, calibration, and threshold work. This final evaluation used pure predictions only.

## C. Test A — original frozen test

| Metric | MiniLM candidate |
|---|---:|
| Accuracy | 0.5459 |
| Macro precision | 0.5552 |
| Macro recall | 0.5889 |
| Macro F1 | 0.5649 |
| Weighted F1 | 0.5409 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| happiness | 0.6324 | 0.6935 | 0.6615 | 62 |
| sadness | 0.6182 | 0.5312 | 0.5714 | 64 |
| loneliness | 0.6250 | 0.9091 | 0.7407 | 11 |
| anxiety | 0.7308 | 0.6441 | 0.6847 | 59 |
| anger | 0.5309 | 0.6935 | 0.6014 | 62 |
| cognitive_fog | 0.4375 | 0.3443 | 0.3853 | 61 |
| neutral | 0.3115 | 0.3065 | 0.3089 | 62 |

The candidate underperformed v3 on this more general/non-domain test.

## D. Test B — independent ElderMeds domain test

This is the primary application-domain evaluation.

| Metric | MiniLM candidate |
|---|---:|
| Accuracy | 0.8142857 |
| Macro precision | 0.8278 |
| Macro recall | 0.8143 |
| Macro F1 | 0.8166 |
| Weighted F1 | 0.8166 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| happiness | 1.0000 | 0.9500 | 0.9744 | 20 |
| sadness | 0.8421 | 0.8000 | 0.8205 | 20 |
| loneliness | 0.7826 | 0.9000 | 0.8372 | 20 |
| anxiety | 0.8889 | 0.8000 | 0.8421 | 20 |
| anger | 0.9333 | 0.7000 | 0.8000 | 20 |
| cognitive_fog | 0.6957 | 0.8000 | 0.7442 | 20 |
| neutral | 0.6522 | 0.7500 | 0.6977 | 20 |

The unrounded accuracy exceeds 0.80, and Macro F1 exceeds 0.75.

## E. v3 versus MiniLM

### Overall metrics

| Test/metric | v3 | MiniLM | Difference |
|---|---:|---:|---:|
| Original accuracy | 0.6142 | 0.5459 | −0.0682 |
| Original Macro F1 | 0.6422 | 0.5649 | −0.0774 |
| Original macro recall | 0.6490 | 0.5889 | −0.0602 |
| Original weighted F1 | 0.6118 | 0.5409 | −0.0709 |
| Domain accuracy | 0.6000 | 0.8143 | +0.2143 |
| Domain Macro F1 | 0.5869 | 0.8166 | +0.2297 |
| Domain macro recall | 0.6000 | 0.8143 | +0.2143 |
| Domain weighted F1 | 0.5869 | 0.8166 | +0.2297 |

### Per-class F1 comparison

| Test/class | v3 | MiniLM | Difference |
|---|---:|---:|---:|
| Original happiness | 0.7500 | 0.6615 | −0.0885 |
| Original sadness | 0.6875 | 0.5714 | −0.1161 |
| Original loneliness | 0.8696 | 0.7407 | −0.1288 |
| Original anxiety | 0.7395 | 0.6847 | −0.0548 |
| Original anger | 0.5323 | 0.6014 | +0.0691 |
| Original cognitive fog | 0.4553 | 0.3853 | −0.0700 |
| Original neutral | 0.4615 | 0.3089 | −0.1526 |
| Domain happiness | 0.4242 | 0.9744 | +0.5501 |
| Domain sadness | 0.4865 | 0.8205 | +0.3340 |
| Domain loneliness | 0.6667 | 0.8372 | +0.1705 |
| Domain anxiety | 0.5455 | 0.8421 | +0.2967 |
| Domain anger | 0.7556 | 0.8000 | +0.0444 |
| Domain cognitive fog | 0.5500 | 0.7442 | +0.1942 |
| Domain neutral | 0.6800 | 0.6977 | +0.0177 |

## F. Confusion analysis

On the domain test, the requested boundary errors were:

- cognitive_fog→neutral: 3; neutral→cognitive_fog: 4
- cognitive_fog→anxiety: 0; anxiety→cognitive_fog: 1
- sadness→loneliness: 3; loneliness→sadness: 0
- anger→anxiety: 2; anxiety→anger: 1

On the original test, cognitive fog and neutral remained difficult: cognitive_fog→neutral was 18 and neutral→cognitive_fog was 14. Cognitive_fog↔anxiety was 4 in each direction; sadness→loneliness was 3 and loneliness→sadness 1; anger→anxiety was 1 and anxiety→anger 7. No retraining, relabeling, or tuning occurred after this analysis.

## G–H. Required final table and weak-class safety

Per-class rows below use the primary independent domain test.

| Metric | V3 | MiniLM |
|---|---:|---:|
| Original Accuracy | 0.6142 | 0.5459 |
| Original Macro F1 | 0.6422 | 0.5649 |
| Domain Accuracy | 0.6000 | 0.8143 |
| Domain Macro F1 | 0.5869 | 0.8166 |
| Happiness F1 | 0.4242 | 0.9744 |
| Sadness F1 | 0.4865 | 0.8205 |
| Loneliness F1 | 0.6667 | 0.8372 |
| Anxiety F1 | 0.5455 | 0.8421 |
| Anger F1 | 0.7556 | 0.8000 |
| Cognitive Fog F1 | 0.5500 | 0.7442 |
| Neutral F1 | 0.6800 | 0.6977 |

No weak class collapsed on the primary domain test; all seven domain F1 values improved. The original test shows a material neutral regression of 0.1526 and smaller regressions for most other classes. This is an important domain-specialization trade-off, not hidden by the stronger domain result.

## I. Deployment cost

| Characteristic | v3 SVM | MiniLM candidate |
|---|---:|---:|
| Classifier/model artifact | approximately 5.39 MB | 22.7 KB classifier |
| Embedding dependency | none | approximately 91.6 MB MiniLM cache |
| CPU startup | lightweight | approximately 0.59 seconds |
| Mean single-text CPU inference | approximately 5.26 ms | approximately 8.41 ms |

MiniLM is heavier but remains CPU-practical for the current text check-in workload.

## J. Target result and research conclusion

- Genuine independent-domain accuracy ≥0.80: **YES** (`0.8142857`)
- Independent-domain Macro F1 ≥0.75: **YES** (`0.8166`)
- Better than v3 on the primary ElderMeds domain test: **YES**
- Better than v3 on the original general held-out test: **NO**
- Was hierarchy useful: **NO**; the winning candidate is flat
- Clinical claim: none; these are software classification results

The candidate shows genuine domain specialization: substantially stronger ElderMeds conversational performance with weaker performance on the original broader held-out corpus.

## K. Final recommendation

**A. Recommend MiniLM candidate for v4 deployment review.**

Reason: it genuinely achieved the primary domain targets, improved all seven domain-class F1 scores, showed no critical domain-class collapse, and has acceptable CPU cost. The recommendation is for human-reviewed v4 deployment work, not automatic deployment. Reviewers should explicitly accept the documented loss on the original general test—especially neutral—before approving production replacement.

No model or application artifact was deployed or modified by this evaluation.
