# Stage 6A v4 / v5 Comparison and Promotion Assessment

## Validation and baseline comparability

The v5 calibrated candidate scored **0.6266 accuracy / 0.6599 macro F1 / 0.4950 worst-class F1** on its text-family-disjoint model-selection validation partition (316 records). The v4 artifact also scored those same rows, but its training corpus contains all 316 texts. Its 0.7152 / 0.7295 score is resubstitution performance, not a valid held-out baseline, and is excluded from comparative conclusions.

Therefore a leakage-free v4-versus-v5 validation comparison on this partition is **Not Evaluated**. The candidate cannot be called improved based on its own validation results alone.

## Previously exposed locked benchmark

The 381-row `locked_test_v2.csv` benchmark was scored during procedural reruns after model selection while correcting validation handling and adding reliability outputs. Exact and >=0.90 near-duplicate overlap with either model's training data is zero. v4 reproduced its stored result: accuracy 0.5459, macro F1 0.5649. On the final valid artifact, v5 scored accuracy 0.5538 and macro F1 0.5745. No benchmark result was used for fitting, calibration, candidate selection, threshold choice, or configuration changes. This small difference (+0.0079 accuracy, +0.0096 macro F1) is exploratory only because the benchmark was examined in prior work. It is not fresh independent evidence.

| Class | v4 F1 | v5 F1 | Change |
|---|---:|---:|---:|
| happiness | 0.6615 | 0.6165 | -0.0450 |
| sadness | 0.5714 | 0.6116 | +0.0401 |
| loneliness | 0.7407 | 0.7273 | -0.0135 |
| anxiety | 0.6847 | 0.7455 | +0.0608 |
| anger | 0.6014 | 0.5352 | -0.0662 |
| cognitive_fog | 0.3853 | 0.4202 | +0.0348 |
| neutral | 0.3089 | 0.3652 | +0.0563 |

Test supports are 62 happiness, 64 sadness, 11 loneliness, 59 anxiety, 62 anger, 61 cognitive_fog, and 62 neutral. Class-level changes are descriptive; small/previously exposed test data does not support a promotion decision.

Measured CPU batch end-to-end latency, including shared MiniLM encoding, was 17.38 ms/record for v5 and 17.28 ms/record for v4. These measurements used batches up to 32 on the experiment machine and do not establish production service latency. v5 calibration metrics on the model-selection validation partition were log loss 1.1458, Brier 0.5496, and top-label 10-bin ECE 0.1400; calibration is not strong enough to remove the other promotion blockers.

## Runtime and promotion

The serialized v5 candidate loaded successfully through the optional runtime when explicitly selected; it returns the seven labels and a probability vector. `context_aware` remains false. The API and default model remain v4. The v3 rollback is preserved.

Promotion decision: **not eligible**. v5 improvement over v4 on a leakage-free validation partition is not established, it has not demonstrated robust generalization on a fresh benchmark, and it was trained in a Sentence Transformers version outside the repository requirement. Keep the artifact for research only. Do not activate it in production. A new independently collected/annotated benchmark and a dependency-compatible retraining run are needed before reconsideration.
