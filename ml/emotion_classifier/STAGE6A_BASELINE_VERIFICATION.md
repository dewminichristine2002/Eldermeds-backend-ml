# Stage 6A Baseline Verification

## Runtime and artifacts

- The Node-facing emotion prediction path is `backend/ml/emotion_classifier/api_service.py`, backed by `model_runtime.py` (the API implementation in this checkout is FastAPI).
- Default runtime: `minilm_logistic_regression_v4`.
- Explicit rollback: `tfidf_linear_svm_calibrated_v3`.
- v4 encoder: `sentence-transformers/all-MiniLM-L6-v2`, L2-normalized 384-dimensional embeddings.
- v4 classifier artifact: `domain_hierarchical_experiment/candidate_model.joblib`.
- v4 artifact SHA-256 from production metadata: `8c93ccc4a87c975b5a58d3ef8a42be3efc41b71e52653f05879cf45e50d82ee7`; the training run verified the on-disk hash against metadata before using v4 as a baseline.
- Seven runtime labels and order remain: happiness, sadness, loneliness, anxiety, anger, cognitive_fog, neutral.
- v4 output uses classifier `predict_proba`; metadata says its confidence is not calibrated.

## Reproduction

The v4 artifact reproduced on the 56-row `domain_development_evaluation_frozen.csv` partition at accuracy **0.8393** and macro F1 **0.8355**, matching stored historical development values exactly. It also reproduced on `locked_test_v2.csv` at accuracy **0.5459** and macro F1 **0.5649**, matching stored historical values to four decimal places. This locked benchmark is previously exposed. It was scored during procedural reruns while correcting validation handling and adding reliability outputs; its results did not influence fitting, calibration, candidate selection, threshold choice, or configuration changes. The raw locked CSV SHA-256 is `497f53bf895a8426f185b511943c9b2ce7f544fa36b47fd00a4686d393392491`.

The 56-row reproduction uses the frozen domain-development partition and checksum-verified candidate. The separate historical broader-domain value (0.8143 / 0.8166) was not rerun; it remains historical evidence only.

The 316 v5 validation records have **316/316 exact text overlap** with the v4 artifact's 2,620-row development training split. The v4 score previously computed on those rows (0.7152 accuracy / 0.7295 macro F1) is resubstitution performance and is not a valid held-out comparator. The v4 training split has zero exact and zero >=0.90 char-TFIDF near-duplicate overlap with the 381 locked test rows. The v5 development/training data also has zero exact and zero detected >=0.90 near-duplicate overlap with that test.

## Environment recorded during v5 experiment

Python 3.12.1; scikit-learn 1.8.0; Sentence Transformers 6.0.0; pandas 3.0.2; NumPy 2.0.0; joblib 1.5.3. The checked-in emotion requirements allow scikit-learn >=1.3 and Sentence Transformers >=3,<6. The experiment environment therefore does not match the declared Sentence Transformers bound. The optional v5 runtime checks exact scikit-learn and Sentence Transformers versions and fails closed on mismatch. This must be resolved in a controlled packaging environment before v5 could be considered for rollout.

## Limitations

No test set can be considered newly blind: the locked benchmark has previously been examined and was scored during procedural reruns. Its scores did not influence the experiment choices. The original v4 preprocessing and checkpoint files are not modified.
