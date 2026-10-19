# Stage 7A Dataset Provenance and Quality Report

The reproducible audit is in `stage7a_experiments/20261008T_STAGE7A_FINAL/dataset_quality_inventory.json`; the capped reviewer queue and near-duplicate candidates are beside it. Audit command: `python stage7a_quality_audit.py --run-id <unique-run-id> --replay-v5`.

## Findings

- Seven CSV datasets were inventoried by schema, row count, class counts, missing data, exact normalized duplicates/conflicts, very short and long text, negation counts, IDs, and SHA-256.
- The 1,582-row development corpus has seven supported classes, no missing labels/text, no exact normalized duplicate/conflict families, and 34 very short rows. It contains 220 negation-bearing rows. A character 3–5 gram cosine scan at 0.90 found one candidate near-duplicate pair; its existing labels agree. It remains a candidate, not an automatic merge.
- The 104,395-row prepared GoEmotions file has 48,540 duplicate rows and 33,595 normalized text families with multiple labels. Its source citations, licensing evidence, and source-to-project-label mapping are absent from the local data README. It remains excluded from new training pending provenance and annotation review.
- The separate 5,937-row dataset contains 1,937 `fear` and 2,000 `joy` labels unsupported by the active seven-label schema, as well as three conflicting duplicate families. No remapping was inferred.
- The 59-row project-authored hardening file has four classes and should be treated as already represented in development history, not blindly merged again.
- `training_dataset_v2.csv` contains the locked rows and was excluded from queue sampling. `locked_test_v2.csv` and `project_evaluation_set.csv` were inventoried but excluded from review queue generation and model replay.
- No available emotion dataset has participant, session, or conversation identifiers or paired turns. Grouped splitting can control text-family leakage, but cannot establish participant-independent or conversational generalization.

## Review queue

`annotation_review_queue.csv` contains 500 deterministically sampled candidates from 79,527 duplicate/short/near-duplicate candidate rows, with text, current label, source file, stable pseudonymous ID, reason, pending status, and blank reviewer fields. It includes both sides of the one development near-duplicate candidate pair. Evaluation files and the legacy full training file are excluded. No labels were modified. The cap means the queue is an initial review tranche, not a complete adjudication of every candidate. Keep provenance-unverified public-derived rows out of training until usage is verified.
