# Stage 7B Annotation Workflow

## Current queue

The source queue is `stage7a_experiments/20261008T_STAGE7A_FINAL/annotation_review_queue.csv` (500 rows). It is still pending: zero human decisions, zero adjudications, and no agreement estimate. `stage7b_stage8_experiments/review_status.json` has a live queue summary by seven-label class, reason, source, duplicate flag, conflict flag, context availability, and triage priority. The `common_confusion_family` flag is only a triage heuristic from the historical v5 model-selection confusion matrix; it does not relabel or prioritize training data for model selection.

## Start reviewing

From `backend/ml/emotion_classifier`:

```powershell
$env:ELDERMEDS_REVIEWER_KEY = "use-a-local-secret-of-at-least-16-characters"
python -m annotation.review_tool review
```

The tool prompts for reviewer identity using a hidden terminal input, stores only a keyed pseudonym, shows text/original label/reason, accepts one of seven labels or ambiguous/context/rejected/adjudicated states, and appends progress to `stage7b_stage8_experiments/review_events.jsonl`. It skips samples already reviewed by that pseudonym, so the reviewer can stop and resume. Do not put identifying or health details in notes; email and phone-like values are rejected. The audit log stores sample IDs and decisions, not source text.

```powershell
python -m annotation.review_tool summary --summary-output stage7b_stage8_experiments/review_status.json
```

Independent reviewers should use separate identities and the same secret key. Adjudication is allowed only after two independent `reviewed` events exist for that sample. Independent agreement or explicit adjudication is needed for label correction; two reviewers agreeing on ambiguous/rejected/context-needed status is required to exclude that row. Disagreements are not silently resolved.

## Build a versioned dataset

```powershell
python -m annotation.review_tool build-dataset --output-dir stage7b_stage8_experiments/datasets/development_dataset_v4
```

The builder refuses to write a version with no accepted human resolution, refuses unsupported labels or loss of any of the seven classes, never modifies the source CSV, preserves row provenance, writes a correction/exclusion manifest, and preserves unresolved disagreements in the manifest. It does not collapse duplicates. The current review log is empty, so no v4 dataset was generated.
