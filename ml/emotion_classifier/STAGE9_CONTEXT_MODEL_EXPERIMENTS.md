# Stage 9 Context Model Experiments

## Status

`WAITING_FOR_VALID_LABELED_DATA`. No supervised contextual training was run and no artifact/metric was produced. The existing MiniLM v4 production environment, active default, and v3 rollback remain unchanged.

## Candidate protocol

Once eligible records exist, `train_contextual.py` compares three input contracts on the same participant/session and duplicate-family-disjoint fit/calibration/validation partitions:

1. `answer_only`: current answer only.
2. `question_answer`: current question and answer.
3. `multi_turn`: up to two earlier question-answer pairs, then current question and answer.

Each format uses the same cached local MiniLM checkpoint, L2-normalized embeddings, and three bounded Logistic Regression settings. Candidate selection uses internal validation Macro F1 (then accuracy as tie-breaker); sigmoid calibration uses a separate calibration partition. The resulting reported validation is model-selection evidence, not independent evaluation. There is no artificial conversation synthesis.

Participant IDs, sessions, exact text families, and near-duplicate families stay within one split. The split manifest contains hashed sample, participant, and session IDs; the training data checksum and manifest hash are recorded with each candidate. Artifacts receive unique Stage 9 IDs and remain outside the production registry.

## Run when authorized data exists

From `backend/ml/emotion_classifier` in the reviewed context-experiment environment:

```powershell
& .\.venv\Scripts\python.exe -m annotation.context_review_queue --source <approved-sessions.jsonl> --output <restricted-queue.csv> --assert-consent-confirmed
& .\.venv\Scripts\python.exe -m annotation.review_tool review --queue <restricted-queue.csv> --events <restricted-events.jsonl>
& .\.venv\Scripts\python.exe -m annotation.review_tool build-context-dataset --queue <restricted-queue.csv> --events <restricted-events.jsonl> --source-jsonl <approved-sessions.jsonl> --output-dir <new-dataset-version>
& .\.venv\Scripts\python.exe train_contextual.py --data <new-dataset-version>\context_emotion_training.csv --output-root stage9_experiments
```

The reviewer and dataset commands require actual approved consent records and real human review decisions. They do not prefill either. The trainer refuses missing consent, synthetic/public source, missing authorization/provenance, missing question-bank ID/version, single-review annotations, test partitions, invalid context, and missing seven-class coverage. Dataset/session/participant boundaries must support each split; otherwise it exits without activating anything.

For a separately collected and double-reviewed benchmark, create it with `--independent-evaluation`. Keep its labels frozen and unavailable during candidate selection. After candidate training, run:

```powershell
& .\.venv\Scripts\python.exe evaluate_context_models.py --data <frozen-benchmark>\context_emotion_training.csv --training-split-manifest <run>\split_manifest.csv --candidate <run>\answer_only_logistic_regression.joblib --candidate-sha256 <sha256> --candidate <run>\question_answer_logistic_regression.joblib --candidate-sha256 <sha256> --candidate <run>\multi_turn_logistic_regression.joblib --candidate-sha256 <sha256> --output <new-evaluation.json>
```

The evaluator requires `independent_test`, all seven classes at the configured minimum support (default five), eligible human provenance, no participant overlap with the candidate split manifest, and matching candidate checksums. It reports accuracy, macro precision/recall/F1, weighted F1, per-class scores, confusion matrices, log loss, Brier score, ECE, clarification proxy rate, participant-cluster bootstrap interval, and measured per-record model inference latency (embedding plus classifier; input formatting is excluded). These metrics have not been calculated because there is no qualifying benchmark.
