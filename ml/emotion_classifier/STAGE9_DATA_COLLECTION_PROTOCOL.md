# Stage 9 Conversational Data Collection Protocol

## Governance before recruitment

Before collection, the research lead must document the study purpose, intended model-development use, participant population, data controller, retention period, withdrawal/deletion plan, access list, storage location, and whether institutional ethics review or other local approvals are required. No approval is asserted by this repository. Obtain the required approval and informed consent before collection. Explain that participation is voluntary, declining has no effect on care, each question may be skipped, and the participant may stop at any time. Use accessible language and a consent process appropriate to the participant's capacity and communication needs.

Do not export production chat history or treat application use as consent for model research. Do not collect names, contact details, medication histories, diagnoses, or unrelated narrative. If identifiers are volunteered, stop/skip as appropriate and remove them in an approved de-identification step before research review. Keep consent documents and any re-identification key separate from the pseudonymous dataset under stricter access control.

## Session process

Use the approved adaptive five-turn interaction. For each answered turn capture the exact bank `question_id`, question text/version, exact answer, sequence number, opaque participant-group and session pseudonyms, and only the prior two answered question-answer turns. Record a skipped prompt as a skip event outside the supervised examples; never invent an answer or backfill context. Do not infer emotion labels from model predictions. The local data schema and queue utility are preparation tools; they do not collect application data or grant consent.

Export the active question bank read-only with `node backend/scripts/exportAdaptiveQuestionBank.js <new-output-path>` after configuring the backend database connection. The export includes the database's actual IDs, text, tags, response types, active status, and branch codes; its content SHA-256 is its version. Reviewers should use that frozen export when curating session records. Exporting a queue requires an explicit `--assert-consent-confirmed` flag and valid per-record authorization references.

## Annotation and storage

Store raw session JSONL and review queue CSV only in an approved encrypted, access-controlled location. Limit access to named study staff and audit access. Two trained reviewers independently inspect the current question, answer, and up to two prior turns, blinded to model predictions. Record the seven-class decision or ambiguous/needs-context/rejected outcome. A distinct third reviewer adjudicates disagreement with a rationale. The append-only review log stores pseudonymous reviewer IDs and decisions but not conversation text. Agreement and exclusions are computed from review events; no decisions are prefilled.

The source, queue, events, and derived training CSV all contain sensitive conversational content and must follow the approved retention/deletion schedule. The training manifest omits raw conversation text. If a participant withdraws, apply the approved deletion process to source and derived records and rebuild hashes/splits; do not promise a fixed retention period until governance sets one.

## Operational commands

```powershell
python -m annotation.context_review_queue --source <approved-sessions.jsonl> --output <restricted-queue.csv> --assert-consent-confirmed
python -m annotation.review_tool review --queue <restricted-queue.csv> --events <restricted-events.jsonl>
python -m annotation.review_tool summary --queue <restricted-queue.csv> --events <restricted-events.jsonl>
python -m annotation.review_tool build-context-dataset --queue <restricted-queue.csv> --events <restricted-events.jsonl> --source-jsonl <approved-sessions.jsonl> --output-dir <new-version-directory>
```

For a separately recruited and independently annotated benchmark, create a different new version directory and add `--independent-evaluation`; this marks records `independent_test`, which the training command rejects. Do not use this set for candidate selection. At evaluation time, compare its participant-group hashes with the candidate training split manifest.

Use the classifier virtual environment and run commands from `backend/ml/emotion_classifier`. The output directory must be new. None of these commands should be run on production conversations without the documented authorization and governance.
