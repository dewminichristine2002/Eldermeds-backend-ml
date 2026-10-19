# Stage 10A: Collect and Prepare Conversational Data

This guide prepares data for later human annotation. It does not train a model, change MiniLM v4, activate v6, or make a synthetic row eligible for research.

## What the application currently does

The adaptive emotional check-in starts with question 1 and accepts at most five submitted assessment answers. It selects each prompt from `adaptive_question_bank`; each saved turn contains the user answer, question ID/text, question number, and runtime/model metadata. `adaptive_chat_sessions` is linked to an application `user_id`. The repository schema does not record research consent, research-use scope, or withdrawal status, and there is no existing consent-controlled conversation export. Do not export those tables for this work.

## Before any real collection

1. Obtain the required ethics/governance approval and informed consent for this specific emotion-model research use. Explain voluntariness, skipping, stopping, data use, storage, retention, and withdrawal in accessible language.
2. Have an authorized study-data custodian create pseudonymous participant and session IDs at the approved de-identification boundary. Keep identity mapping and consent documents in separate, access-controlled storage. Never place application `user_id`, names, contact information, diagnoses, or care notes in the collection JSONL or CSV.
3. Store an authorization ledger only in the approved restricted location. Add a row after verifying signed/recorded consent and scope. `authorization_reference` must be an opaque link to evidence kept elsewhere; it is not itself proof of consent.
4. Freeze the exact question-bank export used for collection. The source turns must retain the exact `question_id`, question text, and matching content-derived `question_bank_version`.
5. Capture only actual answered turns. Keep turn numbers in order from 1 to 5. Do not invent answers, backfill skipped questions, pair different sessions, or include later turns as context.

## Input files

Use `stage10a/conversation_collection_template.csv` as the flattened output/annotation template. It has five blank example rows; fill each row with one actual, authorized, de-identified answered turn. Keep `previous_turns_json` to the exact previous one or two Q/A pairs from that same session. Leave `emotion_label` blank and `annotation_status` as `pending` until human review. Keep Stage 9's richer JSONL as the validated interchange input for the exporter.

The consent ledger follows `stage10a/consent_ledger_template.csv` and must include:

- pseudonymous `participant_group_id` and `session_id`;
- `consent_status=consented`;
- `research_scope=emotion_model_research`;
- `withdrawn=false`;
- the matching opaque `authorization_reference`.

Create each source JSONL row using the Stage 9 schema (`eldermeds-context-emotion-1.1`): `sample_id`, `participant_group_id`, `session_id`, 1-based `turn_index`, `question_id`, `question_bank_version`, `current_question`, `current_answer`, `previous_turns`, `consent_or_use_authorization_reference`, `data_origin=consented_participant`, `source_provenance`, and `dataset_split=unassigned`. Do not include raw identity fields or model predictions. See the synthetic fixture for format only.

## Validate and export locally

From `backend/ml/emotion_classifier`, using the project Python environment:

```powershell
& .\.venv\Scripts\python.exe stage10a_collection.py export `
  --source <restricted-deidentified-sessions.jsonl> `
  --consent-ledger <restricted-consent-ledger.csv> `
  --question-bank-export <frozen-question-bank-export.json> `
  --output <new-restricted-output.csv> `
  --assert-authorization-verified
```

The command reads local files only. It does not connect to the application database. It validates the consent ledger against every participant/session and authorization reference; checks active assessment question IDs, exact question text and exact bank version; requires each session's turns to be contiguous, unique, and in chronological order; enforces same-session/same-participant context and a maximum of two previous turns; and rejects future-turn leakage. It exports only an explicit allowlist and refuses to overwrite an existing output. Store both input files and the output under the approved access controls; the output contains sensitive conversation text even though direct identifiers are excluded.

Test the format fixture without research data:

```powershell
& .\.venv\Scripts\python.exe stage10a_collection.py validate-synthetic `
  --source stage10a\synthetic_five_turn_fixture.jsonl
```

It must report `SYNTHETIC_FORMAT_FIXTURE_VALID`, `research_data: false`, and `training_eligible: false`. Never add its text or labels to real collection data.

## Human annotation after collection

After export, two trained reviewers independently use the existing Stage 9 contextual review workflow. They see the current question/answer and up to two prior turns and annotate according to `STAGE9_ANNOTATION_GUIDELINES.md`; do not show model predictions. A distinct third reviewer adjudicates disagreement. Keep event logs text-free, source/queue/output access-controlled, and follow the approved withdrawal/deletion process. Build a versioned dataset only from resolved human reviews with the existing Stage 9 builder. Stage 10A does not start training; the Stage 9 training gate and independent benchmark requirements remain in force.
