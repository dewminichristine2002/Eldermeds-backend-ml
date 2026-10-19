# ElderMeds Stage 10A Readiness Report

## Outcome

**Stage 10A collection preparation is implemented. Real-data readiness remains dependent on approvals and a consented, already de-identified source.** No participant records were accessed, no live database was queried, no model training was run, and production model configuration was not changed.

## Inspection findings

- `backend/src/controllers/adaptiveChatController.js` defines `TOTAL_ADAPTIVE_QUESTIONS = 5`, assigns question numbers from submitted turns, rejects a sixth answer, and completes the aggregate after five answers.
- `backend/src/repositories/adaptiveChatRepository.js` reads/writes user-linked sessions and stores question IDs/text and user answers. Migration `1747800000000_component4_adaptive_chat.js` defines `user_id`, `is_complete`, and turn answer fields. The checked schema does not establish research-consent status, emotion-model research scope, or withdrawal state.
- The active question bank is `adaptive_question_bank`; Stage 9 supplies a content-versioned exporter and the contextual review queue/guidelines/builder.
- **No existing consent-controlled export of production conversations was found.** Application use or the presence of stored chat turns does not show research consent. The Stage 10A utility therefore reads only a separately prepared de-identified JSONL source and an authorization ledger; it has no database connector and cannot retrieve application sessions.

## Delivered workflow

`stage10a_collection.py` provides a local, read-only-input export to a new CSV. It requires an explicit authorization assertion, consent-ledger match for every pseudonymous participant/session, a non-withdrawn emotion-model-research scope, and a frozen question-bank export. It validates the exact question ID/text/version, one-to-five contiguous chronological turn numbering, same-session context, exact prior-turn history (up to two turns), and rejects future-turn leakage. It uses an allowlisted output schema and exclusive file creation.

The collection template is `stage10a/conversation_collection_template.csv`. The separate ledger header is `stage10a/consent_ledger_template.csv`. `stage10a/synthetic_five_turn_fixture.jsonl` contains one explicitly synthetic five-turn example for format validation only; it has no emotion labels and is not research or training data. Instructions are in `STAGE10A_COLLECTION_GUIDE.md`.

## Verification

The full Python emotion-classifier suite passed: **82 tests** (including the public-data experiment tests). These checks cover the synthetic-only marker, successful allowlisted export with matching consent and question-bank fixtures, pending/withdrawn/wrong-scope rejection, authorization mismatch rejection, question-bank version/text rejection, source chronology, repeated-question rejection, session isolation, future-turn rejection, and no-overwrite behavior. The standalone fixture validator passed and reported five ordered turns in one session, `research_data: false`, and `training_eligible: false`; the CSV template has 20 consistent columns across its five blank rows. Backend `npm run test:all` passed **273 tests** (241 unit, 19 integration, 11 NFR, 2 E2E). The separate Node-to-Python runtime integration check passed startup, concurrent startup, readiness, prediction, unavailable-service fallback and shutdown; it reported production model `minilm_logistic_regression_v4`.

No export against live records was attempted. The question-bank consistency test uses synthetic bank rows; no real bank export was generated. Stage 10A is a file-based research-preparation workflow, not an application-data extraction mechanism or consent system.

## Remaining prerequisites

1. Obtain and document applicable ethics/governance approval and informed consent for emotion-model research use.
2. Assign an authorized custodian to de-identify sessions and maintain the separate consent/withdrawal ledger and identity mapping.
3. Freeze the relevant actual question-bank export and use its exact IDs, wording and content version.
4. Collect only actual answered turns with no fabricated or future context; use the Stage 9 two-review plus independent-adjudication process.
5. Continue to keep contextual training gated until Stage 9 dataset sufficiency and an independent benchmark are met.

MiniLM v4 remains active and unchanged; contextual v6 remains unavailable. No model was trained or promoted.
