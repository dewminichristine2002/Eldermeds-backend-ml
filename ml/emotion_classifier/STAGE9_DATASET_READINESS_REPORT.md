# Stage 9 Dataset Readiness Report

## Finding

**Status: `WAITING_FOR_VALID_LABELED_DATA`.** No consented, human-reviewed, multi-turn conversation training dataset is present in the repository. The default Stage 9 training input `data/context_emotion_training.csv` is absent. No participant-level independent test set is available, and no Stage 9 model artifact or performance result was created.

The checked-in emotion datasets and previous Stage 7A/7B annotation queue are single-turn utterance/label data. The Stage 7B queue report records 500 pending source examples and zero human review decisions; it is not contextual participant data. Public/project single-turn rows cannot be turned into conversations by pairing unrelated examples. No repository record indicates consent or study-use authorization for these historical examples.

The application has database-backed `adaptive_chat_sessions` and `adaptive_chat_turns` code storing user-linked conversation turns, exact questions/answers, model version and detected state. That is application data, and `detected_state` is a model output rather than a human research annotation. The migration definitions inspected do not establish research consent scope, double-review decisions, or training-use provenance. The live database was not queried; the existence of user chat history is not treated as research consent.

## Question bank

The active question bank is the PostgreSQL `adaptive_question_bank`, read by `adaptiveQuestionBankRepository.js`. Its source/migrations expose question IDs/codes/text, phase/category/subcategory/target, question type, trigger keywords, response type, quick replies, assessment dimension/status, and positive/negative/neutral/follow-up branch codes. `scripts/exportAdaptiveQuestionBank.js` performs a read-only export and assigns a content-derived SHA-256 version. It was not executed here because no live question-bank export was authorized/performed; therefore no current row count or question-bank version is asserted.

## Eligibility decision

The training gate now requires every record to be a consented participant conversation with opaque authorization reference, source provenance, actual question-bank ID/version, same-session chronological history, and eligible independent human annotation. Model-predicted labels, one-reviewer labels, synthetic fixtures, public single-turn rows, and independent-test rows fail eligibility. `train_contextual.py` reports `WAITING_FOR_VALID_LABELED_DATA` and creates no files when the input is absent.

## Required next inputs

1. Governance owner documents required approvals, consent text/process, scope of model research use, data controller, and retention/deletion schedule.
2. Authorized staff collect pseudonymous sessions under that approved protocol and export the exact question-bank version.
3. Two independent trained reviewers annotate each eligible turn; a distinct third reviewer resolves disagreements. No model predictions are shown as labels.
4. A separate frozen independent benchmark is curated and kept away from candidate selection.

No privacy-sensitive production chat records were read or exported for this audit.
