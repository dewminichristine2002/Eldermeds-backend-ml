# Stage 8 Data Collection and Annotation Protocol

## Governance before collection

Obtain informed consent appropriate to the study population and intended model-development use. Determine whether institutional ethics review or other local approvals are required before recruitment; this repository does not establish that approval or consent has been obtained. Participation must be voluntary, with a clear withdrawal and deletion process. Do not collect model-training data from production users without separate authorization and governance.

## Minimize and protect data

Collect only the question, answer, turn order, pseudonymous participant group, session key, annotation labels, and provenance needed for the prespecified study. Do not store names, contact details, medication histories, diagnoses, or unrelated health narrative in the training dataset. Store the re-identification key separately under restricted access; training tables contain only pseudonyms. Encrypt source data at rest/in transit, restrict access to named study staff, audit access, and publish no raw text. Define a retention period and deletion procedure before recruitment.

## Session and annotation procedure

Use the five existing check-in questions as the interaction context; preserve exact question wording and current answer with integer turn indices. Include up to two earlier question/answer turns, but never future turns. Do not generate a conversation from unrelated utterances. Two trained reviewers independently label the emotional state explicitly expressed by the participant using the Stage 7A seven-class definitions. They must not see the model's predicted label. Use `ambiguous` or `needs_context` where the words do not support a stable single label. A third trained adjudicator resolves disagreements with a rationale. Keep the original individual reviews and adjudication history.

## Quality and independent evaluation

Check consent scope, de-identification, schema, session/participant ownership, chronological order, duplicate families, label distribution, and reviewer agreement before dataset release. Split by participant, then group duplicate/near-duplicate families so no person or text family crosses fit, calibration, validation, or test. Freeze a fresh benchmark under a custodian who is not involved in candidate selection; prespecify per-class metrics, calibration, confidence intervals, minimum class support, subgroup analyses only where justified, and failure review. Keep benchmark labels unavailable to training staff until the final protocol is locked.

## Retention and deletion

Set a study-specific retention schedule with approval owners and secure deletion verification. Remove withdrawn participant records from source and derived versions according to the approved protocol, then rebuild hashed manifests and splits. Keep only the minimum audit evidence needed to demonstrate deletion and annotation provenance.
