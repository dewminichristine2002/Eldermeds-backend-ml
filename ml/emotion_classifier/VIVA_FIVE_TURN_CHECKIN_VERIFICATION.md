# Five-Turn Check-In Verification

## Runtime path

1. The existing adaptive chat UI submits the current question ID and answer text to the backend.
2. `adaptiveChatController.respondAdaptiveChat` validates that the supplied user, session, and question match the active session, then obtains prior turns from that same session.
3. `analyzeNarrative` requests emotion prediction from the Python service through `emotionPredictionService`; a runtime failure uses the existing `rule_fallback` path. The raw model emotion, confidence, model version, and source remain distinct from the contextual interpretation.
4. The controller supplies current question semantics, answer text, answer polarity, v4/fallback result, previous interpreted state, and up to two earlier question/answer summaries to `interpretContextualAnswer`.
5. The resulting seven-class interpreted emotion is stored for that turn, recorded with explainable metadata, and used by the adaptive question selector to choose the next question.
6. On turn five, `aggregateAdaptiveSessionResult` aggregates all five stored interpreted states and evidence strengths. That final aggregate is passed to existing risk assessment, support directive, and activity recommendation services. Caregiver safeguards were not edited.
7. The existing frontend receives `final_emotional_state` and the recommended activity from the same completion response. Intermediate turns now also receive `clarification_suggested`; the existing UI does not yet render a new clarification interaction.

## Five-turn test trace

The focused suite exercises five ordered responses through the real contextual interpretation function and existing aggregator:

| Turn | Example answer | Interpreted state | Source intent |
|---:|---|---|---|
| 1 | “I have felt low today.” | sadness | explicit current self-report |
| 2 | “No, I have not felt lonely.” | neutral | direct negation, not opposite-class inference |
| 3 | “Yes, very worried.” to a direct worry prompt | anxiety | question-aware affirmation |
| 4 | “Sometimes, I lose track of what I’m doing.” | cognitive fog | weak current evidence; clarification suggested |
| 5 | “I feel happy today.” | happiness | explicit current self-report |

The aggregator validates the required five numbered turns, preserves source/evidence weights, and selects a final state without simply repeating turn five. Production activity selection uses that aggregate state. The exact final state depends on the evidence weights, and its confidence is an accumulated-evidence score, not a probability or clinical measure.

## Model available and fallback

`npm run test:ml-runtime` started the owned local Python service, loaded `minilm_logistic_regression_v4`, passed readiness and a real Node prediction request, exercised unavailable-service fallback, and shut down the service it owned. Smoke output: startup/readiness/concurrent ensures/Node prediction/fallback/shutdown all passed.

The service-available path preserves the v4 model version and raw source/confidence. The fallback path remains tagged `rule_fallback`; rules are not represented as MiniLM output. The unchanged v4 registry/checksum tests passed in the Python suite.

## Test results

- Python classifier suite: **82 passed**.
- `npm run test:all`: **27 unit suites / 331 tests**, **2 integration suites / 19 tests**, **2 NFR suites / 11 tests**, and **2 E2E suites / 2 tests** all passed.
- Focused emotion/evidence/context suites: **134 passed**.

This is code-level integration verification. The database-backed adaptive chat controller was not exercised against live participant records; no live data was accessed. MiniLM v4 remains active and contextual v6 was not enabled.
