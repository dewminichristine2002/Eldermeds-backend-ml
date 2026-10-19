# Viva Emotion Detection: Before and After

## Scope

This is an application behavior improvement for the existing five-turn check-in. It is not model training or a clinical validation study. MiniLM v4 remains the production predictor; no model weights, model registry selection, caregiver alert criteria, or activity recommendation contracts were changed.

## Original root causes

1. **Answer valence was mistaken for response polarity.** The old flow used an emotion-oriented `answerPolarity` value to select question bank `positiveResponseEffect` / `negativeResponseEffect`. `Yes, very worried` can have negative emotional valence even though it affirms the worry question, so the selected effect could oppose anxiety.
2. **Short replies lacked a question-level target interpretation.** A bare yes/no did not distinguish direct questions such as “Are you feeling lonely?” from broader wellbeing or protective-factor questions.
3. **Explicit evidence patterns were incomplete and fragile.** The expanded dynamic regular expression contained incorrectly escaped word boundaries, making direct self-reports undetectable. Broad word matching would have created false positives, so the repair uses first-person and narrowly scoped indirect patterns, with negation and time scope.
4. **Prior turns were not provided to interpretation.** Only the previous interpreted emotion was available. The controller now supplies up to two earlier question/answer/state summaries as audit context; current interpretation does not inherit their label.

## Files changed for this improvement

- `backend/src/services/explicitEmotionEvidenceService.js`
- `backend/src/services/contextualAnswerInterpretationService.js`
- `backend/src/controllers/adaptiveChatController.js`
- `backend/tests/unit/services/vivaEmotionRecognition.test.js`

Other modified/untracked workspace files predate this task; they were preserved. No frontend file was modified for this task.

## Improvements

- Classifies clear affirmative, negative, tentative, and ambiguous short replies separately from emotional valence.
- Uses an explicit question target only when the question text names that target. A direct “yes” to a clear worry question can support anxiety; “no” rejects the queried concern without inventing its opposite.
- Keeps direct current self-reports above contextual inference, respects negation and temporal resolution, identifies mixed evidence for clarification, and recognizes a small allowlist of indirect social and cognitive expressions.
- Returns explainable evidence source, reason, confidence weight, and `clarificationSuggested` metadata. The adaptive response includes `clarification_suggested`; the five-turn final state and activity continue to come from the established aggregator/recommender pipeline.
- Controller includes up to two earlier turns in interpretation metadata, without treating them as a prediction input or allowing them to override a current answer.

## Illustrative case results

The before run of the same viva fixture file was **34/89 passing** (55 failed). The final fixture file includes 70 direct examples (10 each for six non-neutral emotion classes and 10 neutral), 13 question/ambiguity examples, 6 negation/temporal examples, and one alternative-emotion example: **90/90 passing** after the final added “not anymore” clarification case. These are manually reviewed software fixtures, not accuracy estimates.

Examples:

| Question / answer | Before behavior | Final behavior |
|---|---|---|
| “Did anything make you feel worried today?” / “Yes, very worried.” | Valence could choose a negative response effect, potentially opposing anxiety. | Anxiety; direct question affirmation with worry language. |
| “Have you been feeling lonely?” / “No, I have been feeling happy.” | Mixed valence and keyword overlap could mislead context. | Happiness from the explicit current self-report. |
| “Are you feeling lonely?” / “Not anymore.” | Short “not” could be treated as a negative response effect. | Neutral current state, with clarification suggested. |
| “How are you feeling?” / “I miss my family.” | Model/fallback keyword outcome only. | Loneliness from a narrowly scoped first-person social expression. |
| “Have you been feeling confused?” / “Sometimes, I lose track of what I’m doing.” | Short answer could get neutral or unrelated class. | Cognitive fog with weak evidence and clarification suggested. |
| “How are you feeling?” / “I feel happy but also worried.” | Could retain one class without surfacing conflict. | Neutral and clarification suggested for mixed current emotions. |

## Remaining limitations

- Indirect-language coverage is deliberately narrow; unsupported idioms and sarcasm may remain ambiguous.
- A tentative answer such as “sometimes” receives weak target evidence only when the question clearly asks about that same state. “I think so” stays uncertain.
- `clarification_suggested` is returned as metadata; the existing UI does not launch a new follow-up flow from that flag.
- Python/Node model loading emits pre-existing scikit-learn artifact-version and dependency deprecation warnings. The registered v4 checksum and selected runtime passed existing integrity checks; no artifacts were rewritten intentionally.

## Verification

- Focused interpretation/evidence/viva suites: **134/134 passing**.
- Python emotion-classifier suite: **82 tests passing**.
- Full backend `npm run test:all`: **331 unit + 19 integration + 11 NFR + 2 E2E passing**.
- Local Node-to-Python smoke: startup, readiness, concurrent ensure, Node prediction, unavailable-service fallback, and owned-service shutdown passed; loaded model was `minilm_logistic_regression_v4`.
- No model training or model activation occurred.
