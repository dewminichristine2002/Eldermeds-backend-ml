# Viva Emotion Recognition Test Cases

These are illustrative, manually reviewed software regression fixtures. They do not estimate clinical performance or accuracy in a real population.

## Case inventory

`backend/tests/unit/services/vivaEmotionRecognition.test.js` contains 90 table-driven cases:

| Group | Cases |
|---|---:|
| Direct happiness statements | 10 |
| Direct sadness statements | 10 |
| Direct loneliness statements | 10 |
| Direct anxiety statements | 10 |
| Direct anger statements | 10 |
| Direct cognitive fog statements | 10 |
| Neutral/ordinary statements | 10 |
| Question-aware, short, uncertain, or mixed answers | 13 |
| Negation and historical/current time scope | 6 |
| Negated class with explicit alternative emotion | 1 |
| **Total** | **90** |

## Coverage

- Direct self-reports and narrowly scoped indirect phrases
- Yes/no and very short answers, interpreted against explicit question wording
- Tentative/ambiguous phrases (`sometimes`, `a little`, `I think so`, `I'm fine`)
- Negation (`not lonely`, `no longer sad`, `not happy`, `don't feel worried anymore`)
- Past versus current state (`used to`, `not anymore`, current change language)
- Mixed current emotions and clarification suggestions
- Existing interpretation tests also cover low-confidence / fallback evidence, retained meaningful ML output, previous-state preservation for curated clarification questions, and five-turn aggregation

## Results

- Before targeted changes: **34/89 passed**, **55 failed** in the original viva fixture run.
- After targeted changes and the added `Not anymore` case: **90/90 passed**.
- Focused total including existing explicit-evidence and contextual interpretation tests: **134/134 passed**.

The before and after values are regression counts for this checked-in fixture set only. They are not an accuracy benchmark. The full backend suite is reported in `VIVA_EMOTION_DETECTION_BEFORE_AFTER.md`.
