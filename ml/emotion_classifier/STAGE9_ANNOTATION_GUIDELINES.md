# Stage 9 Contextual Emotion Annotation Guidelines

Guideline version: `eldermeds-context-emotion-guidelines-1.0`. The prior single-utterance guideline remains `eldermeds-emotion-guidelines-1.0`; Stage 9 context-review events record this contextual version. The seven output labels and their authoritative order remain `happiness`, `sadness`, `loneliness`, `anxiety`, `anger`, `cognitive_fog`, `neutral`.

## Review sequence

Read the exact current question, current answer, and up to two earlier question-answer turns in chronological order. Judge the participant's currently expressed emotion. Do not infer a diagnosis, enduring trait, risk level, or future emotion. Do not view model predictions during first or second review. Preserve the original annotation if shown; do not silently overwrite it. Record `ambiguous` or `needs_context` when a single supported label is not defensible. Exclude unresolved examples from training.

## Label definitions and boundaries

- **happiness:** explicit pleasure, gratitude, hope, joy, or contentment. A positive event without expressed affect can be neutral.
- **sadness:** low mood, grief, disappointment, or emotional pain. Do not infer it from a loss or illness mentioned without affect.
- **loneliness:** expressed disconnection, isolation, or wanting meaningful company. Being physically alone by choice or circumstance is not sufficient.
- **anxiety:** expressed worry, fear, dread, nervousness, or distress about uncertainty. Confusion about a fact alone may instead be cognitive_fog.
- **anger:** expressed irritation, frustration, resentment, or hostility. Disagreement or a negative event alone is not sufficient.
- **cognitive_fog:** explicit confusion, forgetfulness, concentration difficulty, or mental slowness. Never infer it from age, spelling, grammar, or diagnosis.
- **neutral:** factual/logistical speech without clear expressed affect. It is not a substitute for ambiguity or missing context.

## Context-specific guidance

- **Short answers:** interpret “yes”, “not anymore”, “sometimes”, and similar replies only against the paired question and relevant history. If the antecedent remains unclear, choose `needs_context` or `ambiguous`.
- **Negation and tense:** preserve negation, time, and speaker. “I am not worried” is not anxiety; “I used to feel lonely” does not assert current loneliness.
- **Question dependence:** the question gives semantic context, not ground truth. Do not copy the question's emotion into an answer that contradicts it.
- **Indirect expression:** label only affect clearly expressed through the participant's words; do not overread idioms or events.
- **Mixed emotion:** choose the dominant explicitly expressed current emotion only when reasonably clear; otherwise mark ambiguous.
- **Change across turns:** label the current answer, not the session's aggregate trend. Earlier turns clarify references but do not override the current answer.
- **loneliness vs sadness:** disconnection/wanting company supports loneliness; grief/low mood without a social-connection theme supports sadness. Both may be present; abstain when dominance is unclear.
- **anxiety vs cognitive_fog:** worry/fear supports anxiety; explicit memory/concentration/confusion difficulty supports cognitive_fog. Do not infer cognitive_fog from a hesitant short reply.
- **happiness vs neutral:** a positive fact (“the sun came out”) is not automatically happiness; explicit enjoyment/contentment is.

## Independent review and adjudication

Two reviewers annotate independently using pseudonyms. Agreement is eligible for the accepted dataset only when both labels match. A disagreement requires adjudication by a distinct third reviewer who sees prior decisions and documents a concise rationale. Ambiguous, needs-context, and rejected outcomes remain excluded; model outputs are never labels. The audit log contains no question or answer text and remains append-only.
