# Stage 7A Emotion Annotation Guidelines

## Scope and review rule

These guidelines are for human review of the existing seven supported labels. The generated queue is a set of candidates for review, not a relabeled dataset. Keep the existing label unchanged until a reviewer records a decision and a short rationale. Use `abstain_or_escalate` when the text does not support one label reliably; do not force a label to complete a row.

Reviewers should see the utterance alone unless the source dataset provides legitimate, consented context. Do not infer emotion from age, diagnosis, medication, identity, or other protected or clinical attributes. The current datasets contain no participant or conversation identifiers, so they cannot support claims about an individual's longitudinal state.

## Label definitions

| Label | Apply when | Common boundary |
|---|---|---|
| `happiness` | The speaker expresses positive affect, pleasure, gratitude, hope, or contentment. | A factual positive event without expressed feeling may be neutral. |
| `sadness` | The speaker expresses low mood, grief, disappointment, or emotional pain. | Mentioning illness or loss without an expressed emotional reaction is not enough by itself. |
| `loneliness` | The speaker expresses social isolation, disconnection, or wanting meaningful company. | Being physically alone for a practical reason is not necessarily loneliness. |
| `anxiety` | The speaker expresses worry, fear, dread, nervousness, or uncertainty-related distress. | A future plan or question without distress is not anxiety. |
| `anger` | The speaker expresses irritation, frustration, resentment, or hostility. | Firm disagreement or a negative fact without anger is not enough. |
| `cognitive_fog` | The speaker directly reports confusion, forgetfulness, difficulty concentrating, or mental slowness. | Do not infer cognitive fog from grammar, spelling, age, or a diagnosis. |
| `neutral` | The text is primarily factual, logistical, or conversational and does not clearly express one of the other states. | Do not use neutral as a substitute for unclear or missing context; abstain when ambiguity is material. |

## Decision steps

1. Read the complete text and identify explicit affect, not merely emotion-related vocabulary.
2. Preserve negation, tense, speaker, and target. “I am not worried” is not anxiety; “I used to feel lonely” is not necessarily a current state.
3. Distinguish the speaker's emotion from an emotion attributed to another person or quoted in a story.
4. If multiple labels are genuinely present and the schema requires one label, select the dominant explicitly expressed state and note the secondary state. If dominance cannot be resolved, abstain/escalate.
5. Treat sarcasm, idioms, very short replies, missing antecedents, and context-dependent phrases as ambiguous unless the text itself resolves their meaning.
6. For exact or near-duplicate families, review all members together. Keep contradictory labels visible; do not silently collapse examples.
7. Record one outcome: `confirm_existing`, `change_label`, `abstain_or_escalate`, or `needs_context`. A label change requires a rationale and reviewer identifier kept in the controlled annotation system, not in a public report.

## Minimum quality controls

- Two independent reviewers for label conflicts and ambiguous cases.
- Adjudication by a third reviewer when the first two disagree.
- Track agreement and per-class disagreement before changing the training corpus.
- Preserve source attribution, license/usage evidence, original label, final label, reviewer decisions, and a dated change log.
- Keep evaluation data out of annotation-driven training and model selection. A separate independent benchmark needs a frozen protocol and authorized blinded labels.

The current queue contains source text from non-evaluation datasets and stable pseudonymous sample IDs. It contains no reviewer decisions yet. The provenance and license documentation for several source files is incomplete, so those records should not enter a new training corpus until their use is verified.
