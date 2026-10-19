# Public Dataset Label Mapping

## Scope

This mapping is for the isolated DailyDialog public-data experiment only. It does not change ElderMeds production labels or Stage 9 consent-gated dataset validation. The public benchmark has four labels and is **not** an ElderMeds seven-class evaluation.

| DailyDialog native annotation | Experiment label | Decision and reason |
|---|---|---|
| `happiness` | `happiness` | Direct label match. |
| `sadness` | `sadness` | Direct label match. |
| `anger` | `anger` | Direct label match. |
| `no emotion` | `neutral` | Treated as no explicit emotion annotation in this public task. It is not clinical neutrality. |
| `fear` | Excluded | Not automatically mapped to `anxiety`; the construct and annotation are not identical. |
| `disgust` | Excluded | No ElderMeds equivalent. |
| `surprise` | Excluded | No ElderMeds equivalent. |

ElderMeds `loneliness` and `cognitive_fog` have no defensible DailyDialog label mapping and receive no public training examples. The experiment has no valid examples for a full seven-class ElderMeds taxonomy. No mapping is made from general negative sentiment, confusion, or fear to an ElderMeds label.

## Native label counts

The downloaded source contains the original turn annotations. Across all source files: no emotion 85,572; happiness 12,885; surprise 1,823; sadness 1,150; anger 1,022; disgust 353; fear 174. To prevent exact dialogue duplicates from crossing the source splits, 93 validation and 129 test dialogues matching another split exactly were excluded from the prepared benchmark. Counts after this de-duplication are recorded in `public_dataset_experiments/data/processed/dailydialog_public_v1/manifest.json`.

After the leakage removal, mapped examples are neutral 84,168; happiness 12,664; sadness 1,134; anger 988 (98,954 turns total). Excluded labels are surprise 1,803; disgust 347; fear 173 (2,323 turns). The difference from raw source counts is the set of turns in exact cross-split duplicate dialogues removed before mapping.

## Interpretation limits

DailyDialog contains ordinary daily-life text conversations, not ElderMeds users, older-adult participants, clinical interviews, or adaptive five-turn support sessions. The `no emotion` class is the dataset's annotation, not a clinical state. Performance estimates apply only to this four-label public benchmark and its source splits. They cannot establish the model's accuracy on older adults, ElderMeds users, cognitive impairment, care settings, or the seven-label production taxonomy.
