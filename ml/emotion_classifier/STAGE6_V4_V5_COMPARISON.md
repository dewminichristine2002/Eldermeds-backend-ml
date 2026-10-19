# Stage 6 v4/v5 Comparison

## Decision

**No v5 comparison is available.** No validated MiniLM v5 artifact, contextual
training dataset, calibration artifact, or v5 evaluation output exists in the
repository. It would be misleading to create a v4/v5 table with missing v5
values or to call a regenerated split an untouched external test.

## Verified v4 results

| Metric | Broader frozen test (381) | ElderMeds domain test (140) |
|---|---:|---:|
| Accuracy | 0.5459 | 0.8143 |
| Macro precision | 0.5552 | 0.8278 |
| Macro recall | 0.5889 | 0.8143 |
| Macro F1 | 0.5649 | 0.8166 |
| Weighted F1 | 0.5409 | 0.8166 |

The broader and domain tests were previously exposed in Stage 4. They are not
new blind Stage 6 measurements.

## Per-class v4 results

### Broader frozen test

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| happiness | 0.6324 | 0.6935 | 0.6615 | 62 |
| sadness | 0.6182 | 0.5313 | 0.5714 | 64 |
| loneliness | 0.6250 | 0.9091 | 0.7407 | 11 |
| anxiety | 0.7308 | 0.6441 | 0.6847 | 59 |
| anger | 0.5309 | 0.6935 | 0.6014 | 62 |
| cognitive_fog | 0.4375 | 0.3443 | 0.3853 | 61 |
| neutral | 0.3115 | 0.3065 | 0.3089 | 62 |

The weakest broader-test classes are cognitive_fog and neutral. This supports
investigating distribution shift and annotation consistency before claiming that
context alone will solve the error boundary.

## Continuation update — 2026-10-08

The comparison above predates v5 training. The continuation trained an
experimental v5 answer-only artifact and compared it on the previously exposed
locked benchmark. The v5 validation set overlaps all 316 texts with the v4
training corpus, so the same-row v4 score is resubstitution and is excluded from
comparative conclusions. The locked benchmark showed a small mixed-class v5
change but is not fresh blind evidence. v4 stays active and v3 rollback stays
available. See `STAGE6A_V4_V5_COMPARISON.md` and
`STAGE6_MODEL_PROMOTION_REPORT.md` for corrected details.

## Promotion recommendation

Retain v4. A future v5 candidate must be evaluated on the same held-out records
plus a genuinely independent session-disjoint set, with per-class metrics,
calibration, latency, artifact checksums, and rollback verification.
