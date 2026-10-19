# Stage 7B Review Status

Current queue snapshot: 500 pending items, 0 reviewed, 0 ambiguous, 0 needing context, 0 rejected, 0 adjudicated; no reviewer identities or inter-annotator agreement observations exist. See `stage7b_stage8_experiments/review_status.json` for breakdowns.

The queue contains 332 exact-duplicate candidates, 2 near-duplicate candidate rows, 332 label-conflict rows, and 332 very-short/context-dependent candidates (reasons overlap). 164 queue rows' conflicting family labels match one of the historically common confusion pairs used for triage. Source counts are 464 GoEmotions-prepared rows and 36 development rows. The queue has no loneliness examples (0) and cannot be treated as representative. Missing question/session context is not recorded in source files; all 500 are marked `not_available_in_source_queue`, with short-text flags kept separate.

No label corrections have been applied. No agreement statistic is available. The queue is a triage tranche, not completed annotation. Public-derived source licensing/mapping still requires verification before training use.
