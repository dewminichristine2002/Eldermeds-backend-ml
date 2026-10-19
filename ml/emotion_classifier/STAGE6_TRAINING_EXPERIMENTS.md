# Stage 6 Training Experiments

## Experiment status

No MiniLM v5 candidate was trained or promoted in this run. The repository has
single-turn labeled text, but no valid question-answer or multi-turn labels.
Training a contextual model by fabricating dialogue would introduce unsupported
evidence and invalidate the requested comparison.

## Existing verified baseline

The existing v4 artifact is a frozen MiniLM embedding plus balanced Logistic
Regression model. Its recorded evaluation is preserved in
`STAGE4_V4_DEPLOYMENT_REVIEW.md` and `domain_hierarchical_experiment/`.

| Dataset provenance | Rows | Accuracy | Macro F1 | Status |
|---|---:|---:|---:|---|
| ElderMeds domain test | 140 | 0.8143 | 0.8166 | Previously evaluated v4 |
| Broader frozen test | 381 | 0.5459 | 0.5649 | Previously evaluated v4 |

These are historical v4 results, not Stage 6 or v5 measurements.

## Candidate matrix

| Candidate | Input | Result |
|---|---|---|
| A | Answer-only MiniLM | Existing v4 baseline only |
| B | Question + answer MiniLM | Not trained: no paired question/answer labels |
| C | Two-turn contextual MiniLM | Not trained: no session/turn labels |
| D | MiniLM + calibrated linear SVM | Not run in Stage 6; would require a fresh, valid calibration protocol |
| E | Class-weighted Logistic Regression | Existing v4 classifier configuration |

No hyperparameter search, calibration fit, or test-set selection was performed in
Stage 6. This is intentional: the available context-free datasets cannot support
the proposed contextual claims.

## Reproducible commands

Audit available data:

```powershell
cd backend/ml/emotion_classifier
.\.venv\Scripts\python.exe audit_dataset.py --data-dir data --output results\stage6_dataset_audit.json
```

Run the existing reproducible Stage 4 training/evaluation protocol only when a
research rerun is authorized and its frozen-test safeguards are understood:

```powershell
cd backend/ml/emotion_classifier
.\.venv\Scripts\python.exe harden_emotion_model.py
```

The Stage 4 final evaluation is marked rerun-prohibited in its completion record;
it was not rerun as a Stage 6 promotion experiment.

## Continuation update — 2026-10-08

The rerun-prohibited `harden_emotion_model.py` path above was not invoked. A
separate `train_v5_answer_only.py` experiment used the development partition,
grouped text families, and kept calibration separate. The v5 artifact and
per-run evidence are under `v5_experiments/`; the authoritative run is
`20261008T175725Z`. v5 is experimental and not active. See
`STAGE6A_MODEL_TRAINING_REPORT.md` for results and
`STAGE6_MODEL_PROMOTION_REPORT.md` for the decision.
