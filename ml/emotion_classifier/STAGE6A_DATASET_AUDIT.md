# Stage 6A Dataset Audit

Audit command: `python audit_dataset.py --data-dir data --output results/stage6a_dataset_audit.json` from this directory. The output records schemas, raw content hashes, class counts, duplicate counts, empty/short/long text, negation markers, source counts, and exact normalized overlaps. No source CSV was modified.

## Dataset findings

| Dataset | Rows | Key findings | Stage 6A use |
|---|---:|---|---|
| `development_dataset_v3.csv` | 1,582 | Seven supported labels; 1,582 normalized unique texts; 0 empty, duplicate, or conflicting text families; 34 at most two tokens; source counts 1,427 GoEmotions, 37 domain relabel, 118 project-authored | Used as the only fitting/model-selection source |
| `training_dataset_v2.csv` | 1,904 | Seven labels; 1,904 unique texts; 44 very short; its 381 locked records are contained in this file | Not merged; source history shows it contains the frozen split |
| `locked_test_v2.csv` | 381 | Seven labels; loneliness support 11; no exact overlap with development | One-time post-selection comparison only; previously exposed |
| `goemotions_prepared_dataset.csv` | 104,395 | Six classes, no loneliness; 48,540 duplicate rows; 33,595 normalized text families have conflicting/multiple labels | Excluded; not safe to merge without source-level annotation review |
| `prepared_emotion_dataset.csv` | 5,937 | Labels are anger/fear/joy, not the seven-class map; three conflicting duplicate families | Excluded; unsupported labels were not guessed or remapped |
| `project_authored_hardening.csv` | 59 | Four classes only; already represented through the development preparation history | Not separately merged |
| `project_evaluation_set.csv` | 70 | Seven classes with `expected_emotion,text`; single-turn benchmark only | Excluded from fitting, calibration, and selection |

The machine-readable counts, schemas, SHA-256 values, and exact cross-file overlap counts are in `results/stage6a_dataset_audit.json`. In particular, exact normalized overlap is 1,523 rows between the development and full training CSVs because the latter includes the held-out records; the development source itself has zero exact overlap with the locked file. GoEmotions also overlaps the locked file, another reason it was not mixed into the experiment.

The eligible development class-distribution chart is `results/stage6a_development_class_distribution.png`, generated from the audited counts above.

## Leakage controls and limitations

The experiment uses only the development CSV. It normalizes whitespace while preserving punctuation, case, and negation; validates all labels; removes conflicting normalized text families; and forms text-family groups using exact matching plus character TF-IDF cosine similarity (3–5 character n-grams, threshold 0.90). Deterministic `StratifiedGroupKFold` partitions are fit 949 / validation 316 / calibration 317. No group crosses these partitions. The independent text-family scan found no exact or >=0.90 near-duplicate family shared between development and the locked test; it found two near-duplicate pairs within the combined scan, both internal to one side.

The files have no participant, user, session, or conversation identifiers. The split controls text-family leakage only; participant-independent generalization cannot be claimed. The validation partition was used to select among candidates, so its score is a model-selection result, not an unbiased external estimate. The locked test is previously exposed and offers no new blind estimate.

No augmented examples were generated. No private responses were copied into reports; experiment prediction rows use SHA-256 record identifiers.
