# Public Conversational Emotion Experiment — DailyDialog

## Status

The experiment is isolated from ElderMeds production and Stage 9 participant data. Dataset preparation and three frozen MiniLM context candidates are complete. The result is a reduced-compute public benchmark, not an ElderMeds model. Versioned metrics, split manifest, checksummed artifact copies and SVG figures are in `public_dataset_experiments/results/dailydialog_public_v1/`. The full local run, including prepared rows, remains under the ignored `runs/` directory.

## Dataset and license

The selected public source is ConvLab's DailyDialog dataset card and archive: [source and dataset card](https://huggingface.co/datasets/ConvLab/dailydialog), [archive](https://huggingface.co/datasets/ConvLab/dailydialog/resolve/main/data.zip). The card identifies CC BY-NC-SA 4.0, describes 13,118 dialogues / 102,979 utterances, and lists official train, validation, and test splits. Citation: Li, Yanran, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang Cao, and Shuzi Niu (2017), “DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset,” IJCNLP 2017 ([paper](https://aclanthology.org/I17-1099/)). Archive SHA-256: `b1f38f3bf5431f9384107011a943f1a3446027721292e3a248a894b7766fa9d3`.

EmpatheticDialogues was also inspected: its archived repository and dataset card describe about 25,000 conversations, situation/emotion context labels, and CC BY-NC 4.0 ([repository](https://github.com/facebookresearch/EmpatheticDialogues), [dataset card](https://huggingface.co/datasets/facebook/empathetic_dialogues), Rashkin et al. 2019, [ACL paper](https://aclanthology.org/P19-1534/)). DailyDialog was selected because it provides utterance-level emotion annotations and official conversation splits for this turn-classification task; EmpatheticDialogues' situation label was not repurposed as a turn-level label.

## Label mapping and eligibility

The model reports a reduced four-class public label set: `neutral`, `happiness`, `sadness`, `anger`. Native `no emotion` maps to the public benchmark's `neutral` interpretation only; happiness, sadness and anger map directly. Fear, disgust and surprise are excluded. No mapping is made to anxiety, loneliness or cognitive fog. The exact table and counts are in `PUBLIC_DATASET_LABEL_MAPPING.md`.

The public set is not ElderMeds participant data. It has no older-adult/clinical sample provenance, and its annotations do not establish ElderMeds taxonomy validity. Models and metrics are labelled `public_research`, tagged `public_nonclinical_not_eldermeds_data`, and stored outside the production registry.

## Leakage control and partitions

Preprocessing found exact full-dialogue text duplicates crossing official splits. The training copies were retained; 93 matching validation and 129 matching test dialogues were removed before modeling. The prepared benchmark has train 11,118 dialogues, validation 907, test 871, with 98,954 mapped utterance examples total. No dialogue appears in two source splits after this removal.

To keep execution within available CPU resources, the experiment uses a fixed SHA-256 ordering independent of labels to select 2,000 official-train, 500 official-validation, and 500 official-test dialogue groups. Ten percent of selected train groups are held out for probability calibration. Current counts are fit 1,800 dialogues / 13,735 examples; calibration 200 / 1,469; validation 500 / 3,974; test 500 / 3,800. Sample and group hashes are in each run's `split_manifest.json`. All partitions have support for the four mapped labels. This is a deterministic subset, not a full-split benchmark.

## Models and procedure

Three inputs use the cached `sentence-transformers/all-MiniLM-L6-v2` checkpoint in local-only CPU mode, L2-normalized embeddings, and calibrated logistic regression:

1. **A — answer only:** current utterance and speaker role.
2. **B — one previous:** current utterance plus the immediately preceding utterance/speaker.
3. **C — two previous:** current utterance plus up to two preceding utterances/speakers.

Inputs preserve chronological order, include only turns with lower original utterance indices from the same dialogue, and never include labels or future turns. A shared tokenizer budget uses the checkpoint's verified 512-position capacity. Oldest context is removed first if needed; an overlong current utterance is rejected rather than silently truncated.

The original validation split selects among logistic-regression `C={0.1,1.0}` and class-weight `{None, balanced}` settings by Macro F1, with accuracy as tie-breaker. Final model parameters are fitted on the fit partition. Sigmoid calibration uses only the separate calibration partition. The selected and calibrated candidates are frozen before a single evaluation on the deterministic official-test subset. The test set is not used for hyperparameter choice.

## Metrics and outputs

The run reports accuracy, macro precision/recall/F1, per-class precision/recall/F1/support, confusion matrices, log loss, Brier score, 10-bin ECE, measured single-record CPU latency, error changes from A to C, and accuracy/Macro F1 by available prior-turn depth. The fit/validation/calibration/test sample counts are 13,735 / 3,974 / 1,469 / 3,800, from 1,800 / 500 / 200 / 500 disjoint dialogues.

| Representation | Accuracy | Macro F1 | Macro precision | Macro recall | Log loss | Brier | ECE (10 bins) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A — current utterance | 0.834 | 0.275 | 0.547 | 0.275 | 0.446 | 0.246 | 0.028 |
| B — + one previous turn | 0.831 | 0.255 | 0.516 | 0.265 | 0.473 | 0.260 | 0.029 |
| C — + up to two previous turns | 0.828 | 0.261 | 0.480 | 0.269 | 0.468 | 0.260 | 0.031 |

Test supports are neutral 3,144, happiness 531, sadness 62 and anger 63. In A/B/C order, class F1 is: neutral 0.908 / 0.907 / 0.906; happiness 0.103 / 0.055 / 0.030; sadness 0.088 / 0.060 / 0.110; anger 0 / 0 / 0. All candidates have 0 anger recall on this subset. Overall accuracy is dominated by the majority neutral class and should not be read without these per-class results.

Single-record CPU embedding-plus-classifier p50/p95 milliseconds were A 23.7/29.1, B 72.7/573.1, C 83.3/235.3 (40 records, one local run; timing is noisy and hardware-specific). Two-turn context corrected 14 answer-only errors and introduced 35 new errors on the same 3,800 turns. Confusion-matrix SVGs and the comparison SVG are in `public_dataset_experiments/results/dailydialog_public_v1/figures/`.

Post-run quality review caught sklearn's class-column-order assumption in the original log-loss calculation. Log loss and all test metrics were recalculated using the frozen, checksum-verified artifacts on the exact same test dialogue groups; no fitting, calibration or hyperparameter selection was repeated. A missing citation metadata field was added to the three artifacts and their sidecar checksums refreshed before recalculation. The saved report records this post-hoc recalculation.

## Limitations and safe use

This is a public, nonclinical, reduced-taxonomy experiment. It is not a full seven-class model, an ElderMeds conversational evaluation, evidence about older adults, or evidence of clinical effectiveness. The source is English and strongly imbalanced (neutral/no-emotion dominates). The three candidates perform poorly on minority classes and fail to identify anger in the held-out subset. A deterministic subset of official splits was used because full CPU embedding of all three formats was not feasible within the available execution budget. The dataset license is noncommercial/share-alike; local experimental artifacts are not promoted or distributed as production models.

MiniLM v4 remains the production default, v3 remains the explicit rollback, v5 remains experimental, and contextual v6 remains disabled. Stage 9's consent and independent-review requirements are unchanged.
