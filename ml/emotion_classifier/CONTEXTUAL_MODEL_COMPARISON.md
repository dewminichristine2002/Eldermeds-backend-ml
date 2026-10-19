# Contextual Model Comparison

## Evaluation scope

This comparison covers the reduced four-label **DailyDialog public research benchmark** only. It is not comparable to ElderMeds v4/v5 scores from other datasets or label spaces and does not measure ElderMeds seven-class performance. Exact duplicate dialogue text was removed across official splits; fixed subsets of 500 validation and 500 test dialogue groups were selected by a label-independent hash. Calibration used disjoint groups from the official training split.

Versioned outputs are in `public_dataset_experiments/results/dailydialog_public_v1/` (run ID `dailydialog_20261009T061752Z`). The local full run contains 1,800 fit dialogues (13,735 turns), 200 calibration dialogues (1,469 turns), 500 validation dialogues (3,974 turns), and 500 test dialogues (3,800 turns). All four reduced classes are present in each group partition.

## Aggregate test results

| Representation | Test dialogues / turns | Accuracy | Macro F1 | Macro precision | Macro recall | Log loss | Latency p50 / p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| A — current utterance | 500 / 3,800 | 0.834 | 0.275 | 0.547 | 0.275 | 0.446 | 23.7 / 29.1 |
| B — current + previous utterance | 500 / 3,800 | 0.831 | 0.255 | 0.516 | 0.265 | 0.473 | 72.7 / 573.1 |
| C — current + up to two previous utterances | 500 / 3,800 | 0.828 | 0.261 | 0.480 | 0.269 | 0.468 | 83.3 / 235.3 |

Latency is single-record CPU embedding plus calibrated classifier for 40 examples on one local run; it is hardware- and load-sensitive.

| Reduced public label | A F1 | B F1 | C F1 | Test support |
|---|---:|---:|---:|---:|
| neutral | 0.908 | 0.907 | 0.906 | 3,144 |
| happiness | 0.103 | 0.055 | 0.030 | 531 |
| sadness | 0.088 | 0.060 | 0.110 | 62 |
| anger | 0.000 | 0.000 | 0.000 | 63 |

## Interpretation

All candidates have zero anger recall. Accuracy is dominated by neutral examples, so the results show poor minority-class recognition. Answer-only has the best aggregate accuracy and Macro F1 in this bounded experiment. On the same 3,800 turns, two-turn context corrected 14 answer-only errors and introduced 35 new errors; 3,750 predictions agreed. This does not establish that context harms ElderMeds users or that any model is suitable for ElderMeds.

The original log-loss implementation passed non-lexicographic labels to sklearn. Final test probabilities were recalculated with the declared class-column order using the frozen, checksum-verified artifacts and the exact same test dialogue groups. No candidate was retrained or retuned. Artifact citation metadata was repaired and checksums refreshed before the final inference pass.

Per-model confusion matrices, calibration measures, context-depth error analysis, token-budget audit and hyperparameter selection results are in `public_dataset_experiments/results/dailydialog_public_v1/experiment_results.json` and `figures/`.

No result supports performance claims for older adults, ElderMeds sessions, loneliness, anxiety, cognitive fog, the full production taxonomy or clinical outcomes. Contextual v6 remains disabled and production MiniLM v4 remains unchanged.
