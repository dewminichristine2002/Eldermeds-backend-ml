# ElderMeds Component 4 — Final Research Results Summary

## Research status

Engineering through Stage 10A is retained. This task adds an isolated public DailyDialog experiment because no consented, human-annotated elderly conversation benchmark is available. Public data do not satisfy or bypass Stage 9 participant consent, annotation, or independent-evaluation gates.

## Results by evidence type

1. **Existing ElderMeds classifier work:** Historical Stage 5–9 results remain in their reports and original datasets. They are not rerun or combined with the public benchmark here.
2. **Public conversational benchmark:** Three contextual representations were trained and evaluated on a reduced four-label, dialogue-disjoint DailyDialog subset. Test accuracy / Macro F1: answer-only 0.834 / 0.275; one previous turn 0.831 / 0.255; up to two previous turns 0.828 / 0.261. All had zero anger recall and weak minority-label results. Detailed scores are in `CONTEXTUAL_MODEL_COMPARISON.md`.
3. **Validation and leakage:** Exact duplicated dialogues crossing source splits were removed from held-out splits. The run used 1,800 fit, 200 calibration, 500 validation and 500 test dialogue groups. Context was limited to previous turns from the same dialogue with a lower turn index. No future turns entered the input. Hyperparameter selection used validation only; calibration used separate train groups; final metrics use test groups.
4. **Elderly-specific contextual performance:** Unvalidated. DailyDialog does not identify older-adult/clinical participants and is not an ElderMeds evaluation.
5. **Future work:** Requires authorized, consented, de-identified ElderMeds multi-turn collection; independent human annotation and adjudication; participant-disjoint development and validation; and a separately frozen benchmark before any contextual ElderMeds model can be considered.

## Tests and artifact checks

The public candidate artifacts were checksum-verified and passed their local inference smoke checks. The experiment's final test metrics were recomputed with corrected probability-column-safe log loss from the frozen artifacts on the same test groups; no model fitting or retuning occurred during recalculation. The full emotion-classifier Python suite passed **82/82 tests**. Backend `npm run test:all` passed **273/273 tests**: 241 unit, 19 integration, 11 NFR and 2 E2E.

## Non-promotion and production safety

MiniLM v4 remains the production default; v3 rollback is preserved; v5 remains experimental; contextual v6 is not activated. Public candidates remain under `public_dataset_experiments/runs/` and are never registered in the production registry. No private participant conversations or Supabase data were accessed.

## References

- Public experiment methods and results: `PUBLIC_DATASET_EXPERIMENT_REPORT.md`.
- Label mapping: `PUBLIC_DATASET_LABEL_MAPPING.md`.
- Context model comparison: `CONTEXTUAL_MODEL_COMPARISON.md`.
- Dataset sources, license and citation: `PUBLIC_DATASET_SOURCES.md`.

## Thesis-ready methods and results text

**Methods.** A public conversational emotion-recognition experiment used the CC BY-NC-SA 4.0 DailyDialog dataset. Exact full-dialogue duplicates across official splits were removed to prevent overlap. Only happiness, sadness, anger and the dataset's “no emotion” label (treated as a nonclinical neutral class) were retained. Fear, disgust and surprise were excluded; loneliness, anxiety and cognitive fog were not represented. Three input representations (current utterance alone, plus one previous utterance, plus up to two previous utterances) used a local `sentence-transformers/all-MiniLM-L6-v2` encoder and logistic-regression candidates. The deterministic subsets contained 1,800 fit, 200 calibration, 500 validation and 500 test dialogue groups. Validation selected hyperparameters, calibration used its separate train group set, and final evaluation used held-out official-test dialogue groups. All dialogue groups were disjoint.

**Results and interpretation.** Test accuracy / Macro F1 were 0.834 / 0.275 for current utterance only, 0.831 / 0.255 with one previous turn, and 0.828 / 0.261 with up to two previous turns. All candidates had zero recall for anger and weak minority-class F1. Context did not improve aggregate scores on this public subset. These results concern only a reduced four-class nonclinical English benchmark. They do not measure the ElderMeds seven-class task, older-adult performance, or clinical outcomes. Production v4 was not changed and public candidates were not promoted.
