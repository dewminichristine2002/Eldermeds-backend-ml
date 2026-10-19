# Stage 9 Context Model Comparison

## Results status

No Stage 9 candidates were trained, and no independent contextual benchmark exists. Therefore, Stage 9 accuracy, Macro F1, per-class F1, calibration, clarification rate, and latency comparison are **not available**. No performance figure is appropriate yet.

| Candidate | Stage 9 status | Input contract | Independent result |
|---|---|---|---|
| Production `minilm_logistic_regression_v4` | Active unchanged | Answer only | Not re-evaluated on a fresh independent conversational benchmark |
| v3 rollback `tfidf_linear_svm_calibrated_v3` | Explicit rollback unchanged | Answer only | Not evaluated in Stage 9 |
| v5 `minilm_logistic_regression_v5_answer_only` | Experimental, version-gated | Answer only | Historical validation only; not Stage 9 evidence |
| Stage 9 answer-only candidate | Not trained | Current answer | Pending eligible data |
| Stage 9 question-answer candidate | Not trained | Current question + answer | Pending eligible data |
| Stage 9 multi-turn candidate | Not trained | Up to two previous Q/A turns + current Q/A | Pending eligible data |
| Contextual v6 registry entry | Unavailable | Multi-turn | No artifact; remains fail-closed |

Historical figures (v4 locked test Accuracy 0.5459/Macro F1 0.5649; v5 locked test 0.5538/0.5745; v5 validation Macro F1 0.6599) belong to prior work and are not Stage 9 comparisons. The validation value was model selection, not independent evaluation. No claim of contextual improvement or elderly-user clinical validation is made.

The comparison pipeline is `evaluate_context_models.py`. It pairs v4 answer-only inference with explicitly supplied checksummed Stage 9 artifacts on a frozen, double-reviewed `independent_test` set, checks participant hashes against the training split manifest, and produces the prespecified metrics only when each label has sufficient support.
