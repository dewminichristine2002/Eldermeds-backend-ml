# Stage 8 Context Model Evaluation

Status: **WAITING_FOR_VALID_LABELED_DATA**. The data inventory found no participant/session-keyed, human-labelled multi-turn emotion records. The 70-row project evaluation CSV is single-turn and is not conversational training data. No contextual model artifact or contextual performance result was created.

`train_contextual.py` accepts a versioned CSV with JSON `previous_turns`, validates human annotation provenance, groups records by participant plus exact/near-duplicate text families, makes fit/calibration/validation group partitions, and compares a small predefined set of logistic regression candidates for question+answer and two-turn inputs. It writes new experimental artifacts only when valid data contains all seven labels in every partition. Command when data is approved: `python train_contextual.py --data data/context_emotion_training.csv`. With the current checkout it reports `WAITING_FOR_VALID_LABELED_DATA` and writes no model.

No synthetic scores are reported as research results. The historical v5 answer-only result (accuracy 0.6266, macro-F1 0.6599 on 316 repeatedly used model-selection rows) is not an independent comparator. No fresh independent benchmark exists.
