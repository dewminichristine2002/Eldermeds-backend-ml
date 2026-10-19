# Public Dataset Sources and Attribution

## DailyDialog (selected)

- **Dataset repository/card:** [ConvLab/DailyDialog on Hugging Face](https://huggingface.co/datasets/ConvLab/dailydialog)
- **Downloaded archive:** `https://huggingface.co/datasets/ConvLab/dailydialog/resolve/main/data.zip`
- **License shown on the dataset card:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). The experiment is noncommercial research. Attribution and license terms must accompany any distribution of adapted material; the trained models are kept in local experimental storage and are not deployed or distributed by this work.
- **Citation:** Li, Yanran, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang Cao, and Shuzi Niu. 2017. “DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset.” Proceedings of IJCNLP 2017. [Paper](https://aclanthology.org/I17-1099/).
- **Dataset-card count:** 13,118 dialogues and 102,979 utterances: train 11,118 / 87,170; validation 1,000 / 8,069; test 1,000 / 7,740.
- **Annotation:** Each utterance has a dialogue-act annotation and one of seven emotion annotations: no emotion, happiness, sadness, anger, surprise, disgust, or fear. Native annotations are preserved in prepared public records.
- **Local archive SHA-256:** `b1f38f3bf5431f9384107011a943f1a3446027721292e3a248a894b7766fa9d3`.
- **Limitations:** English-only, ordinary daily-life conversations, no older-adult demographic or clinical validation, strong class imbalance, and a taxonomy different from ElderMeds. Exact duplicate dialogue text was detected across supplied splits; 93 validation and 129 test copies were removed to prevent leakage.

The repository card describes its transformation of the original IJCNLP DailyDialog data and identifies its license and split statistics. The paper supplies the dataset citation and design context. [Dataset card](https://huggingface.co/datasets/ConvLab/dailydialog), [original paper](https://aclanthology.org/I17-1099/), [license text](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## EmpatheticDialogues (inspected, not selected)

- **Repository:** [facebookresearch/EmpatheticDialogues](https://github.com/facebookresearch/EmpatheticDialogues) (archived/read-only).
- **Dataset card:** [facebook/empathetic_dialogues on Hugging Face](https://huggingface.co/datasets/facebook/empathetic_dialogues).
- **License shown by the repository/card:** CC BY-NC 4.0.
- **Citation:** Rashkin, Hannah, Eric Michael Smith, Margaret Li, and Y-Lan Boureau. 2019. “Towards Empathetic Open-domain Conversation Models: A New Benchmark and Dataset.” ACL 2019. [Paper](https://aclanthology.org/P19-1534/).
- **Count and structure:** About 25,000 open-domain conversations with an emotion situation/context annotation and conversation-level grouping. The Hugging Face card lists `conv_id`, utterance index, context, prompt, speaker, utterance and related fields.
- **Selection decision:** DailyDialog was preferred because its source provides a manually assigned emotion field at each utterance and official train/validation/test dialogue splits, allowing a direct reduced-label conversational turn-classification experiment. EmpatheticDialogues' situation/emotion field is not treated as a turn-level emotion label here.

## Use boundaries

Neither public source is ElderMeds participant data. The public experiment is not a proxy consent dataset, not evidence of model effectiveness for older adults, and not eligible input for the Stage 9 consented-participant contextual trainer. Review the source licenses and institutional requirements before reuse or distribution; this report records source terms rather than giving legal advice.
