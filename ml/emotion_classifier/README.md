# Component 4 Emotion Classifier

Standalone ML training module for the ElderMeds Component 4 emotion classification workflow.

This module does not connect to the Node.js backend API yet. The current rule-based emotion analyzer remains unchanged.

## Folder Contents

```text
backend/ml/emotion_classifier/
  data/
  prepare_dataset.py
  train_emotion_model.py
  predict_emotion.py
  requirements.txt
  README.md
  label_mapping.json
```

Generated training artifacts are saved in this folder by default:

```text
emotion_classifier.pkl
tfidf_vectorizer.pkl
model_metrics.json
```

## Supported Project Labels

- `happiness`
- `sadness`
- `loneliness`
- `anxiety`
- `anger`
- `cognitive_fog`
- `neutral`

## Dataset Format

Use a CSV file with exactly these required columns:

```csv
text,label
"I felt peaceful after seeing my old photos",happiness
"I could not remember what I wanted to say",cognitive_fog
```

Extra columns are ignored.

## Using Public Emotion Datasets

For PP2/final research training, use a real public emotion dataset instead of a small hand-written demo dataset. Suitable sources include:

- Kaggle Emotion Dataset
- GoEmotions by Google Research
- Another public dataset with clear text and emotion labels

Before training, convert the selected dataset into CSV format with these columns:

```csv
text,label
```

The training script expects these exact column names. Extra columns can remain in the CSV, but `text` and `label` are required.

Public datasets usually use their own raw labels. Those raw labels are mapped into the ElderMeds Component 4 labels through `label_mapping.json`. Rows with labels that are not present in `label_mapping.json` are removed before training.

Example command:

```bash
python train_emotion_model.py data/your_dataset.csv
```

For GoEmotions, labels such as `joy`, `sadness`, `fear`, `nervousness`, `anger`, `annoyance`, `confusion`, `realization`, and `neutral` can be mapped into the project labels. For Kaggle datasets, labels such as `joy`, `sadness`, `anger`, `fear`, and `neutral` can also be mapped depending on the dataset version.

Loneliness should be handled carefully. If the dataset does not contain a direct loneliness-related label, do not force unrelated labels into `loneliness`. Use one of these approaches instead:

- Add a custom supplementary loneliness dataset.
- Keep the current keyword-supported fallback for loneliness phrases such as "alone", "lonely", "isolated", or "miss my family".
- Map to `loneliness` only when the public dataset has labels such as `loneliness`, `isolation`, `isolated`, or `alone`.

For final research reporting, train with a larger public dataset and document how raw dataset labels were mapped into the Component 4 classes.

## Preparing a Public Dataset

Public datasets must be legally downloaded by the student from their original public source. This module does not download datasets automatically.

Use `prepare_dataset.py` to convert a raw public dataset into the required training format:

```csv
text,label
```

Example:

```bash
python prepare_dataset.py --input data/raw_dataset.csv --output data/prepared_emotion_dataset.csv --text-column text --label-column label
```

The preparation script:

1. Loads the input CSV.
2. Selects the supplied text and label columns.
3. Renames them to `text,label`.
4. Removes rows with missing text or label.
5. Lightly cleans text by lowercasing and removing extra spaces.
6. Saves the prepared CSV with only `text,label`.
7. Prints total rows before cleaning, total rows after cleaning, label distribution, and output path.

After preparing the dataset, train with:

```bash
python train_emotion_model.py data/prepared_emotion_dataset.csv
```

Raw labels are mapped into the ElderMeds project labels using `label_mapping.json` during training. Rows with unmapped labels are removed by `train_emotion_model.py`.

## Preparing GoEmotions

GoEmotions uses a multi-label format: each row has a `text` column and separate 0/1 emotion columns such as `joy`, `sadness`, `fear`, `anger`, `confusion`, and `neutral`.

Use `prepare_goemotions_dataset.py` to combine the three GoEmotions CSV files and convert them into the standard training format:

```bash
python prepare_goemotions_dataset.py --raw-dir data/raw --output data/goemotions_prepared_dataset.csv
```

If the raw files are stored outside this module, pass that folder as `--raw-dir`:

```bash
python prepare_goemotions_dataset.py --raw-dir ..\..\..\ml-service\data\raw --output data/goemotions_prepared_dataset.csv
```

The script maps GoEmotions labels into these Component 4 classes:

- `happiness`
- `sadness`
- `anxiety`
- `anger`
- `cognitive_fog`
- `neutral`

When one GoEmotions row has multiple mapped labels, the script duplicates the text once per mapped project label and then removes duplicate `text,label` pairs. This keeps multi-label evidence without forcing the row into only one class.

Train with the prepared GoEmotions dataset:

```bash
python train_emotion_model.py data/goemotions_prepared_dataset.csv
```

GoEmotions improves ML coverage because it includes more public emotion labels than the smaller `Emotion_classify_Data.csv` dataset. It supports training across happiness, sadness, anxiety, anger, cognitive fog, and neutral.

Loneliness is not forced from unrelated GoEmotions labels. Keep loneliness handled by the existing rule-based fallback unless a direct loneliness dataset is added with labels such as `loneliness`, `isolation`, `isolated`, or `alone`.

## Label Mapping

Raw dataset labels are mapped through `label_mapping.json`. Labels that do not map to one of the supported project labels are removed before training.

Update `label_mapping.json` when a dataset uses different names, for example:

```json
{
  "joy": "happiness",
  "fear": "anxiety",
  "confusion": "cognitive_fog"
}
```

## Setup

From this folder:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate with:

```bash
source .venv/bin/activate
```

## Train

Run training with a CSV dataset path:

```bash
python train_emotion_model.py data/your_dataset.csv
```

The training script:

1. Loads the CSV from the command-line argument.
2. Cleans text by lowercasing, removing punctuation, and collapsing extra spaces.
3. Maps raw labels through `label_mapping.json`.
4. Removes unmapped labels.
5. Splits the data into 80% train and 20% test.
6. Uses TF-IDF with `ngram_range=(1, 2)` and `max_features=5000`.
7. Trains Logistic Regression, Linear SVM, and Multinomial Naive Bayes.
8. Evaluates accuracy, macro precision, macro recall, macro F1, and classification reports.
9. Selects the model with the highest macro F1-score.
10. Saves the selected model, vectorizer, and metrics.

To save artifacts elsewhere:

```bash
python train_emotion_model.py path\to\dataset.csv --output-dir path\to\output
```

To use a different label mapping file:

```bash
python train_emotion_model.py path\to\dataset.csv --label-mapping path\to\label_mapping.json
```

## Predict

After training, classify one text input:

```bash
python predict_emotion.py "I miss my family and the house feels quiet"
```

This prints the predicted project emotion label. If the selected model supports probabilities, such as Logistic Regression or Naive Bayes, it also prints class probabilities.

To use model artifacts from another directory:

```bash
python predict_emotion.py "I feel calm today" --model-path path\to\emotion_classifier.pkl --vectorizer-path path\to\tfidf_vectorizer.pkl
```

## FastAPI ML Service

Run the standalone emotion classifier API from this folder:

```bash
uvicorn api_service:app --host 0.0.0.0 --port 8001
```

Health endpoint:

```http
GET /health
```

Prediction endpoint:

```http
POST /predict-emotion
Content-Type: application/json

{
  "text": "I felt worried and could not relax today"
}
```

Example prediction texts for testing:

```text
I felt worried and could not relax today
I feel sad and tired today
I enjoyed remembering my old songs
I feel angry because things went wrong
I cannot remember what I was doing
```

The ML model is trained using GoEmotions mapped into the ElderMeds Component 4 project classes. The final system should use the ML model together with the existing rule-based fallback for safety, especially for elderly-specific loneliness detection and low-confidence cases.

## Notes

- `LinearSVC` does not provide probability scores by default, so prediction output may contain only the label.
- If any class has too few examples for stratified splitting, the script falls back to a regular 80/20 split.
- This module is intentionally separate from the existing backend until API integration is explicitly requested.
