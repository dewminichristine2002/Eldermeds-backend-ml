# Emotion Classifier Data Folder

Place converted public emotion datasets here for local training.

Expected CSV format:

```csv
text,label
```

Example training command from `backend/ml/emotion_classifier/`:

```bash
python train_emotion_model.py data/your_dataset.csv
```

Do not commit large raw public datasets unless the project explicitly allows it. Keep the dataset source and label mapping notes in research documentation.
