"""CPU-compatible development-only transformer comparison for Stage 4."""

import copy
import argparse
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

HERE = Path(__file__).resolve().parent
LABELS = ["happiness", "sadness", "loneliness", "anxiety", "anger", "cognitive_fog", "neutral"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
MODELS = {"distilroberta": "distilbert/distilroberta-base", "deberta_v3_small": "microsoft/deberta-v3-small"}
CONFIGS = [(1e-5, 3), (2e-5, 4), (3e-5, 5)]
MAX_LENGTH = 48
BATCH_SIZE = 32


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def metrics(labels, predictions):
    return {"accuracy": accuracy_score(labels, predictions),
            "macro_precision": precision_score(labels, predictions, average="macro", labels=range(7), zero_division=0),
            "macro_recall": recall_score(labels, predictions, average="macro", labels=range(7), zero_division=0),
            "macro_f1": f1_score(labels, predictions, average="macro", labels=range(7), zero_division=0),
            "weighted_f1": f1_score(labels, predictions, average="weighted", labels=range(7), zero_division=0)}


class TextDataset(Dataset):
    def __init__(self, frame, tokenizer):
        self.encodings = tokenizer(frame.text.tolist(), truncation=True, padding=True, max_length=MAX_LENGTH)
        self.labels = [LABEL_TO_ID[label] for label in frame.label]
    def __len__(self): return len(self.labels)
    def __getitem__(self, index):
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item


def make_splits(data):
    # Join obvious char-ngram near duplicates into deterministic families before
    # splitting so templated variants cannot cross development partitions.
    vectors = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(data.text)
    distances, indexes = NearestNeighbors(n_neighbors=4, metric="cosine").fit(vectors).kneighbors(vectors)
    parents = list(range(len(data)))
    def find(value):
        while parents[value] != value:
            parents[value] = parents[parents[value]]; value = parents[value]
        return value
    def union(left, right):
        left, right = find(left), find(right)
        if left != right: parents[max(left, right)] = min(left, right)
    for row in range(len(data)):
        for distance, neighbor in zip(distances[row, 1:], indexes[row, 1:]):
            if 1.0 - distance >= 0.88: union(row, int(neighbor))
    groups = np.array([find(index) for index in range(len(data))])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    assignments = np.full(len(data), -1)
    for fold, (_, indexes) in enumerate(splitter.split(data.text, data.label, groups)):
        assignments[indexes] = fold
    split = np.where(assignments == 0, "selection", np.where(assignments == 1, "calibration", "train"))
    data = data.assign(split=split)
    data.to_csv(HERE / "advanced_development_split.csv", index=False)
    return data[data.split == "train"], data[data.split == "selection"], data[data.split == "calibration"]


def evaluate(model, loader, device):
    model.eval(); logits_parts=[]; labels=[]
    with torch.no_grad():
        for batch in loader:
            truth = batch.pop("labels")
            output = model(**{k:v.to(device) for k,v in batch.items()})
            logits_parts.append(output.logits.cpu()); labels.extend(truth.numpy().tolist())
    logits = torch.cat(logits_parts).numpy()
    predictions = logits.argmax(axis=1)
    return metrics(labels, predictions), logits, np.asarray(labels)


def train_once(model_id, train, selection, learning_rate, epochs, seed):
    seed_all(seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=len(LABELS), id2label={i:l for i,l in enumerate(LABELS)}, label2id=LABEL_TO_ID)
    model = model.float()
    # CPU feasibility: fine-tune the classification head and upper two encoder
    # layers while preserving lower pretrained representations.
    base = model.base_model
    for parameter in base.parameters(): parameter.requires_grad = False
    encoder_layers = getattr(getattr(base, "encoder", None), "layer", None)
    if encoder_layers is None:
        encoder_layers = getattr(getattr(base, "transformer", None), "layer", None)
    if encoder_layers is not None:
        for layer in encoder_layers[-2:]:
            for parameter in layer.parameters(): parameter.requires_grad = True
    model.to(device)
    train_loader = DataLoader(TextDataset(train, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
    selection_loader = DataLoader(TextDataset(selection, tokenizer), batch_size=BATCH_SIZE)
    counts = train.label.value_counts(); weights = torch.tensor([len(train)/(len(LABELS)*counts.get(label,1)) for label in LABELS], dtype=torch.float32, device=device)
    loss_fn = CrossEntropyLoss(weight=weights)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(len(train_loader)*epochs*0.1), len(train_loader)*epochs)
    best_metrics = None; best_state = None; best_epoch = 0; stale = 0; started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            truth = batch.pop("labels").to(device); optimizer.zero_grad()
            logits = model(**{k:v.to(device) for k,v in batch.items()}).logits
            loss_fn(logits, truth).backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
        current, _, _ = evaluate(model, selection_loader, device)
        print(json.dumps({"model":model_id,"lr":learning_rate,"planned_epochs":epochs,"epoch":epoch,**current}), flush=True)
        if best_metrics is None or current["macro_f1"] > best_metrics["macro_f1"]:
            best_metrics = current; best_state = copy.deepcopy(model.state_dict()); best_epoch = epoch; stale = 0
        else:
            stale += 1
            if stale >= 1: break
    model.load_state_dict(best_state)
    duration = time.perf_counter() - started
    return model, tokenizer, {**best_metrics, "best_epoch":best_epoch, "seconds":duration}


def svm_reference(frame):
    pipeline = joblib.load(HERE / "baseline_v3" / "emotion_pipeline_v3.pkl")
    predictions = pipeline.predict(frame.text)
    truth = frame.label.map(LABEL_TO_ID).to_numpy(); predicted = np.array([LABEL_TO_ID[x] for x in predictions])
    return metrics(truth, predicted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=list(MODELS), default=None)
    args = parser.parse_args()
    data = pd.read_csv(HERE / "advanced_development.csv")
    train, selection, calibration = make_splits(data)
    split_frame = pd.concat([train, selection, calibration]).sort_index()
    split_distribution = split_frame.groupby(["split","label"]).size().rename("count").reset_index()
    split_distribution.to_csv(HERE / "advanced_split_distribution.csv", index=False)
    result_path = HERE / "transformer_hyperparameter_results.csv"
    if result_path.exists():
        rows = pd.read_csv(result_path).to_dict("records")
    else:
        rows = [{"architecture":"v3_svm_reference","checkpoint":"preserved_v3","learning_rate":None,"epochs":None,"seed":42,**svm_reference(selection)}]
    winner = None
    for architecture, model_id in MODELS.items():
        if args.architecture and architecture != args.architecture: continue
        for learning_rate, epochs in CONFIGS:
            completed = any(row["architecture"] == architecture and float(row["learning_rate"]) == learning_rate and int(row["epochs"]) == epochs for row in rows if pd.notna(row.get("learning_rate")))
            if completed: continue
            model, tokenizer, result = train_once(model_id, train, selection, learning_rate, epochs, 42)
            row = {"architecture":architecture,"checkpoint":model_id,"learning_rate":learning_rate,"epochs":epochs,"seed":42,**result}
            rows.append(row); pd.DataFrame(rows).to_csv(result_path, index=False)
            if winner is None or result["macro_f1"] > winner["result"]["macro_f1"]:
                winner = {"architecture":architecture,"checkpoint":model_id,"learning_rate":learning_rate,"epochs":epochs,"result":result,
                          "state":copy.deepcopy(model.state_dict()),"tokenizer":tokenizer}
            del model
    pd.DataFrame(rows).to_csv(result_path, index=False)
    transformer_rows = [row for row in rows if row["architecture"] != "v3_svm_reference"]
    comparison = pd.DataFrame(transformer_rows).sort_values("macro_f1",ascending=False).groupby("architecture",as_index=False).first()
    comparison = pd.concat([pd.DataFrame([rows[0]]),comparison],ignore_index=True)
    comparison.to_csv(HERE / "transformer_model_comparison.csv",index=False)
    if winner is not None:
        winner_model = AutoModelForSequenceClassification.from_pretrained(winner["checkpoint"],num_labels=7,id2label={i:l for i,l in enumerate(LABELS)},label2id=LABEL_TO_ID)
        winner_model.load_state_dict(winner["state"])
        output = HERE / f"development_winner_{winner['architecture']}"
        winner_model.save_pretrained(output); winner["tokenizer"].save_pretrained(output)
    global_winner = pd.DataFrame(transformer_rows).sort_values(["macro_f1", "macro_recall", "accuracy"], ascending=False).iloc[0].to_dict()
    selection_metadata = {"architecture": global_winner["architecture"], "checkpoint": global_winner["checkpoint"],
                          "learning_rate": float(global_winner["learning_rate"]), "epochs": int(global_winner["epochs"]),
                          "seed": int(global_winner["seed"]), "result": {key: float(global_winner[key]) for key in ["accuracy","macro_precision","macro_recall","macro_f1","weighted_f1","best_epoch","seconds"]}}
    selection_metadata["selection_basis"] = "development selection partition macro F1 only"
    selection_metadata["final_tests_opened"] = False
    (HERE / "development_selection.json").write_text(json.dumps(selection_metadata,indent=2),encoding="utf-8")
    print(json.dumps(selection_metadata,indent=2),flush=True)


if __name__ == "__main__": main()
