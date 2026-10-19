from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class TabletCountDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, count_to_index: dict[int, int], transform: transforms.Compose) -> None:
        self.frame = frame.reset_index(drop=True)
        self.count_to_index = count_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        count = int(row["count"])
        return self.transform(image), torch.tensor(self.count_to_index[count], dtype=torch.long)


def build_model(class_count: int, pretrained: bool) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, class_count)
    return model


def load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required_columns = {"image_path", "count"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {', '.join(sorted(missing))}")

    frame = frame.copy()
    frame["image_path"] = frame["image_path"].astype(str)
    frame["count"] = frame["count"].astype(int)
    frame = frame[frame["image_path"].map(lambda value: Path(value).exists())].copy()
    if frame.empty:
        raise ValueError("No training rows remain after checking image paths.")
    return frame


def split_manifest(frame: pd.DataFrame, val_split: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "split" in frame.columns:
        train_frame = frame[frame["split"].astype(str).str.lower() == "train"].copy()
        val_frame = frame[frame["split"].astype(str).str.lower().isin(["val", "valid", "validation", "test"])].copy()
        if not train_frame.empty and not val_frame.empty:
            return train_frame, val_frame

    stratify = frame["count"] if frame["count"].value_counts().min() >= 2 else None
    return train_test_split(frame, test_size=val_split, random_state=42, stratify=stratify)


def limit_rows(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame

    groups = []
    per_count = max(1, max_rows // max(1, frame["count"].nunique()))
    for _, group in frame.groupby("count"):
        groups.append(group.sample(n=min(per_count, len(group)), random_state=42))

    limited = pd.concat(groups).sample(frac=1, random_state=42).reset_index(drop=True)
    if len(limited) > max_rows:
        limited = limited.sample(n=max_rows, random_state=42).reset_index(drop=True)
    return limited


def train(args: argparse.Namespace) -> None:
    frame = load_manifest(args.manifest)
    counts = sorted(int(value) for value in frame["count"].unique())
    count_to_index = {count: index for index, count in enumerate(counts)}
    index_to_count = {index: count for count, index in count_to_index.items()}
    train_frame, val_frame = split_manifest(frame, args.val_split)
    train_frame = limit_rows(train_frame, args.max_train_rows)
    val_frame = limit_rows(val_frame, args.max_val_rows)

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.22, contrast=0.2, saturation=0.18),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_loader = DataLoader(
        TabletCountDataset(train_frame, count_to_index, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        TabletCountDataset(val_frame, count_to_index, val_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_model(len(counts), args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        model.eval()
        correct = 0
        total = 0
        off_by_one = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                predicted = outputs.argmax(dim=1)
                correct += (predicted == targets).sum().item()
                total += targets.numel()

                predicted_counts = torch.tensor([index_to_count[int(index)] for index in predicted.cpu()], device=targets.device)
                target_counts = torch.tensor([index_to_count[int(index)] for index in targets.cpu()], device=targets.device)
                off_by_one += (torch.abs(predicted_counts - target_counts) <= 1).sum().item()

        train_loss = total_loss / max(1, len(train_frame))
        accuracy = correct / max(1, total)
        off_by_one_accuracy = off_by_one / max(1, total)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_accuracy={accuracy:.4f} off_by_one_accuracy={off_by_one_accuracy:.4f}"
        )

        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "architecture": "mobilenet_v3_small",
                    "task": "tablet_count_classification",
                    "counts": counts,
                    "val_accuracy": accuracy,
                    "off_by_one_accuracy": off_by_one_accuracy,
                },
                args.output,
            )

    print(f"Saved best model to {args.output} with val_accuracy={best_accuracy:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an intake tablet-count classifier.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
