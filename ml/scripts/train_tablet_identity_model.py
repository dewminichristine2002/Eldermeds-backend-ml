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


class TabletIdentityDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, label_to_index: dict[str, int], transform: transforms.Compose) -> None:
        self.frame = frame.reset_index(drop=True)
        self.label_to_index = label_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        label = str(row["medicine_name"])
        return self.transform(image), torch.tensor(self.label_to_index[label], dtype=torch.long)


def build_model(class_count: int, pretrained: bool, architecture: str) -> nn.Module:
    if architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, class_count)
        return model

    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, class_count)
        return model

    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, class_count)
    return model


def load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required_columns = {"image_path", "medicine_name"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {', '.join(sorted(missing))}")

    manifest_dir = path.resolve().parent
    ml_root = Path(__file__).resolve().parents[1]

    def resolve_manifest_path(value: str) -> str:
        candidate = Path(str(value))
        if candidate.is_absolute():
            return str(candidate)
        if candidate.exists():
            return str(candidate.resolve())
        if (ml_root / candidate).exists():
            return str((ml_root / candidate).resolve())
        return str((manifest_dir / candidate).resolve())

    frame = frame.copy()
    frame["image_path"] = frame["image_path"].astype(str).map(resolve_manifest_path)
    frame["medicine_name"] = frame["medicine_name"].astype(str).str.strip()
    frame = frame[
        (frame["medicine_name"] != "")
        & frame["image_path"].map(lambda value: Path(value).exists())
    ].copy()
    if frame.empty:
        raise ValueError("No training rows remain after checking image paths.")
    if frame["medicine_name"].nunique() < 2:
        raise ValueError("At least two medicine_name classes are required for identity training.")
    return frame


def split_manifest(frame: pd.DataFrame, val_split: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "split" in frame.columns:
        split_values = frame["split"].astype(str).str.lower()
        train_frame = frame[split_values == "train"].copy()
        val_frame = frame[split_values.isin(["val", "valid", "validation", "test"])].copy()
        if not train_frame.empty and not val_frame.empty:
            return train_frame, val_frame

    class_counts = frame["medicine_name"].value_counts()
    stratify = frame["medicine_name"] if class_counts.min() >= 2 else None
    return train_test_split(frame, test_size=val_split, random_state=42, stratify=stratify)


def limit_rows(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame

    groups = []
    per_class = max(1, max_rows // max(1, frame["medicine_name"].nunique()))
    for _, group in frame.groupby("medicine_name"):
        groups.append(group.sample(n=min(per_class, len(group)), random_state=42))

    limited = pd.concat(groups).sample(frac=1, random_state=42).reset_index(drop=True)
    if len(limited) > max_rows:
        limited = limited.sample(n=max_rows, random_state=42).reset_index(drop=True)
    return limited


def train(args: argparse.Namespace) -> None:
    frame = load_manifest(args.manifest)
    labels = sorted(str(value) for value in frame["medicine_name"].unique())
    label_to_index = {label: index for index, label in enumerate(labels)}
    index_to_label = {index: label for label, index in label_to_index.items()}
    train_frame, val_frame = split_manifest(frame, args.val_split)
    train_frame = limit_rows(train_frame, args.max_train_rows)
    val_frame = limit_rows(val_frame, args.max_val_rows)

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop((224, 224), scale=(0.72, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(14),
            transforms.ColorJitter(brightness=0.24, contrast=0.22, saturation=0.2),
            transforms.RandomPerspective(distortion_scale=0.12, p=0.25),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.16, scale=(0.015, 0.08), ratio=(0.35, 2.8)),
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
        TabletIdentityDataset(train_frame, label_to_index, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        TabletIdentityDataset(val_frame, label_to_index, val_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_model(len(labels), args.pretrained, args.architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    class_weights = None
    if args.class_weights:
        counts = train_frame["medicine_name"].value_counts()
        weights = [1.0 / max(1, int(counts.get(label, 1))) for label in labels]
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

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
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                predicted = outputs.argmax(dim=1)
                correct += (predicted == targets).sum().item()
                total += targets.numel()

        train_loss = total_loss / max(1, len(train_frame))
        accuracy = correct / max(1, total)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_accuracy={accuracy:.4f}")

        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "architecture": args.architecture,
                    "task": "tablet_identity_classification",
                    "labels": labels,
                    "index_to_label": index_to_label,
                    "val_accuracy": accuracy,
                },
                args.output,
            )

    print(f"Saved best model to {args.output} with val_accuracy={best_accuracy:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tablet identity classifier.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--architecture", choices=["mobilenet_v3_small", "efficientnet_b0", "resnet18"], default="efficientnet_b0")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--class-weights", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
