from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import models, transforms


class TabletMetricDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, label_to_index: dict[str, int], transform: transforms.Compose) -> None:
        self.frame = frame.reset_index(drop=True)
        self.label_to_index = label_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        label = self.label_to_index[str(row["medicine_name"])]
        return self.transform(image), torch.tensor(label, dtype=torch.long)


class BalancedClassBatchSampler(Sampler[list[int]]):
    def __init__(self, labels: list[int], classes_per_batch: int, samples_per_class: int, batches_per_epoch: int) -> None:
        self.labels = labels
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.batches_per_epoch = batches_per_epoch
        self.by_label: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            self.by_label[int(label)].append(index)
        self.valid_labels = [label for label, indexes in self.by_label.items() if indexes]

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        rng = np.random.default_rng()
        for _ in range(self.batches_per_epoch):
            class_count = min(self.classes_per_batch, len(self.valid_labels))
            selected_labels = rng.choice(self.valid_labels, size=class_count, replace=False)
            batch = []
            for label in selected_labels:
                indexes = self.by_label[int(label)]
                replace = len(indexes) < self.samples_per_class
                batch.extend(rng.choice(indexes, size=self.samples_per_class, replace=replace).tolist())
            rng.shuffle(batch)
            yield batch


class MetricIdentityModel(nn.Module):
    def __init__(self, architecture: str, embedding_dim: int, class_count: int, pretrained: bool) -> None:
        super().__init__()
        self.architecture = architecture
        self.embedding_dim = embedding_dim
        self.encoder, feature_dim = build_encoder(architecture, pretrained)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.12),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, class_count)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(images)
        embeddings = F.normalize(self.projector(features), p=2, dim=1)
        logits = self.classifier(embeddings)
        return embeddings, logits


def build_encoder(architecture: str, pretrained: bool) -> tuple[nn.Module, int]:
    if architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        feature_dim = model.classifier[-1].in_features
        model.classifier = nn.Identity()
        return model, feature_dim

    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim

    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    feature_dim = model.classifier[0].in_features
    model.classifier = nn.Identity()
    return model, feature_dim


def resolve_path(value: str, manifest_dir: Path, ml_root: Path) -> str:
    candidate = Path(str(value))
    if candidate.is_absolute():
        return str(candidate)
    if candidate.exists():
        return str(candidate.resolve())
    if (ml_root / candidate).exists():
        return str((ml_root / candidate).resolve())
    return str((manifest_dir / candidate).resolve())


def load_manifest(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"image_path", "medicine_name"} - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {', '.join(sorted(missing))}")

    manifest_dir = path.resolve().parent
    ml_root = Path(__file__).resolve().parents[1]
    frame = frame.copy()
    frame["image_path"] = frame["image_path"].astype(str).map(lambda value: resolve_path(value, manifest_dir, ml_root))
    frame["medicine_name"] = frame["medicine_name"].astype(str).str.strip()
    frame = frame[
        (frame["medicine_name"] != "")
        & frame["image_path"].map(lambda value: Path(value).exists())
    ].copy()
    if split and "split" in frame.columns:
        split_values = {item.strip().lower() for item in split.split(",") if item.strip()}
        frame = frame[frame["split"].astype(str).str.lower().isin(split_values)].copy()
    if frame.empty:
        raise ValueError("No training rows remain after filtering.")
    return frame.reset_index(drop=True)


def split_train_val(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "split" in frame.columns:
        train_frame = frame[frame["split"].astype(str).str.lower() == "train"].copy()
        val_frame = frame[frame["split"].astype(str).str.lower().isin(["val", "valid", "validation", "test"])].copy()
        if not train_frame.empty and not val_frame.empty:
            return train_frame.reset_index(drop=True), val_frame.reset_index(drop=True)

    shuffled = frame.sample(frac=1, random_state=42).reset_index(drop=True)
    val_size = max(1, int(round(len(shuffled) * 0.2)))
    return shuffled.iloc[val_size:].copy(), shuffled.iloc[:val_size].copy()


def batch_hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    distances = torch.cdist(embeddings, embeddings, p=2)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive_mask = same & ~eye
    negative_mask = ~same

    if not positive_mask.any() or not negative_mask.any():
        return embeddings.new_tensor(0.0)

    hardest_positive = distances.masked_fill(~positive_mask, -1.0).max(dim=1).values
    hardest_negative = distances.masked_fill(~negative_mask, 1e6).min(dim=1).values
    valid = hardest_positive >= 0
    if not valid.any():
        return embeddings.new_tensor(0.0)
    return F.relu(hardest_positive[valid] - hardest_negative[valid] + margin).mean()


def train(args: argparse.Namespace) -> None:
    frame = load_manifest(args.manifest, "")
    if args.max_rows > 0 and len(frame) > args.max_rows:
        frame = frame.sample(n=args.max_rows, random_state=42).reset_index(drop=True)
    train_frame, val_frame = split_train_val(frame)
    labels = sorted(str(value) for value in frame["medicine_name"].unique())
    label_to_index = {label: index for index, label in enumerate(labels)}

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop((224, 224), scale=(0.72, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(18),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.18),
            transforms.RandomPerspective(distortion_scale=0.12, p=0.25),
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

    train_labels = [label_to_index[str(value)] for value in train_frame["medicine_name"]]
    batches_per_epoch = max(1, min(args.batches_per_epoch, len(train_frame) // max(1, args.classes_per_batch * args.samples_per_class)))
    sampler = BalancedClassBatchSampler(train_labels, args.classes_per_batch, args.samples_per_class, batches_per_epoch)

    train_loader = DataLoader(
        TabletMetricDataset(train_frame, label_to_index, train_transform),
        batch_sampler=sampler,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        TabletMetricDataset(val_frame, label_to_index, val_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = MetricIdentityModel(args.architecture, args.embedding_dim, len(labels), args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_val_acc = 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        seen = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            embeddings, logits = model(images)
            ce_loss = criterion(logits, targets)
            triplet_loss = batch_hard_triplet_loss(embeddings, targets, args.margin)
            loss = (args.ce_weight * ce_loss) + (args.triplet_weight * triplet_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            seen += images.size(0)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                _, logits = model(images)
                predicted = logits.argmax(dim=1)
                correct += (predicted == targets).sum().item()
                total += targets.numel()

        val_acc = correct / max(1, total)
        print(f"epoch={epoch} train_loss={train_loss / max(1, seen):.4f} val_acc={val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "architecture": args.architecture,
                    "embedding_dim": args.embedding_dim,
                    "task": "tablet_metric_identity",
                    "labels": labels,
                    "val_accuracy": val_acc,
                },
                args.output,
            )

    print(f"Saved best metric identity model to {args.output} with val_acc={best_val_acc:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ePillID-style metric embedding model for tablet identity.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--architecture", choices=["mobilenet_v3_small", "efficientnet_b0", "resnet18"], default="efficientnet_b0")
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--classes-per-batch", type=int, default=16)
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--batches-per-epoch", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.35)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--triplet-weight", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
