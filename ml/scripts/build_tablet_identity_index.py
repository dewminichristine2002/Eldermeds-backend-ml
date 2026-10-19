from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class IdentityIndexDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: transforms.Compose) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, str, int]:
        row = self.frame.iloc[index]
        image_path = str(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), str(row["medicine_name"]), image_path, int(index)


def resolve_path(value: str, manifest_dir: Path, ml_root: Path) -> str:
    candidate = Path(str(value))
    if candidate.is_absolute():
        return str(candidate)
    if candidate.exists():
        return str(candidate.resolve())
    if (ml_root / candidate).exists():
        return str((ml_root / candidate).resolve())
    return str((manifest_dir / candidate).resolve())


def load_manifest(path: Path, split: str, max_rows: int) -> pd.DataFrame:
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

    if max_rows > 0 and len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=42).copy()

    if frame.empty:
        raise ValueError("No index rows remain after filtering.")
    return frame.reset_index(drop=True)


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


class MetricIdentityModel(nn.Module):
    def __init__(self, architecture: str, embedding_dim: int, class_count: int) -> None:
        super().__init__()
        self.encoder, feature_dim = build_encoder(architecture, pretrained=False)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.12),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, class_count)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        return F.normalize(self.projector(features), p=2, dim=1)


def load_metric_model(model_path: Path, device: torch.device) -> tuple[nn.Module, str, int]:
    checkpoint = torch.load(model_path, map_location=device)
    if checkpoint.get("task") != "tablet_metric_identity":
        raise ValueError(f"Unsupported metric model task: {checkpoint.get('task')}")

    labels = [str(value) for value in checkpoint["labels"]]
    architecture = str(checkpoint.get("architecture") or "efficientnet_b0")
    embedding_dim = int(checkpoint.get("embedding_dim") or 512)
    model = MetricIdentityModel(architecture, embedding_dim, len(labels))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, architecture, embedding_dim


def embed_batch(model: nn.Module, images: torch.Tensor) -> np.ndarray:
    outputs = model(images)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    features = outputs.detach().cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-8)


def build_index(args: argparse.Namespace) -> None:
    frame = load_manifest(args.manifest, args.split, args.max_rows)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    loader = DataLoader(
        IdentityIndexDataset(frame, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    metric_model_path = str(args.metric_model.resolve()) if args.metric_model else ""
    if args.metric_model:
        if not args.metric_model.exists():
            raise FileNotFoundError(f"Metric model not found: {args.metric_model}")
        model, architecture, feature_dim = load_metric_model(args.metric_model, device)
        embedding_model = "tablet_metric_identity"
    else:
        model, feature_dim = build_encoder(args.architecture, args.pretrained)
        architecture = args.architecture
        embedding_model = "pretrained_encoder"
        model.to(device)
        model.eval()

    embedding_batches = []
    labels: list[str] = []
    image_paths: list[str] = []
    row_indexes: list[int] = []

    with torch.no_grad():
        for images, batch_labels, batch_paths, batch_indexes in loader:
            images = images.to(device)
            embedding_batches.append(embed_batch(model, images))
            labels.extend(str(item) for item in batch_labels)
            image_paths.extend(str(item) for item in batch_paths)
            row_indexes.extend(int(item) for item in batch_indexes)

    embeddings = np.concatenate(embedding_batches, axis=0).astype(np.float32)
    feature_dim = int(embeddings.shape[1])
    metadata_columns = [
        column
        for column in frame.columns
        if column not in {"image_path", "medicine_name", "split"}
    ]
    metadata_values = []
    for row_index in row_indexes:
        row = frame.iloc[row_index]
        metadata_values.append([str(row.get(column, "") or "") for column in metadata_columns])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        embeddings=embeddings,
        labels=np.asarray(labels, dtype=object),
        image_paths=np.asarray(image_paths, dtype=object),
        architecture=architecture,
        feature_dim=feature_dim,
        embedding_model=embedding_model,
        metric_model_path=metric_model_path,
        source_manifest=str(args.manifest),
        metadata_columns=np.asarray(metadata_columns, dtype=object),
        metadata_values=np.asarray(metadata_values, dtype=object),
    )
    print(f"Saved identity index with {len(labels)} images and {len(set(labels))} labels to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a visual retrieval index for tablet identity matching.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default="train,val", help="Comma-separated split values to include.")
    parser.add_argument("--architecture", choices=["mobilenet_v3_small", "efficientnet_b0", "resnet18"], default="efficientnet_b0")
    parser.add_argument("--metric-model", type=Path, help="Optional trained tablet_metric_identity checkpoint for ePillID-style embeddings.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    build_index(args)


if __name__ == "__main__":
    main()
