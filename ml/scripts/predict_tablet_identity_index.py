from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models, transforms


def scalar_value(index, key: str, default: str = "") -> str:
    if key not in index:
        return default
    value = index[key]
    return str(value.item() if getattr(value, "shape", ()) == () else value)


def build_encoder(architecture: str) -> tuple[nn.Module, int]:
    if architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        feature_dim = model.classifier[-1].in_features
        model.classifier = nn.Identity()
        return model, feature_dim
    if architecture == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    feature_dim = model.classifier[0].in_features
    model.classifier = nn.Identity()
    return model, feature_dim


class MetricIdentityModel(nn.Module):
    def __init__(self, architecture: str, embedding_dim: int, class_count: int) -> None:
        super().__init__()
        self.encoder, feature_dim = build_encoder_without_weights(architecture)
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


def build_encoder_without_weights(architecture: str) -> tuple[nn.Module, int]:
    if architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        feature_dim = model.classifier[-1].in_features
        model.classifier = nn.Identity()
        return model, feature_dim
    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim
    model = models.mobilenet_v3_small(weights=None)
    feature_dim = model.classifier[0].in_features
    model.classifier = nn.Identity()
    return model, feature_dim


def load_metric_model(model_path: Path, device: torch.device) -> nn.Module:
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
    return model


def build_embedding_model(architecture: str, metric_model_path: str, device: torch.device) -> nn.Module:
    if metric_model_path:
        return load_metric_model(Path(metric_model_path), device)
    model, _ = build_encoder(architecture)
    model.to(device)
    model.eval()
    return model


def embed_image(image_path: Path, model: nn.Module, device: torch.device) -> np.ndarray:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        features = outputs.detach().cpu().numpy().astype(np.float32)[0]
    return features / max(float(np.linalg.norm(features)), 1e-8)


def predict(args: argparse.Namespace) -> None:
    index = np.load(args.index, allow_pickle=True)
    embeddings = index["embeddings"].astype(np.float32)
    labels = [str(item) for item in index["labels"].tolist()]
    image_paths = [str(item) for item in index["image_paths"].tolist()]
    metadata_columns = [str(item) for item in index["metadata_columns"].tolist()] if "metadata_columns" in index else []
    metadata_values = index["metadata_values"].tolist() if "metadata_values" in index else []
    architecture = scalar_value(index, "architecture", "efficientnet_b0")
    metric_model_path = str(args.metric_model or scalar_value(index, "metric_model_path", ""))
    if metric_model_path and not Path(metric_model_path).exists():
        raise FileNotFoundError(f"Metric model not found: {metric_model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_embedding_model(architecture, metric_model_path, device)
    query = embed_image(args.image, model, device)
    similarities = embeddings @ query
    top_indexes = np.argsort(-similarities)[: max(args.top_k * 8, args.top_k)]

    label_scores: dict[str, dict[str, object]] = {}
    label_score_lists: dict[str, list[float]] = {}
    for index_value in top_indexes:
        label = labels[int(index_value)]
        similarity = float(similarities[int(index_value)])
        label_score_lists.setdefault(label, []).append(similarity)
        existing = label_scores.get(label)
        if existing is None or similarity > float(existing.get("similarity") or -1):
            metadata = {}
            if metadata_columns and int(index_value) < len(metadata_values):
                metadata = {
                    metadata_columns[column_index]: str(value)
                    for column_index, value in enumerate(metadata_values[int(index_value)])
                    if column_index < len(metadata_columns) and str(value)
                }
            label_scores[label] = {
                "medicineName": label,
                "confidence": round(float((similarity + 1.0) / 2.0), 4),
                "similarity": round(similarity, 4),
                "referenceImage": image_paths[int(index_value)],
                "metadata": metadata,
            }

    ranked_labels = sorted(label_scores.values(), key=lambda item: float(item.get("similarity") or -1), reverse=True)
    predictions = []
    for item in ranked_labels:
        score_list = label_score_lists.get(str(item.get("medicineName"))) or []
        if len(score_list) > 1:
            item["meanSimilarity"] = round(float(np.mean(score_list)), 4)
            item["referenceCount"] = int(len(score_list))
        predictions.append(item)
        if len(predictions) >= args.top_k:
            break

    top_similarity = float(predictions[0].get("similarity") or 0) if predictions else 0.0
    second_similarity = float(predictions[1].get("similarity") or -1) if len(predictions) > 1 else -1.0
    margin = top_similarity - second_similarity if second_similarity >= -1 else top_similarity

    print(json.dumps({
        "image": str(args.image),
        "architecture": architecture,
        "metricModel": metric_model_path,
        "predictions": predictions,
        "topSimilarity": round(top_similarity, 4),
        "secondSimilarity": round(second_similarity, 4) if second_similarity >= -1 else None,
        "similarityMargin": round(float(margin), 4),
        "marginThreshold": args.margin_threshold,
        "ambiguous": bool(len(predictions) > 1 and margin < args.margin_threshold),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict medicine identity from a visual retrieval index.")
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--margin-threshold", type=float, default=0.045)
    parser.add_argument("--metric-model", type=Path, help="Override or supply the metric model checkpoint used by a metric index.")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    predict(args)


if __name__ == "__main__":
    main()
