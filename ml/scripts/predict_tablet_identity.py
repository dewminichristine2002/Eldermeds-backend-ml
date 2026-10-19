from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


def load_model(model_path: Path, device: torch.device) -> tuple[nn.Module, list[str]]:
    checkpoint = torch.load(model_path, map_location=device)
    labels = [str(value) for value in checkpoint["labels"]]
    architecture = checkpoint.get("architecture") or "mobilenet_v3_small"
    if checkpoint.get("task") != "tablet_identity_classification":
        raise ValueError(f"Unsupported task: {checkpoint.get('task')}")

    if architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, len(labels))
    elif architecture == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, len(labels))
    elif architecture == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, len(labels))
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, labels


def predict(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, labels = load_model(args.model, device)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(args.image).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).squeeze(0)
        values, indexes = torch.topk(probabilities, k=min(args.top_k, len(labels)))

    result = {
        "image": str(args.image),
        "predictions": [
            {
                "medicineName": labels[int(index.item())],
                "confidence": round(float(value.item()), 4),
            }
            for value, index in zip(values, indexes)
        ],
    }
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict medicine name from a tablet photo.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    predict(args)


if __name__ == "__main__":
    main()
