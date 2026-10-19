"""
Generate confusion matrix charts for intake monitoring.

Outputs:
- models/intake_tablet_count_confusion_matrix.json
- models/intake_tablet_count_confusion_matrix.png
- models/intake_monitoring_binary_confusion_matrix.png, when yolo_intake_metrics.json exists

Run:
  python generate_intake_monitoring_confusion_matrix.py --cpu
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DEFAULT_MANIFEST = ROOT / "data" / "public" / "pills-detection-count-manifest-dose.csv"
DEFAULT_MODEL = MODELS_DIR / "tablet_count_classifier.pt"
DEFAULT_OUTPUT_JSON = MODELS_DIR / "intake_tablet_count_confusion_matrix.json"
DEFAULT_OUTPUT_CHART = MODELS_DIR / "intake_tablet_count_confusion_matrix.png"
DEFAULT_BINARY_METRICS = MODELS_DIR / "yolo_intake_metrics.json"
DEFAULT_BINARY_CHART = MODELS_DIR / "intake_monitoring_binary_confusion_matrix.png"


class TabletCountDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, count_to_index: dict[int, int], transform: transforms.Compose) -> None:
        self.frame = frame.reset_index(drop=True)
        self.count_to_index = count_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        image = Image.open(row["resolved_image_path"]).convert("RGB")
        count = int(row["count"])
        return self.transform(image), self.count_to_index[count]


def resolve_path(value: str | Path, base: Path = ROOT) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else base / path


def load_manifest(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required_columns = {"image_path", "count"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {', '.join(sorted(missing))}")

    frame = frame.copy()
    frame["count"] = frame["count"].astype(int)
    frame["resolved_image_path"] = frame["image_path"].map(lambda value: resolve_path(str(value)))

    if split and "split" in frame.columns:
        requested = split.lower()
        split_aliases = {
            "validation": {"val", "valid", "validation", "test"},
            "val": {"val", "valid", "validation", "test"},
            "test": {"test"},
            "train": {"train"},
            "all": {"train", "val", "valid", "validation", "test"},
        }
        accepted = split_aliases.get(requested, {requested})
        frame = frame[frame["split"].astype(str).str.lower().isin(accepted)].copy()

    frame = frame[frame["resolved_image_path"].map(lambda value: Path(value).exists())].copy()
    if frame.empty:
        raise ValueError(f"No rows found for split={split!r} with existing image files.")
    return frame


def build_model(class_count: int) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, class_count)
    return model


def load_count_model(model_path: Path, device: torch.device) -> tuple[nn.Module, list[int]]:
    checkpoint = torch.load(model_path, map_location=device)
    if checkpoint.get("architecture") != "mobilenet_v3_small":
        raise ValueError(f"Unsupported architecture: {checkpoint.get('architecture')}")

    counts = [int(value) for value in checkpoint["counts"]]
    model = build_model(len(counts))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, counts


def evaluate_count_model(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, counts = load_count_model(args.model, device)
    count_to_index = {count: index for index, count in enumerate(counts)}
    index_to_count = {index: count for count, index in count_to_index.items()}
    frame = load_manifest(args.manifest, args.split)
    frame = frame[frame["count"].isin(counts)].copy()
    if frame.empty:
        raise ValueError(f"No rows found for model count labels: {counts}")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    loader = DataLoader(
        TabletCountDataset(frame, count_to_index, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, targets in loader:
            outputs = model(images.to(device))
            predicted_indexes = outputs.argmax(dim=1).cpu().tolist()
            target_indexes = [int(value) for value in targets]
            y_true.extend(index_to_count[index] for index in target_indexes)
            y_pred.extend(index_to_count[index] for index in predicted_indexes)

    labels = counts
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[f"{count} tablet" if count == 1 else f"{count} tablets" for count in labels],
        output_dict=True,
        zero_division=0,
    )

    return {
        "task": "intake_tablet_count_classification",
        "model": str(args.model),
        "manifest": str(args.manifest),
        "split": args.split,
        "rows": len(y_true),
        "labels": labels,
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": {
            "rows": labels,
            "columns": labels,
            "values": matrix,
        },
        "classification_report": report,
    }


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, image_font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=image_font)
    return box[2] - box[0], box[3] - box[1]


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    image_font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    width, height = text_size(draw, text, image_font)
    x1, y1, x2, y2 = box
    draw.text((x1 + (x2 - x1 - width) / 2, y1 + (y2 - y1 - height) / 2), text, font=image_font, fill=fill)


def draw_confusion_matrix_chart(
    matrix: list[list[int]],
    labels: list[str],
    title: str,
    output_path: Path,
    subtitle: str = "",
) -> None:
    size = len(labels)
    cell = 160 if size <= 3 else 112
    left = 220
    top = 185
    right = 120
    bottom = 120
    width = max(left + (size * cell) + right, 760)
    height = top + (size * cell) + bottom
    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)

    title_font = font(28, bold=True)
    subtitle_font = font(16)
    label_font = font(16, bold=True)
    cell_font = font(24, bold=True)
    axis_font = font(18, bold=True)

    draw.text((left, 34), title, font=title_font, fill="#102a43")
    if subtitle:
        draw.text((left, 76), subtitle, font=subtitle_font, fill="#52616b")

    max_value = max([value for row in matrix for value in row] or [1])
    min_blue = (229, 242, 255)
    max_blue = (28, 91, 150)

    for row_index, row in enumerate(matrix):
        row_total = max(1, sum(row))
        for col_index, value in enumerate(row):
            intensity = value / max(1, max_value)
            color = tuple(
                int(min_blue[channel] + (max_blue[channel] - min_blue[channel]) * intensity)
                for channel in range(3)
            )
            x1 = left + col_index * cell
            y1 = top + row_index * cell
            x2 = x1 + cell
            y2 = y1 + cell
            draw.rectangle((x1, y1, x2, y2), fill=color, outline="#ffffff", width=3)
            pct = round((value / row_total) * 100)
            text_fill = "#ffffff" if intensity >= 0.46 else "#102a43"
            draw_centered_text(draw, (x1, y1 - 12, x2, y2 - 8), str(value), cell_font, text_fill)
            draw_centered_text(draw, (x1, y1 + 32, x2, y2 + 36), f"{pct}%", subtitle_font, text_fill)

    for col_index, label in enumerate(labels):
        x1 = left + col_index * cell
        draw_centered_text(draw, (x1, top - 52, x1 + cell, top - 10), label, label_font, "#102a43")

    for row_index, label in enumerate(labels):
        y1 = top + row_index * cell
        draw_centered_text(draw, (34, y1, left - 16, y1 + cell), label, label_font, "#102a43")

    axis_width, _ = text_size(draw, "Predicted", axis_font)
    draw.text((left + (size * cell - axis_width) / 2, top - 92), "Predicted", font=axis_font, fill="#102a43")
    draw.text((34, top + (size * cell) + 38), "Rows: true labels | Columns: predicted labels", font=subtitle_font, fill="#52616b")
    draw.text((34, top + (size * cell) / 2 - 12), "Actual", font=axis_font, fill="#102a43")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def maybe_draw_binary_chart(metrics_path: Path, output_path: Path) -> bool:
    if not metrics_path.exists():
        return False

    payload = json.loads(metrics_path.read_text(encoding="utf8"))
    confusion_payload = payload.get("metrics", {}).get("confusion_matrix", {})
    matrix = confusion_payload.get("matrix")
    labels = confusion_payload.get("labels")
    if not matrix or not labels:
        return False

    pretty_labels = [str(label).replace("_", " ").title() for label in labels]
    subtitle = f"YOLO intake monitor, evaluated images: {payload.get('evaluated_images', 'unknown')}"
    draw_confusion_matrix_chart(matrix, pretty_labels, "Intake Monitoring Confusion Matrix", output_path, subtitle)
    return True


def print_tablet_count_summary(result: dict[str, Any]) -> None:
    report = result["classification_report"]
    matrix = result["confusion_matrix"]["values"]
    labels = result["labels"]
    macro = report["macro avg"]
    weighted = report["weighted avg"]

    print("Tablet Count Confusion Matrix Metrics")
    print(f"Split: {result['split']}")
    print(f"Evaluated images: {result['rows']}")
    print(f"Labels: {', '.join(str(label) for label in labels)}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Macro Precision: {macro['precision']:.4f}")
    print(f"Macro Recall/Sensitivity: {macro['recall']:.4f}")
    print(f"Macro F1-score: {macro['f1-score']:.4f}")
    print(f"Weighted Precision: {weighted['precision']:.4f}")
    print(f"Weighted Recall: {weighted['recall']:.4f}")
    print(f"Weighted F1-score: {weighted['f1-score']:.4f}")
    print("Confusion Matrix:")
    print(f"Rows/Actual: {labels}")
    print(f"Columns/Predicted: {labels}")
    for label, row in zip(labels, matrix):
        print(f"Actual {label}: {row}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate intake monitoring confusion matrix charts.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-chart", type=Path, default=DEFAULT_OUTPUT_CHART)
    parser.add_argument("--binary-metrics", type=Path, default=DEFAULT_BINARY_METRICS)
    parser.add_argument("--binary-chart", type=Path, default=DEFAULT_BINARY_CHART)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    args.manifest = resolve_path(args.manifest)
    args.model = resolve_path(args.model)
    args.output_json = resolve_path(args.output_json)
    args.output_chart = resolve_path(args.output_chart)
    args.binary_metrics = resolve_path(args.binary_metrics)
    args.binary_chart = resolve_path(args.binary_chart)

    result = evaluate_count_model(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf8")

    count_labels = [f"{count}" for count in result["labels"]]
    draw_confusion_matrix_chart(
        result["confusion_matrix"]["values"],
        count_labels,
        "Intake Tablet Count Confusion Matrix",
        args.output_chart,
        f"Validation split: {result['rows']} images | Accuracy: {result['accuracy']:.3f}",
    )

    print(f"[intake-confusion] Wrote {args.output_json}")
    print(f"[intake-confusion] Wrote {args.output_chart}")

    if maybe_draw_binary_chart(args.binary_metrics, args.binary_chart):
        print(f"[intake-confusion] Wrote {args.binary_chart}")

    print()
    print_tablet_count_summary(result)


if __name__ == "__main__":
    main()
