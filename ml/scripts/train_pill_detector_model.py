from __future__ import annotations

import argparse
import os
from pathlib import Path


def train(args: argparse.Namespace) -> None:
    os.environ.setdefault("YOLO_CONFIG_DIR", str((Path.cwd() / ".ultralytics").resolve()))

    from ultralytics import YOLO

    model = YOLO(args.base_model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        workers=args.workers,
        patience=args.patience,
        plots=args.plots,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO pill detector for tablet counting.")
    parser.add_argument("--data", type=Path, default=Path("pill-detector-data.yaml"))
    parser.add_argument("--base-model", default="yolo11n.pt")
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--name", default="pill_detector")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
