from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def predict(args: argparse.Namespace) -> None:
    os.environ.setdefault("YOLO_CONFIG_DIR", str((Path.cwd() / ".ultralytics").resolve()))

    from ultralytics import YOLO

    image = Image.open(args.image).convert("RGB")
    model = YOLO(str(args.model))
    result = model.predict(
        np.asarray(image),
        imgsz=args.image_size,
        conf=args.confidence,
        iou=args.iou,
        max_det=args.max_det,
        verbose=False,
        device=args.device,
    )[0]

    boxes = result.boxes
    detections = []
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        width, height = image.size
        for box, confidence in zip(xyxy, confidences):
            x1, y1, x2, y2 = [float(value) for value in box]
            detections.append(
                {
                    "x": round(max(0.0, x1) / max(1, width), 4),
                    "y": round(max(0.0, y1) / max(1, height), 4),
                    "width": round(max(0.0, x2 - x1) / max(1, width), 4),
                    "height": round(max(0.0, y2 - y1) / max(1, height), 4),
                    "confidence": round(float(confidence), 4),
                }
            )

    print(
        json.dumps(
            {
                "image": str(args.image),
                "count": len(detections),
                "detections": detections,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Count pills using a trained YOLO detector.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    predict(args)


if __name__ == "__main__":
    main()
