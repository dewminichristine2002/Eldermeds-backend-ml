from __future__ import annotations

import base64
import json
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _decode_image(image_base64: str) -> Image.Image:
    value = str(image_base64 or "").strip()
    if "," in value and value.lower().startswith("data:"):
        value = value.split(",", 1)[1]
    image_bytes = base64.b64decode(value, validate=False)
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def _neighbor_count(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    return (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    )


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    cleaned = mask.astype(bool)
    # Remove isolated sensor noise without aggressively merging nearby tablets.
    cleaned = _neighbor_count(cleaned) >= 3
    # Fill tiny holes inside bright/colored tablet blobs.
    cleaned = _neighbor_count(cleaned) >= 4
    cleaned = cleaned | (_neighbor_count(cleaned) >= 7)
    return cleaned


def _estimate_perimeter(component_mask: np.ndarray) -> int:
    padded = np.pad(component_mask.astype(bool), 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    boundary = center & (~up | ~down | ~left | ~right)
    return int(boundary.sum())


def _find_components(mask: np.ndarray, image_width: int, image_height: int) -> list[dict[str, Any]]:
    visited = np.zeros_like(mask, dtype=bool)
    components: list[dict[str, Any]] = []
    image_area = image_width * image_height
    min_area = max(35, int(image_area * 0.00035))
    max_area = max(min_area + 1, int(image_area * 0.08))

    for start_y, start_x in np.argwhere(mask):
        sy = int(start_y)
        sx = int(start_x)
        if visited[sy, sx] or not mask[sy, sx]:
            continue

        stack = [(sy, sx)]
        visited[sy, sx] = True
        pixels: list[tuple[int, int]] = []
        touches_border = False

        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            if x <= 1 or y <= 1 or x >= image_width - 2 or y >= image_height - 2:
                touches_border = True

            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if ny < 0 or nx < 0 or ny >= image_height or nx >= image_width:
                    continue
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                stack.append((ny, nx))

        area = len(pixels)
        if area < min_area or touches_border:
            continue

        ys, xs = zip(*pixels)
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        box_width = max(1, x2 - x1 + 1)
        box_height = max(1, y2 - y1 + 1)
        if box_width < 5 or box_height < 5:
            continue

        bbox_area = box_width * box_height
        fill_ratio = area / max(1, bbox_area)
        aspect = max(box_width / box_height, box_height / box_width)

        component_mask = np.zeros_like(mask, dtype=bool)
        component_mask[np.asarray(ys), np.asarray(xs)] = True
        perimeter = _estimate_perimeter(component_mask)
        circularity = (4.0 * np.pi * area) / max(1.0, float(perimeter * perimeter))

        # Keep larger merged blobs so the later estimator can split them.
        if area > max_area and aspect < 1.8:
            continue

        if fill_ratio < 0.22 or aspect > 5.2:
            continue

        shape_score = min(1.0, max(0.35, (fill_ratio * 0.55) + (min(circularity, 1.0) * 0.45)))
        size_score = min(1.0, max(0.35, area / max(min_area, 1)))
        confidence = round(float((shape_score * 0.75) + (size_score * 0.25)), 4)

        components.append(
            {
                "x": round(x1 / image_width, 4),
                "y": round(y1 / image_height, 4),
                "width": round(box_width / image_width, 4),
                "height": round(box_height / image_height, 4),
                "pixelArea": int(area),
                "boxPixelWidth": int(box_width),
                "boxPixelHeight": int(box_height),
                "areaRatio": round(area / (image_width * image_height), 4),
                "fillRatio": round(float(fill_ratio), 4),
                "aspect": round(float(aspect), 4),
                "circularity": round(float(circularity), 4),
                "confidence": confidence,
            }
        )

    components.sort(key=lambda item: item["confidence"], reverse=True)
    return components[:12]


def _component_iou(first: dict[str, Any], second: dict[str, Any]) -> float:
    ax1 = float(first.get("x") or 0)
    ay1 = float(first.get("y") or 0)
    ax2 = ax1 + float(first.get("width") or 0)
    ay2 = ay1 + float(first.get("height") or 0)
    bx1 = float(second.get("x") or 0)
    by1 = float(second.get("y") or 0)
    bx2 = bx1 + float(second.get("width") or 0)
    by2 = by1 + float(second.get("height") or 0)

    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _find_components_cv2(arr: np.ndarray) -> list[dict[str, Any]]:
    try:
        import cv2
    except Exception:
        return []

    height, width, _ = arr.shape
    image_area = max(1, width * height)
    min_area = max(35, int(image_area * 0.00035))
    max_area = max(min_area + 1, int(image_area * 0.08))

    rgb_u8 = np.clip(arr, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    masks = []
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.extend([otsu, cv2.bitwise_not(otsu)])

    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        3,
    )
    masks.extend([adaptive, cv2.bitwise_not(adaptive)])

    sat_threshold = max(22, int(np.percentile(saturation, 68)))
    masks.append((saturation > sat_threshold).astype(np.uint8) * 255)

    median_gray = float(np.median(blurred))
    lower = int(max(0, 0.66 * median_gray))
    upper = int(min(255, 1.33 * median_gray + 20))
    edges = cv2.Canny(blurred, lower, upper)
    masks.append(cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1))

    morph_kernel = np.ones((5, 5), np.uint8)
    candidates: list[dict[str, Any]] = []
    for raw_mask in masks:
        mask_u8 = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, morph_kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, morph_kernel)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area or area > max_area:
                continue

            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_width < 6 or box_height < 6:
                continue
            if x <= 1 or y <= 1 or x + box_width >= width - 1 or y + box_height >= height - 1:
                continue

            bbox_area = max(1, box_width * box_height)
            fill_ratio = area / bbox_area
            aspect = max(box_width / max(1, box_height), box_height / max(1, box_width))
            if fill_ratio < 0.12 or aspect > 6.5:
                continue

            perimeter = max(1.0, float(cv2.arcLength(contour, True)))
            circularity = (4.0 * np.pi * area) / max(1.0, perimeter * perimeter)
            hull = cv2.convexHull(contour)
            hull_area = max(1.0, float(cv2.contourArea(hull)))
            solidity = area / hull_area
            if solidity < 0.42:
                continue

            score = (
                min(1.0, area / max(min_area * 16, 1))
                + min(1.0, fill_ratio)
                + min(1.0, solidity)
                + min(1.0, circularity)
                - max(0.0, aspect - 3.5) * 0.12
            )
            confidence = round(float(max(0.35, min(1.0, score / 4.0))), 4)

            candidates.append(
                {
                    "x": round(x / width, 4),
                    "y": round(y / height, 4),
                    "width": round(box_width / width, 4),
                    "height": round(box_height / height, 4),
                    "pixelArea": int(round(area)),
                    "boxPixelWidth": int(box_width),
                    "boxPixelHeight": int(box_height),
                    "areaRatio": round(float(area / image_area), 4),
                    "fillRatio": round(float(fill_ratio), 4),
                    "aspect": round(float(aspect), 4),
                    "circularity": round(float(circularity), 4),
                    "solidity": round(float(solidity), 4),
                    "confidence": confidence,
                    "source": "background-independent-contour",
                }
            )

    candidates.sort(key=lambda item: float(item.get("confidence") or 0), reverse=True)
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        if any(_component_iou(candidate, existing) > 0.35 for existing in selected):
            continue
        selected.append(candidate)
        if len(selected) >= 12:
            break

    return selected


def _component_count_estimate(
    components: list[dict[str, Any]],
    image_area: int,
    expected_count: float | None = None,
) -> tuple[int, list[dict[str, Any]], float]:
    if not components:
        return 0, [], 0.0

    areas = np.asarray([float(item["pixelArea"]) for item in components], dtype=np.float32)
    compact_areas = [
        float(item["pixelArea"])
        for item in components
        if float(item.get("circularity") or 0) >= 0.22 and float(item.get("aspect") or 99) <= 2.8
    ]
    if compact_areas:
        reference_area = float(np.median(compact_areas))
    else:
        reference_area = float(np.median(areas))

    # Real palm photos vary a lot. Bound the reference area to avoid tiny noise
    # causing huge counts and avoid one merged blob being treated as one tablet.
    min_reference = max(35.0, image_area * 0.00055)
    max_reference = max(min_reference, image_area * 0.025)
    reference_area = float(np.clip(reference_area, min_reference, max_reference))

    estimated_components: list[dict[str, Any]] = []
    total = 0
    confidence_values: list[float] = []
    for item in components:
        area = float(item["pixelArea"])
        aspect = float(item.get("aspect") or 1)
        area_ratio = area / max(reference_area, 1.0)
        long_axis_ratio = max(float(item.get("boxPixelWidth") or 1), float(item.get("boxPixelHeight") or 1)) / max(
            min(float(item.get("boxPixelWidth") or 1), float(item.get("boxPixelHeight") or 1)),
            1.0,
        )

        estimated_count = 1
        if area_ratio >= 1.7 or (area_ratio >= 1.35 and aspect > 2.0):
            estimated_count = int(round(area_ratio))
        if long_axis_ratio > 2.8 and area_ratio > 1.2:
            estimated_count = max(estimated_count, int(round(long_axis_ratio / 1.8)))
        if (
            len(components) == 1
            and expected_count is not None
            and expected_count > 1
            and aspect > 2.1
            and area >= image_area * 0.008
        ):
            estimated_count = max(estimated_count, min(int(round(expected_count)), int(round(aspect / 1.15))))
        estimated_count = max(1, min(6, estimated_count))

        next_item = {
            **item,
            "estimatedCount": estimated_count,
            "areaToReference": round(float(area_ratio), 3),
        }
        estimated_components.append(next_item)
        total += estimated_count
        confidence_values.append(float(item.get("confidence") or 0) / max(1, estimated_count))

    confidence = round(float(np.mean(confidence_values)) if confidence_values else 0.0, 4)
    return total, estimated_components, confidence


def _build_count_status(detected_count: float, expected_count: float | None) -> dict[str, Any]:
    if expected_count is None:
        return {}

    count_matches = abs(float(detected_count) - float(expected_count)) < 0.001
    if detected_count <= 0:
        status = "uncertain"
    elif count_matches:
        status = "okay"
    elif float(detected_count) > float(expected_count):
        status = "overdose"
    else:
        status = "underdose"

    return {
        "expectedCount": expected_count,
        "countMatches": count_matches,
        "doseComparison": status,
        "status": status,
    }


def _resolve_model_path() -> str:
    configured_path = os.getenv("INTAKE_COUNT_MODEL_PATH", "").strip()
    if configured_path:
        return configured_path

    model_dir = Path(__file__).resolve().parents[1] / "ml" / "models"
    for name in (
        "tablet_count_classifier.pt",
        "tablet_count_classifier_public.pt",
        "tablet_count_classifier_public_smoke.pt",
    ):
        candidate = model_dir / name
        if candidate.exists():
            return str(candidate)

    return ""


def _resolve_detector_model_path() -> str:
    configured_path = os.getenv("INTAKE_DETECTOR_MODEL_PATH", "").strip()
    if configured_path:
        return configured_path

    backend_dir = Path(__file__).resolve().parents[1]
    for candidate in (
        backend_dir / "ml" / "models" / "pill_detector.pt",
        backend_dir / "ml" / "runs" / "detect" / "pill_detector" / "weights" / "best.pt",
    ):
        if candidate.exists():
            return str(candidate)

    return ""


def _predict_with_detector(image: Image.Image, model_path: str) -> dict[str, Any]:
    if not model_path:
        return {
            "available": False,
            "error": "No trained pill detector model found.",
        }

    resolved_model_path = Path(model_path).resolve()
    if not resolved_model_path.exists():
        return {
            "available": False,
            "error": f"Detector model file not found: {resolved_model_path}",
        }

    try:
        os.environ.setdefault("YOLO_CONFIG_DIR", str((Path(__file__).resolve().parents[1] / "ml" / ".ultralytics").resolve()))
        from ultralytics import YOLO
    except Exception as error:
        return {
            "available": False,
            "error": f"Detector dependencies are not available: {error}",
            "modelPath": str(resolved_model_path),
        }

    try:
        try:
            confidence_threshold = float(os.getenv("INTAKE_DETECTOR_CONFIDENCE", "0.55"))
        except ValueError:
            confidence_threshold = 0.55

        model = YOLO(str(resolved_model_path))
        result = model.predict(
            np.asarray(image),
            imgsz=416,
            conf=confidence_threshold,
            iou=0.45,
            max_det=24,
            verbose=False,
            device="cpu",
        )[0]
        boxes = result.boxes
        width, height = image.size
        detections: list[dict[str, Any]] = []

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
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

        detections.sort(key=lambda item: item["confidence"], reverse=True)
        count = len(detections)
        confidence = float(np.mean([item["confidence"] for item in detections])) if detections else 0.0
        return {
            "available": True,
            "count": count,
            "confidence": round(confidence, 4),
            "objects": detections,
            "modelPath": str(resolved_model_path),
            "confidenceThreshold": confidence_threshold,
        }
    except Exception as error:
        return {
            "available": False,
            "error": str(error),
            "modelPath": str(resolved_model_path),
        }


def _predict_with_trained_model(image: Image.Image, model_path: str) -> dict[str, Any]:
    if not model_path:
        return {
            "available": False,
            "error": "No trained tablet-count model found.",
        }

    resolved_model_path = Path(model_path).resolve()
    if not resolved_model_path.exists():
        return {
            "available": False,
            "error": f"Model file not found: {resolved_model_path}",
        }

    try:
        import torch
        from torch import nn
        from torchvision import models, transforms
    except Exception as error:
        return {
            "available": False,
            "error": f"Model dependencies are not available: {error}",
        }

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(resolved_model_path, map_location=device)
        counts = [int(value) for value in checkpoint["counts"]]
        if checkpoint.get("architecture") != "mobilenet_v3_small":
            return {
                "available": False,
                "error": f"Unsupported model architecture: {checkpoint.get('architecture')}",
            }

        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, len(counts))
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1).squeeze(0)
            values, indexes = torch.topk(probabilities, k=min(3, len(counts)))

        predictions = [
            {
                "count": counts[int(index.item())],
                "confidence": round(float(value.item()), 4),
            }
            for value, index in zip(values, indexes)
        ]
        top_prediction = predictions[0] if predictions else {"count": 0, "confidence": 0}
        return {
            "available": True,
            "count": int(top_prediction["count"]),
            "confidence": float(top_prediction["confidence"]),
            "predictions": predictions,
            "modelPath": str(resolved_model_path),
        }
    except Exception as error:
        return {
            "available": False,
            "error": str(error),
            "modelPath": str(resolved_model_path),
        }


def _fuse_counts(
    image_count: int,
    image_confidence: float,
    model_prediction: dict[str, Any],
    detector_prediction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        model_confidence_threshold = float(os.getenv("INTAKE_COUNT_MODEL_CONFIDENCE", "0.70"))
    except ValueError:
        model_confidence_threshold = 0.70

    if detector_prediction and detector_prediction.get("available"):
        detector_count = int(detector_prediction.get("count") or 0)
        detector_confidence = float(detector_prediction.get("confidence") or 0)
        if detector_count > 0:
            model_count = int(model_prediction.get("count") or 0)
            model_confidence = float(model_prediction.get("confidence") or 0)
            model_is_confident = bool(model_prediction.get("available")) and model_confidence >= model_confidence_threshold
            if (
                image_count > 0
                and image_count != detector_count
                and model_is_confident
                and model_count == image_count
            ):
                return {
                    "count": image_count,
                    "confidence": round(float((image_confidence + model_confidence) / 2), 4),
                    "countSource": "hybrid-agreement-detector-disagreement",
                    "modelAccepted": True,
                    "modelAgreement": True,
                    "modelConfidenceThreshold": model_confidence_threshold,
                    "detectorAccepted": False,
                    "detectorMissed": False,
                }
            if image_count > 0 and image_count != detector_count and detector_confidence < 0.6 and image_confidence >= 0.35:
                return {
                    "count": image_count,
                    "confidence": image_confidence,
                    "countSource": "image-processing-detector-disagreement",
                    "modelAccepted": False,
                    "modelAgreement": None,
                    "modelConfidenceThreshold": model_confidence_threshold,
                    "detectorAccepted": False,
                    "detectorMissed": False,
                }
            return {
                "count": detector_count,
                "confidence": detector_confidence,
                "countSource": "pill-detector",
                "modelAccepted": False,
                "modelAgreement": None,
                "modelConfidenceThreshold": model_confidence_threshold,
                "detectorAccepted": True,
                "detectorMissed": False,
            }

    if not model_prediction.get("available"):
        return {
            "count": image_count,
            "confidence": image_confidence,
            "countSource": "image-processing",
            "modelAccepted": False,
            "modelAgreement": None,
            "modelConfidenceThreshold": model_confidence_threshold,
            "detectorAccepted": False,
            "detectorMissed": bool(detector_prediction and detector_prediction.get("available")),
        }

    model_count = int(model_prediction.get("count") or 0)
    model_confidence = float(model_prediction.get("confidence") or 0)
    model_agrees = model_count == image_count
    if model_confidence < model_confidence_threshold:
        return {
            "count": image_count,
            "confidence": image_confidence,
            "countSource": "image-processing-model-low-confidence",
            "modelAccepted": False,
            "modelAgreement": model_agrees,
            "modelConfidenceThreshold": model_confidence_threshold,
            "detectorAccepted": False,
            "detectorMissed": bool(detector_prediction and detector_prediction.get("available")),
        }

    if model_agrees:
        return {
            "count": model_count,
            "confidence": round(float((image_confidence + model_confidence) / 2), 4),
            "countSource": "hybrid-agreement",
            "modelAccepted": True,
            "modelAgreement": True,
            "modelConfidenceThreshold": model_confidence_threshold,
            "detectorAccepted": False,
            "detectorMissed": bool(detector_prediction and detector_prediction.get("available")),
        }

    return {
        "count": model_count,
        "confidence": model_confidence,
        "countSource": "ai-model",
        "modelAccepted": True,
        "modelAgreement": False,
        "modelConfidenceThreshold": model_confidence_threshold,
        "detectorAccepted": False,
        "detectorMissed": bool(detector_prediction and detector_prediction.get("available")),
    }


def analyze(image_base64: str, expected_count: float | None = None) -> dict[str, Any]:
    image = _decode_image(image_base64)
    original_image = image.copy()
    image.thumbnail((520, 520))
    arr = np.asarray(image).astype(np.float32)
    height, width, _ = arr.shape

    border = np.concatenate(
        [
            arr[: max(2, height // 18), :, :].reshape(-1, 3),
            arr[-max(2, height // 18) :, :, :].reshape(-1, 3),
            arr[:, : max(2, width // 18), :].reshape(-1, 3),
            arr[:, -max(2, width // 18) :, :].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(border, axis=0)
    distance = np.linalg.norm(arr - background, axis=2)
    brightness = arr.mean(axis=2)
    saturation = arr.max(axis=2) - arr.min(axis=2)
    background_brightness = float(np.mean(background))
    median_brightness = float(np.median(brightness))
    median_saturation = float(np.median(saturation))
    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]

    skin_like_mask = (
        (red > 65)
        & (green > 35)
        & (blue > 20)
        & (red >= green * 0.88)
        & (red >= blue * 1.05)
        & ((red - blue) > 8)
        & (saturation > 12)
    )

    bright_threshold = max(background_brightness + 18, median_brightness + 20, float(np.percentile(brightness, 72)))
    dark_threshold = min(background_brightness - 28, median_brightness - 25, float(np.percentile(brightness, 28)))
    saturation_threshold = max(42.0, median_saturation + 18.0)

    bright_tablet_mask = (brightness > bright_threshold) & (saturation < 72) & (brightness > 105)
    dark_tablet_mask = (brightness < dark_threshold) & (distance > 26) & (brightness > 18) & (saturation < 105)
    colored_tablet_mask = (saturation > saturation_threshold) & (distance > 24) & (brightness > 38) & (brightness < 248)
    non_skin_object_mask = (distance > 42) & (~skin_like_mask) & (brightness > 32) & (brightness < 252)
    mask = bright_tablet_mask | dark_tablet_mask | colored_tablet_mask | non_skin_object_mask

    margin_y = max(1, height // 60)
    margin_x = max(1, width // 60)
    usable_area = np.zeros_like(mask, dtype=bool)
    usable_area[margin_y : height - margin_y, margin_x : width - margin_x] = True
    mask = mask & usable_area

    mask = _clean_mask(mask)
    legacy_components = _find_components(mask, width, height)
    contour_components = _find_components_cv2(arr)
    components = []
    for component in [*legacy_components, *contour_components]:
        if any(_component_iou(component, existing) > 0.35 for existing in components):
            continue
        components.append(component)
    components.sort(key=lambda item: float(item.get("confidence") or 0), reverse=True)
    components = components[:12]
    image_processing_count, estimated_components, image_processing_confidence = _component_count_estimate(
        components,
        width * height,
        expected_count,
    )
    detector_prediction = _predict_with_detector(original_image, _resolve_detector_model_path())
    model_prediction = _predict_with_trained_model(original_image, _resolve_model_path())
    fused_count = _fuse_counts(image_processing_count, image_processing_confidence, model_prediction, detector_prediction)

    result: dict[str, Any] = {
        "detectedCount": fused_count["count"],
        "confidence": fused_count["confidence"],
        "countSource": fused_count["countSource"],
        "objects": estimated_components,
        "imageProcessing": {
            "detectedCount": image_processing_count,
            "rawComponentCount": len(components),
            "legacyComponentCount": len(legacy_components),
            "contourComponentCount": len(contour_components),
            "confidence": image_processing_confidence,
            "maskCoverage": round(float(mask.mean()), 4),
            "backgroundRgb": [int(round(value)) for value in background.tolist()],
            "thresholds": {
                "bright": round(float(bright_threshold), 2),
                "dark": round(float(dark_threshold), 2),
                "saturation": round(float(saturation_threshold), 2),
            },
        },
        "modelAnalysis": {
            **model_prediction,
            "accepted": fused_count["modelAccepted"],
            "agreement": fused_count["modelAgreement"],
            "confidenceThreshold": fused_count["modelConfidenceThreshold"],
        },
        "detectorAnalysis": {
            **detector_prediction,
            "accepted": fused_count.get("detectorAccepted", False),
            "missed": fused_count.get("detectorMissed", False),
        },
    }
    result.update(_build_count_status(fused_count["count"], expected_count))

    return result


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        expected_raw = payload.get("expectedCount")
        expected_count = None
        if expected_raw is not None:
            expected_count = float(expected_raw)
        result = analyze(str(payload.get("imageBase64") or ""), expected_count)
        print(json.dumps(result))
    except Exception as error:
        print(json.dumps({"error": str(error), "status": "uncertain", "detectedCount": 0}))
        sys.exit(1)


if __name__ == "__main__":
    main()
