from __future__ import annotations

import base64
import json
import math
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


COLOR_REFERENCES = {
    "white": (238, 238, 230),
    "yellow": (235, 204, 70),
    "orange": (224, 130, 45),
    "red": (202, 65, 65),
    "pink": (235, 120, 165),
    "purple": (150, 90, 180),
    "blue": (70, 135, 210),
    "green": (80, 170, 105),
    "brown": (125, 85, 55),
    "gray": (145, 150, 150),
    "grey": (145, 150, 150),
    "black": (45, 45, 45),
}


SHAPE_ALIASES = {
    "circle": "round",
    "circular": "round",
    "round": "round",
    "ellipse": "oval",
    "elliptical": "oval",
    "oval": "oval",
    "capsule": "capsule",
    "oblong": "capsule",
    "tablet": "square",
    "square": "square",
    "diamond": "diamond",
    "triangle": "triangle",
}


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    except Exception:
        pass
    return fallback


def _decode_image(image_base64: str) -> Image.Image:
    value = str(image_base64 or "").strip()
    if "," in value and value.lower().startswith("data:"):
        value = value.split(",", 1)[1]
    image_bytes = base64.b64decode(value, validate=False)
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def _resolve_detector_model_path() -> str:
    configured_path = os.getenv("TABLET_IDENTITY_DETECTOR_MODEL_PATH", "").strip()
    if configured_path:
        return configured_path

    configured_intake_path = os.getenv("INTAKE_DETECTOR_MODEL_PATH", "").strip()
    if configured_intake_path:
        return configured_intake_path

    backend_dir = Path(__file__).resolve().parents[1]
    for candidate in (
        backend_dir / "ml" / "models" / "pill_detector.pt",
        backend_dir / "ml" / "runs" / "detect" / "pill_detector" / "weights" / "best.pt",
    ):
        if candidate.exists():
            return str(candidate)

    return ""


def _detect_pill_objects(image: Image.Image, model_path: str) -> dict[str, Any]:
    if not model_path:
        return {
            "available": False,
            "error": "No trained pill detector model found.",
            "objects": [],
        }

    resolved_model_path = Path(model_path).resolve()
    if not resolved_model_path.exists():
        return {
            "available": False,
            "error": f"Detector model file not found: {resolved_model_path}",
            "objects": [],
        }

    try:
        os.environ.setdefault("YOLO_CONFIG_DIR", str((Path(__file__).resolve().parents[1] / "ml" / ".ultralytics").resolve()))
        from ultralytics import YOLO
    except Exception as error:
        return {
            "available": False,
            "error": f"Detector dependencies are not available: {error}",
            "modelPath": str(resolved_model_path),
            "objects": [],
        }

    try:
        confidence_threshold = _safe_float(os.getenv("TABLET_IDENTITY_DETECTOR_CONFIDENCE", os.getenv("INTAKE_DETECTOR_CONFIDENCE", "0.45")), 0.45)
        model = YOLO(str(resolved_model_path))
        result = model.predict(
            np.asarray(image),
            imgsz=416,
            conf=confidence_threshold,
            iou=0.45,
            max_det=8,
            verbose=False,
            device="cpu",
        )[0]
        boxes = result.boxes
        width, height = image.size
        objects: list[dict[str, Any]] = []

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            for box, confidence in zip(xyxy, confidences):
                x1, y1, x2, y2 = [float(value) for value in box]
                box_width = max(0.0, x2 - x1)
                box_height = max(0.0, y2 - y1)
                if box_width <= 1 or box_height <= 1:
                    continue
                objects.append(
                    {
                        "x": round(max(0.0, x1) / max(1, width), 4),
                        "y": round(max(0.0, y1) / max(1, height), 4),
                        "width": round(box_width / max(1, width), 4),
                        "height": round(box_height / max(1, height), 4),
                        "confidence": round(float(confidence), 4),
                    }
                )

        objects.sort(key=lambda item: float(item.get("confidence") or 0), reverse=True)
        return {
            "available": True,
            "objects": objects,
            "modelPath": str(resolved_model_path),
            "confidenceThreshold": confidence_threshold,
        }
    except Exception as error:
        return {
            "available": False,
            "error": str(error),
            "modelPath": str(resolved_model_path),
            "objects": [],
        }


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
    cleaned = _neighbor_count(mask.astype(bool)) >= 3
    cleaned = cleaned | (_neighbor_count(cleaned) >= 7)
    return cleaned


def _mask_from_box(width: int, height: int, box: dict[str, Any], pad_ratio: float = 0.06) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    x = _safe_float(box.get("x"))
    y = _safe_float(box.get("y"))
    box_width = _safe_float(box.get("width"), 1.0)
    box_height = _safe_float(box.get("height"), 1.0)
    pad_x = box_width * pad_ratio
    pad_y = box_height * pad_ratio
    x1 = max(0, int(round((x - pad_x) * width)))
    y1 = max(0, int(round((y - pad_y) * height)))
    x2 = min(width, int(round((x + box_width + pad_x) * width)))
    y2 = min(height, int(round((y + box_height + pad_y) * height)))
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def _box_pixel_bounds(width: int, height: int, box: dict[str, Any], pad_ratio: float = 0.0) -> tuple[int, int, int, int] | None:
    x = _safe_float(box.get("x"))
    y = _safe_float(box.get("y"))
    box_width = _safe_float(box.get("width"), 1.0)
    box_height = _safe_float(box.get("height"), 1.0)
    pad_x = box_width * pad_ratio
    pad_y = box_height * pad_ratio
    x1 = max(0, int(round((x - pad_x) * width)))
    y1 = max(0, int(round((y - pad_y) * height)))
    x2 = min(width, int(round((x + box_width + pad_x) * width)))
    y2 = min(height, int(round((y + box_height + pad_y) * height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _component_from_mask(mask: np.ndarray, source: str, confidence: Any = None) -> dict[str, Any] | None:
    if not mask.any():
        return None
    height, width = mask.shape
    x_values = np.where(mask.any(axis=0))[0]
    y_values = np.where(mask.any(axis=1))[0]
    if len(x_values) == 0 or len(y_values) == 0:
        return None
    x1, x2 = int(x_values.min()), int(x_values.max())
    y1, y2 = int(y_values.min()), int(y_values.max())
    box_width = max(1, x2 - x1 + 1)
    box_height = max(1, y2 - y1 + 1)
    area = int(mask.sum())
    bbox_area = max(1, box_width * box_height)
    aspect = max(box_width / box_height, box_height / box_width)
    perimeter = _estimate_perimeter(mask)
    circularity = (4.0 * math.pi * area) / max(1.0, float(perimeter * perimeter))
    fill_ratio = area / bbox_area
    vertices = 0
    solidity = fill_ratio
    ellipse_aspect = None

    try:
        import cv2

        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            contour_area = max(1.0, float(cv2.contourArea(contour)))
            perimeter_cv = max(1.0, float(cv2.arcLength(contour, True)))
            circularity = (4.0 * math.pi * contour_area) / max(1.0, perimeter_cv * perimeter_cv)
            hull = cv2.convexHull(contour)
            solidity = contour_area / max(1.0, float(cv2.contourArea(hull)))
            approx = cv2.approxPolyDP(contour, 0.035 * perimeter_cv, True)
            vertices = int(len(approx))
            if len(contour) >= 5:
                (_, _), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
                ellipse_aspect = max(axis_a, axis_b) / max(1.0, min(axis_a, axis_b))
    except Exception:
        pass

    return {
        "area": area,
        "x": round(x1 / width, 4),
        "y": round(y1 / height, 4),
        "width": round(box_width / width, 4),
        "height": round(box_height / height, 4),
        "aspect": round(float(aspect), 4),
        "fillRatio": round(float(fill_ratio), 4),
        "circularity": round(float(circularity), 4),
        "solidity": round(float(solidity), 4),
        "ellipseAspect": round(float(ellipse_aspect), 4) if ellipse_aspect else None,
        "vertices": vertices,
        "mask": mask,
        "source": source,
        "confidence": confidence,
    }


def _component_from_box(width: int, height: int, box: dict[str, Any]) -> dict[str, Any] | None:
    bounds = _box_pixel_bounds(width, height, box, pad_ratio=0.0)
    if bounds is None:
        return None
    x1, y1, x2, y2 = bounds
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    mask = np.zeros((height, width), dtype=bool)
    yy, xx = np.ogrid[y1:y2, x1:x2]
    center_x = x1 + box_width / 2.0
    center_y = y1 + box_height / 2.0
    radius_x = max(1.0, box_width * 0.43)
    radius_y = max(1.0, box_height * 0.43)
    inner = (((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2) <= 1.0
    mask[y1:y2, x1:x2] = inner
    component = _component_from_mask(mask, "detector-box-inner-object", box.get("confidence"))
    if component is not None:
        component["objectPixelsOnly"] = True
    return component


def _center_score(x: int, y: int, box_width: int, box_height: int, width: int, height: int) -> float:
    center_x = x + box_width / 2.0
    center_y = y + box_height / 2.0
    distance = math.sqrt(((center_x / max(1, width)) - 0.5) ** 2 + ((center_y / max(1, height)) - 0.5) ** 2)
    return max(0.0, 1.0 - min(1.0, distance * 2.0))


def _refine_component_inside_box(arr: np.ndarray, box: dict[str, Any]) -> dict[str, Any] | None:
    try:
        import cv2
    except Exception:
        return None

    height, width, _ = arr.shape
    bounds = _box_pixel_bounds(width, height, box, pad_ratio=0.02)
    if bounds is None:
        return None

    x1, y1, x2, y2 = bounds
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    if box_width < 10 or box_height < 10:
        return None

    rgb_u8 = np.clip(arr, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    grab_mask = np.zeros((height, width), np.uint8)
    rect = (x1, y1, box_width, box_height)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(bgr, grab_mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return None

    foreground = (grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD)
    box_mask = np.zeros((height, width), dtype=bool)
    box_mask[y1:y2, x1:x2] = True
    foreground &= box_mask

    kernel_size = max(3, int(round(min(box_width, box_height) * 0.04)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_u8 = foreground.astype(np.uint8) * 255
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    component = _largest_component(mask_u8.astype(bool))
    if component is None:
        return None

    box_area = box_width * box_height
    area = int(component.get("area") or 0)
    if area < max(25, int(box_area * 0.12)) or area > int(box_area * 0.96):
        return None

    component["source"] = "detector-box-grabcut-object"
    component["confidence"] = box.get("confidence")
    component["objectPixelsOnly"] = True
    return component


def _object_component_no_background_cv2(arr: np.ndarray, allowed_mask: np.ndarray | None = None) -> dict[str, Any] | None:
    try:
        import cv2
    except Exception:
        return None

    height, width, _ = arr.shape
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
    edge_kernel = np.ones((5, 5), np.uint8)
    masks.append(cv2.dilate(edges, edge_kernel, iterations=1))

    if allowed_mask is not None:
        allowed_u8 = allowed_mask.astype(np.uint8) * 255
    else:
        allowed_u8 = np.ones((height, width), dtype=np.uint8) * 255

    best: dict[str, Any] | None = None
    best_score = -1.0
    image_area = max(1, width * height)
    min_area = max(35, int(image_area * 0.00035))
    max_area = max(min_area + 1, int(image_area * 0.86))
    morph_kernel = np.ones((5, 5), np.uint8)

    for raw_mask in masks:
        mask_u8 = cv2.bitwise_and(raw_mask, allowed_u8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, morph_kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, morph_kernel)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area or area > max_area:
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_width < 5 or box_height < 5:
                continue
            bbox_area = max(1, box_width * box_height)
            fill_ratio = area / bbox_area
            aspect = max(box_width / max(1, box_height), box_height / max(1, box_width))
            if fill_ratio < 0.12 or aspect > 6.5:
                continue

            perimeter = max(1.0, float(cv2.arcLength(contour, True)))
            circularity = (4.0 * math.pi * area) / max(1.0, perimeter * perimeter)
            hull = cv2.convexHull(contour)
            hull_area = max(1.0, float(cv2.contourArea(hull)))
            solidity = area / hull_area
            approx = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
            contour_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
            contour_mask = cv2.bitwise_and(contour_mask, allowed_u8)

            score = (
                min(1.0, area / max(min_area * 16, 1))
                + min(1.0, fill_ratio)
                + min(1.0, solidity)
                + _center_score(x, y, box_width, box_height, width, height)
                - max(0.0, aspect - 3.5) * 0.12
            )
            if score <= best_score:
                continue

            angle = 0.0
            ellipse_aspect = None
            if len(contour) >= 5:
                (_, _), (axis_a, axis_b), raw_angle = cv2.fitEllipse(contour)
                angle = float(raw_angle)
                ellipse_aspect = max(axis_a, axis_b) / max(1.0, min(axis_a, axis_b))

            best_score = score
            best = {
                "area": int(round(area)),
                "x": round(x / width, 4),
                "y": round(y / height, 4),
                "width": round(box_width / width, 4),
                "height": round(box_height / height, 4),
                "aspect": round(float(aspect), 4),
                "fillRatio": round(float(fill_ratio), 4),
                "circularity": round(float(circularity), 4),
                "solidity": round(float(solidity), 4),
                "ellipseAspect": round(float(ellipse_aspect), 4) if ellipse_aspect else None,
                "vertices": int(len(approx)),
                "angle": round(float(angle), 2),
                "mask": contour_mask.astype(bool),
                "source": "object-contour",
                "objectScore": round(float(score), 4),
            }

    return best


def _object_component_no_background(arr: np.ndarray, allowed_mask: np.ndarray | None = None) -> dict[str, Any] | None:
    component = _object_component_no_background_cv2(arr, allowed_mask)
    if component is not None:
        return component

    brightness = arr.mean(axis=2)
    saturation = arr.max(axis=2) - arr.min(axis=2)
    bright_mask = brightness > np.percentile(brightness, 72)
    dark_mask = brightness < np.percentile(brightness, 28)
    color_mask = saturation > max(22, np.percentile(saturation, 68))
    mask = (bright_mask | dark_mask | color_mask)
    if allowed_mask is not None:
        mask = mask & allowed_mask
    mask = _clean_mask(mask)
    component = _largest_component(mask)
    if component is not None:
        component["source"] = "object-percentile"
    return component


def _object_components_no_background_cv2(arr: np.ndarray, allowed_mask: np.ndarray | None = None, max_components: int = 12) -> list[dict[str, Any]]:
    try:
        import cv2
    except Exception:
        return []

    height, width, _ = arr.shape
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

    if allowed_mask is not None:
        allowed_u8 = allowed_mask.astype(np.uint8) * 255
    else:
        allowed_u8 = np.ones((height, width), dtype=np.uint8) * 255

    image_area = max(1, width * height)
    min_area = max(35, int(image_area * 0.00035))
    max_area = max(min_area + 1, int(image_area * 0.22))
    morph_kernel = np.ones((5, 5), np.uint8)
    candidates: list[dict[str, Any]] = []

    for raw_mask in masks:
        mask_u8 = cv2.bitwise_and(raw_mask, allowed_u8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, morph_kernel)
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
            circularity = (4.0 * math.pi * area) / max(1.0, perimeter * perimeter)
            hull = cv2.convexHull(contour)
            hull_area = max(1.0, float(cv2.contourArea(hull)))
            solidity = area / hull_area
            if solidity < 0.45:
                continue

            approx = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
            contour_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
            contour_mask = cv2.bitwise_and(contour_mask, allowed_u8)

            angle = 0.0
            ellipse_aspect = None
            if len(contour) >= 5:
                (_, _), (axis_a, axis_b), raw_angle = cv2.fitEllipse(contour)
                angle = float(raw_angle)
                ellipse_aspect = max(axis_a, axis_b) / max(1.0, min(axis_a, axis_b))

            score = (
                min(1.0, area / max(min_area * 16, 1))
                + min(1.0, fill_ratio)
                + min(1.0, solidity)
                + _center_score(x, y, box_width, box_height, width, height)
                - max(0.0, aspect - 3.5) * 0.12
            )

            candidates.append(
                {
                    "area": int(round(area)),
                    "x": round(x / width, 4),
                    "y": round(y / height, 4),
                    "width": round(box_width / width, 4),
                    "height": round(box_height / height, 4),
                    "aspect": round(float(aspect), 4),
                    "fillRatio": round(float(fill_ratio), 4),
                    "circularity": round(float(circularity), 4),
                    "solidity": round(float(solidity), 4),
                    "ellipseAspect": round(float(ellipse_aspect), 4) if ellipse_aspect else None,
                    "vertices": int(len(approx)),
                    "angle": round(float(angle), 2),
                    "mask": contour_mask.astype(bool),
                    "source": "multi-object-contour",
                    "objectScore": round(float(score), 4),
                }
            )

    candidates.sort(key=lambda item: float(item.get("objectScore") or 0), reverse=True)
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        if any(_component_iou(candidate, existing) > 0.35 for existing in selected):
            continue
        selected.append(candidate)
        if len(selected) >= max_components:
            break

    selected.sort(key=lambda item: (float(item.get("y") or 0), float(item.get("x") or 0)))
    return selected


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


def _largest_component_cv2(mask: np.ndarray) -> dict[str, Any] | None:
    try:
        import cv2
    except Exception:
        return None

    height, width = mask.shape
    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((5, 5), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = max(1, width * height)
    min_area = max(45, int(image_area * 0.00045))
    best: dict[str, Any] | None = None

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue

        x, y, box_width, box_height = cv2.boundingRect(contour)
        if x <= 1 or y <= 1 or x + box_width >= width - 1 or y + box_height >= height - 1:
            continue

        bbox_area = max(1, box_width * box_height)
        fill_ratio = area / bbox_area
        aspect = max(box_width / max(1, box_height), box_height / max(1, box_width))
        if fill_ratio < 0.16 or aspect > 6.0:
            continue

        perimeter = max(1.0, float(cv2.arcLength(contour, True)))
        circularity = (4.0 * math.pi * area) / max(1.0, perimeter * perimeter)
        hull = cv2.convexHull(contour)
        hull_area = max(1.0, float(cv2.contourArea(hull)))
        solidity = area / hull_area
        approx = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
        component_mask_u8 = np.zeros_like(mask_u8)
        cv2.drawContours(component_mask_u8, [contour], -1, 255, thickness=-1)

        angle = 0.0
        ellipse_aspect = None
        if len(contour) >= 5:
            (_, _), (axis_a, axis_b), raw_angle = cv2.fitEllipse(contour)
            angle = float(raw_angle)
            ellipse_aspect = max(axis_a, axis_b) / max(1.0, min(axis_a, axis_b))

        candidate = {
            "area": int(round(area)),
            "x": round(x / width, 4),
            "y": round(y / height, 4),
            "width": round(box_width / width, 4),
            "height": round(box_height / height, 4),
            "aspect": round(float(aspect), 4),
            "fillRatio": round(float(fill_ratio), 4),
            "circularity": round(float(circularity), 4),
            "solidity": round(float(solidity), 4),
            "ellipseAspect": round(float(ellipse_aspect), 4) if ellipse_aspect else None,
            "vertices": int(len(approx)),
            "angle": round(float(angle), 2),
            "mask": component_mask_u8.astype(bool),
        }

        if best is None or candidate["area"] > best["area"]:
            best = candidate

    return best


def _estimate_perimeter(component_mask: np.ndarray) -> int:
    padded = np.pad(component_mask.astype(bool), 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    boundary = center & (~up | ~down | ~left | ~right)
    return int(boundary.sum())


def _largest_component(mask: np.ndarray) -> dict[str, Any] | None:
    cv2_component = _largest_component_cv2(mask)
    if cv2_component is not None:
        return cv2_component

    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best: dict[str, Any] | None = None
    min_area = max(45, int(width * height * 0.00045))

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
            if x <= 1 or y <= 1 or x >= width - 2 or y >= height - 2:
                touches_border = True

            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if ny < 0 or nx < 0 or ny >= height or nx >= width:
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
        bbox_area = box_width * box_height
        fill_ratio = area / max(1, bbox_area)
        aspect = max(box_width / box_height, box_height / box_width)

        if fill_ratio < 0.18 or aspect > 6.0:
            continue

        component_mask = np.zeros_like(mask, dtype=bool)
        component_mask[np.asarray(ys), np.asarray(xs)] = True
        perimeter = _estimate_perimeter(component_mask)
        circularity = (4.0 * math.pi * area) / max(1.0, float(perimeter * perimeter))

        candidate = {
            "area": int(area),
            "x": round(x1 / width, 4),
            "y": round(y1 / height, 4),
            "width": round(box_width / width, 4),
            "height": round(box_height / height, 4),
            "aspect": round(float(aspect), 4),
            "fillRatio": round(float(fill_ratio), 4),
            "circularity": round(float(circularity), 4),
            "solidity": round(float(fill_ratio), 4),
            "vertices": 0,
            "mask": component_mask,
        }

        if best is None or candidate["area"] > best["area"]:
            best = candidate

    return best


def _rgb_to_hsv(rgb: np.ndarray) -> tuple[float, float, float]:
    values = [float(value) / 255.0 for value in rgb]
    r, g, b = values
    max_v = max(values)
    min_v = min(values)
    delta = max_v - min_v
    if delta == 0:
        hue = 0.0
    elif max_v == r:
        hue = (60 * ((g - b) / delta) + 360) % 360
    elif max_v == g:
        hue = 60 * ((b - r) / delta + 2)
    else:
        hue = 60 * ((r - g) / delta + 4)
    saturation = 0.0 if max_v == 0 else delta / max_v
    return hue, saturation, max_v


def _erode_object_mask(mask: np.ndarray) -> np.ndarray:
    try:
        import cv2

        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return mask
        object_size = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        kernel_size = max(3, int(round(object_size * 0.035)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        if eroded.sum() >= max(12, int(mask.sum() * 0.35)):
            return eroded
    except Exception:
        pass

    neighbors = _neighbor_count(mask.astype(bool))
    eroded = mask & (neighbors >= 9)
    if eroded.sum() >= max(12, int(mask.sum() * 0.35)):
        return eroded
    return mask


def _pill_object_color(arr: np.ndarray, component: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    mask = component["mask"].astype(bool)
    inner_mask = _erode_object_mask(mask)
    pixels = arr[inner_mask]
    if len(pixels) < 12:
        inner_mask = mask
        pixels = arr[inner_mask]

    if len(pixels) == 0:
        return np.asarray([0.0, 0.0, 0.0]), {
            "pixelCount": 0,
            "innerPixelCount": 0,
            "usedInnerMask": False,
            "backgroundExcluded": True,
        }

    brightness = pixels.mean(axis=1)
    lower = np.percentile(brightness, 8)
    upper = np.percentile(brightness, 92)
    filtered = pixels[(brightness >= lower) & (brightness <= upper)]
    if len(filtered) >= max(12, int(len(pixels) * 0.45)):
        pixels = filtered

    median_rgb = np.median(pixels, axis=0)
    distances = np.linalg.norm(pixels - median_rgb, axis=1)
    cutoff = np.percentile(distances, 82)
    stable_pixels = pixels[distances <= cutoff]
    if len(stable_pixels) >= max(12, int(len(pixels) * 0.45)):
        pixels = stable_pixels
        median_rgb = np.median(pixels, axis=0)

    return median_rgb, {
        "pixelCount": int(mask.sum()),
        "innerPixelCount": int(inner_mask.sum()),
        "colorPixelCount": int(len(pixels)),
        "usedInnerMask": bool(inner_mask is not mask and inner_mask.sum() < mask.sum()),
        "backgroundExcluded": True,
    }


def _classify_color(rgb: np.ndarray) -> tuple[str, float]:
    rgb_tuple = tuple(float(value) for value in rgb)
    brightness = sum(rgb_tuple) / 3.0
    saturation = max(rgb_tuple) - min(rgb_tuple)
    hue, hsv_saturation, value = _rgb_to_hsv(rgb)

    if value < 0.22:
        return "black", 0.82
    if value > 0.78 and hsv_saturation < 0.16:
        return "white", 0.86
    if hsv_saturation < 0.16:
        return "gray", 0.72

    if hue < 14 or hue >= 345:
        return "red", 0.78
    if hue < 28:
        return "orange", 0.78
    if hue < 58:
        return "yellow", 0.8
    if hue < 165:
        return "green", 0.78
    if hue < 255:
        return "blue", 0.78
    if hue < 292:
        return "purple", 0.74
    if hue < 345:
        return "pink", 0.72

    best_name = "unknown"
    best_distance = float("inf")
    for name, reference in COLOR_REFERENCES.items():
        distance = math.sqrt(sum((rgb_tuple[index] - reference[index]) ** 2 for index in range(3)))
        if distance < best_distance:
            best_name = name
            best_distance = distance

    confidence = max(0.25, min(0.92, 1.0 - (best_distance / 260.0)))
    return best_name, round(float(confidence), 4)


def _classify_shape(component: dict[str, Any]) -> tuple[str, float]:
    aspect = float(component.get("aspect") or 1)
    ellipse_aspect = float(component.get("ellipseAspect") or 0)
    shape_aspect = max(aspect, ellipse_aspect or 1.0)
    fill_ratio = float(component.get("fillRatio") or 0)
    circularity = float(component.get("circularity") or 0)
    solidity = float(component.get("solidity") or fill_ratio)
    vertices = int(component.get("vertices") or 0)

    if vertices == 3:
        return "triangle", 0.82
    if vertices == 4 and shape_aspect <= 1.35 and fill_ratio < 0.78:
        return "diamond", 0.72
    if shape_aspect >= 2.2:
        return "capsule", round(min(0.94, 0.54 + ((shape_aspect - 2.0) * 0.16) + (solidity * 0.12)), 4)
    if shape_aspect >= 1.34:
        return "oval", round(min(0.93, 0.48 + min(shape_aspect, 2.0) * 0.18 + solidity * 0.08), 4)
    if shape_aspect >= 1.18 and circularity < 0.84:
        return "oval", round(min(0.9, 0.42 + min(shape_aspect, 1.5) * 0.2 + solidity * 0.08), 4)
    if circularity >= 0.68 and shape_aspect <= 1.18:
        round_confidence = 0.5 + circularity * 0.42 + max(0.0, 1.16 - shape_aspect) * 0.18
        return "round", round(min(0.94, round_confidence), 4)
    if shape_aspect >= 1.2:
        return "oval", round(min(0.86, 0.42 + min(shape_aspect, 1.8) * 0.18 + solidity * 0.08), 4)
    if fill_ratio < 0.52:
        return "triangle", 0.5
    if circularity >= 0.62:
        round_confidence = 0.46 + circularity * 0.36 + max(0.0, 1.12 - shape_aspect) * 0.12
        return "round", round(min(0.86, round_confidence), 4)
    return "square", round(min(0.82, 0.42 + fill_ratio * 0.34 + solidity * 0.08), 4)


def _crop_component(image: Image.Image, component: dict[str, Any]) -> Image.Image:
    width, height = image.size
    x1 = int(float(component.get("x") or 0) * width)
    y1 = int(float(component.get("y") or 0) * height)
    x2 = int((float(component.get("x") or 0) + float(component.get("width") or 1)) * width)
    y2 = int((float(component.get("y") or 0) + float(component.get("height") or 1)) * height)
    pad = max(4, int(max(x2 - x1, y2 - y1) * 0.08))
    return image.crop((max(0, x1 - pad), max(0, y1 - pad), min(width, x2 + pad), min(height, y2 + pad)))


def _average_hash(image: Image.Image, hash_size: int = 8) -> str:
    gray = image.convert("L").resize((hash_size, hash_size))
    arr = np.asarray(gray).astype(np.float32)
    threshold = float(arr.mean())
    bits = arr > threshold
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bool(bit))
    return f"{value:0{hash_size * hash_size // 4}x}"


def _visual_features(image: Image.Image) -> dict[str, Any]:
    gray = np.asarray(image.convert("L").filter(ImageFilter.FIND_EDGES)).astype(np.float32)
    edge_density = float((gray > max(18.0, float(gray.mean()) + float(gray.std()))).mean())
    return {
        "averageHash": _average_hash(image),
        "edgeDensity": round(edge_density, 4),
        "hasVisibleDetail": edge_density >= 0.035,
    }


def _normalize_shape(value: Any) -> str:
    return SHAPE_ALIASES.get(str(value or "").strip().lower(), str(value or "").strip().lower())


def _normalize_color(value: Any) -> str:
    color = str(value or "").strip().lower()
    return "gray" if color == "grey" else color


def _normalize_name(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _resolve_identity_model_path() -> str:
    configured_path = os.getenv("TABLET_IDENTITY_MODEL_PATH", "").strip()
    if configured_path:
        return configured_path

    model_dir = Path(__file__).resolve().parents[1] / "ml" / "models"
    candidate = model_dir / "tablet_identity_classifier.pt"
    return str(candidate) if candidate.exists() else ""


def _resolve_identity_index_path() -> str:
    configured_path = os.getenv("TABLET_IDENTITY_INDEX_PATH", "").strip()
    if configured_path:
        return configured_path

    model_dir = Path(__file__).resolve().parents[1] / "ml" / "models"
    for candidate in (
        model_dir / "pillbox_tablet_identity_index.npz",
        model_dir / "tablet_identity_index.npz",
    ):
        if candidate.exists():
            return str(candidate)
    return ""


def _index_scalar_value(index: Any, key: str, fallback: str = "") -> str:
    if key not in index:
        return fallback
    value = index[key]
    return str(value.item() if getattr(value, "shape", ()) == () else value)


def _resolve_identity_metric_model_path(index: Any | None = None) -> str:
    configured_path = os.getenv("TABLET_IDENTITY_METRIC_MODEL_PATH", "").strip()
    if configured_path:
        return configured_path

    if index is not None:
        indexed_path = _index_scalar_value(index, "metric_model_path", "").strip()
        if indexed_path and Path(indexed_path).exists():
            return indexed_path

    model_dir = Path(__file__).resolve().parents[1] / "ml" / "models"
    candidate = model_dir / "tablet_metric_identity_model.pt"
    return str(candidate) if candidate.exists() else ""


def _build_feature_encoder(architecture: str, pretrained: bool = True):
    from torch import nn
    from torchvision import models

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


def _load_metric_embedding_model(model_path: str, device: Any):
    import torch
    import torch.nn.functional as functional
    from torch import nn

    class MetricIdentityRuntimeModel(nn.Module):
        def __init__(self, architecture: str, embedding_dim: int, class_count: int) -> None:
            super().__init__()
            self.encoder, feature_dim = _build_feature_encoder(architecture, pretrained=False)
            self.projector = nn.Sequential(
                nn.Linear(feature_dim, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.12),
                nn.Linear(embedding_dim, embedding_dim),
            )
            self.classifier = nn.Linear(embedding_dim, class_count)

        def forward(self, images):
            features = self.encoder(images)
            return functional.normalize(self.projector(features), p=2, dim=1)

    resolved_model_path = Path(model_path).resolve()
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Metric model file not found: {resolved_model_path}")

    checkpoint = torch.load(resolved_model_path, map_location=device)
    if checkpoint.get("task") != "tablet_metric_identity":
        raise ValueError(f"Unsupported metric model task: {checkpoint.get('task')}")

    labels = [str(value) for value in checkpoint["labels"]]
    architecture = str(checkpoint.get("architecture") or "efficientnet_b0")
    embedding_dim = int(checkpoint.get("embedding_dim") or 512)
    model = MetricIdentityRuntimeModel(architecture, embedding_dim, len(labels))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, architecture, str(resolved_model_path)


def _build_index_embedding_model(architecture: str, metric_model_path: str, device: Any):
    if metric_model_path:
        return _load_metric_embedding_model(metric_model_path, device)

    model, _ = _build_feature_encoder(architecture, pretrained=True)
    model.to(device)
    model.eval()
    return model, architecture, ""


def _predict_with_identity_index(image: Image.Image, index_path: str, top_k: int = 8) -> dict[str, Any]:
    if not index_path:
        return {
            "available": False,
            "error": "No tablet identity retrieval index found.",
        }

    resolved_index_path = Path(index_path).resolve()
    if not resolved_index_path.exists():
        return {
            "available": False,
            "error": f"Identity index file not found: {resolved_index_path}",
        }

    try:
        import torch
        from torchvision import transforms
    except Exception as error:
        return {
            "available": False,
            "error": f"Retrieval dependencies are not available: {error}",
            "indexPath": str(resolved_index_path),
        }

    try:
        index = np.load(resolved_index_path, allow_pickle=True)
        embeddings = index["embeddings"].astype(np.float32)
        labels = [str(item) for item in index["labels"].tolist()]
        image_paths = [str(item) for item in index["image_paths"].tolist()]
        metadata_columns = [str(item) for item in index["metadata_columns"].tolist()] if "metadata_columns" in index else []
        metadata_values = index["metadata_values"].tolist() if "metadata_values" in index else []
        architecture = _index_scalar_value(index, "architecture", "efficientnet_b0")
        embedding_model = _index_scalar_value(index, "embedding_model", "pretrained_encoder")
        configured_metric_model = os.getenv("TABLET_IDENTITY_METRIC_MODEL_PATH", "").strip()
        metric_model_path = configured_metric_model
        if not metric_model_path and embedding_model == "tablet_metric_identity":
            metric_model_path = _resolve_identity_metric_model_path(index)
        if embedding_model == "tablet_metric_identity" and not metric_model_path:
            return {
                "available": False,
                "error": "Identity index was built with a metric model, but no metric model checkpoint is available.",
                "indexPath": str(resolved_index_path),
            }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, architecture, metric_model_path = _build_index_embedding_model(architecture, metric_model_path, device)
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tensor)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            query = outputs.detach().cpu().numpy().astype(np.float32)[0]
        query = query / max(float(np.linalg.norm(query)), 1e-8)

        similarities = embeddings @ query
        top_indexes = np.argsort(-similarities)[: max(top_k * 8, top_k)]
        label_scores: dict[str, dict[str, Any]] = {}
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

        ranked_labels = sorted(
            label_scores.values(),
            key=lambda item: float(item.get("similarity") or -1),
            reverse=True,
        )
        predictions = []
        for item in ranked_labels:
            score_list = label_score_lists.get(str(item.get("medicineName"))) or []
            if len(score_list) > 1:
                item["meanSimilarity"] = round(float(np.mean(score_list)), 4)
                item["referenceCount"] = int(len(score_list))
            predictions.append(item)
            if len(predictions) >= top_k:
                break

        top_similarity = float(predictions[0].get("similarity") or 0) if predictions else 0.0
        second_similarity = float(predictions[1].get("similarity") or -1) if len(predictions) > 1 else -1.0
        margin = top_similarity - second_similarity if second_similarity >= -1 else top_similarity
        try:
            margin_threshold = float(os.getenv("TABLET_IDENTITY_RETRIEVAL_MARGIN", "0.045"))
        except ValueError:
            margin_threshold = 0.045

        return {
            "available": True,
            "predictions": predictions,
            "topSimilarity": round(top_similarity, 4),
            "secondSimilarity": round(second_similarity, 4) if second_similarity >= -1 else None,
            "similarityMargin": round(float(margin), 4),
            "marginThreshold": margin_threshold,
            "ambiguous": bool(len(predictions) > 1 and margin < margin_threshold),
            "ambiguousReason": "Top visual matches are too close to safely choose one pill." if len(predictions) > 1 and margin < margin_threshold else "",
            "indexPath": str(resolved_index_path),
            "architecture": architecture,
            "embeddingModel": "tablet_metric_identity" if metric_model_path else "pretrained_encoder",
            "metricModelPath": metric_model_path,
            "indexSize": int(len(labels)),
            "classCount": int(len(set(labels))),
        }
    except Exception as error:
        return {
            "available": False,
            "error": str(error),
            "indexPath": str(resolved_index_path),
        }


def _predict_with_trained_identity_model(image: Image.Image, model_path: str, top_k: int = 5) -> dict[str, Any]:
    if not model_path:
        return {
            "available": False,
            "error": "No trained tablet identity model found.",
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
            "modelPath": str(resolved_model_path),
        }

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(resolved_model_path, map_location=device)
        labels = [str(value) for value in checkpoint["labels"]]
        architecture = checkpoint.get("architecture") or "mobilenet_v3_small"
        if checkpoint.get("task") != "tablet_identity_classification":
            return {
                "available": False,
                "error": f"Unsupported model task: {checkpoint.get('task')}",
                "modelPath": str(resolved_model_path),
            }

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
            return {
                "available": False,
                "error": f"Unsupported model architecture: {architecture}",
                "modelPath": str(resolved_model_path),
            }

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
            values, indexes = torch.topk(probabilities, k=min(top_k, len(labels)))

        top_probability = float(values[0].item()) if len(values) > 0 else 0.0
        second_probability = float(values[1].item()) if len(values) > 1 else 0.0
        probability_margin = top_probability - second_probability
        try:
            margin_threshold = float(os.getenv("TABLET_IDENTITY_MODEL_MARGIN", "0.15"))
        except ValueError:
            margin_threshold = 0.15

        predictions = [
            {
                "medicineName": labels[int(index.item())],
                "confidence": round(float(value.item()), 4),
            }
            for value, index in zip(values, indexes)
        ]
        return {
            "available": True,
            "predictions": predictions,
            "topProbability": round(top_probability, 4),
            "secondProbability": round(second_probability, 4),
            "probabilityMargin": round(probability_margin, 4),
            "marginThreshold": margin_threshold,
            "ambiguous": bool(len(predictions) > 1 and probability_margin < margin_threshold),
            "ambiguousReason": "Top classifier predictions are too close to safely choose one pill." if len(predictions) > 1 and probability_margin < margin_threshold else "",
            "modelPath": str(resolved_model_path),
            "architecture": architecture,
            "valAccuracy": checkpoint.get("val_accuracy"),
        }
    except Exception as error:
        return {
            "available": False,
            "error": str(error),
            "modelPath": str(resolved_model_path),
        }


def _candidate_score(candidate: dict[str, Any], detected_color: str, detected_shape: str) -> float:
    color = _normalize_color(candidate.get("color"))
    shape = _normalize_shape(candidate.get("shape"))

    score = 0.0
    if color and detected_color and color == detected_color:
        score += 0.58
    elif color and detected_color and {color, detected_color} <= {"gray", "grey", "white"}:
        score += 0.32

    if shape and detected_shape and shape == detected_shape:
        score += 0.36
    elif shape and detected_shape and {shape, detected_shape} <= {"oval", "capsule"}:
        score += 0.18
    if not color and not shape:
        score = max(score, 0.05)

    return round(min(0.96, score), 4)


def _candidate_matches_detected_appearance(candidate: dict[str, Any], detected_color: str, detected_shape: str) -> bool:
    color = _normalize_color(candidate.get("color"))
    shape = _normalize_shape(candidate.get("shape"))
    return bool(color and shape and color == _normalize_color(detected_color) and shape == _normalize_shape(detected_shape))


def _appearance_value_known(value: str) -> bool:
    return bool(value and value not in {"unknown", "not specified", "none", "n/a"})


def _colors_compatible(first: str, second: str) -> bool:
    if not (_appearance_value_known(first) and _appearance_value_known(second)):
        return True
    return first == second or {first, second} <= {"gray", "white"}


def _shapes_compatible(first: str, second: str) -> bool:
    if not (_appearance_value_known(first) and _appearance_value_known(second)):
        return True
    return first == second or {first, second} <= {"oval", "capsule"}


def _candidate_appearance_compatible(candidate: dict[str, Any], detected_color: str, detected_shape: str) -> bool:
    color = _normalize_color(candidate.get("color"))
    shape = _normalize_shape(candidate.get("shape"))
    normalized_detected_color = _normalize_color(detected_color)
    normalized_detected_shape = _normalize_shape(detected_shape)
    return _colors_compatible(color, normalized_detected_color) and _shapes_compatible(shape, normalized_detected_shape)


def _filter_matches_by_compatible_appearance(
    matches: list[dict[str, Any]],
    detected_color: str,
    detected_shape: str,
) -> list[dict[str, Any]]:
    return [
        match
        for match in matches
        if _candidate_appearance_compatible(match, detected_color, detected_shape)
    ]


def _filter_matches_by_detected_appearance(
    matches: list[dict[str, Any]],
    detected_color: str,
    detected_shape: str,
) -> list[dict[str, Any]]:
    return [
        match
        for match in matches
        if _candidate_matches_detected_appearance(match, detected_color, detected_shape)
    ]


def _is_appearance_ambiguous(matches: list[dict[str, Any]]) -> bool:
    if len(matches) < 2:
        return False
    top = float(matches[0].get("confidence") or 0)
    second = float(matches[1].get("confidence") or 0)
    return abs(top - second) < 0.035


def _matches_from_predictions(
    predictions: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_by_name: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        normalized_name = _normalize_name(candidate.get("medicineName"))
        if normalized_name:
            candidate_by_name.setdefault(normalized_name, []).append(candidate)

    matched = []
    unmatched = []
    for prediction in predictions or []:
        prediction_name = prediction.get("medicineName")
        matching_candidates = candidate_by_name.get(_normalize_name(prediction_name)) or []
        if matching_candidates:
            for candidate in matching_candidates:
                matched.append(
                    {
                        "id": candidate.get("id"),
                        "medicineName": candidate.get("medicineName") or prediction_name or "Medicine",
                        "dosageMg": candidate.get("dosageMg"),
                        "color": candidate.get("color") or "",
                        "shape": candidate.get("shape") or "",
                        "confidence": prediction.get("confidence") or 0,
                        "referenceImage": prediction.get("referenceImage"),
                        "metadata": prediction.get("metadata") or {},
                    }
                )
        else:
            unmatched.append(
                {
                    "medicineName": prediction_name or "Unknown tablet",
                    "confidence": prediction.get("confidence") or 0,
                    "referenceImage": prediction.get("referenceImage"),
                    "metadata": prediction.get("metadata") or {},
                }
            )
    return matched, unmatched


def _has_ambiguous_saved_candidate_tie(matches: list[dict[str, Any]]) -> bool:
    if len(matches) < 2:
        return False

    top_confidence = max(float(match.get("confidence") or 0) for match in matches)
    tied_top_matches = [
        match
        for match in matches
        if abs(float(match.get("confidence") or 0) - top_confidence) < 0.0001
    ]
    return len({str(match.get("id") or "") for match in tied_top_matches}) > 1


def _accepted_prediction_result(
    source: str,
    prediction: dict[str, Any],
    threshold: float,
    candidates: list[dict[str, Any]],
    base: dict[str, Any],
) -> dict[str, Any] | None:
    if not prediction.get("available"):
        return None
    if prediction.get("ambiguous"):
        return None

    matched_by_name, unmatched = _matches_from_predictions(prediction.get("predictions") or [], candidates)
    detected_color = str(base.get("detectedColor") or "")
    detected_shape = str(base.get("detectedShape") or "")
    appearance_matched = _filter_matches_by_detected_appearance(matched_by_name, detected_color, detected_shape)
    appearance_compatible = _filter_matches_by_compatible_appearance(matched_by_name, detected_color, detected_shape)
    matched = appearance_matched or appearance_compatible
    appearance_consistent = bool(matched)

    if matched_by_name and not matched:
        return None

    if candidates and not matched:
        return None

    if _has_ambiguous_saved_candidate_tie(matched):
        return None

    best = matched[0] if matched else (unmatched[0] if unmatched else None)
    confidence = float(best.get("confidence") or 0) if best else 0.0
    if not best or confidence < threshold:
        return None

    saved_match = bool(matched)
    same_appearance_count = int(base.get("sameAppearanceCandidateCount") or 0)
    if same_appearance_count > 1 and source not in {"visual-retrieval", "trained-model"}:
        return None

    return {
        **base,
        "status": "matched" if saved_match else "dataset-match",
        "confidence": round(confidence, 4),
        "match": best,
        "matches": matched[:5] if saved_match else [],
        "identitySource": source,
        "fineDetailMatched": saved_match and same_appearance_count > 1,
        "matchedSavedMedicine": saved_match,
        "appearanceConsistent": appearance_consistent,
        "appearanceWarning": "" if appearance_consistent else "Model guess did not fully match detected color/shape metadata.",
        "datasetMatches": unmatched[:5],
        "retrievalAnalysis": prediction if source == "visual-retrieval" else base.get("retrievalAnalysis"),
        "modelAnalysis": prediction if source == "trained-model" else base.get("modelAnalysis"),
    }


def _appearance_result_for_component(
    image: Image.Image,
    arr: np.ndarray,
    component: dict[str, Any],
    candidates: list[dict[str, Any]],
    object_index: int = 0,
    detector_object: dict[str, Any] | None = None,
    retrieval_confidence_threshold: float = 0.72,
    model_confidence_threshold: float = 0.55,
    identity_index_path: str = "",
    identity_model_path: str = "",
) -> dict[str, Any]:
    median_rgb, object_color_stats = _pill_object_color(arr, component)
    detected_color, color_confidence = _classify_color(median_rgb)
    detected_shape, shape_confidence = _classify_shape(component)
    crop = _crop_component(image, component)
    visual_features = _visual_features(crop)

    matches = []
    for candidate in candidates:
        if not _candidate_matches_detected_appearance(candidate, detected_color, detected_shape):
            continue
        score = _candidate_score(candidate, detected_color, detected_shape)
        matches.append(
            {
                "id": candidate.get("id"),
                "medicineName": candidate.get("medicineName") or "Medicine",
                "dosageMg": candidate.get("dosageMg"),
                "color": candidate.get("color") or "",
                "shape": candidate.get("shape") or "",
                "confidence": round(score * ((color_confidence * 0.65) + (shape_confidence * 0.35)), 4),
            }
        )

    matches.sort(key=lambda item: item["confidence"], reverse=True)
    best = matches[0] if matches else None
    confidence = float(best["confidence"]) if best else 0.0
    ambiguous = len(matches) > 1 or _is_appearance_ambiguous(matches)
    match = None if ambiguous else best
    appearance_status = "ambiguous" if ambiguous else "matched" if best and confidence >= 0.28 else "uncertain"

    base_result = {
        "objectIndex": object_index,
        "status": appearance_status,
        "confidence": round(confidence, 4),
        "match": match,
        "matches": matches[:5],
        "ambiguous": ambiguous,
        "detectedColor": detected_color,
        "detectedShape": detected_shape,
        "detectedRgb": [int(round(value)) for value in median_rgb.tolist()],
        "appearanceConfidence": round(float((color_confidence + shape_confidence) / 2.0), 4),
        "tabletBox": {key: value for key, value in component.items() if key != "mask"},
        "detectorObject": detector_object,
        "objectColorAnalysis": object_color_stats,
        "visualFeatures": visual_features,
        "sameAppearanceCandidateCount": len(matches),
        "requiresFineDetailMatch": ambiguous,
        "identitySource": "appearance-matcher",
        "appearanceAnalysis": {
            "status": appearance_status,
            "confidence": round(confidence, 4),
            "ambiguous": ambiguous,
            "match": match,
            "matches": matches[:5],
            "reason": "Several saved medicines look similar. Please compare the close matches." if ambiguous else "",
        },
    }

    retrieval_prediction = _predict_with_identity_index(crop, identity_index_path)
    retrieval_result = _accepted_prediction_result(
        "visual-retrieval",
        retrieval_prediction,
        retrieval_confidence_threshold,
        candidates,
        base_result,
    )
    if retrieval_result:
        return {
            **base_result,
            "status": retrieval_result.get("status") or "matched",
            "confidence": retrieval_result.get("confidence") or 0,
            "match": retrieval_result.get("match"),
            "matches": retrieval_result.get("matches") or [],
            "identitySource": "visual-retrieval",
            "fineDetailMatched": retrieval_result.get("fineDetailMatched", False),
            "appearanceConsistent": retrieval_result.get("appearanceConsistent", False),
            "appearanceWarning": retrieval_result.get("appearanceWarning", ""),
            "retrievalAnalysis": retrieval_prediction,
        }

    model_prediction = _predict_with_trained_identity_model(crop, identity_model_path)
    model_result = _accepted_prediction_result(
        "trained-model",
        model_prediction,
        model_confidence_threshold,
        candidates,
        base_result,
    )
    if model_result:
        return {
            **base_result,
            "status": model_result.get("status") or "matched",
            "confidence": model_result.get("confidence") or 0,
            "match": model_result.get("match"),
            "matches": model_result.get("matches") or [],
            "identitySource": "trained-model",
            "fineDetailMatched": model_result.get("fineDetailMatched", False),
            "appearanceConsistent": model_result.get("appearanceConsistent", False),
            "appearanceWarning": model_result.get("appearanceWarning", ""),
            "modelAnalysis": model_prediction,
        }

    return {
        **base_result,
        "retrievalAnalysis": retrieval_prediction,
        "modelAnalysis": model_prediction,
    }


def _component_for_detector_object(
    arr: np.ndarray,
    width: int,
    height: int,
    detector_object: dict[str, Any],
) -> dict[str, Any] | None:
    allowed_mask = _mask_from_box(width, height, detector_object)
    component = _object_component_no_background(arr, allowed_mask)
    if component is None:
        component = _refine_component_inside_box(arr, detector_object)
    if component is None:
        component = _component_from_box(width, height, detector_object)
    return component


def _detected_medicine_summary(object_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in object_results:
        match = item.get("match") or {}
        confidence = float(match.get("confidence") or item.get("confidence") or 0)
        medicine_id = str(match.get("id") or "").strip()
        medicine_name = str(match.get("medicineName") or "").strip()
        if not medicine_id or not medicine_name or confidence < 0.28:
            continue

        group = grouped.setdefault(
            medicine_id,
            {
                "id": match.get("id"),
                "medicineName": medicine_name,
                "dosageMg": match.get("dosageMg"),
                "color": match.get("color") or item.get("detectedColor") or "",
                "shape": match.get("shape") or item.get("detectedShape") or "",
                "count": 0,
                "confidenceTotal": 0.0,
                "appearanceConsistentCount": 0,
                "objectIndexes": [],
            },
        )
        group["count"] += 1
        group["confidenceTotal"] += confidence
        if item.get("appearanceConsistent", True):
            group["appearanceConsistentCount"] += 1
        group["objectIndexes"].append(item.get("objectIndex"))

    summary = []
    for group in grouped.values():
        count = max(1, int(group.pop("count") or 0))
        confidence_total = float(group.pop("confidenceTotal") or 0.0)
        appearance_consistent_count = int(group.pop("appearanceConsistentCount") or 0)
        summary.append(
            {
                **group,
                "count": count,
                "confidence": round(confidence_total / count, 4),
                "appearanceConsistent": appearance_consistent_count == count,
            }
        )

    summary.sort(key=lambda item: (str(item.get("medicineName") or "").lower(), str(item.get("id") or "")))
    return summary


def analyze(image_base64: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    image = _decode_image(image_base64)
    original_image = image.copy()
    try:
        retrieval_confidence_threshold = float(os.getenv("TABLET_IDENTITY_RETRIEVAL_CONFIDENCE", "0.72"))
    except ValueError:
        retrieval_confidence_threshold = 0.72
    try:
        model_confidence_threshold = float(os.getenv("TABLET_IDENTITY_MODEL_CONFIDENCE", "0.55"))
    except ValueError:
        model_confidence_threshold = 0.55
    identity_index_path = _resolve_identity_index_path()
    identity_model_path = _resolve_identity_model_path()

    detector_prediction = _detect_pill_objects(original_image, _resolve_detector_model_path())

    image.thumbnail((560, 560))
    arr = np.asarray(image).astype(np.float32)
    height, width, _ = arr.shape

    detector_objects = detector_prediction.get("objects") or []
    object_components: list[dict[str, Any]] = []
    for detector_object in detector_objects[:12]:
        object_component = _component_for_detector_object(arr, width, height, detector_object)
        if object_component is not None:
            object_components.append({"component": object_component, "detectorObject": detector_object})

    for fallback_component in _object_components_no_background_cv2(arr, max_components=12):
        if any(_component_iou(fallback_component, item["component"]) > 0.35 for item in object_components):
            continue
        object_components.append({"component": fallback_component, "detectorObject": None})

    object_components = object_components[:12]
    detected_object_results: list[dict[str, Any]] = []
    for object_index, object_item in enumerate(object_components):
        object_component = object_item.get("component")
        detector_object = object_item.get("detectorObject")
        detected_object_results.append(
            _appearance_result_for_component(
                image,
                arr,
                object_component,
                candidates,
                object_index,
                detector_object,
                retrieval_confidence_threshold,
                model_confidence_threshold,
                identity_index_path,
                identity_model_path,
            )
        )

    detected_medicines = _detected_medicine_summary(detected_object_results)

    allowed_mask = _mask_from_box(width, height, detector_objects[0]) if detector_objects else None
    component = _object_component_no_background(arr, allowed_mask)
    if component is None and detector_objects:
        component = _refine_component_inside_box(arr, detector_objects[0])
    if component is None and detector_objects:
        component = _component_from_box(width, height, detector_objects[0])

    if component is None:
        skipped_model_reason = "Pill object was not isolated, so model matching was skipped to avoid using background pixels."
        base_result = {
            "detectedColor": "unknown",
            "detectedShape": "unknown",
            "objectDetection": {
                **detector_prediction,
                "selectedObject": None,
            },
            "retrievalAnalysis": {
                "available": False,
                "error": skipped_model_reason,
                "confidenceThreshold": retrieval_confidence_threshold,
            },
            "modelAnalysis": {
                "available": False,
                "error": skipped_model_reason,
                "confidenceThreshold": model_confidence_threshold,
            },
            "appearanceAnalysis": {
                "status": "uncertain",
                "confidence": 0,
                "error": "Could not find a clear tablet in the photo.",
            },
            "detectedObjects": detected_object_results,
            "detectedMedicines": detected_medicines,
        }
        return {
            **base_result,
            "status": "uncertain",
            "confidence": 0,
            "matches": [],
            "identitySource": "appearance-matcher",
            "error": "Could not find a clear tablet in the photo.",
        }

    median_rgb, object_color_stats = _pill_object_color(arr, component)
    detected_color, color_confidence = _classify_color(median_rgb)
    detected_shape, shape_confidence = _classify_shape(component)
    crop = _crop_component(image, component)
    visual_features = _visual_features(crop)
    retrieval_prediction = _predict_with_identity_index(crop, identity_index_path)
    model_prediction = _predict_with_trained_identity_model(crop, identity_model_path)

    matches = []
    for candidate in candidates:
        if not _candidate_matches_detected_appearance(candidate, detected_color, detected_shape):
            continue
        score = _candidate_score(candidate, detected_color, detected_shape)
        matches.append(
            {
                "id": candidate.get("id"),
                "medicineName": candidate.get("medicineName") or "Medicine",
                "dosageMg": candidate.get("dosageMg"),
                "color": candidate.get("color") or "",
                "shape": candidate.get("shape") or "",
                "confidence": round(score * ((color_confidence * 0.65) + (shape_confidence * 0.35)), 4),
            }
        )

    matches.sort(key=lambda item: item["confidence"], reverse=True)
    best = matches[0] if matches else None
    confidence = float(best["confidence"]) if best else 0.0
    appearance_ambiguous = len(matches) > 1 or _is_appearance_ambiguous(matches)
    appearance_match = None if appearance_ambiguous else best
    ambiguity_reason = (
        "Several saved medicines look similar. Please compare the close matches."
        if appearance_ambiguous
        else ""
    )

    base_result = {
        "detectedColor": detected_color,
        "detectedShape": detected_shape,
        "detectedRgb": [int(round(value)) for value in median_rgb.tolist()],
        "appearanceConfidence": round(float((color_confidence + shape_confidence) / 2.0), 4),
        "tabletBox": {key: value for key, value in component.items() if key != "mask"},
        "objectColorAnalysis": object_color_stats,
        "objectDetection": {
            **detector_prediction,
            "selectedObject": detector_objects[0] if detector_objects else None,
            "objectSource": component.get("source") or "object-mask",
        },
        "visualFeatures": visual_features,
        "sameAppearanceCandidateCount": len(matches),
        "requiresFineDetailMatch": appearance_ambiguous,
        "ambiguityReason": ambiguity_reason,
        "retrievalAnalysis": {
            **retrieval_prediction,
            "confidenceThreshold": retrieval_confidence_threshold,
        },
        "modelAnalysis": {
            **model_prediction,
            "confidenceThreshold": model_confidence_threshold,
        },
        "appearanceAnalysis": {
            "status": "ambiguous" if appearance_ambiguous else "matched" if best and confidence >= 0.28 else "uncertain",
            "confidence": round(confidence, 4),
            "ambiguous": appearance_ambiguous,
            "match": appearance_match,
            "matches": matches[:5],
            "reason": ambiguity_reason,
        },
        "detectedObjects": detected_object_results,
        "detectedMedicines": detected_medicines,
    }

    retrieval_result = _accepted_prediction_result(
        "visual-retrieval",
        retrieval_prediction,
        retrieval_confidence_threshold,
        candidates,
        base_result,
    )
    if retrieval_result:
        return retrieval_result

    model_result = _accepted_prediction_result(
        "trained-model",
        model_prediction,
        model_confidence_threshold,
        candidates,
        base_result,
    )
    if model_result:
        return model_result

    return {
        **base_result,
        "status": "ambiguous" if appearance_ambiguous else "matched" if best and confidence >= 0.28 else "uncertain",
        "confidence": round(confidence, 4),
        "match": appearance_match,
        "matches": matches[:5],
        "ambiguous": appearance_ambiguous,
        "error": ambiguity_reason if appearance_ambiguous else None,
        "identitySource": "appearance-matcher",
    }


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            candidates = []
        print(json.dumps(analyze(str(payload.get("imageBase64") or ""), candidates)))
    except Exception as error:
        print(json.dumps({"status": "uncertain", "confidence": 0, "error": str(error), "matches": []}))
        sys.exit(1)


if __name__ == "__main__":
    main()
