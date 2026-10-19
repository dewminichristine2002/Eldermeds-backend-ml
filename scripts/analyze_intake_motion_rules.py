from __future__ import annotations

import json
import math
import sys
from typing import Any


def _point(frame: dict[str, Any], path: list[str]) -> dict[str, float] | None:
    current: Any = frame
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if not isinstance(current, dict):
        return None

    try:
        return {
            "x": float(current["x"]),
            "y": float(current["y"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _number(frame: dict[str, Any], path: list[str]) -> float | None:
    current: Any = frame
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]

    try:
        value = float(current)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None
    return value


def _distance(a: dict[str, float], b: dict[str, float]) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _median(values: list[float]) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _timestamp(frame: dict[str, Any], fallback_index: int) -> float:
    try:
        return float(frame.get("timestampMs", fallback_index * 100))
    except (TypeError, ValueError):
        return fallback_index * 100


def _smooth_distances(usable: list[dict[str, Any]]) -> list[dict[str, Any]]:
    smoothed: list[dict[str, Any]] = []
    for index, item in enumerate(usable):
        window = usable[max(0, index - 1) : min(len(usable), index + 2)]
        left_cheek_window = [float(entry["distanceToLeftCheek"]) for entry in window if entry.get("distanceToLeftCheek") is not None]
        right_cheek_window = [float(entry["distanceToRightCheek"]) for entry in window if entry.get("distanceToRightCheek") is not None]
        next_item = {
            **item,
            "rawDistanceToMouth": item["distanceToMouth"],
            "distanceToMouth": _median([float(entry["distanceToMouth"]) for entry in window]),
        }
        mouth_window = [
            float(entry["mouthOpenRatio"])
            for entry in window
            if entry.get("mouthOpenRatio") is not None
        ]
        if mouth_window:
            next_item["rawMouthOpenRatio"] = item.get("mouthOpenRatio")
            next_item["mouthOpenRatio"] = _median(mouth_window)
        if left_cheek_window:
            next_item["rawDistanceToLeftCheek"] = item.get("distanceToLeftCheek")
            next_item["distanceToLeftCheek"] = _median(left_cheek_window)
        if right_cheek_window:
            next_item["rawDistanceToRightCheek"] = item.get("distanceToRightCheek")
            next_item["distanceToRightCheek"] = _median(right_cheek_window)
        smoothed.append(next_item)
    return smoothed


def _longest_near_pause_ms(usable: list[dict[str, Any]], near_threshold: float, max_gap_ms: float = 550) -> float:
    longest = 0.0
    segment_start = None
    previous_timestamp = None

    for item in usable:
        timestamp = float(item["timestampMs"])
        is_near = item["distanceToMouth"] <= near_threshold
        if not is_near:
            if segment_start is not None and previous_timestamp is not None:
                longest = max(longest, previous_timestamp - segment_start)
            segment_start = None
            previous_timestamp = None
            continue

        if segment_start is None:
            segment_start = timestamp
        elif previous_timestamp is not None and timestamp - previous_timestamp > max_gap_ms:
            longest = max(longest, previous_timestamp - segment_start)
            segment_start = timestamp

        previous_timestamp = timestamp

    if segment_start is not None and previous_timestamp is not None:
        longest = max(longest, previous_timestamp - segment_start)

    return longest


def _mouth_values(items: list[dict[str, Any]]) -> list[float]:
    return [
        float(item["mouthOpenRatio"])
        for item in items
        if item.get("mouthOpenRatio") is not None
    ]


def analyze_motion(frames: list[dict[str, Any]], swallow_confirmed: bool = False) -> dict[str, Any]:
    usable: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        mouth = _point(frame, ["face", "mouthCenter"])
        face_left = _point(frame, ["face", "leftCheek"])
        face_right = _point(frame, ["face", "rightCheek"])
        mouth_open_ratio = _number(frame, ["face", "mouthOpenRatio"])
        if not mouth:
            continue

        hand_candidates: list[dict[str, float]] = []
        for key in ("wrist", "indexTip", "thumbTip", "middleTip", "pinkyTip"):
            point = _point(frame, ["hand", key])
            if point:
                hand_candidates.append(point)

        if not hand_candidates:
            continue

        hand_point = min(hand_candidates, key=lambda point: _distance(point, mouth))
        wrist = _point(frame, ["hand", "wrist"])

        face_scale = 0.28
        if face_left and face_right:
            face_scale = max(0.08, _distance(face_left, face_right))

        left_cheek_distance = _distance(hand_point, face_left) / face_scale if face_left else None
        right_cheek_distance = _distance(hand_point, face_right) / face_scale if face_right else None

        usable_frame = {
            "timestampMs": _timestamp(frame, index),
            "distanceToMouth": _distance(hand_point, mouth) / face_scale,
        }
        if left_cheek_distance is not None:
            usable_frame["distanceToLeftCheek"] = left_cheek_distance
        if right_cheek_distance is not None:
            usable_frame["distanceToRightCheek"] = right_cheek_distance
        if wrist:
            usable_frame["wristY"] = wrist["y"]
        if mouth_open_ratio is not None:
            usable_frame["mouthOpenRatio"] = mouth_open_ratio
        usable.append(usable_frame)

    usable.sort(key=lambda item: item["timestampMs"])

    if len(usable) < 6:
        return {
            "status": "uncertain",
            "confidence": 0,
            "handToMouthDetected": False,
            "mouthPauseDetected": False,
            "swallowDetected": bool(swallow_confirmed),
            "videoSwallowDetected": False,
            "swallowConfirmed": bool(swallow_confirmed),
            "message": "Not enough hand and mouth landmark frames for rule-based motion analysis.",
            "framesUsed": len(usable),
        }

    usable = _smooth_distances(usable)
    distances = [item["distanceToMouth"] for item in usable]
    face_gap_values = []
    for item in usable:
        cheek_distances = [value for value in [item.get("distanceToLeftCheek"), item.get("distanceToRightCheek")] if value is not None]
        if cheek_distances:
            face_gap_values.append(min(cheek_distances) - item["distanceToMouth"])
    first_window = distances[: max(2, len(distances) // 4)]
    last_window = distances[-max(2, len(distances) // 4) :]
    start_distance = sum(first_window) / len(first_window)
    closest_index = min(range(len(usable)), key=lambda index: usable[index]["distanceToMouth"])
    closest_distance = distances[closest_index]
    end_distance = sum(last_window) / len(last_window)
    closest_progress = closest_index / max(1, len(usable) - 1)

    near_threshold = 0.95
    approach_delta = 0.35
    before_closest = distances[: closest_index + 1]
    after_closest = distances[closest_index:]
    farthest_before_closest = max(before_closest) if before_closest else start_distance
    farthest_after_closest = max(after_closest) if after_closest else end_distance
    movement_span = max(distances) - closest_distance
    approach_distance = max(start_distance, farthest_before_closest) - closest_distance
    approach_ratio = max(start_distance, farthest_before_closest) / max(closest_distance, 0.001)
    near_frame_count = sum(1 for distance in distances if distance <= near_threshold)
    near_frame_ratio = near_frame_count / len(distances)
    face_touch_frame_count = sum(
        1
        for item in usable
        if item["distanceToMouth"] <= near_threshold
        and min(
            [value for value in [item.get("distanceToLeftCheek"), item.get("distanceToRightCheek")] if value is not None]
            or [float("inf")]
        ) - item["distanceToMouth"] < 0.08
    )
    face_touch_ratio = face_touch_frame_count / len(distances)
    face_touch_detected = face_touch_frame_count >= 2 and face_touch_ratio >= 0.18

    pause_ms = _longest_near_pause_ms(usable, near_threshold)
    mouth_pause_detected = pause_ms >= 500 or (near_frame_count >= 5 and near_frame_ratio >= 0.22)
    wrist_items = [item for item in usable if item.get("wristY") is not None]
    wrist_values = [float(item["wristY"]) for item in wrist_items]
    wrist_elevation_delta = 0.0
    longest_wrist_rise_streak = 0
    if len(wrist_values) >= 3:
        first_wrist_window = wrist_values[: max(2, len(wrist_values) // 4)]
        start_wrist_y = sum(first_wrist_window) / len(first_wrist_window)
        highest_later_wrist_y = min(wrist_values[1:])
        wrist_elevation_delta = start_wrist_y - highest_later_wrist_y
        current_streak = 0
        for previous_y, next_y in zip(wrist_values, wrist_values[1:]):
            if previous_y - next_y >= 0.012:
                current_streak += 1
                longest_wrist_rise_streak = max(longest_wrist_rise_streak, current_streak)
            else:
                current_streak = 0
    wrist_elevation_detected = (
        wrist_elevation_delta >= 0.06
        or (wrist_elevation_delta >= 0.035 and longest_wrist_rise_streak >= 2)
    )

    clear_approach = (
        approach_distance >= approach_delta
        and approach_ratio >= 1.35
        and closest_progress >= 0.08
    )
    near_with_motion = (
        mouth_pause_detected
        and movement_span >= 0.45
        and near_frame_count >= 4
        and closest_progress >= 0.05
    )
    hand_mouth_proximity_detected = closest_distance <= near_threshold
    hand_to_mouth_detected = hand_mouth_proximity_detected and not face_touch_detected and (clear_approach or near_with_motion or wrist_elevation_detected)

    moved_away_after_pause = max(end_distance, farthest_after_closest) - closest_distance >= 0.22
    mouth_landmark_frames = sum(1 for item in usable if item.get("mouthOpenRatio") is not None)
    mouth_values = _mouth_values(usable)
    near_items = [item for item in usable if item["distanceToMouth"] <= near_threshold]
    near_mouth_values = _mouth_values(near_items)
    after_closest_mouth_values = _mouth_values(usable[closest_index:])
    mouth_open_span = max(mouth_values) - min(mouth_values) if mouth_values else 0.0
    near_mouth_open_span = max(near_mouth_values) - min(near_mouth_values) if len(near_mouth_values) >= 2 else 0.0
    post_mouth_open_span = (
        max(after_closest_mouth_values) - min(after_closest_mouth_values)
        if len(after_closest_mouth_values) >= 2
        else 0.0
    )
    mouth_close_after_open = (
        len(after_closest_mouth_values) >= 3
        and max(after_closest_mouth_values[: max(1, len(after_closest_mouth_values) // 2)])
        - min(after_closest_mouth_values[-max(1, len(after_closest_mouth_values) // 2) :])
        >= 0.018
    )
    mouth_activity_detected = (
        mouth_landmark_frames >= 5
        and (
            near_mouth_open_span >= 0.022
            or post_mouth_open_span >= 0.025
            or mouth_open_span >= 0.04
            or mouth_close_after_open
        )
    )
    video_swallow_detected = hand_to_mouth_detected and mouth_pause_detected and mouth_activity_detected
    swallow_detected = bool(swallow_confirmed) or video_swallow_detected
    motion_available = wrist_elevation_detected and hand_mouth_proximity_detected and mouth_pause_detected
    if face_touch_detected and not video_swallow_detected:
        motion_available = False
    completed = motion_available
    confidence_parts = [
        0.34 if wrist_elevation_detected else max(0.0, min(0.24, wrist_elevation_delta / 0.18)),
        0.34 if hand_mouth_proximity_detected else max(0.0, min(0.24, approach_distance / 1.4)),
        0.32 if mouth_pause_detected else max(0.0, min(0.24, pause_ms / 1800)),
    ]
    confidence = round(sum(confidence_parts), 4)
    if face_touch_detected and not video_swallow_detected:
        message = "The hand touched the face, but it did not clearly reach the mouth. Move the hand directly to the mouth and pause there before calling it medicine intake."
    elif not wrist_elevation_detected:
        message = "Wrist elevation was not clear. Start with the hand lower, then raise it toward the mouth."
    elif not hand_mouth_proximity_detected:
        message = "Hand came into view, but it did not get close enough to the mouth. Start lower, move to the mouth, and pause for one second."
    elif not mouth_pause_detected:
        message = "Hand reached the mouth, but the near-mouth pause was too short. Hold at the mouth for one second."
    else:
        message = "Intake motion detected: wrist raised, hand reached the mouth, and paused near the mouth."

    return {
        "status": "completed" if completed else "needs-confirmation",
        "confidence": confidence,
        "motionAvailable": motion_available,
        "faceTouchDetected": face_touch_detected,
        "wristElevationDetected": wrist_elevation_detected,
        "handMouthProximityDetected": hand_mouth_proximity_detected,
        "mouthDwellDetected": mouth_pause_detected,
        "handToMouthDetected": hand_to_mouth_detected,
        "mouthPauseDetected": mouth_pause_detected,
        "swallowDetected": swallow_detected,
        "videoSwallowDetected": video_swallow_detected,
        "mouthActivityDetected": mouth_activity_detected,
        "movedAwayAfterPause": moved_away_after_pause,
        "swallowConfirmed": bool(swallow_confirmed),
        "message": message,
        "framesUsed": len(usable),
        "metrics": {
            "startDistance": round(start_distance, 4),
            "closestDistance": round(closest_distance, 4),
            "endDistance": round(end_distance, 4),
            "closestProgress": round(closest_progress, 4),
            "pauseMs": round(pause_ms, 2),
            "nearFrameCount": near_frame_count,
            "nearFrameRatio": round(near_frame_ratio, 4),
            "movementSpan": round(movement_span, 4),
            "approachDistance": round(approach_distance, 4),
            "approachRatio": round(approach_ratio, 4),
            "wristElevationDelta": round(wrist_elevation_delta, 4),
            "wristRiseStreak": longest_wrist_rise_streak,
            "wristFrameCount": len(wrist_values),
            "farthestBeforeClosest": round(farthest_before_closest, 4),
            "farthestAfterClosest": round(farthest_after_closest, 4),
            "mouthLandmarkFrames": mouth_landmark_frames,
            "mouthOpenSpan": round(mouth_open_span, 4),
            "nearMouthOpenSpan": round(near_mouth_open_span, 4),
            "postMouthOpenSpan": round(post_mouth_open_span, 4),
            "mouthCloseAfterOpen": mouth_close_after_open,
            "faceTouchFrameCount": face_touch_frame_count,
            "faceTouchRatio": round(face_touch_ratio, 4),
            "mouthToFaceGap": round(min(face_gap_values), 4) if face_gap_values else None,
        },
    }


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        frames = payload.get("frames") or []
        if not isinstance(frames, list):
            raise ValueError("frames must be a list of landmark frames.")
        print(json.dumps(analyze_motion(frames, bool(payload.get("swallowConfirmed")))))
    except Exception as error:
        print(json.dumps({"status": "uncertain", "confidence": 0, "error": str(error)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
