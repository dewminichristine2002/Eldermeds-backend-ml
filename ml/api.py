"""Deployable FastAPI runtime for ElderMeds intake-monitoring ML.

Start with: uvicorn ml.api:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_intake_motion_rules import analyze_motion  # noqa: E402
from analyze_palm_pills import analyze as analyze_palm  # noqa: E402
from analyze_tablet_identity import analyze as analyze_tablet_identity  # noqa: E402

app = FastAPI(title="ElderMeds Intake ML Service", version="1.0.0")


class PalmRequest(BaseModel):
    imageBase64: str = Field(min_length=1)
    expectedCount: float | None = None


class IdentityRequest(BaseModel):
    imageBase64: str = Field(min_length=1)
    candidates: list[dict[str, Any]] = Field(default_factory=list)


class MotionRequest(BaseModel):
    frames: list[dict[str, Any]] = Field(min_length=1)
    swallowConfirmed: bool = False


class MotionVideoRequest(BaseModel):
    videoBase64: str = Field(min_length=1)
    extension: str = "mp4"
    swallowConfirmed: bool = False


def _clean_base64(value: str) -> str:
    return value.split(",", 1)[1] if "," in value else value


def _extract_frames(video_base64: str, extension: str, force_opencv: bool = False) -> dict[str, Any]:
    suffix = "".join(char for char in extension.lower() if char.isalnum()) or "mp4"
    with tempfile.TemporaryDirectory(prefix="eldermeds-motion-") as temp:
        temp_dir = Path(temp)
        video_path = temp_dir / f"motion.{suffix}"
        output_path = temp_dir / "landmarks.json"
        try:
            video_path.write_bytes(base64.b64decode(_clean_base64(video_base64), validate=True))
        except Exception as error:
            raise ValueError("videoBase64 is not valid base64 data.") from error

        command = [
            sys.executable,
            str(SCRIPTS / "extract_mediapipe_motion_landmarks.py"),
            "--video", str(video_path),
            "--output", str(output_path),
            "--frame-step", os.getenv("INTAKE_MOTION_VIDEO_FRAME_STEP", "2"),
            "--max-sampled-frames", os.getenv("INTAKE_MOTION_VIDEO_MAX_SAMPLED_FRAMES", "90"),
            "--max-seconds", os.getenv("INTAKE_MOTION_VIDEO_MAX_SECONDS", "12"),
            "--max-width", os.getenv("INTAKE_MOTION_VIDEO_MAX_WIDTH", "640"),
        ]
        env = os.environ.copy()
        if force_opencv:
            env["ELDERMEDS_FORCE_OPENCV_MOTION"] = "1"
        completed = subprocess.run(command, capture_output=True, text=True, env=env, timeout=75)
        if completed.returncode != 0 or not output_path.exists():
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "Motion extraction failed.")
        return json.loads(output_path.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "eldermeds-intake-ml", "version": app.version}


@app.post("/analyze/palm")
def palm(request: PalmRequest) -> dict[str, Any]:
    try:
        return analyze_palm(request.imageBase64, request.expectedCount)
    except Exception as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/analyze/tablet-identity")
def tablet_identity(request: IdentityRequest) -> dict[str, Any]:
    try:
        return analyze_tablet_identity(request.imageBase64, request.candidates)
    except Exception as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/analyze/motion")
def motion(request: MotionRequest) -> dict[str, Any]:
    return analyze_motion(request.frames, request.swallowConfirmed)


@app.post("/analyze/motion-video")
def motion_video(request: MotionVideoRequest) -> dict[str, Any]:
    try:
        extraction = _extract_frames(request.videoBase64, request.extension)
        if extraction.get("extractionMode") != "opencv-motion-fallback" and (
            int(extraction.get("handFrameCount", 0)) < 6 or int(extraction.get("faceFrameCount", 0)) < 3
        ):
            primary = extraction
            extraction = _extract_frames(request.videoBase64, request.extension, force_opencv=True)
            extraction["primaryExtractionMode"] = primary.get("extractionMode", "mediapipe")
            extraction["primaryFaceFrameCount"] = primary.get("faceFrameCount", 0)
            extraction["primaryHandFrameCount"] = primary.get("handFrameCount", 0)

        frames = extraction.get("frames") or []
        if not frames:
            raise HTTPException(status_code=422, detail="No hand and mouth landmarks were detected.")
        result = analyze_motion(frames, request.swallowConfirmed)
        result.update({key: value for key, value in extraction.items() if key != "frames"})
        result["extractedFrameCount"] = len(frames)
        return result
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
