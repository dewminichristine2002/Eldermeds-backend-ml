import re
import string
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from model_runtime import PROJECT_CLASSES, load_runtime


MODULE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="ElderMeds Emotion Classifier", version="4.0.0")
SERVICE_VERSION = "1.0.0"


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str | None = None
    context: "PredictionContext | None" = None


class PredictionContext(BaseModel):
    """Optional future context input; current models intentionally ignore it."""

    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(..., min_length=1, max_length=200)
    participant_group_id: str | None = Field(default=None, min_length=1, max_length=200)
    turn_index: int | None = Field(default=None, ge=1)
    question: str | None = Field(default=None, max_length=2000)
    previous_turns: list[str | dict[str, object]] = Field(default_factory=list, max_length=2)


def clean_text(text):
    normalized = "" if text is None else str(text).lower()
    normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", normalized).strip()


runtime = load_runtime()


def model_files_ready():
    return runtime.ready and set(runtime.classes_) == set(PROJECT_CLASSES)


def runtime_context_aware():
    metadata = runtime.metadata or {}
    try:
        from model_registry import get_registration
        registration = get_registration(str(runtime.model_version))
    except ValueError:
        return False
    return (registration.get("context_aware") is True
            and metadata.get("context_aware") is True
            and metadata.get("input_format") == registration.get("input_format"))


@app.get("/health")
def health():
    ready = model_files_ready()
    return {
        "success": ready,
        "service": "emotion_classifier",
        "service_version": SERVICE_VERSION,
        "model_loaded": runtime.classifier is not None,
        "metadata_loaded": runtime.metadata is not None,
        "ready": ready,
        "model_version": runtime.model_version,
        "model_status": "ready" if ready else "unavailable",
        "artifact_integrity": runtime.artifact_integrity,
        "supported_classes": PROJECT_CLASSES,
        "context_aware": runtime_context_aware(),
        "confidence_calibrated": bool((runtime.metadata or {}).get("confidence_calibrated", False)),
        "error": runtime.error,
    }


@app.post("/predict-emotion")
def predict_emotion(request: PredictionRequest):
    if not model_files_ready():
        raise HTTPException(status_code=503, detail="Emotion model artifacts are missing, invalid, or incompatible.")
    cleaned_text = clean_text(request.text)
    if not cleaned_text:
        raise HTTPException(status_code=422, detail="Text is required and cannot be empty.")

    context = request.context.model_dump() if request.context else None
    context_aware = runtime_context_aware()
    if context_aware:
        if not context or not context.get("participant_group_id") or context.get("turn_index") is None:
            raise HTTPException(status_code=422, detail="Contextual inference requires scoped participant and turn identifiers.")
    try:
        probabilities = runtime.predict_proba([request.text.strip()], context=context)[0]
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail="Prediction context is invalid for the selected model.") from exc
    winning_index = int(probabilities.argmax())
    emotion = str(runtime.classes_[winning_index])
    confidence = float(probabilities[winning_index])
    if emotion not in PROJECT_CLASSES:
        raise HTTPException(status_code=503, detail="Emotion model returned an unsupported class.")

    confidence_calibrated = bool((runtime.metadata or {}).get("confidence_calibrated", False))
    token_count = len(cleaned_text.split())
    short_answer = token_count <= 2
    ordered = sorted((float(value) for value in probabilities), reverse=True)
    probability_margin = ordered[0] - ordered[1] if len(ordered) > 1 else None
    close_competition = bool(confidence_calibrated and probability_margin is not None and probability_margin < 0.08)
    short_answer_without_context = short_answer and not context_aware
    clarification_suggested = short_answer_without_context or close_competition
    uncertainty_reasons = []
    if short_answer_without_context:
        uncertainty_reasons.append("very_short_answer")
        if request.context is None or not (request.context.question or "").strip():
            uncertainty_reasons.append("question_context_missing")
        else:
            uncertainty_reasons.append("selected_model_is_answer_only")
    if close_competition:
        uncertainty_reasons.append("close_competing_calibrated_probabilities")

    return {
        "success": True,
        "emotion": emotion,
        "confidence": max(0.0, min(1.0, confidence)),
        "source": "ml_model",
        "model_version": runtime.model_version,
        "supported_classes": PROJECT_CLASSES,
        "confidence_type": "model_probability",
        "confidence_calibrated": confidence_calibrated,
        "context_aware": runtime_context_aware(),
        "clarification_suggested": clarification_suggested,
        "uncertainty": {
            "needs_clarification": clarification_suggested,
            "reasons": uncertainty_reasons,
            "confidence_calibrated": confidence_calibrated,
            "top_probability_margin": probability_margin if confidence_calibrated else None,
        },
    }


@app.get("/live")
def live():
    """Process liveness; /health remains the model readiness check."""
    return {"success": True, "service": "emotion_classifier", "service_version": SERVICE_VERSION}
