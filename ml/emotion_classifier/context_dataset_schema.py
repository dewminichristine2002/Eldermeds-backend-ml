"""Versioned validators for privacy-minimized conversational emotion records."""
from __future__ import annotations

import re
from typing import Any

CONTEXT_SCHEMA_VERSION = "eldermeds-context-emotion-1.1"
LEGACY_CONTEXT_SCHEMA_VERSIONS = {"eldermeds-context-emotion-1.0"}
EMOTION_LABELS = {"happiness", "sadness", "loneliness", "anxiety", "anger", "cognitive_fog", "neutral"}
ANNOTATION_STATUSES = {"reviewed", "adjudicated"}
COLLECTION_ANNOTATION_STATUSES = ANNOTATION_STATUSES | {"pending", "ambiguous", "needs_context", "rejected"}
ANNOTATION_SOURCES = {"human_single_review", "human_double_review", "human_adjudication"}
TRAINING_ANNOTATION_SOURCES = {"human_double_review", "human_adjudication"}
DATASET_SPLITS = {"unassigned", "development", "fit", "calibration", "validation", "test", "independent_test"}
DATA_ORIGINS = {"consented_participant", "synthetic_test", "public_single_turn", "project_single_turn", "unknown"}
MAX_FIELD_CHARS = 2000
MAX_PREVIOUS_TURNS = 2
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$")
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?:\+?\d[\s().-]*){8,}")


class ContextRecordValidationError(ValueError):
    """Raised when a conversational record is incomplete, unsafe, or ineligible."""


def _required_text(record: dict[str, Any], field: str, maximum=MAX_FIELD_CHARS):
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ContextRecordValidationError(f"{field} must be a non-empty string")
    if len(value) > maximum:
        raise ContextRecordValidationError(f"{field} exceeds {maximum} characters")
    return value.strip()


def _valid_identifier(value: Any, field: str):
    if not isinstance(value, str) or not IDENTIFIER_PATTERN.fullmatch(value.strip()):
        raise ContextRecordValidationError(f"{field} must be a pseudonymous stable identifier")


def validate_context_record(record: dict[str, Any], *, require_annotation: bool = True) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ContextRecordValidationError("record must be an object")
    allowed_fields = {
        "schema_version", "sample_id", "participant_group_id", "session_id", "turn_index",
        "current_question", "current_answer", "previous_turns", "emotion_label",
        "annotation_source", "annotation_status", "annotation_guideline_version",
        "annotation_reviewers", "consent_or_use_authorization_reference", "dataset_split",
        "data_origin", "source_provenance", "question_id", "question_bank_version",
        "annotation_reason",
    }
    if set(record) - allowed_fields:
        raise ContextRecordValidationError("record has unsupported fields")
    for key in ("sample_id", "participant_group_id", "session_id"):
        _required_text(record, key, 200)
        _valid_identifier(record[key], key)
    schema_version = record.get("schema_version", CONTEXT_SCHEMA_VERSION)
    if schema_version != CONTEXT_SCHEMA_VERSION and schema_version not in LEGACY_CONTEXT_SCHEMA_VERSIONS:
        raise ContextRecordValidationError("unsupported context schema version")
    turn_index = record.get("turn_index")
    if not isinstance(turn_index, int) or isinstance(turn_index, bool) or turn_index < 1:
        raise ContextRecordValidationError("turn_index must be a positive integer")
    if turn_index > 5:
        raise ContextRecordValidationError("turn_index cannot exceed the five-turn check-in limit")
    current_question = _required_text(record, "current_question")
    current_answer = _required_text(record, "current_answer")
    for text in (current_question, current_answer):
        if EMAIL_PATTERN.search(text) or PHONE_PATTERN.search(text):
            raise ContextRecordValidationError("contact identifiers must be removed before research data preparation")

    label = record.get("emotion_label")
    if require_annotation and label is None:
        raise ContextRecordValidationError("human emotion label and completed annotation are required")
    if label is not None and label not in EMOTION_LABELS:
        raise ContextRecordValidationError("emotion_label must be one of the seven supported labels")
    annotation_source = record.get("annotation_source")
    annotation_status = record.get("annotation_status")
    if require_annotation and (annotation_source not in ANNOTATION_SOURCES or annotation_status not in ANNOTATION_STATUSES):
        raise ContextRecordValidationError("completed human annotation status and source are required")
    if annotation_source is not None and annotation_source not in ANNOTATION_SOURCES:
        raise ContextRecordValidationError("annotation_source must describe human annotation")
    if annotation_status is not None and annotation_status not in COLLECTION_ANNOTATION_STATUSES:
        raise ContextRecordValidationError("annotation_status must be reviewed or adjudicated")

    reviewers = record.get("annotation_reviewers", [])
    if not isinstance(reviewers, list):
        raise ContextRecordValidationError("annotation_reviewers must be a list of pseudonymous reviewer IDs")
    for reviewer in reviewers:
        _valid_identifier(reviewer, "annotation reviewer")
    if len(reviewers) != len(set(reviewers)):
        raise ContextRecordValidationError("annotation_reviewers must be unique")
    auth_ref = record.get("consent_or_use_authorization_reference")
    if auth_ref is not None:
        _valid_identifier(auth_ref, "consent_or_use_authorization_reference")
    split = record.get("dataset_split", "unassigned")
    if split not in DATASET_SPLITS:
        raise ContextRecordValidationError("dataset_split is not recognized")
    origin = record.get("data_origin", "unknown")
    if origin not in DATA_ORIGINS:
        raise ContextRecordValidationError("data_origin is not recognized")
    if record.get("question_id") is not None:
        _valid_identifier(str(record["question_id"]), "question_id")
    if record.get("question_bank_version") is not None and not re.fullmatch(r"sha256:[a-f0-9]{64}", str(record["question_bank_version"])):
        raise ContextRecordValidationError("question_bank_version must be a SHA-256 content version")
    if record.get("source_provenance") is not None:
        _required_text(record, "source_provenance", 500)
    if record.get("annotation_guideline_version") is not None:
        _required_text(record, "annotation_guideline_version", 100)

    previous = record.get("previous_turns", [])
    if not isinstance(previous, list) or len(previous) > MAX_PREVIOUS_TURNS:
        raise ContextRecordValidationError("previous_turns must be a list of at most two records")
    indices = []
    for turn in previous:
        if not isinstance(turn, dict):
            raise ContextRecordValidationError("each previous turn must be an object")
        if set(turn) - {"participant_group_id", "session_id", "turn_index", "question", "answer"}:
            raise ContextRecordValidationError("previous turn has unsupported fields")
        if turn.get("participant_group_id") != record["participant_group_id"]:
            raise ContextRecordValidationError("cross-participant context is forbidden")
        if turn.get("session_id") != record["session_id"]:
            raise ContextRecordValidationError("cross-session context is forbidden")
        idx = turn.get("turn_index")
        if not isinstance(idx, int) or isinstance(idx, bool) or idx < 1 or idx >= turn_index:
            raise ContextRecordValidationError("previous turns must have turn_index values before the current turn")
        _required_text(turn, "question")
        _required_text(turn, "answer")
        if any(EMAIL_PATTERN.search(turn[field]) or PHONE_PATTERN.search(turn[field]) for field in ("question", "answer")):
            raise ContextRecordValidationError("contact identifiers must be removed from previous turns")
        indices.append(idx)
    if indices != sorted(set(indices)):
        raise ContextRecordValidationError("previous turns must be unique and chronologically ordered")

    return {**record, "schema_version": schema_version, "current_question": current_question,
            "current_answer": current_answer}


def validate_context_dataset(records: list[dict[str, Any]], *, require_training_eligible: bool = False,
                             near_duplicate_threshold: float = 0.92,
                             allowed_training_splits: set[str] | None = None,
                             require_previous_turns_present: bool = False) -> dict[str, Any]:
    """Validate sequence ownership/order and report duplicate families without editing inputs."""
    if not isinstance(records, list) or not records:
        raise ContextRecordValidationError("dataset must contain at least one record")
    if len(records) > 20000:
        raise ContextRecordValidationError("dataset exceeds the audited 20,000-record validation bound")
    validated = [validate_context_record(item, require_annotation=require_training_eligible) for item in records]
    sample_ids = [item["sample_id"] for item in validated]
    if len(sample_ids) != len(set(sample_ids)):
        raise ContextRecordValidationError("sample_id values must be globally unique")
    session_owner = {}
    session_turns = {}
    for item in validated:
        owner, session = item["participant_group_id"], item["session_id"]
        if session in session_owner and session_owner[session] != owner:
            raise ContextRecordValidationError("a session_id cannot belong to multiple participant groups")
        session_owner[session] = owner
        turns = session_turns.setdefault((owner, session), {})
        if item["turn_index"] in turns:
            raise ContextRecordValidationError("turn_index must be unique within a participant session")
        turns[item["turn_index"]] = item

    for item in validated:
        turns = session_turns[(item["participant_group_id"], item["session_id"])]
        for previous in item.get("previous_turns", []):
            referenced = turns.get(previous["turn_index"])
            if referenced is None:
                if require_previous_turns_present:
                    raise ContextRecordValidationError("previous turn is not present in the same session dataset")
                continue
            if referenced["current_question"] != previous["question"] or referenced["current_answer"] != previous["answer"]:
                raise ContextRecordValidationError("previous turn question-answer pair does not match its source turn")
        if require_training_eligible:
            if item.get("data_origin") != "consented_participant":
                raise ContextRecordValidationError("training requires genuine consented participant records")
            if not item.get("consent_or_use_authorization_reference"):
                raise ContextRecordValidationError("training requires an authorization reference")
            if item.get("annotation_source") not in TRAINING_ANNOTATION_SOURCES:
                raise ContextRecordValidationError("training requires independent double review or adjudication")
            if ((item["annotation_source"] == "human_double_review" and item["annotation_status"] != "reviewed")
                    or (item["annotation_source"] == "human_adjudication" and item["annotation_status"] != "adjudicated")):
                raise ContextRecordValidationError("annotation source and final status are inconsistent")
            reviewers = item.get("annotation_reviewers", [])
            minimum = 3 if item["annotation_source"] == "human_adjudication" else 2
            if len(reviewers) < minimum:
                raise ContextRecordValidationError("required independent reviewer evidence is missing")
            for field in ("annotation_guideline_version", "source_provenance", "question_id", "question_bank_version"):
                if not item.get(field):
                    raise ContextRecordValidationError(f"{field} is required for training records")
            allowed_splits = allowed_training_splits or {"unassigned", "development"}
            if item.get("dataset_split", "unassigned") not in allowed_splits:
                raise ContextRecordValidationError("dataset split is not allowed for this training/evaluation operation")

    texts = [" ".join(f"{item['current_question']} {item['current_answer']}".casefold().split()) for item in validated]
    exact = {}
    exact_pairs = []
    for sample_id, text in zip(sample_ids, texts):
        if text in exact:
            exact_pairs.append([exact[text], sample_id])
        else:
            exact[text] = sample_id
    near_pairs = []
    if len(validated) > 1:
        if not 0.5 <= near_duplicate_threshold < 1:
            raise ContextRecordValidationError("near_duplicate_threshold must be between 0.5 and 1")
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.neighbors import NearestNeighbors
            matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit_transform(texts)
            neighbors = NearestNeighbors(metric="cosine", algorithm="brute", radius=1 - near_duplicate_threshold,
                                         n_jobs=1).fit(matrix)
            distances, indexes = neighbors.radius_neighbors(matrix, return_distance=True)
            for index, (dists, candidates) in enumerate(zip(distances, indexes)):
                for distance, candidate in zip(dists, candidates):
                    candidate = int(candidate)
                    if candidate > index and texts[index] != texts[candidate]:
                        near_pairs.append([sample_ids[index], sample_ids[candidate], round(1 - float(distance), 6)])
        except ValueError:
            # Character n-grams are undefined for punctuation-only examples; exact checks still apply.
            near_pairs = []
    return {"records": len(validated), "participant_groups": len({item["participant_group_id"] for item in validated}),
            "sessions": len(session_turns), "exact_duplicate_pairs": exact_pairs,
            "near_duplicate_pairs": near_pairs, "near_duplicate_threshold": near_duplicate_threshold,
            "training_eligible": bool(require_training_eligible)}


def model_input_from_record(record: dict[str, Any], input_format: str = "multi_turn", tokenizer=None,
                            max_tokens: int = 256, context_window_turns: int = MAX_PREVIOUS_TURNS) -> str:
    validated = validate_context_record(record)
    from context_builder import build_context_text

    return build_context_text(
        answer=validated["current_answer"], question=validated["current_question"],
        session_id=validated["session_id"], participant_group_id=validated["participant_group_id"],
        current_turn_index=validated["turn_index"], previous_turns=validated.get("previous_turns", []),
        input_format=input_format, max_previous_turns=context_window_turns, tokenizer=tokenizer,
        max_tokens=max_tokens,
    )
