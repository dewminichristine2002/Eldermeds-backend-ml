"""Deterministic, bounded context formatting for future contextual models."""

from collections.abc import Mapping, Sequence

DEFAULT_MAX_PREVIOUS_TURNS = 2
DEFAULT_MAX_CHARS = 2000
DEFAULT_MAX_TOKENS = 256
INPUT_FORMATS = {"legacy", "answer_only", "question_answer", "multi_turn"}


class ContextValidationError(ValueError):
    """Raised when a prediction context cannot be safely associated."""


def build_context_text(
    *,
    answer: str,
    session_id: str,
    question: str | None = None,
    previous_turns: Sequence[str] | None = None,
    max_previous_turns: int = DEFAULT_MAX_PREVIOUS_TURNS,
    max_chars: int = DEFAULT_MAX_CHARS,
    input_format: str = "legacy",
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
    tokenizer=None,
    participant_group_id: str | None = None,
    current_turn_index: int | None = None,
) -> str:
    """Build chronological context without adding labels or future turns."""
    if not isinstance(session_id, str) or not session_id.strip():
        raise ContextValidationError("session_id is required")
    if not isinstance(answer, str) or not answer.strip():
        raise ContextValidationError("answer is required")
    if max_previous_turns < 0 or max_chars < 1:
        raise ContextValidationError("context limits must be non-negative and bounded")
    if input_format not in INPUT_FORMATS:
        raise ContextValidationError("input_format must be legacy, answer_only, question_answer, or multi_turn")
    if max_tokens is not None and max_tokens < 1:
        raise ContextValidationError("max_tokens must be positive")

    turns = list(previous_turns or [])
    structured_indices = []
    for turn in turns:
        if not isinstance(turn, (str, Mapping)):
            raise ContextValidationError("previous_turns must contain strings or turn objects")
        if isinstance(turn, Mapping):
            if not isinstance(turn.get("question"), str) or not isinstance(turn.get("answer"), str):
                raise ContextValidationError("structured previous turns require question and answer")
            if not session_id or not participant_group_id or current_turn_index is None:
                raise ContextValidationError("structured history requires participant, session, and current turn identifiers")
            if session_id and turn.get("session_id") != session_id:
                raise ContextValidationError("previous turn belongs to a different session")
            if participant_group_id and turn.get("participant_group_id") != participant_group_id:
                raise ContextValidationError("previous turn belongs to a different participant group")
            turn_index = turn.get("turn_index")
            if (not isinstance(turn_index, int) or isinstance(turn_index, bool)
                    or turn_index < 1 or turn_index >= current_turn_index):
                raise ContextValidationError("previous turn must have an earlier integer turn_index")
            structured_indices.append(turn_index)
    if structured_indices and structured_indices != sorted(set(structured_indices)):
        raise ContextValidationError("structured previous turns must be unique and chronological")

    history_parts = []
    current_parts = []
    normalized_question = " ".join(question.split()) if isinstance(question, str) else ""
    normalized_answer = " ".join(answer.split())
    if input_format == "answer_only":
        current_parts = [normalized_answer]
    elif input_format == "legacy":
        for turn in turns[-max_previous_turns:] if max_previous_turns else []:
            if isinstance(turn, str):
                normalized = " ".join(turn.split())
                if normalized:
                    history_parts.append(f"Previous: {normalized}")
            else:
                history_parts.append(f"Previous question: {' '.join(turn['question'].split())}\nPrevious answer: {' '.join(turn['answer'].split())}")
        if normalized_question:
            current_parts.append(f"Question: {normalized_question}")
        current_parts.append(f"Answer: {normalized_answer}")
    else:
        if not normalized_question:
            raise ContextValidationError("current question is required for this input format")
        if input_format == "question_answer":
            current_parts.extend((f"Question: {normalized_question}", f"Answer: {normalized_answer}"))
        else:
            selected = turns[-max_previous_turns:] if max_previous_turns else []
            for turn in selected:
                if isinstance(turn, str):
                    normalized = " ".join(turn.split())
                    if normalized:
                        history_parts.append(f"Previous: {normalized}")
                else:
                    q = " ".join(turn["question"].split())
                    a = " ".join(turn["answer"].split())
                    history_parts.append(f"Previous question: {q}\nPrevious answer: {a}")
            current_parts.extend((f"Question: {normalized_question}", f"Answer: {normalized_answer}"))

    def token_count(text):
        if max_tokens is None:
            return 0
        if tokenizer is not None:
            return len(tokenizer.encode(text, add_special_tokens=False))
        return len(text.split())

    def compose():
        return "\n".join(history_parts + current_parts)

    text = compose()
    # Only prior context is truncated. Current question/answer and their negation are kept intact.
    while history_parts and ((max_tokens is not None and token_count(text) > max_tokens)
                             or len(text) > max_chars):
        history_parts.pop(0)
        text = compose()
    if (max_tokens is not None and token_count(text) > max_tokens) or len(text) > max_chars:
        raise ContextValidationError("current question and answer exceed the model input budget")
    return text


def build_context_from_payload(
    payload: Mapping[str, object],
    *,
    max_previous_turns: int = DEFAULT_MAX_PREVIOUS_TURNS,
    max_chars: int = DEFAULT_MAX_CHARS,
    input_format: str = "legacy",
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
    tokenizer=None,
) -> str:
    """Validate and format a request-scoped context payload."""
    if not isinstance(payload, Mapping):
        raise ContextValidationError("context must be an object")
    previous_turns = payload.get("previous_turns")
    if previous_turns is not None and (
        isinstance(previous_turns, (str, bytes)) or not isinstance(previous_turns, Sequence)
    ):
        raise ContextValidationError("previous_turns must be a list")
    return build_context_text(
        answer=str(payload.get("answer") or ""),
        session_id=str(payload.get("session_id") or ""),
        question=payload.get("question") if isinstance(payload.get("question"), str) else None,
        previous_turns=previous_turns,
        max_previous_turns=max_previous_turns,
        max_chars=max_chars,
        input_format=input_format,
        max_tokens=max_tokens,
        tokenizer=tokenizer,
        participant_group_id=payload.get("participant_group_id") if isinstance(payload.get("participant_group_id"), str) else None,
        current_turn_index=payload.get("turn_index") if isinstance(payload.get("turn_index"), int) else None,
    )
