"""Startup-loaded, checksum-verified emotion model runtime with v3 rollback."""

import hashlib
import json
import os
import re
from pathlib import Path

import joblib


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_CLASSES = [
    "happiness", "sadness", "loneliness", "anxiety", "anger",
    "cognitive_fog", "neutral",
]
V4_VERSION = "minilm_logistic_regression_v4"
V5_VERSION = "minilm_logistic_regression_v5_answer_only"
V3_VERSION = "tfidf_linear_svm_calibrated_v3"
V4_ARTIFACT = MODULE_DIR / "domain_hierarchical_experiment" / "candidate_model.joblib"
V4_METADATA = MODULE_DIR / "production_model_metadata_v4.json"
V3_ARTIFACT = MODULE_DIR / "advanced_experiment" / "baseline_v3" / "emotion_pipeline_v3.pkl"
V3_METADATA = MODULE_DIR / "advanced_experiment" / "baseline_v3" / "selected_model_metadata_v3.json"
CONTEXTUAL_VERSION = "minilm_logistic_regression_v6_contextual"


def _read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_model_inputs(texts, contexts, metadata, tokenizer=None, max_tokens=256):
    """Apply a model's declared input contract; answer-only artifacts ignore context."""
    input_format = metadata.get("input_format", "answer-only text")
    context_aware = metadata.get("context_aware") is True
    if input_format in {"answer-only text", "answer_only"}:
        return [str(text) for text in texts]
    if not context_aware or input_format not in {"question_answer", "multi_turn"}:
        raise ValueError("model input-format metadata is incompatible")
    from context_builder import build_context_text

    if contexts is None:
        raise ValueError("contextual model requires request-scoped context for every answer")
    if isinstance(contexts, dict):
        contexts = [contexts] * len(texts)
    if len(contexts) != len(texts):
        raise ValueError("context count must match answer count")
    if any(not isinstance(context, dict) for context in contexts):
        raise ValueError("each contextual model input requires a context object")
    for context in contexts:
        if not all(context.get(key) not in (None, "") for key in ("session_id", "participant_group_id", "turn_index")):
            raise ValueError("contextual model requests require participant, session, and turn identifiers")
        if any(not isinstance(turn, dict) for turn in context.get("previous_turns", [])):
            raise ValueError("contextual model requests require structured previous turns")
    return [
        build_context_text(
            answer=str(text), question=context.get("question"),
            session_id=context.get("session_id", ""),
            participant_group_id=context.get("participant_group_id"),
            current_turn_index=context.get("turn_index"),
            previous_turns=context.get("previous_turns", []),
            input_format=input_format, tokenizer=tokenizer, max_tokens=max_tokens,
            max_previous_turns=int(metadata.get("context_window_turns", 2)),
        )
        for text, context in zip(texts, contexts)
    ]


class EmotionModelRuntime:
    """Load one selected model once and expose a common probability interface."""

    def __init__(self, requested_version=None):
        self.requested_version = requested_version or os.getenv("EMOTION_MODEL_VERSION", V4_VERSION)
        self.model_version = None
        self.metadata = None
        self.classifier = None
        self.embedding_model = None
        self.error = None
        self.artifact_integrity = "not_verified"
        try:
            from model_registry import MODEL_REGISTRY

            registration = MODEL_REGISTRY.get(self.requested_version)
            if registration is None:
                raise ValueError(f"Unsupported EMOTION_MODEL_VERSION: {self.requested_version}")
            if registration["status"] == "unavailable_waiting_for_validated_data":
                raise ValueError(f"Model {self.requested_version} is registered but has no validated artifact")
            if self.requested_version == V4_VERSION:
                self._load_v4()
            elif self.requested_version == V5_VERSION:
                self._load_v5()
            elif self.requested_version == V3_VERSION:
                self._load_v3()
            else:
                raise ValueError(f"Unsupported EMOTION_MODEL_VERSION: {self.requested_version}")
        except Exception as exc:  # startup health exposes failure; endpoint returns 503
            self.error = f"{type(exc).__name__}: {exc}"

    def _validate_classes(self, classes):
        if set(map(str, classes)) != set(PROJECT_CLASSES):
            raise ValueError("Model label mapping does not match the seven supported classes")

    def _load_v4(self):
        metadata = _read_json(V4_METADATA)
        if metadata.get("model_version") != V4_VERSION:
            raise ValueError("v4 metadata version mismatch")
        if metadata.get("supported_classes") != PROJECT_CLASSES:
            raise ValueError("v4 metadata class ordering mismatch")
        from model_registry import MODEL_REGISTRY
        expected_hash = MODEL_REGISTRY[V4_VERSION]["artifact_sha256"]
        if metadata.get("artifact_sha256") != expected_hash or _sha256(V4_ARTIFACT) != expected_hash:
            raise ValueError("v4 classifier checksum mismatch")
        bundle = joblib.load(V4_ARTIFACT)
        classifier = bundle.get("model") if isinstance(bundle, dict) else None
        if bundle.get("architecture") != "flat" or classifier is None:
            raise ValueError("v4 classifier bundle is incompatible")
        self._validate_classes(classifier.classes_)

        # Import lazily so the explicit v3 rollback path does not require MiniLM.
        from sentence_transformers import SentenceTransformer

        embedding_model = SentenceTransformer(
            metadata["embedding_model"], device="cpu", local_files_only=True
        )
        self.metadata = metadata
        self.classifier = classifier
        self.embedding_model = embedding_model
        self.model_version = V4_VERSION
        self.artifact_integrity = "verified"

    def _load_v3(self):
        metadata = _read_json(V3_METADATA)
        from model_registry import MODEL_REGISTRY
        if metadata.get("model_version") != V3_VERSION:
            raise ValueError("v3 metadata version mismatch")
        if metadata.get("supported_classes") != PROJECT_CLASSES:
            raise ValueError("v3 metadata class ordering mismatch")
        if _sha256(V3_ARTIFACT) != MODEL_REGISTRY[V3_VERSION]["artifact_sha256"]:
            raise ValueError("v3 classifier checksum mismatch")
        pipeline = joblib.load(V3_ARTIFACT)
        self._validate_classes(pipeline.classes_)
        self.metadata = metadata
        self.classifier = pipeline
        self.model_version = V3_VERSION
        self.artifact_integrity = "verified"

    def _load_v5(self):
        artifact_value = os.getenv("EMOTION_V5_ARTIFACT_PATH")
        if not artifact_value:
            raise ValueError("EMOTION_V5_ARTIFACT_PATH is required for the experimental v5 model")
        artifact = Path(artifact_value).expanduser().resolve()
        metadata_path = artifact.with_name("metadata.json")
        metadata = _read_json(metadata_path)
        if metadata.get("model_version") != V5_VERSION:
            raise ValueError("v5 metadata version mismatch")
        if metadata.get("artifact_file") != artifact.name:
            raise ValueError("v5 artifact filename mismatch")
        if _sha256(artifact) != metadata.get("artifact_sha256"):
            raise ValueError("v5 classifier checksum mismatch")
        if metadata.get("embedding_model") != "sentence-transformers/all-MiniLM-L6-v2":
            raise ValueError("v5 embedding checkpoint is unsupported")
        if metadata.get("embedding_normalization") != "L2 normalized":
            raise ValueError("v5 embedding preprocessing is unsupported")
        import sklearn
        import sentence_transformers

        if metadata.get("scikit_learn_version") != sklearn.__version__:
            raise ValueError("v5 scikit-learn version differs from the training environment")
        if metadata.get("sentence_transformers_version") != sentence_transformers.__version__:
            raise ValueError("v5 Sentence Transformers version differs from the training environment")
        if metadata.get("input_format") != "answer-only text" or metadata.get("context_aware") is not False:
            raise ValueError("v5 input contract is incompatible")
        self._validate_classes(metadata.get("supported_classes", []))
        bundle = joblib.load(artifact)
        classifier = bundle.get("model") if isinstance(bundle, dict) else None
        if (bundle.get("architecture") != "flat" or classifier is None
                or bundle.get("model_version") != V5_VERSION
                or bundle.get("supported_classes") != PROJECT_CLASSES
                or bundle.get("embedding_model") != metadata.get("embedding_model")
                or bundle.get("embedding_normalization") != metadata.get("embedding_normalization")):
            raise ValueError("v5 classifier bundle is incompatible")
        self._validate_classes(classifier.classes_)

        from sentence_transformers import SentenceTransformer

        embedding_model = SentenceTransformer(
            metadata["embedding_model"], device="cpu", local_files_only=True
        )
        self.metadata = metadata
        self.classifier = classifier
        self.embedding_model = embedding_model
        self.model_version = V5_VERSION
        self.artifact_integrity = "verified"

    @property
    def ready(self):
        return self.error is None and self.classifier is not None and self.metadata is not None

    @property
    def classes_(self):
        return self.classifier.classes_ if self.classifier is not None else []

    def predict_proba(self, texts, context=None):
        """Return scores for answer-only inference.

        ``context`` is intentionally accepted as a forward-compatible seam for
        a validated contextual model. The current v3/v4 artifacts are
        answer-only models, so context is not incorporated or retained.
        """
        if not self.ready:
            raise RuntimeError(self.error or "Emotion model is not ready")
        if self.model_version in {V4_VERSION, V5_VERSION, CONTEXTUAL_VERSION}:
            encoder_texts = format_model_inputs(
                texts, context, self.metadata or {},
                tokenizer=getattr(self.embedding_model, "tokenizer", None),
                max_tokens=max(1, int(getattr(self.embedding_model, "max_seq_length", 258)) - 2),
            )
            if self.model_version == V5_VERSION:
                encoder_texts = [re.sub(r"\s+", " ", str(text)).strip() for text in encoder_texts]
            features = self.embedding_model.encode(
                encoder_texts, batch_size=32, show_progress_bar=False,
                normalize_embeddings=True,
            )
            return self.classifier.predict_proba(features)
        return self.classifier.predict_proba(list(texts))


def load_runtime(requested_version=None):
    return EmotionModelRuntime(requested_version=requested_version)
