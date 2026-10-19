"""Explicit model identifiers and immutable artifact expectations."""
from model_runtime import PROJECT_CLASSES, V3_VERSION, V4_VERSION, V5_VERSION

MODEL_REGISTRY = {
    V3_VERSION: {
        "status": "rollback",
        "artifact": "advanced_experiment/baseline_v3/emotion_pipeline_v3.pkl",
        "artifact_sha256": "b239e1ee1a257c4693e7400282d1154806e8eaf9851656eceed36d89cb0a033b",
        "input_format": "answer_only", "context_aware": False,
        "supported_classes": tuple(PROJECT_CLASSES),
    },
    V4_VERSION: {
        "status": "production_default",
        "artifact": "domain_hierarchical_experiment/candidate_model.joblib",
        "metadata": "production_model_metadata_v4.json",
        "artifact_sha256": "8c93ccc4a87c975b5a58d3ef8a42be3efc41b71e52653f05879cf45e50d82ee7",
        "input_format": "answer_only", "context_aware": False,
        "supported_classes": tuple(PROJECT_CLASSES),
    },
    V5_VERSION: {
        "status": "experimental_disabled_by_default",
        "artifact": "v5_experiments/<run>/minilm_logistic_regression_v5_answer_only.joblib",
        "artifact_sha256": "verified_from_adjacent_metadata",
        "input_format": "answer_only", "context_aware": False,
        "supported_classes": tuple(PROJECT_CLASSES),
        "training_scikit_learn": "1.8.0", "training_sentence_transformers": "6.0.0",
    },
    "minilm_logistic_regression_v6_contextual": {
        "status": "unavailable_waiting_for_validated_data",
        "artifact": None, "input_format": "multi_turn", "context_aware": True,
        "supported_classes": tuple(PROJECT_CLASSES),
    },
}


def get_registration(model_version: str):
    try:
        return MODEL_REGISTRY[model_version]
    except KeyError as exc:
        raise ValueError(f"Unsupported EMOTION_MODEL_VERSION: {model_version}") from exc
