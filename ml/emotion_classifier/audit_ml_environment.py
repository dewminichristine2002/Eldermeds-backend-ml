"""Report installed ML packages and model artifact integrity without loading models."""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGES = ["scikit-learn", "sentence-transformers", "transformers", "torch", "numpy", "scipy",
            "joblib", "Flask", "fastapi", "uvicorn", "pandas", "httpx"]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    v3 = ROOT / "advanced_experiment" / "baseline_v3" / "emotion_pipeline_v3.pkl"
    v4 = ROOT / "domain_hierarchical_experiment" / "candidate_model.joblib"
    v5_metadata = ROOT / "v5_experiments" / "20261008T175725Z" / "metadata.json"
    report = {
        "python": platform.python_version(), "implementation": platform.python_implementation(),
        "packages": {name: package_version(name) for name in PACKAGES},
        "artifacts": {
            "v3": {"path": str(v3.relative_to(ROOT)), "sha256": file_hash(v3)},
            "v4": {"path": str(v4.relative_to(ROOT)), "sha256": file_hash(v4)},
            "v5": json.loads(v5_metadata.read_text(encoding="utf-8")),
        },
        "requirements": (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines(),
        "notes": [
            "v5 metadata records scikit-learn 1.8.0 and sentence-transformers 6.0.0; current installed versions are compared in packages.",
            "v4 metadata does not record its training dependency versions. It loaded and passed runtime tests in the current environment, with sklearn artifact compatibility warnings observed.",
            "No Flask emotion endpoint was found. The Node emotion prediction client uses the FastAPI /predict-emotion endpoint on port 8001.",
        ],
    }
    output = ROOT / "stage7b_stage8_experiments" / "environment_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
