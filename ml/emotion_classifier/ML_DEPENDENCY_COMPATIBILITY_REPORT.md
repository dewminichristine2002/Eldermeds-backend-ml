# ML Dependency Compatibility Report

Audited the active `backend/ml/emotion_classifier/.venv` on 2026-10-09 (Python 3.12.1): scikit-learn 1.9.0, sentence-transformers 5.7.0, transformers 5.15.1, torch 2.13.0, NumPy 2.5.2, SciPy 1.18.1, joblib 1.5.3, pandas 3.0.5, FastAPI 0.141.1, Uvicorn 0.52.4, httpx 0.28.1; Flask is not installed. Detailed versions and artifact hashes are in `stage7b_stage8_experiments/environment_audit.json`.

The current broad `requirements.txt` permits scikit-learn >=1.3 and sentence-transformers <6, which does not describe the v5 build environment. v5 metadata requires scikit-learn 1.8.0 and sentence-transformers 6.0.0. Loading it under the current environment is rejected by runtime compatibility checks; the prior Stage 7A replay nevertheless emitted sklearn pickle-version warnings and matched outputs, which is not a compatibility guarantee. v4 lacks recorded training dependency versions; it loads and passes runtime tests in this venv, while pickle warnings are emitted. Its artifact checksum is verified before deserialization.

Direct pins are provided in `requirements-v4-runtime.txt`, `requirements-v5-experiment.txt`, and `requirements-context-experiment.txt`. These are not complete hash locks; the original v5 PyTorch/Transformers wheel and transitive versions were not captured. A clean network installation was not attempted. Recreate isolated environments with Python 3.12.1 and the respective requirements file; run the model compatibility tests before using any artifact. Do not suppress pickle warnings or load an artifact from a mismatched environment for promotion.

Recreation commands from this directory (use a Python 3.12.1 interpreter):

```powershell
py -3.12 -m venv .venv-v4-runtime
.\.venv-v4-runtime\Scripts\python.exe -m pip install -r requirements-v4-runtime.txt
py -3.12 -m venv .venv-v5-experiment
.\.venv-v5-experiment\Scripts\python.exe -m pip install -r requirements-v5-experiment.txt
py -3.12 -m venv .venv-context-experiment
.\.venv-context-experiment\Scripts\python.exe -m pip install -r requirements-context-experiment.txt
```

The v5 recipe intentionally requests sentence-transformers 6.0.0 even though the legacy broad requirements file says `<6`; keep the environments isolated. These commands are provided but were not executed in this restricted session.
