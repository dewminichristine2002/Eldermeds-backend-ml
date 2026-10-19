"""

Phase 9 — continuous learning pipeline (no formal migration files).



1. Export consented anonymized feedback corpus

2. Apply differential privacy to aggregate feedback statistics

3. Export training dataset from PostgreSQL

4. Run drift monitor (PSI + accuracy drop + quarterly schedule)

5. Retrain when triggers fire (or when --force flag is passed)



Run from backend/:  npm run ml:retrain

"""

from __future__ import annotations



import json

import os

import subprocess

import sys

from datetime import datetime, timezone

from pathlib import Path





ROOT = Path(__file__).resolve().parent

BACKEND_ROOT = ROOT.parent

RETRAIN_STATE_PATH = ROOT / "models" / "retrain_state.json"

DRIFT_REPORT_PATH = ROOT / "models" / "drift_monitor_report.json"





def resolve_python() -> str:

    if os.environ.get("ML_PYTHON_PATH"):

        return os.environ["ML_PYTHON_PATH"]



    venv_win = ROOT / ".venv" / "Scripts" / "python.exe"

    if venv_win.exists():

        return str(venv_win)



    venv_unix = ROOT / ".venv" / "bin" / "python"

    if venv_unix.exists():

        return str(venv_unix)



    return sys.executable





def run_step(label: str, command: list[str], cwd: Path) -> None:

    print(f"[retrain] {label}...")

    result = subprocess.run(command, cwd=str(cwd), check=False)

    if result.returncode != 0:

        raise RuntimeError(f"{label} failed with exit code {result.returncode}")





def should_retrain(force: bool) -> bool:

    if force:

        return True

    if not DRIFT_REPORT_PATH.exists():

        return True

    report = json.loads(DRIFT_REPORT_PATH.read_text(encoding="utf8"))

    return report.get("recommended_action") == "retrain"





def write_retrain_state() -> None:

    RETRAIN_STATE_PATH.parent.mkdir(exist_ok=True)

    payload = {

        "lastRetrainAt": datetime.now(timezone.utc).isoformat(),

        "triggerSource": "retrain_pipeline.py",

    }

    RETRAIN_STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf8")





def main() -> None:

    skip_export = "--skip-export" in sys.argv

    force = "--force" in sys.argv



    if not skip_export:

        run_step("Export consented anonymized feedback", ["node", "ml/scripts/exportFeedbackDataset.js"], BACKEND_ROOT)

        run_step("Apply differential privacy to feedback aggregates", [resolve_python(), "feedback_privacy.py"], ROOT)

        run_step("Export dataset from PostgreSQL", ["node", "ml/scripts/exportDataset.js"], BACKEND_ROOT)



    run_step("Drift monitor (PSI + accuracy + quarterly)", [resolve_python(), "drift_monitor.py"], ROOT)



    if should_retrain(force):

        run_step("Train production XGBoost model", [resolve_python(), "train_production_model.py"], ROOT)

        write_retrain_state()

        print("[retrain] Retraining completed. Restart API to load new baseline_model.joblib (XGBoost).")

    else:

        print("[retrain] No retraining triggers active; monitoring only.")





if __name__ == "__main__":

    main()

