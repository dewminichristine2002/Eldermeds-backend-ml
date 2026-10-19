from __future__ import annotations

from pathlib import Path

import pandas as pd

from feature_schema import FEATURE_COLUMNS, TARGET_COLUMN, dataframe_to_features, derive_adr_target


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FAERS_PATH = DATA_DIR / "faers_adrs.csv"
FAERS_METADATA_PATH = DATA_DIR / "faers_build_metadata.json"
OMOP_PATH = DATA_DIR / "omop_synthetic_adrs.csv"
LEGACY_TRAINING_PATH = DATA_DIR / "legacy" / "medicine_safety_training_dataset.csv"
EXPORTED_DATA_PATH = DATA_DIR / "medicine_safety_dataset.csv"

SPEC_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def _read_csv_if_exists(path: Path, source_label: str) -> tuple[pd.DataFrame, str | None]:
    if not path.exists():
        return pd.DataFrame(), None
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(), None
    df = df.copy()
    df["_data_source"] = source_label
    return df, source_label


def resolve_faers_path() -> Path:
    """Prefer metadata output or newest enriched CSV when faers_adrs.csv is stale/locked."""
    candidates: list[Path] = []
    if FAERS_METADATA_PATH.exists():
        try:
            import json

            meta = json.loads(FAERS_METADATA_PATH.read_text(encoding="utf-8"))
            meta_path = Path(meta.get("output", ""))
            if meta_path.exists():
                candidates.append(meta_path)
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    candidates.extend(sorted(DATA_DIR.glob("faers_adrs*.csv"), key=lambda p: p.stat().st_mtime, reverse=True))
    if FAERS_PATH.exists():
        candidates.append(FAERS_PATH)

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        try:
            header = resolved.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        except OSError:
            continue
        if "risk_label_eval" in header or path == FAERS_PATH.resolve():
            return resolved

    if FAERS_PATH.exists():
        return FAERS_PATH
    raise FileNotFoundError(
        f"No FAERS dataset found under {DATA_DIR}. Run `npm run ml:build-faers` first."
    )


def load_faers() -> tuple[pd.DataFrame, str | None]:
    path = resolve_faers_path()
    return _read_csv_if_exists(path, "FAERS")


def load_omop() -> tuple[pd.DataFrame, str | None]:
    return _read_csv_if_exists(OMOP_PATH, "OMOP CDM synthetic")


def load_optional_app_export() -> tuple[pd.DataFrame, list[str]]:
    """Optional small real-app supplement (RxNorm-normalized checks). Disabled by default."""
    import os

    if os.environ.get("ML_INCLUDE_APP_EXPORT", "").lower() not in {"1", "true", "yes"}:
        return pd.DataFrame(), []

    if not EXPORTED_DATA_PATH.exists():
        return pd.DataFrame(), []

    df = pd.read_csv(EXPORTED_DATA_PATH)
    if df.empty:
        return pd.DataFrame(), []
    df = df.copy()
    df["_data_source"] = EXPORTED_DATA_PATH.name
    return df, [EXPORTED_DATA_PATH.name]


def _merge_training_frames() -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    sources: list[str] = []

    faers_df, faers_label = load_faers()
    if faers_label:
        frames.append(faers_df)
        sources.append(faers_label)

    omop_df, omop_label = load_omop()
    if omop_label:
        frames.append(omop_df)
        sources.append(omop_label)

    export_df, export_names = load_optional_app_export()
    if not export_df.empty:
        frames.append(export_df)
        sources.extend(export_names)

    if not frames:
        raise FileNotFoundError(
            "No training data found. Run `npm run ml:build-faers` first, "
            "or set ML_INCLUDE_APP_EXPORT=1 with medicine_safety_dataset.csv."
        )

    merged = pd.concat(frames, ignore_index=True, sort=False)
    if merged.empty:
        raise ValueError("Merged training dataset is empty.")
    return merged, sources


def load_raw_training_data() -> tuple[pd.DataFrame, list[str]]:
    return _merge_training_frames()


def load_training_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    merged, sources = _merge_training_frames()
    X = dataframe_to_features(merged)
    y = derive_adr_target(merged)
    if y.nunique() < 2:
        raise ValueError(
            f"Target '{TARGET_COLUMN}' needs at least two classes. "
            "Add severe ADR examples or check label mapping."
        )
    return X, y, sources
