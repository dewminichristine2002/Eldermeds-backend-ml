"""
Build ML training data from OpenFDA FAERS (real labels) + local RxNorm / SIDER / DDInter.

Architecture:
  FAERS        -> adr_event (y), patient demographics
  RxNorm       -> ingredient normalization (activesubstance / medicinalproduct)
  SIDER/MedDRA -> sider_adr_count feature
  DDInter      -> ddi_pair_count, ddi_severity_max features (co-drugs in same report)

Output: backend/ml/data/faers_adrs.csv

Run: npm run ml:build-faers
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import re
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

from feature_schema import FEATURE_COLUMNS, TARGET_COLUMN


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_PATH = DATA_DIR / "faers_adrs.csv"
METADATA_PATH = DATA_DIR / "faers_build_metadata.json"


def write_csv_atomic(df: pd.DataFrame, target_path: Path) -> Path:
    """Write via temp file to reduce OneDrive / editor lock failures on Windows."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
    try:
        df.to_csv(temp_path, index=False)
        temp_path.replace(target_path)
        return target_path
    except PermissionError:
        fallback = target_path.with_name(
            f"{target_path.stem}_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}{target_path.suffix}"
        )
        df.to_csv(fallback, index=False)
        print(
            f"[faers] WARNING: could not overwrite {target_path} (file locked). "
            f"Wrote fallback -> {fallback}"
        )
        print("[faers] Close faers_adrs.csv in Excel/editor, then re-run or rename the fallback file.")
        return fallback
    finally:
        if temp_path.exists() and not target_path.exists():
            try:
                temp_path.replace(target_path)
            except OSError:
                pass

DEFAULT_DATASET_ROOT = Path(r"C:\Users\thyag\OneDrive\Desktop\Y4S1\Research\Datasets")
DATASET_ROOT = Path(os.environ.get("MED_DATASET_ROOT", str(DEFAULT_DATASET_ROOT)))

LIMIT_PER_QUERY = max(10, min(100, int(os.environ.get("FAERS_LIMIT_PER_QUERY", "100"))))
_requested_pages = int(os.environ.get("FAERS_SKIP_PAGES", "20"))
# Full fetch defaults to 20 pages × 100 rows; FAERS_QUICK_TEST=1 allows a fast smoke test.
SKIP_PAGES = max(1, _requested_pages)
if os.environ.get("FAERS_QUICK_TEST") != "1" and _requested_pages < 20:
    SKIP_PAGES = 20
REQUEST_SLEEP_SEC = float(os.environ.get("FAERS_REQUEST_SLEEP", "0.25"))
TARGET_ROW_COUNT = max(0, int(os.environ.get("FAERS_TARGET_ROWS", "10000")))
OPENFDA_BASE = "https://api.fda.gov/drug/event.json"

# Section — ADR types for FAERS search (MedDRA PT terms)
ADR_REACTION_QUERIES = [
    {"reaction": "anaphylaxis", "adr_event": 1, "allergy_severity_max": 3, "category": "severe_allergy"},
    {"reaction": "angioedema", "adr_event": 1, "allergy_severity_max": 3, "category": "severe_allergy"},
    {"reaction": "hypersensitivity", "adr_event": 0, "allergy_severity_max": 2, "category": "allergy"},
    {"reaction": "rash", "adr_event": 0, "allergy_severity_max": 1, "category": "allergic_skin"},
    {"reaction": "hepatotoxicity", "adr_event": 1, "allergy_severity_max": 0, "category": "hepatotoxicity"},
    {"reaction": "acute kidney injury", "adr_event": 1, "allergy_severity_max": 0, "category": "nephrotoxicity"},
    {"reaction": "stevens-johnson syndrome", "adr_event": 1, "allergy_severity_max": 0, "category": "critical_adr"},
    {"reaction": "drug interaction", "adr_event": 1, "allergy_severity_max": 0, "category": "drug_interaction"},
    {"reaction": "seizure", "adr_event": 1, "allergy_severity_max": 0, "category": "neurological_adr"},
    {"reaction": "haemorrhage", "adr_event": 1, "allergy_severity_max": 0, "category": "bleeding"},
]

NEGATIVE_CONTROL_QUERY = "serious:2"

RXNORM_TTY_PRIORITY = {
    "IN": 1,
    "PIN": 2,
    "MIN": 3,
    "SCD": 4,
    "SBD": 5,
    "BN": 6,
}

DDINTER_LEVEL_ORDINAL = {"minor": 1, "moderate": 2, "major": 3}

# FAERS chronic flags — keyword extraction from drugindication + narrative (100% FAERS-derived)
CONDITION_KEYWORDS = {
    "has_renal_disease": ["renal", "kidney", "nephrop", "ckd", "dialysis", "nephro"],
    "has_hepatic_disease": ["hepat", "liver", "cirrhosis", "hep b", "hep c", "jaundice"],
    "has_diabetes": ["diabet", "insulin", "hyperglycaem", "hyperglycem", "type 2 dm", "type 1 dm"],
    "has_cardiovascular": [
        "cardiac",
        "heart",
        "hypertension",
        "coronary",
        "atrial",
        "arrhythmia",
        "anticoagul",
        "myocard",
        "stroke",
    ],
    "has_epilepsy": ["epilep", "seizure", "anticonvuls", "levetirac", "convulsion"],
}

# Evaluation-only 3-class labels (NOT the RF training target — hybrid system uses rules at runtime).
# Warning = pharmacovigilance middle ground: DDI signal present but no severe ADR recorded.
RISK_LABEL_NAMES = ("Safe", "Warning", "Dangerous")
RISK_LABEL_TO_CODE = {"Safe": 0, "Warning": 1, "Dangerous": 2}

MAX_PATIENT_AGE_YEARS = 120


def normalize_drug_name(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|iu|units?)\b", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_medicinal_product(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text)
    return text.strip()


class RxNormIndex:
    """Normalized drug name -> RxNorm ingredient (IN preferred)."""

    def __init__(self) -> None:
        self.by_name: dict[str, dict] = {}
        self.in_by_cui: dict[str, str] = {}

    @classmethod
    def from_zip(cls, zip_path: Path) -> "RxNormIndex":
        index = cls()
        if not zip_path.exists():
            print(f"[faers] RxNorm zip missing: {zip_path}")
            return index

        print(f"[faers] Loading RxNorm from {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as zf:
            entry = next((n for n in zf.namelist() if n.endswith("RXNCONSO.RRF")), None)
            if not entry:
                raise FileNotFoundError("RXNCONSO.RRF not found in RxNorm zip")
            with zf.open(entry) as handle:
                for raw_line in handle:
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
                    parts = line.split("|")
                    if len(parts) < 15:
                        continue
                    rxcui, lat, sab, tty, raw_name = parts[0], parts[1], parts[11], parts[12], parts[14]
                    if lat != "ENG" or sab != "RXNORM" or not raw_name.strip():
                        continue
                    normalized = normalize_drug_name(raw_name)
                    if not normalized:
                        continue
                    candidate = {
                        "rxnorm_cui": rxcui,
                        "ingredient_name": raw_name.strip(),
                        "tty": tty,
                        "normalized_name": normalized,
                    }
                    current = index.by_name.get(normalized)
                    if current is None or RXNORM_TTY_PRIORITY.get(tty, 99) < RXNORM_TTY_PRIORITY.get(
                        current["tty"], 99
                    ):
                        index.by_name[normalized] = candidate
                    if tty == "IN":
                        index.in_by_cui[rxcui] = raw_name.strip()

        print(f"[faers] RxNorm names indexed: {len(index.by_name)}")
        return index

    def resolve(self, *names: object) -> dict | None:
        for name in names:
            normalized = normalize_drug_name(clean_medicinal_product(name))
            if not normalized:
                continue
            hit = self.by_name.get(normalized)
            if hit:
                ingredient = hit["ingredient_name"]
                if hit["tty"] != "IN":
                    in_name = self.in_by_cui.get(hit["rxnorm_cui"])
                    if in_name:
                        ingredient = in_name
                return {
                    "rxnorm_cui": hit["rxnorm_cui"],
                    "ingredient_name": ingredient,
                    "normalized_ingredient": normalize_drug_name(ingredient),
                    "rxnorm_tty": hit["tty"],
                }
            # first token fallback (e.g. "vancomycin manufacturer unknown")
            token = normalized.split(" ")[0]
            if token and token != normalized:
                hit = self.by_name.get(token)
                if hit:
                    ingredient = hit["ingredient_name"]
                    if hit["tty"] != "IN":
                        in_name = self.in_by_cui.get(hit["rxnorm_cui"])
                        if in_name:
                            ingredient = in_name
                    return {
                        "rxnorm_cui": hit["rxnorm_cui"],
                        "ingredient_name": ingredient,
                        "normalized_ingredient": normalize_drug_name(ingredient),
                        "rxnorm_tty": hit["tty"],
                    }
        return None


class SiderIndex:
    """SIDER drug_names.tsv + meddra_all_se.tsv.gz -> side-effect counts."""

    def __init__(self) -> None:
        self.name_to_cid: dict[str, str] = {}
        self.cid_effect_count: dict[str, int] = {}

    @classmethod
    def from_files(cls, drug_names_path: Path, meddra_path: Path) -> "SiderIndex":
        index = cls()
        if drug_names_path.exists():
            for line in drug_names_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip() or "\t" not in line:
                    continue
                cid, raw_name = line.split("\t", 1)
                name = normalize_drug_name(raw_name)
                if name and name not in index.name_to_cid:
                    index.name_to_cid[name] = cid.strip()

        if meddra_path.exists():
            open_fn = gzip.open if str(meddra_path).endswith(".gz") else open
            mode = "rt" if str(meddra_path).endswith(".gz") else "r"
            effect_sets: dict[str, set[str]] = defaultdict(set)
            with open_fn(meddra_path, mode, encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 6:
                        continue
                    cid = parts[0].strip()
                    effect = parts[5].strip().lower()
                    if cid and effect:
                        effect_sets[cid].add(effect)
            index.cid_effect_count = {cid: len(effects) for cid, effects in effect_sets.items()}

        print(f"[faers] SIDER names: {len(index.name_to_cid)}, CIDs with effects: {len(index.cid_effect_count)}")
        return index

    def side_effect_count(self, *names: object) -> int:
        for name in names:
            normalized = normalize_drug_name(name)
            if not normalized:
                continue
            cid = self.name_to_cid.get(normalized)
            if cid:
                return int(self.cid_effect_count.get(cid, 0))
        return 0


class DdinterIndex:
    """Local DDInter CSVs -> pairwise severity lookup."""

    def __init__(self) -> None:
        self.pair_level: dict[str, str] = {}

    @classmethod
    def from_directory(cls, root: Path) -> "DdinterIndex":
        index = cls()
        files = sorted(root.glob("ddinter_downloads_code_*.csv"))
        for file_path in files:
            with file_path.open("r", encoding="utf-8", errors="replace") as handle:
                reader = csv.reader(handle)
                next(reader, None)
                for row in reader:
                    if len(row) < 5:
                        continue
                    drug_a = normalize_drug_name(row[1])
                    drug_b = normalize_drug_name(row[3])
                    level = str(row[4] or "").strip().lower()
                    if not drug_a or not drug_b or not level:
                        continue
                    key = "::".join(sorted([drug_a, drug_b]))
                    current = index.pair_level.get(key)
                    if current is None or DDINTER_LEVEL_ORDINAL.get(level, 0) > DDINTER_LEVEL_ORDINAL.get(current, 0):
                        index.pair_level[key] = level
        print(f"[faers] DDInter interaction pairs: {len(index.pair_level)}")
        return index

    def co_administration_features(self, drug_norms: list[str]) -> tuple[int, int]:
        """Count DDInter hits across all co-administered drug pairs in the report."""
        unique = [d for d in dict.fromkeys(drug_norms) if d]
        if len(unique) < 2:
            return 0, 0
        count = 0
        max_ord = 0
        for i, drug_a in enumerate(unique):
            for drug_b in unique[i + 1 :]:
                key = "::".join(sorted([drug_a, drug_b]))
                level = self.pair_level.get(key)
                if level:
                    count += 1
                    max_ord = max(max_ord, DDINTER_LEVEL_ORDINAL.get(level, 0))
        return count, max_ord


def parse_faers_age(age_raw: object, age_unit: object) -> float:
    """
    Convert FAERS patientonsetage + patientonsetageunit to years.
    Units: 801=year, 802=month, 803=week, 804=day, 800=decade (per FAERS ICH codes).
    """
    if age_raw in (None, ""):
        return 0.0
    try:
        value = float(age_raw)
    except (TypeError, ValueError):
        return 0.0

    unit = str(age_unit or "801").strip()
    if unit == "800":
        years = value * 10.0
    elif unit == "802":
        years = value / 12.0
    elif unit == "803":
        years = value / 52.0
    elif unit == "804":
        years = value / 365.0
    else:
        years = value

    if years < 0 or years > MAX_PATIENT_AGE_YEARS:
        return 0.0
    return round(years, 1)


def extract_chronic_flags_from_report(report: dict) -> dict[str, int]:
    """Derive chronic condition flags from FAERS drug indications and clinical narrative."""
    patient = report.get("patient") or {}
    text_parts: list[str] = []
    for drug in patient.get("drug") or []:
        indication = drug.get("drugindication") or ""
        if indication:
            text_parts.append(str(indication).lower())
    summary = patient.get("summary") or {}
    narrative = summary.get("narrativeincludeclinical") if isinstance(summary, dict) else ""
    if narrative:
        text_parts.append(str(narrative).lower())
    full_text = " ".join(text_parts)

    return {
        condition: int(any(keyword in full_text for keyword in keywords))
        for condition, keywords in CONDITION_KEYWORDS.items()
    }


def ddi_flag_from_row(row: dict) -> int:
    if "ddi_flag" in row and row["ddi_flag"] not in (None, ""):
        return int(row["ddi_flag"])
    return int(int(row.get("ddi_pair_count", 0) or 0) > 0)


def risk_label_eval(row: dict) -> str:
    """
    Evaluation-only Safe/Warning/Dangerous — not the RF training target.

    Derivation (dissertation-defensible, FAERS + DDInter derived):
      adr_event=1                 -> Dangerous
      adr_event=0 and ddi_flag=1  -> Warning  (interaction signal, no severe ADR)
      adr_event=0 and ddi_flag=0  -> Safe
    """
    adr_event = int(row.get(TARGET_COLUMN, 0))
    if adr_event == 1:
        return "Dangerous"
    if ddi_flag_from_row(row) == 1:
        return "Warning"
    return "Safe"


def risk_label_code(row: dict) -> int:
    return RISK_LABEL_TO_CODE[risk_label_eval(row)]


def encode_patient_sex(value: object) -> str:
    text = str(value or "").strip()
    if text in {"1", "male", "m"}:
        return "male"
    if text in {"2", "female", "f"}:
        return "female"
    return "unknown"


def faers_reaction_search(reaction: str) -> str:
    """OpenFDA Lucene search — unquoted terms paginate better than quoted phrases."""
    term = reaction.strip().lower().replace(" ", "+")
    return f"patient.reaction.reactionmeddrapt:{term}"


def fetch_openfda(search: str, limit: int, skip: int) -> list[dict]:
    params = {"search": search, "limit": limit, "skip": skip}
    response = requests.get(OPENFDA_BASE, params=params, timeout=60)
    if response.status_code == 404:
        return []
    response.raise_for_status()
    payload = response.json()
    return payload.get("results", [])


def extract_primary_drug(patient: dict) -> dict | None:
    drugs = patient.get("drug") or []
    if not drugs:
        return None
    primary = next((d for d in drugs if str(d.get("drugcharacterization")) == "1"), drugs[0])
    active = (primary.get("activesubstance") or {}).get("activesubstancename")
    product = primary.get("medicinalproduct")
    return {
        "activesubstance": active,
        "medicinalproduct": product,
        "all_drugs": drugs,
    }


def all_normalized_co_drugs(drug_bundle: dict, rxnorm: RxNormIndex) -> tuple[str, str, list[str]]:
    primary_raw = drug_bundle.get("activesubstance") or drug_bundle.get("medicinalproduct")
    resolved = rxnorm.resolve(primary_raw, drug_bundle.get("medicinalproduct"))
    primary_norm = resolved["normalized_ingredient"] if resolved else normalize_drug_name(clean_medicinal_product(primary_raw))

    co_norms: list[str] = []
    for drug in drug_bundle.get("all_drugs", []):
        active = (drug.get("activesubstance") or {}).get("activesubstancename")
        product = drug.get("medicinalproduct")
        hit = rxnorm.resolve(active, product)
        norm = hit["normalized_ingredient"] if hit else normalize_drug_name(clean_medicinal_product(active or product))
        if norm:
            co_norms.append(norm)
    return primary_norm, (resolved or {}).get("ingredient_name", primary_raw or ""), list(dict.fromkeys(co_norms))


def build_feature_row(
    report: dict,
    query_meta: dict,
    rxnorm: RxNormIndex,
    sider: SiderIndex,
    ddinter: DdinterIndex,
) -> dict | None:
    patient = report.get("patient") or {}
    drug_bundle = extract_primary_drug(patient)
    if not drug_bundle:
        return None

    primary_norm, ingredient_display, co_drugs = all_normalized_co_drugs(drug_bundle, rxnorm)
    if not primary_norm:
        return None

    resolved = rxnorm.resolve(drug_bundle.get("activesubstance"), drug_bundle.get("medicinalproduct"))
    ddi_count, ddi_max = ddinter.co_administration_features(co_drugs)
    sider_count = sider.side_effect_count(
        resolved["ingredient_name"] if resolved else ingredient_display,
        primary_norm,
    )

    from feature_schema import drug_flags_from_names, encode_atc_class

    drug_flags = drug_flags_from_names(primary_norm, ingredient_display)
    atc_class = encode_atc_class("", "", primary_norm, ingredient_display)
    chronic_flags = extract_chronic_flags_from_report(report)
    age = parse_faers_age(patient.get("patientonsetage"), patient.get("patientonsetageunit"))

    row = {
        "patient_age": age,
        "patient_sex": encode_patient_sex(patient.get("patientsex")),
        "num_current_meds": len(drug_bundle.get("all_drugs") or []),
        **chronic_flags,
        # FAERS rows do not include patient allergy-history severity independent of the ADR label,
        # so keep this neutral instead of leaking the queried reaction category into training.
        "allergy_severity_max": 0,
        "allergy_class_overlap": 0,
        "ddi_severity_max": int(ddi_max),
        "ddi_pair_count": int(ddi_count),
        "ddi_flag": int(ddi_count > 0),
        "sider_adr_count": int(sider_count),
        **drug_flags,
        "atc_class_encoded": atc_class,
        TARGET_COLUMN: int(query_meta.get("adr_event", 0)),
        # traceability (ignored by trainer feature matrix)
        "faers_safetyreportid": report.get("safetyreportid", ""),
        "reaction_query": query_meta.get("reaction", ""),
        "reaction_category": query_meta.get("category", ""),
        "ingredient_name": resolved["ingredient_name"] if resolved else ingredient_display,
        "normalized_ingredient": primary_norm,
        "rxnorm_cui": resolved["rxnorm_cui"] if resolved else "",
        "rxnorm_matched": int(bool(resolved)),
        "data_source": "FAERS",
    }
    return row


def collect_rows_for_query(
    search: str,
    query_meta: dict,
    rxnorm: RxNormIndex,
    sider: SiderIndex,
    ddinter: DdinterIndex,
) -> list[dict]:
    rows: list[dict] = []
    seen_ids: set[str] = set()
    for page in range(SKIP_PAGES):
        skip = page * LIMIT_PER_QUERY
        try:
            results = fetch_openfda(search, LIMIT_PER_QUERY, skip)
        except requests.RequestException as exc:
            print(f"[faers] API error ({query_meta.get('reaction', search)}): {exc}")
            break
        if not results:
            break
        page_added = 0
        for report in results:
            report_id = str(report.get("safetyreportid", ""))
            if report_id and report_id in seen_ids:
                continue
            row = build_feature_row(report, query_meta, rxnorm, sider, ddinter)
            if row:
                rows.append(row)
                page_added += 1
                if report_id:
                    seen_ids.add(report_id)
        print(f"[faers]   page {page + 1}/{SKIP_PAGES} skip={skip} fetched={len(results)} kept={page_added}")
        time.sleep(REQUEST_SLEEP_SEC)
    return rows


def finalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Age sanity filter and evaluation-only 3-class risk column."""
    before = len(df)
    df = df[(df["patient_age"] <= MAX_PATIENT_AGE_YEARS) | (df["patient_age"] == 0)].copy()
    dropped = before - len(df)
    if dropped:
        print(f"[faers] Dropped {dropped} rows with implausible patient_age > {MAX_PATIENT_AGE_YEARS}")

    df["ddi_flag"] = (df["ddi_pair_count"].fillna(0).astype(int) > 0).astype(int)
    df["risk_label_eval"] = df.apply(lambda row: risk_label_eval(row.to_dict()), axis=1)
    df["risk_label_code"] = df.apply(lambda row: risk_label_code(row.to_dict()), axis=1)
    return df


def main() -> None:
    rxnorm_zip = Path(
        os.environ.get(
            "MED_RXNORM_ZIP_PATH",
            str(DATASET_ROOT / "RxNorm_full_prescribe_03022026 (1).zip"),
        )
    )
    drug_names = Path(os.environ.get("MED_SIDER_DRUG_NAMES_PATH", str(DATASET_ROOT / "drug_names.tsv")))
    meddra = Path(os.environ.get("MED_SIDER_SIDE_EFFECTS_PATH", str(DATASET_ROOT / "meddra_all_se.tsv.gz")))

    rxnorm = RxNormIndex.from_zip(rxnorm_zip)
    sider = SiderIndex.from_files(drug_names, meddra)
    ddinter = DdinterIndex.from_directory(DATASET_ROOT)

    all_rows: list[dict] = []
    seen_ids: set[str] = set()

    for spec in ADR_REACTION_QUERIES:
        if TARGET_ROW_COUNT and len(all_rows) >= TARGET_ROW_COUNT:
            print(f"[faers] Target row count ({TARGET_ROW_COUNT}) reached — stopping fetch.")
            break
        reaction = spec["reaction"]
        search = faers_reaction_search(reaction)
        print(f"[faers] Fetching {reaction!r} (adr_event={spec['adr_event']}) ...")
        batch = collect_rows_for_query(search, spec, rxnorm, sider, ddinter)
        added = 0
        for row in batch:
            rid = row.get("faers_safetyreportid")
            if rid and rid in seen_ids:
                continue
            all_rows.append(row)
            if rid:
                seen_ids.add(rid)
            added += 1
            if TARGET_ROW_COUNT and len(all_rows) >= TARGET_ROW_COUNT:
                break
        print(f"[faers]   +{added} unique rows (total {len(all_rows)})")

    print(f"[faers] Fetching negative controls ({NEGATIVE_CONTROL_QUERY}) ...")
    negative_meta = {
        "reaction": "non_serious_control",
        "adr_event": 0,
        "allergy_severity_max": 0,
        "category": "control",
    }
    neg_batch = collect_rows_for_query(NEGATIVE_CONTROL_QUERY, negative_meta, rxnorm, sider, ddinter)
    neg_added = 0
    for row in neg_batch:
        rid = row.get("faers_safetyreportid")
        if rid and rid in seen_ids:
            continue
        all_rows.append(row)
        if rid:
            seen_ids.add(rid)
        neg_added += 1
    print(f"[faers]   +{neg_added} control rows")

    if not all_rows:
        raise RuntimeError("No FAERS rows collected. Check network/API limits.")

    df = pd.DataFrame(all_rows)
    if df[TARGET_COLUMN].nunique() < 2:
        raise ValueError("FAERS dataset needs both adr_event=0 and adr_event=1 rows.")

    df = finalize_dataframe(df)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    written_path = write_csv_atomic(df, OUTPUT_PATH)

    rxnorm_match_rate = float(df["rxnorm_matched"].mean()) if "rxnorm_matched" in df.columns else 0.0
    chronic_stats = {
        col: int(df[col].sum()) for col in CONDITION_KEYWORDS if col in df.columns
    }
    ddi_nonzero = int((df["ddi_pair_count"] > 0).sum()) if "ddi_pair_count" in df.columns else 0
    risk_eval_counts = (
        df["risk_label_eval"].value_counts().to_dict() if "risk_label_eval" in df.columns else {}
    )
    metadata = {
        "section": "FAERS_training_pipeline",
        "output": str(written_path),
        "rows": int(len(df)),
        "adr_event_positive": int((df[TARGET_COLUMN] == 1).sum()),
        "adr_event_negative": int((df[TARGET_COLUMN] == 0).sum()),
        "rxnorm_match_rate": rxnorm_match_rate,
        "chronic_flag_positive_counts": chronic_stats,
        "ddi_rows_with_signal": ddi_nonzero,
        "ddi_signal_rate": round(ddi_nonzero / len(df), 4) if len(df) else 0.0,
        "risk_label_eval_distribution": risk_eval_counts,
        "queries": ADR_REACTION_QUERIES,
        "negative_control_query": NEGATIVE_CONTROL_QUERY,
        "limit_per_query": LIMIT_PER_QUERY,
        "skip_pages": SKIP_PAGES,
        "dataset_root": str(DATASET_ROOT),
        "architecture": {
            "labels": "OpenFDA FAERS binary adr_event (severe vs not severe)",
            "chronic_flags": "keyword extraction from FAERS drugindication + narrative",
            "risk_label_eval": "evaluation-only Safe/Warning/Dangerous — not RF training target",
            "normalization": "RxNorm RXNCONSO",
            "features": "DDInter all-pair co-administration + SIDER side-effect counts + patient/drug flags",
            "model": "RandomForest (train_baseline.py)",
            "runtime_risk": "hybrid 0.6 rule + 0.4 ML -> thresholds 34/36",
            "rules": "clinical rule engine (not trained)",
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[faers] Wrote {written_path} ({len(df)} rows)")
    print(
        f"[faers] adr_event: positive={(df[TARGET_COLUMN]==1).sum()}, "
        f"negative={(df[TARGET_COLUMN]==0).sum()}, rxnorm_match={rxnorm_match_rate:.1%}"
    )
    print(f"[faers] chronic flags (positive rows): {chronic_stats}")
    print(f"[faers] DDI signal: {ddi_nonzero}/{len(df)} rows ({metadata['ddi_signal_rate']:.1%})")
    print(f"[faers] risk_label_eval: {risk_eval_counts}")


if __name__ == "__main__":
    main()
