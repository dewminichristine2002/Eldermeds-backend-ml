from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DRUG_CLASS_DATA_PATH = ROOT / "data" / "drug_class_dataset.json"

CONTINUOUS_FEATURES = [
    "patient_age",
    "num_current_meds",
    "ddi_pair_count",
    "sider_adr_count",
]
BINARY_FEATURES = [
    "has_renal_disease",
    "has_hepatic_disease",
    "has_diabetes",
    "has_cardiovascular",
    "has_epilepsy",
    "allergy_class_overlap",
    "drug_hepatic_metabolism",
    "drug_renal_excretion",
    "nti_drug_flag",
    "ddi_flag",
    "rxnorm_matched",
]
ORDINAL_FEATURES = [
    "allergy_severity_max",
    "ddi_severity_max",
]
CATEGORICAL_FEATURES = [
    "patient_sex",
    "atc_class_encoded",
]
FEATURE_COLUMNS = CONTINUOUS_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "adr_event"

# Plain-language clinical explanations for ML explainability (Section 11.3)
CLINICAL_FEATURE_EXPLANATIONS: dict[str, str] = {
    "patient_age": "Older age can increase sensitivity to adverse drug reactions and drug accumulation.",
    "num_current_meds": "Taking many medicines at once raises the chance of interactions and side effects.",
    "ddi_pair_count": "Number of known drug–drug interactions increases combined toxicity risk.",
    "sider_adr_count": "Known side-effect burden from drug databases (SIDER) for this medicine.",
    "has_renal_disease": "Kidney disease may slow drug clearance and worsen adverse effects.",
    "has_hepatic_disease": "Liver disease may impair drug metabolism and increase toxicity risk.",
    "has_diabetes": "Diabetes can alter drug response and complication risk with some medicines.",
    "has_cardiovascular": "Heart or blood-vessel disease may increase risk with certain drug classes.",
    "has_epilepsy": "Seizure disorders may interact with medicines that affect the nervous system.",
    "allergy_class_overlap": "Medicine belongs to a drug class you may already be allergic to.",
    "drug_hepatic_metabolism": "This medicine is mainly processed by the liver — hepatic risk factor.",
    "drug_renal_excretion": "This medicine is mainly cleared by the kidneys — renal risk factor.",
    "nti_drug_flag": "Narrow therapeutic index — small dose changes can cause harm.",
    "ddi_flag": "Binary signal that at least one known drug–drug interaction was detected.",
    "rxnorm_matched": "Medicine name matched successfully to the RxNorm-style knowledge base.",
    "allergy_severity_max": "Severity of documented allergy history (none → severe).",
    "ddi_severity_max": "Highest severity among detected drug–drug interactions.",
    "patient_sex": "Biological sex may influence metabolism and ADR susceptibility.",
    "atc_class_encoded": "WHO ATC therapeutic class — some classes carry higher ADR profiles.",
}


def clinical_explanation_for_feature(feature_name: str, value: object = None, shap_value: float | None = None) -> str:
    base = feature_name.split("_")[0] if feature_name.startswith("patient_sex_") or feature_name.startswith("atc_class_encoded_") else feature_name
    # One-hot expanded names: patient_sex_male → patient_sex
    for key in CLINICAL_FEATURE_EXPLANATIONS:
        if feature_name == key or feature_name.startswith(f"{key}_"):
            base = key
            break
    text = CLINICAL_FEATURE_EXPLANATIONS.get(base, f"Model factor: {feature_name.replace('_', ' ')}.")
    if shap_value is not None:
        direction = "increases" if shap_value > 0 else "decreases" if shap_value < 0 else "does not change"
        text = f"{text} In this check it {direction} estimated ADR risk."
    if value is not None and value != "":
        text = f"{text} (value: {value})"
    return text


def enrich_shap_contributions(contributions: list[dict]) -> list[dict]:
    enriched = []
    cumulative = 0.0
    for item in contributions:
        shap_val = float(item.get("shap", 0))
        cumulative += shap_val
        enriched.append(
            {
                **item,
                "clinicalExplanation": clinical_explanation_for_feature(
                    str(item.get("feature", "")),
                    item.get("value"),
                    shap_val,
                ),
                "waterfallCumulative": round(cumulative, 6),
            }
        )
    return enriched

ATC_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUV")

NTI_TERMS = [
    "warfarin",
    "digoxin",
    "lithium",
    "phenytoin",
    "theophylline",
    "levothyroxine",
    "methotrexate",
    "cyclosporine",
]
RENAL_EXCRETION_TERMS = [
    "metformin",
    "digoxin",
    "lithium",
    "atenolol",
    "nitrofurantoin",
    "allopurinol",
]
HEPATIC_METABOLISM_TERMS = [
    "warfarin",
    "statins",
    "atorvastatin",
    "simvastatin",
    "carbamazepine",
    "phenytoin",
    "paracetamol",
    "acetaminophen",
]

CLASS_ALIAS_MAP = {
    "penicillin": ["penicillin", "penicillin antibiotic", "beta-lactam", "beta lactam"],
    "cephalosporin": ["cephalosporin", "cephalosporins"],
    "macrolide": ["macrolide", "macrolide antibiotic"],
    "fluoroquinolone": ["fluoroquinolone", "quinolone", "quinolone antibiotic"],
    "sulfonamide": ["sulfonamide", "sulfa", "sulfa drug", "sulfa drugs"],
    "nsaid": ["nsaid", "nonsteroidal anti-inflammatory", "non steroidal anti inflammatory"],
}


def normalize_drug_name(value: object) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("(", " ")
        .replace(")", " ")
    )


def count_text_items(value: object) -> int:
    if pd.isna(value):
        return 0
    text = str(value).strip()
    if not text or text.lower() in {"none", "no", "n/a", "na", "nil"}:
        return 0
    parts = [part.strip() for part in text.replace(";", ",").split(",")]
    return sum(1 for part in parts if part)


def chronic_flags_from_text(text: object) -> dict[str, int]:
    normalized = str(text or "").lower()
    return {
        "has_renal_disease": int(any(term in normalized for term in ["kidney", "renal", "ckd", "dialysis", "nephro"])),
        "has_hepatic_disease": int(any(term in normalized for term in ["liver", "hepatic", "cirrhosis", "jaundice"])),
        "has_diabetes": int(any(term in normalized for term in ["diabetes", "diabetic"])),
        "has_cardiovascular": int(
            any(term in normalized for term in ["hypertension", "heart", "cardiac", "cardiovascular", "stroke", "coronary"])
        ),
        "has_epilepsy": int(any(term in normalized for term in ["epilepsy", "seizure", "convulsion"])),
    }


def drug_flags_from_names(*values: object) -> dict[str, int]:
    combined = " ".join(normalize_drug_name(value) for value in values)
    return {
        "drug_hepatic_metabolism": int(any(term in combined for term in HEPATIC_METABOLISM_TERMS)),
        "drug_renal_excretion": int(any(term in combined for term in RENAL_EXCRETION_TERMS)),
        "nti_drug_flag": int(any(term in combined for term in NTI_TERMS)),
    }


def encode_patient_sex(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"m", "male", "1"}:
        return "male"
    if text in {"f", "female", "2"}:
        return "female"
    return "unknown"


def ddi_severity_to_ordinal(value: object) -> int:
    text = str(value or "").strip().lower()
    mapping = {"none": 0, "low": 1, "minor": 1, "medium": 2, "moderate": 2, "high": 3, "major": 3, "severe": 3}
    if text.isdigit():
        return max(0, min(3, int(text)))
    return mapping.get(text, 0)


def allergy_severity_from_row(row: pd.Series) -> int:
    if "allergy_severity_max" in row and pd.notna(row["allergy_severity_max"]):
        return max(0, min(3, int(row["allergy_severity_max"])))

    score = 0
    if str(row.get("has_medicine_allergy", "")).lower() in {"true", "1", "yes"}:
        score = max(score, 1)
    if int(row.get("same_class_allergy", row.get("allergy_class_overlap", 0)) or 0):
        score = max(score, 2)
    if str(row.get("q_past_reaction", "")).strip().lower() in {"yes", "true", "1"}:
        score = max(score, 2)
    return score


@lru_cache(maxsize=1)
def load_drug_class_lookup() -> dict[str, dict]:
    if not DRUG_CLASS_DATA_PATH.exists():
        return {}
    try:
        payload = json.loads(DRUG_CLASS_DATA_PATH.read_text(encoding="utf8"))
    except json.JSONDecodeError:
        return {}

    lookup: dict[str, dict] = {}
    for record in payload.get("records", []):
        normalized_name = normalize_drug_name(
            record.get("normalized_name") or record.get("drug_name") or record.get("ingredient_name")
        )
        if normalized_name and normalized_name not in lookup:
            lookup[normalized_name] = record
    return lookup


def resolve_drug_class(*values: object) -> str:
    lookup = load_drug_class_lookup()
    for value in values:
        normalized = normalize_drug_name(value)
        if normalized and normalized in lookup:
            drug_class = str(lookup[normalized].get("drug_class") or "").strip().lower()
            if drug_class:
                return drug_class
    return "unknown"


def extract_drug_classes_from_text(value: object) -> set[str]:
    text = normalize_drug_name(value)
    if not text:
        return set()

    found: set[str] = set()
    for segment in [part.strip() for part in re.split(r"[,;/\n]+", text) if part.strip()]:
        drug_class = resolve_drug_class(segment)
        if drug_class != "unknown":
            found.add(drug_class)

    for drug_class, aliases in CLASS_ALIAS_MAP.items():
        if any(alias in text for alias in aliases):
            found.add(drug_class)
    return found


def encode_atc_class(*values: object) -> str:
    if any(values):
        for value in values:
            text = str(value or "").strip().upper()
            if text and text[0] in ATC_LETTERS:
                return text[0]

    lookup = load_drug_class_lookup()
    for value in values:
        normalized = normalize_drug_name(value)
        if normalized and normalized in lookup:
            atc_code = str(lookup[normalized].get("atc_code") or "").strip().upper()
            if atc_code and atc_code[0] in ATC_LETTERS:
                return atc_code[0]
    return "unknown"


def same_class_allergy_from_row(row: pd.Series) -> int:
    if "allergy_class_overlap" in row and pd.notna(row["allergy_class_overlap"]):
        return int(bool(row["allergy_class_overlap"]))

    drug_class = resolve_drug_class(
        row.get("normalized_drug_name", ""),
        row.get("ingredient_name", ""),
        row.get("medicine_name", ""),
    )
    if drug_class == "unknown":
        return 0
    profile_classes = extract_drug_classes_from_text(row.get("known_allergies_text", "")) | extract_drug_classes_from_text(
        row.get("q_medicine_name", "")
    )
    return int(drug_class in profile_classes)


def row_to_features(row: pd.Series) -> dict[str, object]:
    chronic = chronic_flags_from_text(row.get("chronic_diseases_text", ""))
    drug_flags = drug_flags_from_names(
        row.get("normalized_drug_name", ""),
        row.get("ingredient_name", ""),
        row.get("medicine_name", ""),
    )

    return {
        "patient_age": float(pd.to_numeric(row.get("patient_age", row.get("age", 0)), errors="coerce") or 0),
        "patient_sex": encode_patient_sex(row.get("patient_sex", row.get("gender", "unknown"))),
        "num_current_meds": int(
            pd.to_numeric(row.get("num_current_meds", count_text_items(row.get("current_medications_text", ""))), errors="coerce") or 0
        ),
        **chronic,
        "allergy_severity_max": allergy_severity_from_row(row),
        "allergy_class_overlap": same_class_allergy_from_row(row),
        "ddi_severity_max": ddi_severity_to_ordinal(row.get("ddi_severity_max", row.get("max_interaction_severity", "none"))),
        "ddi_pair_count": int(pd.to_numeric(row.get("ddi_pair_count", row.get("interaction_count", 0)), errors="coerce") or 0),
        "ddi_flag": int(
            pd.to_numeric(
                row.get(
                    "ddi_flag",
                    int(pd.to_numeric(row.get("ddi_pair_count", row.get("interaction_count", 0)), errors="coerce") or 0) > 0,
                ),
                errors="coerce",
            )
            or 0
        ),
        "sider_adr_count": int(pd.to_numeric(row.get("sider_adr_count", row.get("side_effect_count", 0)), errors="coerce") or 0),
        **drug_flags,
        "rxnorm_matched": int(pd.to_numeric(row.get("rxnorm_matched", 0), errors="coerce") or 0),
        "atc_class_encoded": encode_atc_class(
            row.get("atc_class_encoded", ""),
            row.get("atc_code", ""),
            row.get("normalized_drug_name", ""),
            row.get("ingredient_name", ""),
            row.get("medicine_name", ""),
        ),
    }


def dataframe_to_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = [row_to_features(row) for _, row in df.iterrows()]
    feature_df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    for column in CONTINUOUS_FEATURES:
        feature_df[column] = pd.to_numeric(feature_df[column], errors="coerce").fillna(0)
    for column in BINARY_FEATURES:
        feature_df[column] = pd.to_numeric(feature_df[column], errors="coerce").fillna(0).astype(int)
    for column in ORDINAL_FEATURES:
        feature_df[column] = pd.to_numeric(feature_df[column], errors="coerce").fillna(0).astype(int)
    for column in CATEGORICAL_FEATURES:
        feature_df[column] = feature_df[column].fillna("unknown").astype(str)
    return feature_df


def derive_adr_target(df: pd.DataFrame) -> pd.Series:
    if TARGET_COLUMN in df.columns:
        return pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)

    if "risk_level" in df.columns:
        return df["risk_level"].map({"Dangerous": 1, "Warning": 0, "Safe": 0}).fillna(0).astype(int)

    if "severe_adr" in df.columns:
        return pd.to_numeric(df["severe_adr"], errors="coerce").fillna(0).astype(int)

    raise ValueError(f"Missing target column: expected {TARGET_COLUMN}, risk_level, or severe_adr")


def payload_to_dataframe(payload: dict) -> pd.DataFrame:
    row = {column: payload.get(column) for column in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
