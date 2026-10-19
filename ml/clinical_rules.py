"""
Section 12/13 — formalized clinical rule classes (dissertation pseudocode mirror).

AllergyCrossReactivityRule implements P2 ATC-class cross-reactivity with
severity-adjusted scoring used by the live rule engine (clinicalRuleEngine.js).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class AllergyLevel(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    ANAPHYLACTIC = "anaphylactic"


SEVERITY_MULTIPLIERS = {
    AllergyLevel.MILD: 0.5,
    AllergyLevel.MODERATE: 0.75,
    AllergyLevel.SEVERE: 1.0,
    AllergyLevel.ANAPHYLACTIC: 1.25,
}


class PatientProfile(Protocol):
    allergies: list
    max_allergy_severity: AllergyLevel


class Drug(Protocol):
    atc_class: str


@dataclass(frozen=True)
class AllergyCrossReactivityRule:
    """P2 — same ATC class as documented allergy; score scaled by severity."""

    priority: int = 2
    base_score: float = 65.0

    def condition(self, patient: PatientProfile, drug: Drug) -> bool:
        return any(allergy.atc_class == drug.atc_class for allergy in patient.allergies)

    def score(self, patient: PatientProfile, drug: Drug) -> float:
        multiplier = SEVERITY_MULTIPLIERS[patient.max_allergy_severity]
        return self.base_score * multiplier
