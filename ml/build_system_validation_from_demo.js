const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const {
  buildAnalysis,
  sanitizeAnalysisPayload,
} = require('../src/controllers/allergyController');
const { buildFdaSeriousFeaturePayload } = require('../src/services/mlFeatureBuilder');
const {
  HYBRID_RULE_WEIGHT,
  HYBRID_ML_WEIGHT,
  RISK_THRESHOLDS,
} = require('../src/config/hybridScoring');

const ROOT = path.resolve(__dirname);
const OUTPUT_PATH = path.join(ROOT, 'models', 'system_level_validation_cases.json');
const PYTHON_HELPER = path.join(ROOT, 'score_fda_lr_payload.py');

const CURRENT = {
  alpha: HYBRID_RULE_WEIGHT,
  beta: HYBRID_ML_WEIGHT,
  warning: RISK_THRESHOLDS.warningMin,
  dangerous: RISK_THRESHOLDS.dangerousMin,
};
const CANDIDATE = { alpha: 0.5, beta: 0.5, warning: 15, dangerous: 50 };
const LABELS = ['Safe', 'Caution', 'Dangerous'];

const yesNo = (value) => (value ? 'Yes' : 'No');

const noQuestionnaire = () => ([
  { questionKey: 'pastReaction', answerText: 'No' },
  { questionKey: 'reactionSymptoms', answerText: 'None' },
  { questionKey: 'medicineName', answerText: 'No' },
  { questionKey: 'doctorAdvice', answerText: 'No' },
  { questionKey: 'painkillerAntibioticReaction', answerText: 'No' },
]);

const medicinePayload = (medicineName, overrides = {}) => sanitizeAnalysisPayload({
  inputMethod: 'validation',
  medicineName,
  normalizedDrugName: medicineName,
  dose: overrides.dose || '',
  frequency: overrides.frequency || '',
  takenBefore: overrides.takenBefore ?? 'No',
  hadReactionBefore: overrides.hadReactionBefore ?? 'No',
  symptomMatch: overrides.symptomMatch || '',
  severity: overrides.severity || 'mild',
  takingOtherMedicinesNow: overrides.takingOtherMedicinesNow ?? 'Yes',
  notes: overrides.notes || '',
});

const scenarios = [
  {
    case_id: 'D01',
    scenario_name: 'Penicillin allergy -> Amoxicillin',
    expected_class: 'Dangerous',
    profile: {
      age: '72',
      gender: 'Female',
      hasMedicineAllergy: true,
      knownAllergiesText: 'Penicillin',
      chronicDiseasesText: 'Type 2 Diabetes, Hypertension',
      currentMedicationsText: 'Metformin, Losartan',
      emergencyContact: 'Nimal - Son',
      caregiverDetails: 'Home caregiver',
      reactionSymptomsText: 'Skin rash, itching',
      suspectedMedicineNamesText: 'Amoxicillin',
      avoidedMedicinesText: 'Penicillin antibiotics',
      antibioticPainkillerReaction: 'Yes',
    },
    questionnaireAnswers: [
      { questionKey: 'pastReaction', answerText: 'Yes' },
      { questionKey: 'reactionSymptoms', answerText: 'Skin rash' },
      { questionKey: 'medicineName', answerText: 'Amoxicillin' },
      { questionKey: 'doctorAdvice', answerText: 'Avoid penicillin antibiotics' },
      { questionKey: 'painkillerAntibioticReaction', answerText: 'Yes' },
    ],
    payload: medicinePayload('Amoxicillin', { hadReactionBefore: 'Yes', symptomMatch: 'Skin rash', dose: '500 mg', frequency: 'Three times daily' }),
  },
  {
    case_id: 'D02',
    scenario_name: 'Warfarin -> Ibuprofen',
    expected_class: 'Dangerous',
    profile: {
      age: '68',
      gender: 'Male',
      hasMedicineAllergy: false,
      knownAllergiesText: '',
      chronicDiseasesText: 'Hypertension',
      currentMedicationsText: 'Warfarin',
      emergencyContact: 'Daughter',
      caregiverDetails: '',
      reactionSymptomsText: '',
      suspectedMedicineNamesText: '',
      avoidedMedicinesText: '',
      antibioticPainkillerReaction: 'No',
    },
    questionnaireAnswers: noQuestionnaire(),
    payload: medicinePayload('Ibuprofen', { dose: '400 mg', frequency: 'Twice daily' }),
  },
  {
    case_id: 'S01',
    scenario_name: 'No allergy clean -> Paracetamol',
    expected_class: 'Safe',
    profile: {
      age: '60',
      gender: 'Female',
      hasMedicineAllergy: false,
      knownAllergiesText: '',
      chronicDiseasesText: '',
      currentMedicationsText: '',
      emergencyContact: 'Son',
      caregiverDetails: '',
      reactionSymptomsText: '',
      suspectedMedicineNamesText: '',
      avoidedMedicinesText: '',
      antibioticPainkillerReaction: 'No',
    },
    questionnaireAnswers: noQuestionnaire(),
    payload: medicinePayload('Paracetamol', { dose: '500 mg', frequency: 'As needed', takingOtherMedicinesNow: 'No' }),
  },
  {
    case_id: 'C01',
    scenario_name: 'CKD -> Metformin',
    expected_class: 'Caution',
    profile: {
      age: '74',
      gender: 'Male',
      hasMedicineAllergy: false,
      knownAllergiesText: '',
      chronicDiseasesText: 'Chronic kidney disease',
      currentMedicationsText: 'Losartan, Furosemide',
      emergencyContact: 'Brother',
      caregiverDetails: '',
      reactionSymptomsText: '',
      suspectedMedicineNamesText: '',
      avoidedMedicinesText: '',
      antibioticPainkillerReaction: 'No',
    },
    questionnaireAnswers: noQuestionnaire(),
    payload: medicinePayload('Metformin', { dose: '500 mg', frequency: 'Twice daily' }),
  },
  {
    case_id: 'C02',
    scenario_name: 'Liver disease -> Paracetamol',
    expected_class: 'Caution',
    profile: {
      age: '67',
      gender: 'Female',
      hasMedicineAllergy: false,
      knownAllergiesText: '',
      chronicDiseasesText: 'Chronic liver disease',
      currentMedicationsText: '',
      emergencyContact: 'Husband',
      caregiverDetails: '',
      reactionSymptomsText: '',
      suspectedMedicineNamesText: '',
      avoidedMedicinesText: '',
      antibioticPainkillerReaction: 'No',
    },
    questionnaireAnswers: noQuestionnaire(),
    payload: medicinePayload('Paracetamol', { dose: '500 mg', frequency: 'Twice daily', takingOtherMedicinesNow: 'No' }),
  },
  {
    case_id: 'C03',
    scenario_name: 'Hypertension -> Ibuprofen',
    expected_class: 'Caution',
    profile: {
      age: '66',
      gender: 'Female',
      hasMedicineAllergy: false,
      knownAllergiesText: '',
      chronicDiseasesText: 'Hypertension',
      currentMedicationsText: 'Amlodipine',
      emergencyContact: 'Husband',
      caregiverDetails: '',
      reactionSymptomsText: '',
      suspectedMedicineNamesText: '',
      avoidedMedicinesText: '',
      antibioticPainkillerReaction: 'No',
    },
    questionnaireAnswers: noQuestionnaire(),
    payload: medicinePayload('Ibuprofen', { dose: '400 mg', frequency: 'Twice daily' }),
  },
  {
    case_id: 'S02',
    scenario_name: 'Age 70 + Cetirizine -> Panadol',
    expected_class: 'Safe',
    profile: {
      age: '70',
      gender: 'Female',
      hasMedicineAllergy: false,
      knownAllergiesText: '',
      chronicDiseasesText: '',
      currentMedicationsText: 'Cetirizine',
      emergencyContact: 'Daughter',
      caregiverDetails: '',
      reactionSymptomsText: '',
      suspectedMedicineNamesText: '',
      avoidedMedicinesText: '',
      antibioticPainkillerReaction: 'No',
    },
    questionnaireAnswers: noQuestionnaire(),
    payload: medicinePayload('Panadol', { dose: '500 mg', frequency: 'As needed' }),
  },
  {
    case_id: 'D03',
    scenario_name: 'Aspirin + asthma -> Ibuprofen',
    expected_class: 'Dangerous',
    profile: {
      age: '70',
      gender: 'Male',
      hasMedicineAllergy: true,
      knownAllergiesText: 'Aspirin',
      chronicDiseasesText: 'Asthma',
      currentMedicationsText: 'Salbutamol inhaler',
      emergencyContact: 'Wife',
      caregiverDetails: '',
      reactionSymptomsText: 'Breathing trouble',
      suspectedMedicineNamesText: 'Aspirin',
      avoidedMedicinesText: 'Aspirin and NSAIDs',
      antibioticPainkillerReaction: 'Yes',
    },
    questionnaireAnswers: [
      { questionKey: 'pastReaction', answerText: 'Yes' },
      { questionKey: 'reactionSymptoms', answerText: 'Breathing trouble' },
      { questionKey: 'medicineName', answerText: 'Aspirin' },
      { questionKey: 'doctorAdvice', answerText: 'Avoid aspirin and NSAIDs' },
      { questionKey: 'painkillerAntibioticReaction', answerText: 'Yes' },
    ],
    payload: medicinePayload('Ibuprofen', { dose: '400 mg', frequency: 'Twice daily', hadReactionBefore: 'Yes', symptomMatch: 'Breathing trouble' }),
  },
  {
    case_id: 'D04',
    scenario_name: 'Aspirin allergy -> Naproxen',
    expected_class: 'Dangerous',
    profile: {
      age: '70',
      gender: 'Male',
      hasMedicineAllergy: true,
      knownAllergiesText: 'Aspirin',
      chronicDiseasesText: 'Hypertension',
      currentMedicationsText: 'Amlodipine',
      emergencyContact: 'Wife',
      caregiverDetails: '',
      reactionSymptomsText: 'Facial swelling',
      suspectedMedicineNamesText: 'Aspirin',
      avoidedMedicinesText: 'Aspirin and related NSAIDs',
      antibioticPainkillerReaction: 'Yes',
    },
    questionnaireAnswers: [
      { questionKey: 'pastReaction', answerText: 'Yes' },
      { questionKey: 'reactionSymptoms', answerText: 'Facial swelling' },
      { questionKey: 'medicineName', answerText: 'Aspirin' },
      { questionKey: 'doctorAdvice', answerText: 'Avoid aspirin and related painkillers' },
      { questionKey: 'painkillerAntibioticReaction', answerText: 'Yes' },
    ],
    payload: medicinePayload('Naproxen', { dose: '250 mg', frequency: 'Twice daily', hadReactionBefore: 'Yes', symptomMatch: 'Facial swelling' }),
  },
  {
    case_id: 'S03',
    scenario_name: 'Clean unseen medicine -> Furosemide',
    expected_class: 'Safe',
    profile: {
      age: '60',
      gender: 'Female',
      hasMedicineAllergy: false,
      knownAllergiesText: '',
      chronicDiseasesText: '',
      currentMedicationsText: '',
      emergencyContact: 'Son',
      caregiverDetails: '',
      reactionSymptomsText: '',
      suspectedMedicineNamesText: '',
      avoidedMedicinesText: '',
      antibioticPainkillerReaction: 'No',
    },
    questionnaireAnswers: noQuestionnaire(),
    payload: medicinePayload('Furosemide', { dose: '20 mg', frequency: 'Once daily', takingOtherMedicinesNow: 'No' }),
  },
];

const classify = (score, warning, dangerous) => {
  if (score >= dangerous) return 'Dangerous';
  if (score >= warning) return 'Caution';
  return 'Safe';
};

const computeScore = (ruleScore, mlScore, alpha, beta) => Math.round(alpha * ruleScore + beta * mlScore);

const hasNoPatientSpecificRiskEvidence = (analysisPayload) => {
  const ruleScore = Number(analysisPayload?.dataUsed?.ruleScore ?? analysisPayload?.riskScore ?? 0);
  const riskFactors = Array.isArray(analysisPayload?.riskFactors) ? analysisPayload.riskFactors : [];
  const allergyEvidenceMatches = Array.isArray(analysisPayload?.dataUsed?.allergyEvidenceMatches)
    ? analysisPayload.dataUsed.allergyEvidenceMatches
    : [];
  const interactionCount = Number(analysisPayload?.medicationKnowledge?.interactionCount || 0);
  const sideEffectMatchCount = Number(analysisPayload?.medicationKnowledge?.sideEffectMatchCount || 0);
  const severeReactionSignal = Boolean(analysisPayload?.dataUsed?.severeReactionSignal);
  const chronicRiskFlag = Boolean(analysisPayload?.dataUsed?.chronicRiskFlag);
  const historyPriorCheckCount = Number(analysisPayload?.dataUsed?.historyPriorCheckCount || 0);
  const hasCurrentMedicineSpecificReaction = Boolean(analysisPayload?.dataUsed?.currentMedicineSpecificReaction);

  return (
    ruleScore === 0 &&
    riskFactors.length === 0 &&
    allergyEvidenceMatches.length === 0 &&
    interactionCount === 0 &&
    sideEffectMatchCount === 0 &&
    !severeReactionSignal &&
    !chronicRiskFlag &&
    historyPriorCheckCount === 0 &&
    !hasCurrentMedicineSpecificReaction
  );
};

const replayFinalDecision = (analysisPayload, config) => {
  const ruleScore = Number(analysisPayload?.dataUsed?.ruleScore ?? analysisPayload?.riskScore ?? 0);
  const rawMlScore = Number(analysisPayload?.lr_mlScore ?? 0);
  let combinedRiskScore = computeScore(ruleScore, rawMlScore, config.alpha, config.beta);
  let combinedRiskLevel = classify(combinedRiskScore, config.warning, config.dangerous);
  const ruleRiskLevel = classify(ruleScore, config.warning, config.dangerous);
  const riskFactors = Array.isArray(analysisPayload?.riskFactors) ? analysisPayload.riskFactors : [];

  if ((ruleRiskLevel === 'Dangerous' && combinedRiskLevel !== 'Dangerous') ||
      (ruleRiskLevel === 'Caution' && combinedRiskLevel === 'Safe')) {
    combinedRiskLevel = ruleRiskLevel;
    if (ruleRiskLevel === 'Dangerous') {
      combinedRiskScore = Math.max(combinedRiskScore, config.dangerous);
    } else if (ruleRiskLevel === 'Caution') {
      combinedRiskScore = Math.max(combinedRiskScore, config.warning);
    }
  }

  const hasP1Block = riskFactors.some((factor) => factor.ruleId === 'P1' || factor.factorType === 'allergy_match') ||
    Boolean(analysisPayload?.dataUsed?.p1ShortCircuited);
  const hasNsaidAspirinCrossFactor = riskFactors.some((factor) => factor.factorType === 'nsaid_aspirin_cross_allergy');
  const hasDangerousComboFactor = riskFactors.some((factor) => factor.factorType === 'dangerous_combination');
  const hasHighInteractionFactor = riskFactors.some(
    (factor) => factor.factorType === 'ddinter_interaction' && factor.severity === 'high'
  );
  const histDangerous = Number(analysisPayload?.dataUsed?.historyDangerousCount || 0);
  const histWarning = Number(analysisPayload?.dataUsed?.historyWarningCount || 0);

  if (hasP1Block) {
    combinedRiskLevel = 'Dangerous';
    combinedRiskScore = Math.max(combinedRiskScore, 85);
  } else if (hasNsaidAspirinCrossFactor || hasHighInteractionFactor) {
    combinedRiskLevel = 'Dangerous';
    combinedRiskScore = Math.max(combinedRiskScore, 75);
  } else if (hasDangerousComboFactor && combinedRiskLevel === 'Safe') {
    combinedRiskLevel = 'Caution';
    combinedRiskScore = Math.max(combinedRiskScore, 35);
  }

  if (histDangerous >= 2) {
    combinedRiskLevel = 'Dangerous';
    combinedRiskScore = Math.max(combinedRiskScore, 72);
  } else if (histDangerous >= 1) {
    combinedRiskScore = Math.max(combinedRiskScore, 40);
    if (combinedRiskLevel === 'Safe') {
      combinedRiskLevel = 'Caution';
    }
  } else if (histWarning >= 2 && combinedRiskLevel === 'Safe') {
    combinedRiskLevel = 'Caution';
    combinedRiskScore = Math.max(combinedRiskScore, 28);
  }

  const cleanCase = hasNoPatientSpecificRiskEvidence(analysisPayload);
  if (cleanCase) {
    combinedRiskLevel = 'Safe';
    combinedRiskScore = Math.min(ruleScore, config.warning - 1);
  }

  return {
    score: combinedRiskScore,
    className: combinedRiskLevel,
    cleanCaseGuardrailApplied: cleanCase,
  };
};

const scorePayloadsWithPython = (featurePayloads) => {
  const raw = execFileSync('python', [PYTHON_HELPER], {
    cwd: path.resolve(ROOT, '..'),
    input: JSON.stringify(featurePayloads),
    encoding: 'utf8',
  });
  return JSON.parse(raw);
};

const emptyMetrics = () => ({
  accuracy: 0,
  macro_f1: 0,
  weighted_f1: 0,
  dangerous_recall: 0,
  dangerous_fnr: 0,
  caution_recall: 0,
  safe_false_escalation_rate: 0,
});

const calculateMetrics = (rows, fieldName) => {
  if (!rows.length) {
    return emptyMetrics();
  }

  const support = Object.fromEntries(LABELS.map((label) => [label, 0]));
  const tp = Object.fromEntries(LABELS.map((label) => [label, 0]));
  const fp = Object.fromEntries(LABELS.map((label) => [label, 0]));
  const fn = Object.fromEntries(LABELS.map((label) => [label, 0]));
  let correct = 0;
  let safeEscalations = 0;
  let safeTotal = 0;

  for (const row of rows) {
    const actual = row.expected_class;
    const predicted = row[fieldName];
    support[actual] += 1;
    if (actual === predicted) {
      correct += 1;
      tp[actual] += 1;
    } else {
      fp[predicted] += 1;
      fn[actual] += 1;
    }

    if (actual === 'Safe') {
      safeTotal += 1;
      if (predicted !== 'Safe') {
        safeEscalations += 1;
      }
    }
  }

  const perClass = LABELS.map((label) => {
    const precision = tp[label] + fp[label] === 0 ? 0 : tp[label] / (tp[label] + fp[label]);
    const recall = tp[label] + fn[label] === 0 ? 0 : tp[label] / (tp[label] + fn[label]);
    const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
    return { label, precision, recall, f1, support: support[label] };
  });

  const macroF1 = perClass.reduce((sum, item) => sum + item.f1, 0) / LABELS.length;
  const weightedF1 =
    perClass.reduce((sum, item) => sum + item.f1 * item.support, 0) /
    rows.length;
  const dangerous = perClass.find((item) => item.label === 'Dangerous');
  const caution = perClass.find((item) => item.label === 'Caution');

  return {
    accuracy: correct / rows.length,
    macro_f1: macroF1,
    weighted_f1: weightedF1,
    dangerous_recall: dangerous.recall,
    dangerous_fnr: 1 - dangerous.recall,
    caution_recall: caution.recall,
    safe_false_escalation_rate: safeTotal === 0 ? 0 : safeEscalations / safeTotal,
  };
};

const main = async () => {
  const analyses = [];
  for (const scenario of scenarios) {
    const analysis = await buildAnalysis(
      scenario.payload,
      scenario.profile,
      scenario.questionnaireAnswers,
      [],
      'validation-user'
    );
    analyses.push(analysis);
  }

  const featurePayloads = analyses.map((analysis, index) =>
    buildFdaSeriousFeaturePayload({
      analysisPayload: analysis,
      profile: scenarios[index].profile,
      questionnaireAnswers: scenarios[index].questionnaireAnswers,
    })
  );
  const lrScores = scorePayloadsWithPython(featurePayloads);

  const rows = scenarios.map((scenario, index) => {
    const ruleScore = Number(analyses[index]?.dataUsed?.ruleScore ?? analyses[index]?.riskScore ?? 0);
    const lrMlScore = Number(lrScores[index] ?? 0);
    const analysisForReplay = {
      ...analyses[index],
      lr_mlScore: lrMlScore,
    };
    const currentDecision = replayFinalDecision(analysisForReplay, CURRENT);
    const candidateDecision = replayFinalDecision(analysisForReplay, CANDIDATE);

    return {
      case_id: scenario.case_id,
      scenario_name: scenario.scenario_name,
      expected_class: scenario.expected_class,
      ruleScore,
      lr_mlScore: lrMlScore,
      current_score: currentDecision.score,
      current_class: currentDecision.className,
      candidate_score: candidateDecision.score,
      candidate_class: candidateDecision.className,
      current_clean_case_guardrail_applied: currentDecision.cleanCaseGuardrailApplied,
      candidate_clean_case_guardrail_applied: candidateDecision.cleanCaseGuardrailApplied,
      profile_summary: {
        age: scenario.profile.age,
        gender: scenario.profile.gender,
        has_medicine_allergy: yesNo(scenario.profile.hasMedicineAllergy),
        known_allergies_text: scenario.profile.knownAllergiesText,
        chronic_diseases_text: scenario.profile.chronicDiseasesText,
        current_medications_text: scenario.profile.currentMedicationsText,
      },
      medicine_name: scenario.payload.medicineName,
      feature_payload: featurePayloads[index],
    };
  });

  const payload = {
    generated_at: new Date().toISOString(),
    source: 'frozen_demo_scenarios',
    configurations: {
      current: CURRENT,
      candidate: CANDIDATE,
    },
    assumptions: [
      'Expected classes were taken from the SUS study protocol or mapped conservatively for additional backend validation scenarios.',
      'Scenario labels use Safe/Caution/Dangerous for reporting, while the live backend uses Safe/Warning/Dangerous.',
      'LR scores come from the tuned FDA Logistic Regression artifact using the same FDA-style feature payload schema as runtime medicine-safety ML.',
      'Rule scores come from the live P1-P16 clinical rule path via buildAnalysis() with empty history to avoid contamination from prior checks.',
    ],
    rows,
    metrics: {
      current: calculateMetrics(rows, 'current_class'),
      candidate: calculateMetrics(rows, 'candidate_class'),
    },
  };

  fs.writeFileSync(OUTPUT_PATH, JSON.stringify(payload, null, 2));
  console.log(`[system-validation] Wrote ${OUTPUT_PATH}`);
  console.log(`  Cases: ${rows.length}`);
  console.log(`  Current accuracy=${payload.metrics.current.accuracy.toFixed(3)}, dangerous_recall=${payload.metrics.current.dangerous_recall.toFixed(3)}, safe_false_escalation=${payload.metrics.current.safe_false_escalation_rate.toFixed(3)}`);
  console.log(`  Candidate accuracy=${payload.metrics.candidate.accuracy.toFixed(3)}, dangerous_recall=${payload.metrics.candidate.dangerous_recall.toFixed(3)}, safe_false_escalation=${payload.metrics.candidate.safe_false_escalation_rate.toFixed(3)}`);
};

main().catch((error) => {
  console.error('[system-validation] Failed:', error);
  process.exitCode = 1;
});
