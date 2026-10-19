/**
 * Reproduces the explicit-emotion dilution bug and demonstrates the completed fix.
 *
 * BEFORE = pre-patch semantics reconstructed from git history:
 *   - interpretation had no explicit first-person self-report branch and no negation guard;
 *   - aggregation used weightedEvidence = sourceWeight * recencyWeight with no
 *     evidence-strength factors, so one confident explicit turn was diluted by
 *     several vague contextual turns.
 * AFTER = current working-tree behaviour (explicit evidence strengthened, no hard override).
 *
 * Run: node scripts/reproduce_emotion_patch.js
 */
const {
  interpretContextualAnswer,
  effectFor,
  isShortContextDependentAnswer,
} = require('../src/services/contextualAnswerInterpretationService');
const { determineAnswerPolarity } = require('../src/services/answerPolarityService');
const { aggregateAdaptiveSessionResult, RECENCY_WEIGHTS, RULE_FALLBACK_WEIGHT } = require('../src/services/adaptiveResultAggregator');
const { findExplicitEmotionEvidence } = require('../src/services/explicitEmotionEvidenceService');

const SUPPORTED_EMOTIONS = ['happiness', 'sadness', 'loneliness', 'anxiety', 'anger', 'cognitive_fog', 'neutral'];

function normalizeEmotion(value) {
  const normalized = String(value || '').trim().toLowerCase();
  return SUPPORTED_EMOTIONS.includes(normalized) ? normalized : 'neutral';
}

function emotionFromEffect(effect, previousEmotion) {
  if (!effect) return null;
  if (effect.startsWith('supports:')) return normalizeEmotion(effect.slice('supports:'.length));
  if (effect.startsWith('opposes:')) return previousEmotion === effect.slice('opposes:'.length) ? 'neutral' : previousEmotion;
  if (effect === 'preserve_previous') return normalizeEmotion(previousEmotion);
  return null;
}

// ---------------------------------------------------------------------------
// PRE-PATCH interpretation, reconstructed from git history (no explicit branch).
// ---------------------------------------------------------------------------
function interpretBefore({
  question,
  answerText,
  answerPolarity,
  rawMlEmotion,
  rawMlConfidence,
  rawDetectionSource,
  fallbackEmotion = 'neutral',
  previousInterpretedEmotion = 'neutral',
  confidenceThreshold = 0.3,
}) {
  const rawEmotion = normalizeEmotion(rawMlEmotion);
  const confidence = Number.isFinite(Number(rawMlConfidence)) ? Number(rawMlConfidence) : null;
  const shortAnswer = isShortContextDependentAnswer(answerText);
  const effect = effectFor(question, answerPolarity);
  const contextualEmotion = emotionFromEffect(effect, normalizeEmotion(previousInterpretedEmotion));
  const confidentMeaningfulMl = rawDetectionSource === 'ml_model' && confidence >= confidenceThreshold && rawEmotion !== 'neutral' && !shortAnswer;

  let interpretedEmotion = 'neutral';
  let evidenceSource = 'no_usable_evidence';

  if (confidentMeaningfulMl) {
    interpretedEmotion = rawEmotion;
    evidenceSource = 'raw_ml_retained';
  } else if (contextualEmotion && ['positive', 'negative', 'neutral'].includes(answerPolarity)) {
    interpretedEmotion = contextualEmotion;
    evidenceSource = 'question_context';
  } else if (rawDetectionSource === 'rule_fallback' && normalizeEmotion(fallbackEmotion) !== 'neutral') {
    interpretedEmotion = normalizeEmotion(fallbackEmotion);
    evidenceSource = 'rule_fallback_retained';
  } else if (rawDetectionSource === 'ml_model' && confidence >= confidenceThreshold && rawEmotion !== 'neutral') {
    interpretedEmotion = rawEmotion;
    evidenceSource = 'raw_ml_retained';
  } else if (rawEmotion === 'neutral') {
    interpretedEmotion = 'neutral';
    evidenceSource = 'raw_neutral';
  }

  return {
    interpretedEmotion,
    evidenceSource,
    rawMlEmotion: rawEmotion,
    rawMlConfidence: confidence,
    rawDetectionSource,
    answerPolarity,
    shortContextDependentAnswer: shortAnswer,
    contextualEvidenceWeight: evidenceSource === 'question_context' ? 0.65 : null,
  };
}

// ---------------------------------------------------------------------------
// PRE-PATCH aggregation: weightedEvidence = sourceWeight * recencyWeight only.
// ---------------------------------------------------------------------------
function aggregateBefore(turns) {
  const ordered = [...turns].sort((a, b) => Number(a.questionNumber) - Number(b.questionNumber));
  const scores = Object.fromEntries(SUPPORTED_EMOTIONS.map((emotion) => [emotion, 0]));
  const perTurn = ordered.map((turn, index) => {
    const ci = turn.analysisMetadata?.contextualInterpretation || {};
    const modelConfidence = turn.detectionSource === 'ml_model' ? Number(turn.confidenceScore) : null;
    const ruleWeight = turn.detectionSource === 'rule_fallback' ? RULE_FALLBACK_WEIGHT : null;
    const ctxWeight = ci.evidenceSource === 'question_context' ? Number(ci.contextualEvidenceWeight || 0.65) : null;
    const sourceWeight = ctxWeight ?? modelConfidence ?? ruleWeight;
    const recencyWeight = RECENCY_WEIGHTS[index];
    const weightedEvidence = sourceWeight * recencyWeight;
    scores[turn.detectedState] += weightedEvidence;
    return {
      questionNumber: turn.questionNumber,
      emotion: turn.detectedState,
      source: ci.evidenceSource || turn.detectionSource,
      sourceWeight,
      recencyWeight,
      weightedEvidence: Number(weightedEvidence.toFixed(4)),
    };
  });
  const total = Object.values(scores).reduce((sum, score) => sum + score, 0);
  const ranked = SUPPORTED_EMOTIONS.map((emotion) => ({ emotion, score: scores[emotion] }))
    .sort((a, b) => b.score - a.score || SUPPORTED_EMOTIONS.indexOf(a.emotion) - SUPPORTED_EMOTIONS.indexOf(b.emotion));
  return {
    perTurn,
    scores,
    total: Number(total.toFixed(4)),
    finalEmotionalState: ranked[0].emotion,
    finalConfidence: Number((ranked[0].score / total).toFixed(4)),
    runnerUpEmotion: ranked[1].emotion,
    runnerUpScore: Number((ranked[1].score / total).toFixed(4)),
  };
}

// ---------------------------------------------------------------------------
// Shared session builders.
// ---------------------------------------------------------------------------
function interpretTurn(interpreter, question, answer, raw, previous) {
  const answerPolarity = determineAnswerPolarity(answer, { detectedEmotion: raw.emotion });
  return interpreter({
    question, answerText: answer, answerPolarity,
    rawMlEmotion: raw.emotion, rawMlConfidence: raw.confidence,
    rawDetectionSource: raw.source, fallbackEmotion: 'neutral',
    previousInterpretedEmotion: previous,
  });
}

function buildTurns(interpreter, questions, answers, rawMl) {
  let previous = 'neutral';
  return answers.map((answer, index) => {
    const interpretation = interpretTurn(interpreter, questions[index], answer, rawMl[index], previous);
    previous = interpretation.interpretedEmotion;
    return {
      questionNumber: index + 1,
      questionCode: questions[index].questionCode,
      userAnswer: answer,
      detectedState: interpretation.interpretedEmotion,
      confidenceScore: rawMl[index].source === 'ml_model' ? rawMl[index].confidence : null,
      detectionSource: rawMl[index].source,
      analysisMetadata: { contextualInterpretation: interpretation },
    };
  });
}

const q = (code, target, dim, neg) => ({
  questionId: code.length, questionCode: code, targetState: target, assessmentDimension: dim, negativeResponseEffect: neg,
});

function printTurn(turn) {
  const ci = turn.analysisMetadata.contextualInterpretation;
  console.log(`Q${turn.questionNumber} [${turn.questionCode}] "${turn.userAnswer}"`);
  console.log(`  raw ML: ${ci.rawMlEmotion} (${ci.rawMlConfidence}) -> interpreted: ${turn.detectedState}`);
  console.log(`  source: ${ci.evidenceSource} | strength: ${ci.evidenceStrength ?? 'n/a'} | explicitW: ${ci.explicitEvidenceWeight ?? 'n/a'} | ctxW: ${ci.contextualEvidenceWeight ?? 'n/a'}`);
}

function printBeforeAggregation(before) {
  console.log('\nTurn contributions (PRE-PATCH: sourceWeight x recency only):');
  before.perTurn.forEach((t) => {
    console.log(`  Q${t.questionNumber} ${t.emotion}: src=${t.source} sourceW=${t.sourceWeight} recency=${t.recencyWeight} weighted=${t.weightedEvidence}`);
  });
  console.log('\nEmotion totals:');
  Object.entries(before.scores).forEach(([e, s]) => { if (s > 0) console.log(`  ${e}: ${Number(s.toFixed(4))}`); });
  console.log(`\nFINAL (before): ${before.finalEmotionalState} (conf ${before.finalConfidence}) | runnerUp=${before.runnerUpEmotion} (${before.runnerUpScore})`);
}

function printAfterAggregation(result) {
  console.log('\nTurn contributions (AFTER PATCH: sourceWeight x recency x strengthFactor):');
  result.explanation.turnEvidence.forEach((t) => {
    console.log(`  Q${t.questionNumber} ${t.emotion}: src=${t.evidenceSource} strength=${t.evidenceStrength} factor=${t.evidenceStrengthFactor} recency=${t.recencyWeight} weighted=${t.weightedEvidence}`);
  });
  console.log('\nEmotion totals:');
  Object.entries(result.explanation.emotionScores).forEach(([e, s]) => { if (s > 0) console.log(`  ${e}: ${s}`); });
  console.log(`\nFINAL (after): ${result.finalEmotionalState} (conf ${result.finalConfidence}) | explicitLeader=${result.explanation.explicitLeader} | runnerUp=${result.explanation.runnerUpEmotion} (${result.explanation.runnerUpScore})`);
}

function runSession(title, questions, answers, rawMl) {
  console.log(`\n=== ${title} — BEFORE (pre-patch semantics) ===`);
  const beforeTurns = buildTurns(interpretBefore, questions, answers, rawMl);
  beforeTurns.forEach(printTurn);
  printBeforeAggregation(aggregateBefore(beforeTurns));

  console.log(`\n=== ${title} — AFTER (current implementation) ===`);
  const afterTurns = buildTurns(interpretContextualAnswer, questions, answers, rawMl);
  afterTurns.forEach(printTurn);
  printAfterAggregation(aggregateAdaptiveSessionResult(afterTurns));
}

// ---------------------------------------------------------------------------
// 0. Direct linguistic evidence for the reported sentence.
// ---------------------------------------------------------------------------
console.log('=== EXPLICIT EVIDENCE: "I feel angry with my children" ===');
console.log(JSON.stringify(findExplicitEmotionEvidence('I feel angry with my children'), null, 2));

// ---------------------------------------------------------------------------
// 1. REAL-DEVICE ANGER-DILUTION SESSION (the reported bug).
// MiniLM v4 predicts anger 0.5355 (>= production threshold 0.30); the bug was
// aggregation dilution, not ML failure.
// ---------------------------------------------------------------------------
runSession(
  'REAL-DEVICE SESSION (ANGER DILUTION)',
  [
    q('open_general', 'neutral', 'general_wellbeing', 'supports:sadness'),
    q('neutral_energy', 'neutral', 'energy_motivation', 'supports:sadness'),
    q('neutral_engagement', 'neutral', 'daily_engagement', 'supports:sadness'),
    q('anger_clarify', 'anger', 'clarification', 'preserve_previous'),
    q('neutral_positive', 'neutral', 'positive_protective_factor', 'preserve_previous'),
  ],
  ["It's been a difficult day", 'A little difficult', 'A little difficult', 'I feel angry with my children', 'Not much comfort today'],
  [
    { emotion: 'sadness', confidence: 0.4087, source: 'ml_model' },
    { emotion: 'sadness', confidence: 0.4087, source: 'ml_model' },
    { emotion: 'sadness', confidence: 0.4087, source: 'ml_model' },
    { emotion: 'anger', confidence: 0.5355, source: 'ml_model' },
    { emotion: 'sadness', confidence: 0.3721, source: 'ml_model' },
  ],
);

// ---------------------------------------------------------------------------
// 2. NEGATION: "I'm not angry" must never become user anger.
// ---------------------------------------------------------------------------
console.log('\n\n=== NEGATION CHECK: findExplicitEmotionEvidence("I\'m not angry") ===');
console.log(JSON.stringify(findExplicitEmotionEvidence("I'm not angry"), null, 2));
runSession(
  'NEGATION SESSION (only anger mention is negated)',
  [
    q('open_general', 'neutral', 'general_wellbeing', 'supports:sadness'),
    q('neutral_energy', 'neutral', 'energy_motivation', 'supports:sadness'),
    q('anger_clarify', 'anger', 'clarification', null),
    q('neutral_engagement', 'neutral', 'daily_engagement', 'supports:sadness'),
    q('neutral_positive', 'neutral', 'positive_protective_factor', 'preserve_previous'),
  ],
  ["I'm okay overall", 'Nothing unusual', "I'm not angry", 'Same as always', 'I feel fine'],
  [
    { emotion: 'neutral', confidence: 0.5, source: 'ml_model' },
    { emotion: 'neutral', confidence: 0.5, source: 'ml_model' },
    { emotion: 'anger', confidence: 0.8, source: 'ml_model' },
    { emotion: 'neutral', confidence: 0.5, source: 'ml_model' },
    { emotion: 'neutral', confidence: 0.5, source: 'ml_model' },
  ],
);

// ---------------------------------------------------------------------------
// 3. FIRST-PERSON vs THIRD-PERSON subject handling.
// ---------------------------------------------------------------------------
console.log('\n\n=== SUBJECT HANDLING: first-person vs third-person ===');
console.log('"I feel angry with my children":', JSON.stringify(findExplicitEmotionEvidence('I feel angry with my children')));
console.log('"My son is angry today.":', JSON.stringify(findExplicitEmotionEvidence('My son is angry today.')));
runSession(
  'THIRD-PERSON SESSION (anger about someone else)',
  [
    q('open_general', 'neutral', 'general_wellbeing', 'supports:sadness'),
    q('family_turn', 'neutral', 'social_connection', 'supports:loneliness'),
    q('anger_clarify', 'anger', 'clarification', 'preserve_previous'),
    q('neutral_engagement', 'neutral', 'daily_engagement', 'supports:sadness'),
    q('neutral_positive', 'neutral', 'positive_protective_factor', 'preserve_previous'),
  ],
  ['My day was quiet', 'My son visited briefly', 'My son is angry today.', 'The rest was fine', 'It was an ordinary day'],
  [
    { emotion: 'neutral', confidence: 0.45, source: 'ml_model' },
    { emotion: 'neutral', confidence: 0.4, source: 'ml_model' },
    { emotion: 'anger', confidence: 0.75, source: 'ml_model' },
    { emotion: 'neutral', confidence: 0.5, source: 'ml_model' },
    { emotion: 'neutral', confidence: 0.45, source: 'ml_model' },
  ],
);

// ---------------------------------------------------------------------------
// 4. CONTRAST: SADNESS-DOMINANT conversation stays sadness-dominant.
// ---------------------------------------------------------------------------
runSession(
  'CONTRAST: SADNESS-DOMINANT',
  [
    q('open_general', 'neutral', 'general_wellbeing', 'supports:sadness'),
    q('neutral_energy', 'neutral', 'energy_motivation', 'supports:sadness'),
    q('neutral_engagement', 'neutral', 'daily_engagement', 'supports:sadness'),
    q('neutral_positive', 'neutral', 'positive_protective_factor', 'preserve_previous'),
    q('anger_clarify', 'anger', 'clarification', 'preserve_previous'),
  ],
  ['I feel sad today.', "I've felt down most of the day.", "I didn't enjoy anything.", 'I had very little energy.', 'I feel a little angry about one thing.'],
  [
    { emotion: 'sadness', confidence: 0.8, source: 'ml_model' },
    { emotion: 'sadness', confidence: 0.75, source: 'ml_model' },
    { emotion: 'sadness', confidence: 0.7, source: 'ml_model' },
    { emotion: 'sadness', confidence: 0.65, source: 'ml_model' },
    { emotion: 'anger', confidence: 0.6, source: 'ml_model' },
  ],
);

// ---------------------------------------------------------------------------
// 5. CONTRAST: LONELINESS-DOMINANT conversation stays loneliness-dominant.
// ---------------------------------------------------------------------------
runSession(
  'CONTRAST: LONELINESS-DOMINANT',
  [
    q('open_general', 'neutral', 'general_wellbeing', 'supports:sadness'),
    q('lonely_contact', 'loneliness', 'social_connection', 'supports:loneliness'),
    q('lonely_clarify', 'loneliness', 'clarification', 'preserve_previous'),
    q('lonely_engagement', 'loneliness', 'daily_engagement', 'supports:loneliness'),
    q('lonely_protective', 'loneliness', 'positive_protective_factor', 'preserve_previous'),
  ],
  ['My day was difficult.', 'I feel alone.', "I haven't had anyone to talk to.", 'I miss my family.', 'Nothing made me feel connected today.'],
  [
    { emotion: 'sadness', confidence: 0.5, source: 'ml_model' },
    { emotion: 'loneliness', confidence: 0.8, source: 'ml_model' },
    { emotion: 'loneliness', confidence: 0.7, source: 'ml_model' },
    { emotion: 'loneliness', confidence: 0.75, source: 'ml_model' },
    { emotion: 'loneliness', confidence: 0.6, source: 'ml_model' },
  ],
);

// ---------------------------------------------------------------------------
// 6. CONTRAST: ANXIETY-DOMINANT conversation with explicit self-reports.
// ---------------------------------------------------------------------------
runSession(
  'CONTRAST: ANXIETY-DOMINANT',
  [
    q('open_general', 'neutral', 'general_wellbeing', 'supports:sadness'),
    q('anxious_contact', 'anxiety', 'social_connection', 'supports:anxiety'),
    q('anxious_clarify', 'anxiety', 'clarification', 'preserve_previous'),
    q('anxious_engagement', 'anxiety', 'daily_engagement', 'supports:anxiety'),
    q('anxious_protective', 'anxiety', 'positive_protective_factor', 'preserve_previous'),
  ],
  ['My day was difficult.', 'I feel anxious.', 'I am worried about everything.', 'I feel nervous most evenings.', 'I cannot relax at night.'],
  [
    { emotion: 'sadness', confidence: 0.5, source: 'ml_model' },
    { emotion: 'anxiety', confidence: 0.82, source: 'ml_model' },
    { emotion: 'anxiety', confidence: 0.78, source: 'ml_model' },
    { emotion: 'anxiety', confidence: 0.74, source: 'ml_model' },
    { emotion: 'anxiety', confidence: 0.66, source: 'ml_model' },
  ],
);

// ---------------------------------------------------------------------------
// 7. CONTRAST: HAPPINESS explicit self-reports are honoured (protective state).
// ---------------------------------------------------------------------------
runSession(
  'CONTRAST: HAPPINESS-DOMINANT',
  [
    q('open_general', 'neutral', 'general_wellbeing', 'supports:sadness'),
    q('happy_contact', 'happiness', 'social_connection', 'opposes:loneliness'),
    q('happy_clarify', 'happiness', 'clarification', 'preserve_previous'),
    q('happy_engagement', 'happiness', 'daily_engagement', 'preserve_previous'),
    q('happy_protective', 'happiness', 'positive_protective_factor', 'preserve_previous'),
  ],
  ['I feel happy today.', 'I am glad we talked.', 'Today was good.', 'I enjoyed my meal.', 'Life feels better lately.'],
  [
    { emotion: 'happiness', confidence: 0.85, source: 'ml_model' },
    { emotion: 'happiness', confidence: 0.8, source: 'ml_model' },
    { emotion: 'happiness', confidence: 0.72, source: 'ml_model' },
    { emotion: 'happiness', confidence: 0.7, source: 'ml_model' },
    { emotion: 'happiness', confidence: 0.68, source: 'ml_model' },
  ],
);