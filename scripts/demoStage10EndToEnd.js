const path = require('path');
const axios = require('axios');
const dotenv = require('dotenv');

dotenv.config({ path: path.join(__dirname, '..', '.env') });
const { getPool, query } = require('../src/db/postgres');

const API = process.env.STAGE10_DEMO_API || 'http://localhost:5001/api/emotional-support';
const DEMO_EMAIL = 'component4-stage10-viva-ready@eldermeds.local';
const ANSWERS = {
  happiness: 'I feel happy, joyful, grateful, and wonderful today.',
  loneliness: 'I feel lonely and alone, and I miss having companionship.',
  anxiety: 'I feel anxious, worried, nervous, and tense today.',
};

async function post(url, body) {
  const response = await axios.post(`${API}${url}`, body, { timeout: 15000 });
  return response.data;
}

async function runCheckIn(userId, emotion) {
  const start = await post('/adaptive-chat/start', { user_id: userId });
  const trace = [{ number: start.question_number, code: start.question.question_code, text: start.question.question_text }];
  let current = start;
  for (let number = 1; number <= 5; number += 1) {
    current = await post('/adaptive-chat/respond', {
      user_id: userId,
      session_id: start.session_id,
      question_id: current.question?.question_id || current.next_question?.question_id,
      answer_text: ANSWERS[emotion],
    });
    if (!current.is_complete) {
      trace.push({ number: current.question_number, code: current.next_question.question_code, text: current.next_question.question_text, previousDetectionSource: current.detection_source });
    }
  }
  return { start, final: current, trace };
}

function objectiveResponse(task) {
  if (task.kind === 'multi_recall') return { selectedAnswers: task.studyItems };
  const knownAnswer = task.options.find((option) => ['Carrot', 'Circle', 'Afternoon'].includes(option));
  return { selectedAnswer: knownAnswer || task.options[0] };
}

async function completeRecommendedActivity(userId, checkIn, narrative = '') {
  const recommendation = checkIn.final.recommended_activity;
  const attempt = await post('/adaptive-activities/start', {
    user_id: userId,
    session_id: checkIn.final.session_id,
    activity_code: recommendation.activity_code,
  });
  const response = attempt.category === 'cognitive_engagement'
    ? objectiveResponse(attempt.task)
    : narrative ? { narrative } : {};
  const completion = await post(`/adaptive-activities/attempts/${attempt.attempt_id}/submit`, { user_id: userId, response });
  return { attempt, completion };
}

async function main() {
  const userResult = await query(`
    INSERT INTO users (full_name, email, password_hash)
    VALUES ('Component 4 Stage 10 Viva Demo', $1, 'stage10-demo-not-for-login')
    ON CONFLICT (email) DO UPDATE SET full_name = EXCLUDED.full_name
    RETURNING id
  `, [DEMO_EMAIL]);
  const userId = userResult.rows[0].id;
  const existing = await query('SELECT COUNT(*)::INT AS count FROM adaptive_chat_sessions WHERE user_id = $1', [userId]);
  if (Number(existing.rows[0].count) > 0) throw new Error(`Stage 10 demo sessions already exist for user ${userId}; refusing to duplicate them.`);

  const positive = await runCheckIn(userId, 'happiness');
  const positiveActivity = await completeRecommendedActivity(userId, positive);

  const loneliness1 = await runCheckIn(userId, 'loneliness');
  const loneliness2 = await runCheckIn(userId, 'loneliness');
  const loneliness3 = await runCheckIn(userId, 'loneliness');
  const lonelinessActivity = await completeRecommendedActivity(userId, loneliness3, 'I remember a warm family gathering and familiar songs.');
  const loneliness4 = await runCheckIn(userId, 'loneliness');

  const anxiety = await runCheckIn(userId, 'anxiety');
  const anxietyActivity = await completeRecommendedActivity(userId, anxiety);

  const wellness7d = await axios.get(`${API}/wellness-trends/${userId}`, { params: { period: '7d' }, timeout: 15000 });
  const sessions = await query(`SELECT session_id AS "sessionId", final_emotional_state AS emotion, final_confidence::FLOAT AS "finalConfidence", risk_level AS risk, conversation_engagement AS engagement, recommended_activity AS activity, caregiver_notification_required AS "alertRequired", completed_at AS "completedAt" FROM adaptive_chat_sessions WHERE user_id = $1 AND is_complete = TRUE ORDER BY completed_at`, [userId]);
  const attempts = await query(`SELECT attempt_id AS "attemptId", adaptive_session_id AS "sessionId", category, activity_code AS "activityCode", accuracy_score::FLOAT AS accuracy, is_correct AS "isCorrect", response_time_ms AS "responseTimeMs", recommended_next_difficulty AS "nextDifficulty", completion_status AS status FROM adaptive_activity_attempts WHERE user_id = $1 ORDER BY completed_at`, [userId]);
  const alerts = await query(`SELECT alert_id AS "alertId", adaptive_session_id AS "sessionId", emotional_state AS emotion, matching_concern_count_7d AS occurrences, status, alert_type AS "alertType" FROM emotional_caregiver_alerts WHERE user_id = $1 ORDER BY created_at`, [userId]);

  const summarize = (checkIn) => ({ sessionId: checkIn.final.session_id, q1: checkIn.trace[0], trace: checkIn.trace, finalEmotion: checkIn.final.final_emotional_state, finalConfidence: checkIn.final.final_confidence, risk: checkIn.final.risk_level, activity: checkIn.final.recommended_activity, alertRequired: checkIn.final.caregiver_notification_required, alertCreated: Boolean(checkIn.final.caregiver_alert) });
  console.log(JSON.stringify({
    demoUser: { userId, email: DEMO_EMAIL },
    scenarioA_positive: { ...summarize(positive), completion: positiveActivity.completion },
    scenarioB_loneliness: { ...summarize(loneliness3), completion: lonelinessActivity.completion },
    scenarioC_anxiety: { ...summarize(anxiety), completion: anxietyActivity.completion },
    scenarioD_alertThreshold: [loneliness1, loneliness2, loneliness3, loneliness4].map((item, index) => ({ occurrence: index + 1, risk: item.final.risk_level, alertRequired: item.final.caregiver_notification_required, alertCreated: Boolean(item.final.caregiver_alert) })),
    wellness7d: wellness7d.data,
    database: { sessions: sessions.rows, attempts: attempts.rows, alerts: alerts.rows },
  }, null, 2));
}

main().catch((error) => {
  console.error(error.response?.data || error.message);
  process.exitCode = 1;
}).finally(() => getPool().end());
