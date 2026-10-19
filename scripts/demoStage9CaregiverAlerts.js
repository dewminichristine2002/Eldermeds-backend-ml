const path = require('path');
const dotenv = require('dotenv');

dotenv.config({ path: path.join(__dirname, '..', '.env') });

const { getPool, query, withTransaction } = require('../src/db/postgres');
const { createAdaptiveCaregiverAlert } = require('../src/repositories/adaptiveRiskRepository');
const { assessAdaptiveRisk } = require('../src/services/riskAssessmentService');

const DEMO_EMAIL = 'component4-stage9-demo@eldermeds.local';

async function completeDemoSession(userId, completedAt, sequence) {
  return withTransaction(async (client) => {
    const inserted = await client.query(
      `INSERT INTO adaptive_chat_sessions (
         user_id, current_state, turn_count, is_complete, final_emotional_state,
         final_confidence, conversation_engagement, risk_level,
         caregiver_notification_required, completed_at, support_directive
       ) VALUES ($1, 'loneliness', 5, TRUE, 'loneliness', 0.8, 'engaged',
         'medium', FALSE, $2, $3::JSONB)
       RETURNING session_id AS "sessionId"`,
      [userId, completedAt, JSON.stringify({ stage9_demo: true, sequence })]
    );
    const sessionId = inserted.rows[0].sessionId;
    const assessment = await assessAdaptiveRisk({
      userId, finalEmotionalState: 'loneliness', completedAt, client,
    });
    let alert = null;
    if (assessment.shouldCreateAlert) {
      alert = await createAdaptiveCaregiverAlert(client, {
        userId,
        adaptiveSessionId: sessionId,
        emotionalState: 'loneliness',
        matchingConcernCount7d: assessment.matchingConcernCount7d,
        message: assessment.alertMessage,
        explanation: { ...assessment.explanation, alertCreated: true },
      });
    }
    await client.query(
      `UPDATE adaptive_chat_sessions
       SET risk_level = $2, caregiver_notification_required = $3,
           support_directive = support_directive || $4::JSONB, updated_at = NOW()
       WHERE session_id = $1`,
      [sessionId, assessment.finalRisk, assessment.caregiverNotificationRequired,
        JSON.stringify({ risk_assessment: assessment.explanation })]
    );
    return {
      sequence,
      sessionId,
      risk: assessment.finalRisk,
      matchingCount7d: assessment.matchingConcernCount7d,
      alertRequired: assessment.alertRequired,
      alertCreated: Boolean(alert),
      reason: assessment.alertSuppressedReason,
    };
  });
}

async function main() {
  const userResult = await query(
    `INSERT INTO users (full_name, email, password_hash)
     VALUES ('Stage 9 Caregiver Demo', $1, 'stage9-demo-not-for-login')
     ON CONFLICT (email) DO UPDATE SET full_name = EXCLUDED.full_name
     RETURNING id`,
    [DEMO_EMAIL]
  );
  const userId = userResult.rows[0].id;
  const existing = await query(
    `SELECT COUNT(*)::INT AS count FROM adaptive_chat_sessions
     WHERE user_id = $1 AND support_directive @> '{"stage9_demo":true}'::JSONB`,
    [userId]
  );
  if (Number(existing.rows[0].count) > 0) {
    throw new Error(`Stage 9 demo rows already exist for user ${userId}; refusing to duplicate them.`);
  }

  const day = 86400000;
  const results = [];
  results.push(await completeDemoSession(userId, new Date(Date.now() - 3 * day), 1));
  results.push(await completeDemoSession(userId, new Date(Date.now() - 2 * day), 2));
  results.push(await completeDemoSession(userId, new Date(Date.now() - day), 3));
  results.push(await completeDemoSession(userId, new Date(), 4));

  const sessions = await query(
    `SELECT session_id AS "sessionId", final_emotional_state AS "emotionalState",
       risk_level AS risk, caregiver_notification_required AS "caregiverNotificationRequired",
       completed_at AS "completedAt"
     FROM adaptive_chat_sessions
     WHERE user_id = $1 AND support_directive @> '{"stage9_demo":true}'::JSONB
     ORDER BY completed_at`,
    [userId]
  );
  const alerts = await query(
    `SELECT alert_id AS "alertId", adaptive_session_id AS "adaptiveSessionId",
       emotional_state AS "emotionalState", risk_level AS "riskLevel",
       alert_type AS "alertType", alert_message AS message,
       matching_concern_count_7d AS occurrences, status, explanation, created_at AS "createdAt"
     FROM emotional_caregiver_alerts WHERE user_id = $1 ORDER BY created_at`,
    [userId]
  );
  console.log(JSON.stringify({ userId, results, adaptive_chat_sessions: sessions.rows, emotional_caregiver_alerts: alerts.rows }, null, 2));
}

main()
  .catch((error) => { console.error(error.message); process.exitCode = 1; })
  .finally(() => getPool().end());
