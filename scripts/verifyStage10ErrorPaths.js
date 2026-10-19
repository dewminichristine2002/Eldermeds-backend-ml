const path = require('path');
const axios = require('axios');
const dotenv = require('dotenv');

dotenv.config({ path: path.join(__dirname, '..', '.env') });
const { getPool, query } = require('../src/db/postgres');
const API = process.env.STAGE10_DEMO_API || 'http://localhost:5010/api/emotional-support';

async function capture(name, request, expectedStatus) {
  try {
    await request();
    return { name, passed: false, expectedStatus, actualStatus: 200 };
  } catch (error) {
    const actualStatus = error.response?.status || null;
    return { name, passed: actualStatus === expectedStatus, expectedStatus, actualStatus, message: error.response?.data?.error || error.message };
  }
}

async function main() {
  const user = await query(`SELECT id FROM users WHERE email = 'component4-stage10-viva-ready@eldermeds.local'`);
  const userId = user.rows[0].id;
  const completed = await query(`SELECT session_id FROM adaptive_chat_sessions WHERE user_id = $1 AND is_complete = TRUE ORDER BY completed_at DESC LIMIT 1`, [userId]);
  const attempt = await query(`SELECT attempt_id FROM adaptive_activity_attempts WHERE user_id = $1 AND completed_at IS NOT NULL ORDER BY completed_at DESC LIMIT 1`, [userId]);
  const started = await axios.post(`${API}/adaptive-chat/start`, { user_id: userId });
  const incompleteSessionId = started.data.session_id;

  const results = [];
  results.push(await capture('missing adaptive session', () => axios.post(`${API}/adaptive-chat/respond`, { user_id: userId, session_id: '00000000-0000-4000-8000-000000000000', question_id: 1, answer_text: 'test' }), 404));
  results.push(await capture('completed session reused', () => axios.post(`${API}/adaptive-chat/respond`, { user_id: userId, session_id: completed.rows[0].session_id, question_id: 1, answer_text: 'test' }), 409));
  results.push(await capture('wrong question id', () => axios.post(`${API}/adaptive-chat/respond`, { user_id: userId, session_id: incompleteSessionId, question_id: 999999, answer_text: 'test' }), 409));
  results.push(await capture('activity start without completed session', () => axios.post(`${API}/adaptive-activities/start`, { user_id: userId, session_id: incompleteSessionId, activity_code: 'word_category_easy' }), 409));
  results.push(await capture('duplicate activity submit', () => axios.post(`${API}/adaptive-activities/attempts/${attempt.rows[0].attempt_id}/submit`, { user_id: userId, response: {} }), 409));
  results.push(await capture('invalid wellness period', () => axios.get(`${API}/wellness-trends/${userId}`, { params: { period: '90d' } }), 400));
  results.push(await capture('invalid caregiver alert user', () => axios.get(`${API}/caregiver-alerts/not-a-user`), 400));
  console.log(JSON.stringify({ userId, incompleteSessionId, results, passed: results.filter((item) => item.passed).length, failed: results.filter((item) => !item.passed).length }, null, 2));
}

main().catch((error) => {
  console.error(error.response?.data || error.message);
  process.exitCode = 1;
}).finally(() => getPool().end());
