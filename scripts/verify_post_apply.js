const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });
const db = require('../src/config/db');

(async () => {
  try {
    const tableRes = await db.pool.query(`SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema='public' AND table_name='reminiscence_user_topics') AS exists`);
    const colsRes = await db.pool.query(`SELECT column_name, data_type FROM information_schema.columns WHERE table_schema='public' AND table_name='reminiscence_user_topics' AND column_name IN ('consent_status','consent_recorded_at')`);
    const questions = await db.pool.query(`SELECT question_code, question_text, quick_replies FROM adaptive_question_bank WHERE question_code IN ('lonely_contact_helpful','lonely_daily_engagement','anger_supportive_factor','happy_energy_today','cognitive_engagement_today','neutral_daily_engagement') ORDER BY question_code`);

    console.log(JSON.stringify({
      reminiscence_table_exists: tableRes.rows[0].exists,
      reminiscence_columns: colsRes.rows,
      adaptive_questions: questions.rows
    }, null, 2));
  } catch (err) {
    console.error(JSON.stringify({ success: false, error: String(err && err.message ? err.message : err) }));
    process.exitCode = 2;
  } finally {
    try { await db.pool.end(); } catch (e) {}
  }
})();
