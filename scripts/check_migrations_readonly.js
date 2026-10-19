const path = require('path');
const dotenv = require('dotenv');

// Load backend .env if present (read-only intent)
dotenv.config({ path: path.join(__dirname, '..', '.env') });

const { pool } = require('../src/config/db');

async function run() {
  const results = {};
  try {
    // 1) Check pgmigrations for the two migration filenames
    const mig1 = '1748640000000_component4_reminiscence_user_topics.js';
    const mig2 = '1748650000000_component4_elder_friendly_wording.js';

    const resMig1 = await pool.query(`SELECT name, run_on FROM pgmigrations WHERE name LIKE $1 LIMIT 1`, [
      `%${mig1}%`,
    ]);
    results.migration1_record = resMig1.rows.length > 0 ? resMig1.rows[0] : null;

    const resMig2 = await pool.query(`SELECT name, run_on FROM pgmigrations WHERE name LIKE $1 LIMIT 1`, [
      `%${mig2}%`,
    ]);
    results.migration2_record = resMig2.rows.length > 0 ? resMig2.rows[0] : null;

    // 2) Check reminiscence_user_topics table and columns
    const tableExistsRes = await pool.query(
      `SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = $1) AS exists`,
      ['reminiscence_user_topics']
    );
    results.reminiscence_table_exists = tableExistsRes.rows[0].exists;

    if (results.reminiscence_table_exists) {
      const cols = await pool.query(
        `SELECT column_name, data_type FROM information_schema.columns WHERE table_schema='public' AND table_name=$1 AND column_name IN ('consent_status','consent_recorded_at')`,
        ['reminiscence_user_topics']
      );
      results.reminiscence_columns = cols.rows; // array of found columns
    } else {
      results.reminiscence_columns = [];
    }

    // 3) Check adaptive_question_bank for rewritten question_texts and quick_replies
    const codes = [
      'lonely_contact_helpful',
      'lonely_daily_engagement',
      'anger_supportive_factor',
      'happy_energy_today',
      'cognitive_engagement_today',
      'neutral_daily_engagement',
    ];

    const textUpdates = {
      lonely_contact_helpful: 'Did talking with someone help you feel better today?',
      lonely_daily_engagement: 'Did you spend time doing something you enjoy today?',
      anger_supportive_factor: 'Did anything help the day feel better?',
      happy_energy_today: 'How has your energy felt today?',
      cognitive_engagement_today: 'Did something keep your interest today?',
      neutral_daily_engagement: 'What did you spend time doing today?',
    };

    const quickReplyExpected = {
      anxious_relax_today: ['Yes, there were', 'A little', 'Not really'],
      lonely_daily_engagement: ['Yes, I did', 'A little', 'Not really'],
      happy_energy_today: ['Good energy', 'About usual', 'A bit tired'],
      cognitive_engagement_today: ['Yes, it did', 'A little', 'Not really'],
    };

    results.adaptive_questions = {};
    for (const code of codes) {
      const q = await pool.query(`SELECT question_code, question_text, quick_replies FROM adaptive_question_bank WHERE question_code = $1 LIMIT 1`, [code]);
      if (q.rows.length === 0) {
        results.adaptive_questions[code] = { found: false };
      } else {
        const row = q.rows[0];
        // quick_replies may be jsonb; convert to labels list if present
        let quick = null;
        try {
          quick = row.quick_replies ? JSON.parse(JSON.stringify(row.quick_replies)) : null;
        } catch (e) {
          quick = row.quick_replies;
        }
        results.adaptive_questions[code] = {
          found: true,
          question_text: row.question_text,
          quick_replies: quick,
          text_matches_expected: row.question_text === textUpdates[code] || false,
        };
      }
    }

    // Check expected quick reply updates
    results.quick_reply_checks = {};
    for (const [code, labels] of Object.entries(quickReplyExpected)) {
      const q = await pool.query(`SELECT question_code, quick_replies FROM adaptive_question_bank WHERE question_code = $1 LIMIT 1`, [code]);
      if (q.rows.length === 0) {
        results.quick_reply_checks[code] = { found: false };
      } else {
        const quick = q.rows[0].quick_replies || null;
        // Normalize to labels array
        let labelsFound = null;
        try {
          labelsFound = Array.isArray(quick) ? quick.map((x) => x.label || x) : (quick && quick.map ? quick.map((x) => x.label || x) : null);
        } catch (e) {
          labelsFound = null;
        }
        results.quick_reply_checks[code] = {
          found: true,
          quick_labels: labelsFound,
          matches_expected: JSON.stringify(labelsFound) === JSON.stringify(labels),
        };
      }
    }

    // Done
    console.log(JSON.stringify({ success: true, results }, null, 2));
  } catch (err) {
    console.error(JSON.stringify({ success: false, error: String(err.message) }));
    process.exitCode = 2;
  } finally {
    try { await pool.end(); } catch (e) {}
  }
}

run();
