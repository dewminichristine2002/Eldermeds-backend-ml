const path = require('path');
const dotenv = require('dotenv');

dotenv.config({ path: path.join(__dirname, '..', '.env') });
const { getPool, query } = require('../src/db/postgres');

async function main() {
  const migrations = await query(`
    SELECT name, COUNT(*)::INT AS count
    FROM pgmigrations
    WHERE name LIKE '%component4%'
    GROUP BY name
    ORDER BY name
  `);
  const indexes = await query(`
    SELECT tablename, indexname, indexdef
    FROM pg_indexes
    WHERE schemaname = 'public'
      AND tablename IN ('adaptive_chat_sessions', 'adaptive_chat_turns', 'adaptive_activity_attempts', 'emotional_caregiver_alerts')
    ORDER BY tablename, indexname
  `);
  const constraints = await query(`
    SELECT conrelid::regclass::TEXT AS table_name, conname,
           pg_get_constraintdef(oid) AS definition
    FROM pg_constraint
    WHERE conrelid IN (
      'adaptive_chat_sessions'::regclass, 'adaptive_chat_turns'::regclass,
      'adaptive_activity_attempts'::regclass, 'emotional_caregiver_alerts'::regclass
    )
    ORDER BY table_name, conname
  `);
  const demoUsers = await query(`
    SELECT users.id, users.email,
      (SELECT COUNT(*)::INT FROM adaptive_chat_sessions WHERE user_id = users.id) AS adaptive_sessions,
      (SELECT COUNT(*)::INT FROM adaptive_activity_attempts WHERE user_id = users.id) AS activity_attempts,
      (SELECT COUNT(*)::INT FROM emotional_caregiver_alerts WHERE user_id = users.id) AS emotional_alerts
    FROM users
    WHERE email ILIKE '%demo%' OR full_name ILIKE '%demo%'
    ORDER BY users.id
  `);
  const integrity = await query(`
    SELECT
      (SELECT COUNT(*)::INT FROM emotional_caregiver_alerts alerts LEFT JOIN users ON users.id = alerts.user_id WHERE users.id IS NULL) AS orphan_alert_users,
      (SELECT COUNT(*)::INT FROM adaptive_activity_attempts WHERE accuracy_score IS NOT NULL AND (accuracy_score < 0 OR accuracy_score > 1)) AS invalid_accuracy,
      (SELECT COUNT(*)::INT FROM adaptive_activity_attempts WHERE response_time_ms < 0) AS negative_response_times,
      (SELECT COUNT(*)::INT FROM adaptive_activity_attempts WHERE completion_status NOT IN ('started', 'completed')) AS invalid_completion_status
  `);
  console.log(JSON.stringify({ migrations: migrations.rows, indexes: indexes.rows, constraints: constraints.rows, integrity: integrity.rows[0], demoUsers: demoUsers.rows }, null, 2));
}

main()
  .catch((error) => { console.error(error.message); process.exitCode = 1; })
  .finally(() => getPool().end());
