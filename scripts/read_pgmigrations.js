const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });
const db = require('../src/config/db');

(async () => {
  try {
    const res = await db.pool.query('SELECT name, run_on FROM pgmigrations ORDER BY run_on');
    console.log(JSON.stringify({ success: true, count: res.rows.length, rows: res.rows }, null, 2));
  } catch (err) {
    console.error(JSON.stringify({ success: false, error: String(err && err.message ? err.message : err) }));
    process.exitCode = 2;
  } finally {
    try { await db.pool.end(); } catch (e) {}
  }
})();
