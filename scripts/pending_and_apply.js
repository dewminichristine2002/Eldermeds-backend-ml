const path = require('path');
const fs = require('fs');
const dotenv = require('dotenv');
const { runner } = require('node-pg-migrate');

dotenv.config({ path: path.join(__dirname, '..', '.env') });

const MIG1 = '1748640000000_component4_reminiscence_user_topics.js';
const MIG2 = '1748650000000_component4_elder_friendly_wording.js';

async function main() {
  const poolModule = require('../src/config/db');
  const pool = poolModule.pool;
  try {
    // list files from migrations dir
    const migDir = path.join(__dirname, '..', 'migrations');
    const files = fs.readdirSync(migDir).filter(f => f.endsWith('.js'));
    files.sort();

    // get applied migrations from pgmigrations
    const res = await pool.query(`SELECT name FROM pgmigrations`);
    const applied = res.rows.map(r => r.name);

    const pending = files.filter(f => !applied.includes(f));

    console.log('PENDING_LIST', JSON.stringify(pending));

    // If pending contains exactly the two target migrations (order may vary)
    const expectedSet = new Set([MIG1, MIG2]);
    const pendingSet = new Set(pending);
    if (pending.length === 2 && pending.every(p => expectedSet.has(p))) {
      console.log('APPLYING_MIGRATIONS');
      await runner({
        databaseUrl: {
          host: process.env.DB_HOST,
          port: Number(process.env.DB_PORT || 5432),
          database: process.env.DB_NAME,
          user: process.env.DB_USER,
          password: process.env.DB_PASSWORD,
          ssl: process.env.DB_SSL === 'true' || String(process.env.DB_HOST || '').includes('supabase.com')
            ? { rejectUnauthorized: false }
            : false,
        },
        dir: path.join(__dirname, '..', 'migrations'),
        direction: 'up',
        migrationsTable: 'pgmigrations',
        count: Infinity,
        singleTransaction: true,
        checkOrder: true,
        verbose: true,
      });
      console.log('APPLY_DONE');
    } else {
      if (pending.length === 0) {
        console.log('NO_PENDING');
      } else {
        console.log('OTHER_PENDING', JSON.stringify(pending));
      }
    }
  } catch (err) {
    console.error('ERROR', err && err.stack ? err.stack : err);
    process.exitCode = 2;
  } finally {
    try { await pool.end(); } catch (e) {}
  }
}

main();
