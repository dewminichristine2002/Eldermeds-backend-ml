const path = require('path');
const dotenv = require('dotenv');

dotenv.config({ path: path.join(__dirname, '..', '.env') });
const { getPool, query, withTransaction } = require('../src/db/postgres');

const ALLOWED_EMAILS = Object.freeze([
  'component4-stage9-demo@eldermeds.local',
  'component4-stage10-viva-demo@eldermeds.local',
  'component4-stage10-viva-final@eldermeds.local',
  'component4-stage10-viva-ready@eldermeds.local',
]);

async function main() {
  const rows = await query(
    `SELECT id, email FROM users WHERE email = ANY($1::TEXT[]) ORDER BY id`,
    [ALLOWED_EMAILS]
  );
  console.log('Known Component 4 synthetic users:', rows.rows);
  if (process.argv[2] !== '--confirm') {
    console.log('Dry run only. Re-run with --confirm to delete only the users listed above and their foreign-key-cascaded demo rows.');
    return;
  }
  if (rows.rows.some((row) => !ALLOWED_EMAILS.includes(row.email))) {
    throw new Error('Cleanup scope validation failed.');
  }
  const deleted = await withTransaction((client) => client.query(
    `DELETE FROM users WHERE email = ANY($1::TEXT[]) RETURNING id, email`,
    [ALLOWED_EMAILS]
  ));
  console.log('Deleted known synthetic users:', deleted.rows);
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
}).finally(() => getPool().end());
