const path = require('path');
const dotenv = require('dotenv');
const { runner } = require('node-pg-migrate');

dotenv.config({ path: path.join(__dirname, '..', '.env') });

const direction = process.argv[2] === 'down' ? 'down' : 'up';

runner({
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
  direction,
  migrationsTable: 'pgmigrations',
  count: direction === 'down' ? 1 : Infinity,
  singleTransaction: true,
  checkOrder: true,
  verbose: true,
}).catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
