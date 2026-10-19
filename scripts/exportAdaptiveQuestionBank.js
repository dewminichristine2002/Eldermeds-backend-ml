#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');

dotenv.config({ path: path.resolve(__dirname, '../.env') });
dotenv.config({ path: path.resolve(__dirname, '../../.env') });

const { query, getPool } = require('../src/db/postgres');
const { buildQuestionBankExport } = require('../src/services/questionBankExport');
let poolOpened = false;

async function main() {
  const outputPath = path.resolve(process.argv[2] || path.join(__dirname, '../ml/emotion_classifier/stage9_data_collection/adaptive_question_bank_export.json'));
  const result = await query(`
    SELECT question_id, question_code, question_text, phase, category, sub_category,
           target_state, question_type, trigger_keywords, response_type, quick_replies,
           priority, construct_source, is_active, assessment_dimension, is_assessment,
           positive_next_code, negative_next_code, neutral_next_code, followup_next_code
    FROM adaptive_question_bank
    ORDER BY question_id ASC
  `);
  poolOpened = true;
  const exported = buildQuestionBankExport(result.rows);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify(exported, null, 2)}\n`, { flag: 'wx' });
  console.log(JSON.stringify({ output: outputPath, question_count: exported.question_count,
    question_bank_version: exported.question_bank_version }, null, 2));
}

main().catch((error) => {
  console.error(`Question bank export failed: ${error.message}`);
  process.exitCode = 1;
}).finally(async () => {
  if (poolOpened) {
    try { await getPool().end(); } catch (_) { /* ignore shutdown errors after the export */ }
  }
});
