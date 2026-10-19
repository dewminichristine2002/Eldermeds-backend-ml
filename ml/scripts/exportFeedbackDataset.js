/**
 * Phase 9 — export consented, anonymized feedback for continuous learning.
 * Writes: ml/data/feedback_anonymized.json
 */
const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');

dotenv.config({ path: path.resolve(__dirname, '..', '..', '.env') });

const { pool } = require('../../src/config/db');
const {
  anonymizeReactionRow,
  anonymizeTrainingRow,
} = require('../../src/services/feedbackAnonymizationService');

const outputPath = path.resolve(__dirname, '..', 'data', 'feedback_anonymized.json');

const main = async () => {
  const allergyModel = require('../../src/models/allergyModel');
  const { reactions, checks } = await allergyModel.listConsentedFeedbackForExport();

  const salt = process.env.FEEDBACK_ANONYMIZATION_SALT || 'eldermeds-feedback-anonymization-v1';
  const payload = {
    exportedAt: new Date().toISOString(),
    privacy: {
      anonymized: true,
      consentRequired: true,
      piiStripped: ['user_id', 'raw_input', 'emergency_contact', 'caregiver_details', 'exact_age'],
      differentialPrivacyAppliedOnAggregates: true,
    },
    reactionFeedback: reactions.map((row) => anonymizeReactionRow(row, { salt })),
    checkFeedback: checks.map((row) => anonymizeTrainingRow(row, { salt })),
    counts: {
      reactions: reactions.length,
      checks: checks.length,
      consentedUsersOnly: true,
    },
  };

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(payload, null, 2));

  console.log(`[ML] Exported ${reactions.length} anonymized reactions and ${checks.length} check rows`);
  console.log(`[ML] JSON -> ${outputPath}`);
  await pool.end();
};

main().catch(async (error) => {
  console.error('[ML] Feedback export failed:', error.message);
  try {
    await pool.end();
  } catch {
    // ignore
  }
  process.exit(1);
});
