const path = require('path');
const dotenv = require('dotenv');

dotenv.config({ path: path.resolve(__dirname, '..', '.env') });
const {
  pool,
  getDatabaseStatus,
  getDatabaseTroubleshootingHints,
} = require('../src/config/db');

const seedPatients = [
  {
    user: {
      full_name: 'Shanthi Perera',
      email: 'seed.eldermeds+shanthi@example.com',
      password_hash: 'seed-password-hash',
      phone: '+94 77 123 4567',
      date_of_birth: '1954-03-12',
      blood_type: 'A+',
    },
    profile: {
      age: '72',
      gender: 'Female',
      has_medicine_allergy: true,
      known_allergies_text: 'Penicillin, Sulfa drugs',
      chronic_diseases_text: 'Type 2 Diabetes, Hypertension',
      current_medications_text: 'Metformin, Losartan',
      emergency_contact: 'Nimal Perera - Son - +94 77 123 4567',
      caregiver_details: 'Home caregiver visits twice a week',
      profile_completed: true,
      reaction_symptoms_text: 'Skin rash, itching',
      suspected_medicine_names_text: 'Amoxicillin, Co-amoxiclav',
      avoided_medicines_text: 'Penicillin antibiotics',
      antibiotic_painkiller_reaction: 'Yes',
    },
    questionnaire: {
      pastReaction: 'Yes',
      reactionSymptoms: 'Skin rash and itching after antibiotics',
      medicineName: 'Amoxicillin',
      doctorAdvice: 'Avoid penicillin-class antibiotics',
      painkillerAntibioticReaction: 'Yes',
    },
    medicineCheck: {
      input_method: 'seed',
      raw_input: 'Amoxicillin 500mg',
      medicine_name: 'Amoxicillin',
      normalized_drug_name: 'amoxicillin',
      rxnorm_cui: '723',
      ingredient_name: 'Amoxicillin',
      therapeutic_class: 'Penicillin antibiotic',
      dose: '500mg',
      frequency: 'Three times daily',
      risk_score: 92,
      risk_level: 'Dangerous',
      side_effect_count: 8,
      severe_side_effect_count: 2,
      side_effect_match_count: 3,
      interaction_count: 2,
      max_interaction_severity: 'high',
      knowledge_sources: 'RxNorm; DailyMed; DDInter',
    },
    reactionLog: {
      symptoms: 'Generalized rash and lip swelling',
      severity: 'severe',
      notes: 'Past reaction requiring urgent clinic visit',
    },
  },
  {
    user: {
      full_name: 'Kamal Jayasinghe',
      email: 'seed.eldermeds+kamal@example.com',
      password_hash: 'seed-password-hash',
      phone: '+94 71 987 6543',
      date_of_birth: '1958-08-27',
      blood_type: 'B+',
    },
    profile: {
      age: '68',
      gender: 'Male',
      has_medicine_allergy: true,
      known_allergies_text: 'Aspirin',
      chronic_diseases_text: 'Hypertension, Chronic kidney disease',
      current_medications_text: 'Amlodipine, Metformin',
      emergency_contact: 'Daughter - +94 71 987 6543',
      caregiver_details: 'Monthly nephrology follow-up',
      profile_completed: true,
      reaction_symptoms_text: 'Breathing trouble, wheezing',
      suspected_medicine_names_text: 'Aspirin',
      avoided_medicines_text: 'Aspirin and related NSAIDs',
      antibiotic_painkiller_reaction: 'Yes',
    },
    questionnaire: {
      pastReaction: 'Yes',
      reactionSymptoms: 'Shortness of breath and wheezing',
      medicineName: 'Aspirin',
      doctorAdvice: 'Avoid aspirin and monitor painkillers carefully',
      painkillerAntibioticReaction: 'Yes',
    },
    medicineCheck: {
      input_method: 'seed',
      raw_input: 'Ibuprofen 400mg',
      medicine_name: 'Ibuprofen',
      normalized_drug_name: 'ibuprofen',
      rxnorm_cui: '5640',
      ingredient_name: 'Ibuprofen',
      therapeutic_class: 'NSAID',
      dose: '400mg',
      frequency: 'Twice daily',
      risk_score: 75,
      risk_level: 'Warning',
      side_effect_count: 6,
      severe_side_effect_count: 1,
      side_effect_match_count: 2,
      interaction_count: 1,
      max_interaction_severity: 'medium',
      knowledge_sources: 'RxNorm; DailyMed; DDInter',
    },
    reactionLog: {
      symptoms: 'Chest tightness and wheezing',
      severity: 'moderate',
      notes: 'Previous painkiller reaction reported by family',
    },
  },
  {
    user: {
      full_name: 'Madhavi Silva',
      email: 'seed.eldermeds+madhavi@example.com',
      password_hash: 'seed-password-hash',
      phone: '+94 76 000 1111',
      date_of_birth: '1949-11-05',
      blood_type: 'O+',
    },
    profile: {
      age: '75',
      gender: 'Female',
      has_medicine_allergy: false,
      known_allergies_text: '',
      chronic_diseases_text: 'Arthritis',
      current_medications_text: 'Paracetamol as needed',
      emergency_contact: 'Daughter - +94 76 000 1111',
      caregiver_details: 'Family caregiver',
      profile_completed: true,
      reaction_symptoms_text: '',
      suspected_medicine_names_text: '',
      avoided_medicines_text: '',
      antibiotic_painkiller_reaction: 'No',
    },
    questionnaire: {
      pastReaction: 'No',
      reactionSymptoms: 'None',
      medicineName: 'No',
      doctorAdvice: 'No',
      painkillerAntibioticReaction: 'No',
    },
    medicineCheck: {
      input_method: 'seed',
      raw_input: 'Paracetamol 500mg',
      medicine_name: 'Paracetamol',
      normalized_drug_name: 'paracetamol',
      rxnorm_cui: '161',
      ingredient_name: 'Paracetamol',
      therapeutic_class: 'Analgesic',
      dose: '500mg',
      frequency: 'As needed',
      risk_score: 15,
      risk_level: 'Safe',
      side_effect_count: 2,
      severe_side_effect_count: 0,
      side_effect_match_count: 0,
      interaction_count: 0,
      max_interaction_severity: 'none',
      knowledge_sources: 'RxNorm; DailyMed',
    },
  },
  {
    user: {
      full_name: 'Rashmi Fernando',
      email: 'seed.eldermeds+rashmi@example.com',
      password_hash: 'seed-password-hash',
      phone: '+94 75 123 3333',
      date_of_birth: '1956-02-14',
      blood_type: 'AB+',
    },
    profile: {
      age: '70',
      gender: 'Male',
      has_medicine_allergy: true,
      known_allergies_text: 'Ibuprofen',
      chronic_diseases_text: 'Heart disease, Hypertension',
      current_medications_text: 'Aspirin, Bisoprolol, Atorvastatin',
      emergency_contact: 'Wife - +94 75 123 3333',
      caregiver_details: 'Cardiology clinic review',
      profile_completed: true,
      reaction_symptoms_text: 'Facial swelling',
      suspected_medicine_names_text: 'Ibuprofen',
      avoided_medicines_text: 'Ibuprofen and diclofenac',
      antibiotic_painkiller_reaction: 'Yes',
    },
    questionnaire: {
      pastReaction: 'Yes',
      reactionSymptoms: 'Facial swelling after painkiller',
      medicineName: 'Ibuprofen',
      doctorAdvice: 'Use only doctor-approved pain relief',
      painkillerAntibioticReaction: 'Yes',
    },
    medicineCheck: {
      input_method: 'seed',
      raw_input: 'Diclofenac 50mg',
      medicine_name: 'Diclofenac',
      normalized_drug_name: 'diclofenac',
      rxnorm_cui: '3355',
      ingredient_name: 'Diclofenac',
      therapeutic_class: 'NSAID',
      dose: '50mg',
      frequency: 'Twice daily',
      risk_score: 88,
      risk_level: 'Dangerous',
      side_effect_count: 7,
      severe_side_effect_count: 2,
      side_effect_match_count: 2,
      interaction_count: 2,
      max_interaction_severity: 'high',
      knowledge_sources: 'RxNorm; DailyMed; DDInter',
    },
    reactionLog: {
      symptoms: 'Facial swelling and dizziness',
      severity: 'severe',
      notes: 'Known cross-reaction with NSAIDs',
    },
  },
  {
    user: {
      full_name: 'Sujatha Wijesinghe',
      email: 'seed.eldermeds+sujatha@example.com',
      password_hash: 'seed-password-hash',
      phone: '+94 72 444 5555',
      date_of_birth: '1952-06-21',
      blood_type: 'A-',
    },
    profile: {
      age: '73',
      gender: 'Female',
      has_medicine_allergy: false,
      known_allergies_text: '',
      chronic_diseases_text: 'Type 2 Diabetes',
      current_medications_text: 'Metformin, Gliclazide',
      emergency_contact: 'Son - +94 72 444 5555',
      caregiver_details: 'Lives with family',
      profile_completed: true,
      reaction_symptoms_text: '',
      suspected_medicine_names_text: '',
      avoided_medicines_text: '',
      antibiotic_painkiller_reaction: 'Not sure',
    },
    questionnaire: {
      pastReaction: 'No',
      reactionSymptoms: 'None',
      medicineName: 'No',
      doctorAdvice: 'No',
      painkillerAntibioticReaction: 'Not sure',
    },
    medicineCheck: {
      input_method: 'seed',
      raw_input: 'Azithromycin 500mg',
      medicine_name: 'Azithromycin',
      normalized_drug_name: 'azithromycin',
      rxnorm_cui: '18631',
      ingredient_name: 'Azithromycin',
      therapeutic_class: 'Macrolide antibiotic',
      dose: '500mg',
      frequency: 'Once daily',
      risk_score: 48,
      risk_level: 'Warning',
      side_effect_count: 4,
      severe_side_effect_count: 0,
      side_effect_match_count: 1,
      interaction_count: 1,
      max_interaction_severity: 'medium',
      knowledge_sources: 'RxNorm; DailyMed; DDInter',
    },
  },
  {
    user: {
      full_name: 'Nalin De Costa',
      email: 'seed.eldermeds+nalin@example.com',
      password_hash: 'seed-password-hash',
      phone: '+94 78 888 9999',
      date_of_birth: '1950-01-09',
      blood_type: 'O-',
    },
    profile: {
      age: '76',
      gender: 'Male',
      has_medicine_allergy: true,
      known_allergies_text: 'Sulfa drugs',
      chronic_diseases_text: 'Chronic kidney disease, Hypertension',
      current_medications_text: 'Losartan, Furosemide',
      emergency_contact: 'Brother - +94 78 888 9999',
      caregiver_details: 'Weekly clinic dialysis assessment',
      profile_completed: true,
      reaction_symptoms_text: 'Rash, vomiting',
      suspected_medicine_names_text: 'Co-trimoxazole',
      avoided_medicines_text: 'Sulfonamide antibiotics',
      antibiotic_painkiller_reaction: 'Yes',
    },
    questionnaire: {
      pastReaction: 'Yes',
      reactionSymptoms: 'Rash and vomiting',
      medicineName: 'Co-trimoxazole',
      doctorAdvice: 'Avoid sulfonamide antibiotics',
      painkillerAntibioticReaction: 'Yes',
    },
    medicineCheck: {
      input_method: 'seed',
      raw_input: 'Co-amoxiclav 625mg',
      medicine_name: 'Co-amoxiclav',
      normalized_drug_name: 'co-amoxiclav',
      rxnorm_cui: '199',
      ingredient_name: 'Amoxicillin / Clavulanate',
      therapeutic_class: 'Penicillin antibiotic',
      dose: '625mg',
      frequency: 'Twice daily',
      risk_score: 81,
      risk_level: 'Dangerous',
      side_effect_count: 7,
      severe_side_effect_count: 1,
      side_effect_match_count: 2,
      interaction_count: 2,
      max_interaction_severity: 'high',
      knowledge_sources: 'RxNorm; DailyMed; DDInter',
    },
    reactionLog: {
      symptoms: 'Rash and vomiting',
      severity: 'moderate',
      notes: 'Family reports poor tolerance to antibiotics',
    },
  },
];

const questionnaireEntries = (questionnaire) => [
  ['pastReaction', questionnaire.pastReaction],
  ['reactionSymptoms', questionnaire.reactionSymptoms],
  ['medicineName', questionnaire.medicineName],
  ['doctorAdvice', questionnaire.doctorAdvice],
  ['painkillerAntibioticReaction', questionnaire.painkillerAntibioticReaction],
];

const seedOnePatient = async (client, patient) => {
  const userResult = await client.query(
    `
      INSERT INTO users (full_name, email, password_hash, phone, date_of_birth, blood_type)
      VALUES ($1, $2, $3, $4, $5, $6)
      ON CONFLICT (email) DO UPDATE
      SET
        full_name = EXCLUDED.full_name,
        password_hash = EXCLUDED.password_hash,
        phone = EXCLUDED.phone,
        date_of_birth = EXCLUDED.date_of_birth,
        blood_type = EXCLUDED.blood_type,
        updated_at = NOW()
      RETURNING id
    `,
    [
      patient.user.full_name,
      patient.user.email,
      patient.user.password_hash,
      patient.user.phone,
      patient.user.date_of_birth,
      patient.user.blood_type,
    ]
  );

  const userId = userResult.rows[0].id;

  await client.query(
    `
      INSERT INTO user_routines (user_id)
      VALUES ($1)
      ON CONFLICT (user_id) DO NOTHING
    `,
    [userId]
  );

  await client.query(
    `
      INSERT INTO user_allergy_profiles (
        user_id,
        age,
        gender,
        has_medicine_allergy,
        known_allergies_text,
        chronic_diseases_text,
        current_medications_text,
        emergency_contact,
        caregiver_details,
        profile_completed,
        reaction_symptoms_text,
        suspected_medicine_names_text,
        avoided_medicines_text,
        antibiotic_painkiller_reaction,
        updated_at
      )
      VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, NOW())
      ON CONFLICT (user_id) DO UPDATE
      SET
        age = EXCLUDED.age,
        gender = EXCLUDED.gender,
        has_medicine_allergy = EXCLUDED.has_medicine_allergy,
        known_allergies_text = EXCLUDED.known_allergies_text,
        chronic_diseases_text = EXCLUDED.chronic_diseases_text,
        current_medications_text = EXCLUDED.current_medications_text,
        emergency_contact = EXCLUDED.emergency_contact,
        caregiver_details = EXCLUDED.caregiver_details,
        profile_completed = EXCLUDED.profile_completed,
        reaction_symptoms_text = EXCLUDED.reaction_symptoms_text,
        suspected_medicine_names_text = EXCLUDED.suspected_medicine_names_text,
        avoided_medicines_text = EXCLUDED.avoided_medicines_text,
        antibiotic_painkiller_reaction = EXCLUDED.antibiotic_painkiller_reaction,
        updated_at = NOW()
    `,
    [
      userId,
      patient.profile.age,
      patient.profile.gender,
      patient.profile.has_medicine_allergy,
      patient.profile.known_allergies_text,
      patient.profile.chronic_diseases_text,
      patient.profile.current_medications_text,
      patient.profile.emergency_contact,
      patient.profile.caregiver_details,
      patient.profile.profile_completed,
      patient.profile.reaction_symptoms_text,
      patient.profile.suspected_medicine_names_text,
      patient.profile.avoided_medicines_text,
      patient.profile.antibiotic_painkiller_reaction,
    ]
  );

  for (const [questionKey, answerText] of questionnaireEntries(patient.questionnaire)) {
    await client.query(
      `
        INSERT INTO allergy_questionnaire_answers (user_id, question_key, answer_text)
        VALUES ($1, $2, $3)
        ON CONFLICT (user_id, question_key) DO UPDATE
        SET answer_text = EXCLUDED.answer_text
      `,
      [userId, questionKey, answerText]
    );
  }

  let medicineCheckId;
  const existingCheck = await client.query(
    `
      SELECT id
      FROM medicine_check_history
      WHERE user_id = $1
        AND input_method = $2
        AND raw_input = $3
        AND risk_level = $4
      LIMIT 1
    `,
    [
      userId,
      patient.medicineCheck.input_method,
      patient.medicineCheck.raw_input,
      patient.medicineCheck.risk_level,
    ]
  );

  if (existingCheck.rowCount) {
    medicineCheckId = existingCheck.rows[0].id;
    await client.query(
      `
        UPDATE medicine_check_history
        SET
          medicine_name = $2,
          normalized_drug_name = $3,
          rxnorm_cui = $4,
          ingredient_name = $5,
          therapeutic_class = $6,
          dose = $7,
          frequency = $8,
          risk_score = $9,
          side_effect_count = $10,
          severe_side_effect_count = $11,
          side_effect_match_count = $12,
          interaction_count = $13,
          max_interaction_severity = $14,
          knowledge_sources = $15
        WHERE id = $1
      `,
      [
        medicineCheckId,
        patient.medicineCheck.medicine_name,
        patient.medicineCheck.normalized_drug_name,
        patient.medicineCheck.rxnorm_cui,
        patient.medicineCheck.ingredient_name,
        patient.medicineCheck.therapeutic_class,
        patient.medicineCheck.dose,
        patient.medicineCheck.frequency,
        patient.medicineCheck.risk_score,
        patient.medicineCheck.side_effect_count,
        patient.medicineCheck.severe_side_effect_count,
        patient.medicineCheck.side_effect_match_count,
        patient.medicineCheck.interaction_count,
        patient.medicineCheck.max_interaction_severity,
        patient.medicineCheck.knowledge_sources,
      ]
    );
  } else {
    const insertedCheck = await client.query(
      `
        INSERT INTO medicine_check_history (
          user_id,
          input_method,
          raw_input,
          medicine_name,
          normalized_drug_name,
          rxnorm_cui,
          ingredient_name,
          therapeutic_class,
          dose,
          frequency,
          risk_score,
          risk_level,
          side_effect_count,
          severe_side_effect_count,
          side_effect_match_count,
          interaction_count,
          max_interaction_severity,
          knowledge_sources
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
        RETURNING id
      `,
      [
        userId,
        patient.medicineCheck.input_method,
        patient.medicineCheck.raw_input,
        patient.medicineCheck.medicine_name,
        patient.medicineCheck.normalized_drug_name,
        patient.medicineCheck.rxnorm_cui,
        patient.medicineCheck.ingredient_name,
        patient.medicineCheck.therapeutic_class,
        patient.medicineCheck.dose,
        patient.medicineCheck.frequency,
        patient.medicineCheck.risk_score,
        patient.medicineCheck.risk_level,
        patient.medicineCheck.side_effect_count,
        patient.medicineCheck.severe_side_effect_count,
        patient.medicineCheck.side_effect_match_count,
        patient.medicineCheck.interaction_count,
        patient.medicineCheck.max_interaction_severity,
        patient.medicineCheck.knowledge_sources,
      ]
    );
    medicineCheckId = insertedCheck.rows[0].id;
  }

  if (patient.reactionLog) {
    const existingReaction = await client.query(
      `
        SELECT id
        FROM reaction_logs
        WHERE user_id = $1
          AND medicine_check_id = $2
          AND symptoms = $3
          AND severity = $4
        LIMIT 1
      `,
      [
        userId,
        medicineCheckId,
        patient.reactionLog.symptoms,
        patient.reactionLog.severity,
      ]
    );

    if (existingReaction.rowCount) {
      await client.query(
        `
          UPDATE reaction_logs
          SET notes = $2
          WHERE id = $1
        `,
        [existingReaction.rows[0].id, patient.reactionLog.notes]
      );
    } else {
      await client.query(
        `
          INSERT INTO reaction_logs (user_id, medicine_check_id, symptoms, severity, notes)
          VALUES ($1, $2, $3, $4, $5)
        `,
        [
          userId,
          medicineCheckId,
          patient.reactionLog.symptoms,
          patient.reactionLog.severity,
          patient.reactionLog.notes,
        ]
      );
    }
  }
};

const main = async () => {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    for (const patient of seedPatients) {
      await seedOnePatient(client, patient);
    }
    await client.query('COMMIT');
    console.log(`[DB] Seeded ${seedPatients.length} allergy demo users`);
  } catch (error) {
    await client.query('ROLLBACK');
    throw error;
  } finally {
    client.release();
    await pool.end();
  }
};

main().catch((error) => {
  console.error('[DB] Allergy demo seed failed:', error.message);
  const status = getDatabaseStatus();
  console.error(
    `[DB] Target: ${status.host}:${status.port}/${status.database} as ${status.user}`
  );
  const hints = getDatabaseTroubleshootingHints(error.message);
  if (hints.length) {
    console.error('[DB] Troubleshooting hints:');
    for (const hint of hints) {
      console.error(` - ${hint}`);
    }
  }
  process.exit(1);
});
