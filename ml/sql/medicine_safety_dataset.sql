WITH questionnaire AS (
  SELECT
    user_id,
    MAX(CASE WHEN question_key = 'pastReaction' THEN answer_text END) AS q_past_reaction,
    MAX(CASE WHEN question_key = 'reactionSymptoms' THEN answer_text END) AS q_reaction_symptoms,
    MAX(CASE WHEN question_key = 'medicineName' THEN answer_text END) AS q_medicine_name,
    MAX(CASE WHEN question_key = 'doctorAdvice' THEN answer_text END) AS q_doctor_advice,
    MAX(CASE WHEN question_key = 'painkillerAntibioticReaction' THEN answer_text END) AS q_antibiotic_painkiller_reaction
  FROM allergy_questionnaire_answers
  GROUP BY user_id
),
reaction_summary AS (
  SELECT
    user_id,
    medicine_check_id,
    COUNT(*) AS reaction_count,
    MAX(CASE WHEN LOWER(COALESCE(severity, '')) IN ('severe', 'anaphylactic') THEN 1 ELSE 0 END) AS has_severe_reaction_log
  FROM reaction_logs
  GROUP BY user_id, medicine_check_id
)
SELECT
  mch.id AS medicine_check_id,
  mch.user_id,
  mch.input_method,
  mch.raw_input,
  mch.medicine_name,
  mch.normalized_drug_name,
  mch.rxnorm_cui,
  mch.ingredient_name,
  mch.therapeutic_class,
  mch.dose,
  mch.frequency,
  mch.risk_score,
  mch.risk_level,
  mch.side_effect_count,
  mch.severe_side_effect_count,
  mch.side_effect_match_count,
  mch.interaction_count,
  mch.max_interaction_severity,
  mch.knowledge_sources,
  mch.created_at AS medicine_check_created_at,
  uap.age,
  uap.gender,
  uap.has_medicine_allergy,
  uap.known_allergies_text,
  uap.chronic_diseases_text,
  uap.current_medications_text,
  uap.emergency_contact,
  uap.caregiver_details,
  q.q_past_reaction,
  q.q_reaction_symptoms,
  q.q_medicine_name,
  q.q_doctor_advice,
  q.q_antibiotic_painkiller_reaction,
  COALESCE(rs.reaction_count, 0) AS reaction_count,
  CASE WHEN COALESCE(rs.reaction_count, 0) > 0 THEN 1 ELSE 0 END AS has_reaction_log,
  COALESCE(rs.has_severe_reaction_log, 0) AS has_severe_reaction_log
FROM medicine_check_history mch
LEFT JOIN user_allergy_profiles uap
  ON uap.user_id = mch.user_id
LEFT JOIN questionnaire q
  ON q.user_id = mch.user_id
LEFT JOIN reaction_summary rs
  ON rs.user_id = mch.user_id
 AND rs.medicine_check_id = mch.id
ORDER BY mch.created_at DESC, mch.id DESC;
