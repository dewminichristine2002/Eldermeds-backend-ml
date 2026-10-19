const fs = require('fs');
const path = require('path');
const { parseCsvLine } = require('../../scripts/csvLine');

const defaultSourcePath = 'C:\\Users\\thyag\\Downloads\\WHO ATC-DDD 2021-12-03.csv';
const sourcePath = process.argv[2] || process.env.ATC_SOURCE_CSV || defaultSourcePath;
const outputDir = path.resolve(__dirname, '..', 'data');
const jsonOutputPath = path.join(outputDir, 'drug_class_dataset.json');
const csvOutputPath = path.join(outputDir, 'drug_class_dataset.csv');

const normalizeDrugName = (value) =>
  String(value || '')
    .trim()
    .toLowerCase()
    .replace(/\b\d+(\.\d+)?\s*(mg|mcg|g|ml|tablet|capsule)\b/g, ' ')
    .replace(/[()]/g, ' ')
    .replace(/[^a-z0-9,\s/-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

const isLeafCode = (code) => /^[A-Z]\d{2}[A-Z]{2}\d{2}$/.test(code || '');
const isCombinationName = (name) => /,| and |combinations?/i.test(String(name || ''));

const FAMILY_RULES = [
  { prefixes: ['J01CA', 'J01CE', 'J01CF', 'J01CG', 'J01CR'], drugClass: 'penicillin' },
  { prefixes: ['J01DB', 'J01DC', 'J01DD', 'J01DE', 'J01DI'], drugClass: 'cephalosporin' },
  { prefixes: ['J01FA'], drugClass: 'macrolide' },
  { prefixes: ['J01MA'], drugClass: 'fluoroquinolone' },
  { prefixes: ['J01AA'], drugClass: 'tetracycline' },
  { prefixes: ['J01GB'], drugClass: 'aminoglycoside' },
  { prefixes: ['J01EE', 'J01EC', 'J01ED', 'J01EB'], drugClass: 'sulfonamide' },
  { prefixes: ['M01A', 'M02AA', 'S01BC', 'G02CC'], drugClass: 'nsaid' },
  { prefixes: ['B01AA', 'B01AE', 'B01AF'], drugClass: 'anticoagulant' },
  { prefixes: ['B01AC'], drugClass: 'antiplatelet' },
  { prefixes: ['N02A'], drugClass: 'opioid' },
  { prefixes: ['N05BA', 'N05CD', 'N05CF'], drugClass: 'benzodiazepine' },
  { prefixes: ['R06A'], drugClass: 'antihistamine' },
  { prefixes: ['C09CA', 'C09DA'], drugClass: 'arb' },
  { prefixes: ['C09AA', 'C09BA'], drugClass: 'ace_inhibitor' },
  { prefixes: ['C07'], drugClass: 'beta_blocker' },
  { prefixes: ['C10AA'], drugClass: 'statin' },
  { prefixes: ['A10BA'], drugClass: 'biguanide' },
  { prefixes: ['A10BB'], drugClass: 'sulfonylurea' },
];

const rowsToCsv = (rows) => {
  if (!rows.length) {
    return '';
  }

  const headers = Object.keys(rows[0]);
  const escape = (value) => {
    if (value == null) {
      return '';
    }

    const text = String(value);
    if (/[",\n]/.test(text)) {
      return `"${text.replace(/"/g, '""')}"`;
    }
    return text;
  };

  return [
    headers.join(','),
    ...rows.map((row) => headers.map((header) => escape(row[header])).join(',')),
  ].join('\n');
};

const familyFromAtcCode = (atcCode) => {
  for (const rule of FAMILY_RULES) {
    if (rule.prefixes.some((prefix) => atcCode.startsWith(prefix))) {
      return rule.drugClass;
    }
  }
  return '';
};

const familyScore = (atcCode, route) => {
  const knownFamily = familyFromAtcCode(atcCode) ? 1000 : 0;
  const routeScore = route === 'O' ? 30 : route === 'P' ? 20 : route === 'R' ? 10 : 0;
  return knownFamily + atcCode.length + routeScore;
};

const parentCodeChain = (atcCode) => {
  const codes = [];
  if (atcCode.length >= 5) codes.push(atcCode.slice(0, 5));
  if (atcCode.length >= 4) codes.push(atcCode.slice(0, 4));
  if (atcCode.length >= 3) codes.push(atcCode.slice(0, 3));
  if (atcCode.length >= 1) codes.push(atcCode.slice(0, 1));
  return codes;
};

const buildClassLabel = (atcCode, hierarchy) => {
  for (const parentCode of parentCodeChain(atcCode)) {
    if (hierarchy.has(parentCode)) {
      return hierarchy.get(parentCode);
    }
  }
  return '';
};

const main = () => {
  if (!fs.existsSync(sourcePath)) {
    throw new Error(`ATC source file not found: ${sourcePath}`);
  }

  fs.mkdirSync(outputDir, { recursive: true });

  const lines = fs.readFileSync(sourcePath, 'utf8').split(/\r?\n/).filter(Boolean);
  const [headerLine, ...dataLines] = lines;
  const headers = parseCsvLine(headerLine);

  if (!headers.includes('atc_code') || !headers.includes('atc_name')) {
    throw new Error('ATC source file does not have expected headers.');
  }

  const hierarchy = new Map();
  const grouped = new Map();

  for (const line of dataLines) {
    const parts = parseCsvLine(line);
    if (parts.length !== headers.length) {
      continue;
    }

    const row = headers.reduce((acc, header, index) => {
      acc[header] = parts[index] ?? '';
      return acc;
    }, {});

    const atcCode = String(row.atc_code || '').trim();
    const atcName = String(row.atc_name || '').trim();
    const route = String(row.adm_r || '').trim();

    if (!atcCode || !atcName) {
      continue;
    }

    hierarchy.set(atcCode, atcName);

    if (!isLeafCode(atcCode) || isCombinationName(atcName)) {
      continue;
    }

    const normalizedName = normalizeDrugName(atcName);
    if (!normalizedName) {
      continue;
    }

    const family = familyFromAtcCode(atcCode);
    const groupCode = atcCode.slice(0, 5);
    const classLabel = buildClassLabel(atcCode, hierarchy);
    const candidate = {
      normalized_name: normalizedName,
      drug_name: atcName,
      ingredient_name: atcName,
      atc_code: atcCode,
      atc_group_code: groupCode,
      atc_group_name: hierarchy.get(groupCode) || '',
      atc_class_label: classLabel,
      drug_class: family,
      route,
      score: familyScore(atcCode, route),
    };

    if (!grouped.has(normalizedName)) {
      grouped.set(normalizedName, []);
    }
    grouped.get(normalizedName).push(candidate);
  }

  const records = Array.from(grouped.values())
    .map((candidates) => {
      const best = [...candidates].sort((left, right) => right.score - left.score)[0];
      return {
        normalized_name: best.normalized_name,
        drug_name: best.drug_name,
        ingredient_name: best.ingredient_name,
        atc_code: best.atc_code,
        atc_group_code: best.atc_group_code,
        atc_group_name: best.atc_group_name,
        atc_class_label: best.atc_class_label,
        drug_class: best.drug_class || 'unknown',
      };
    })
    .sort((left, right) => left.normalized_name.localeCompare(right.normalized_name));

  const jsonPayload = {
    generated_at: new Date().toISOString(),
    source_file: sourcePath,
    record_count: records.length,
    records,
  };

  fs.writeFileSync(jsonOutputPath, JSON.stringify(jsonPayload, null, 2));
  fs.writeFileSync(csvOutputPath, rowsToCsv(records));

  console.log(`[ATC] Source -> ${sourcePath}`);
  console.log(`[ATC] Records -> ${records.length}`);
  console.log(`[ATC] JSON -> ${jsonOutputPath}`);
  console.log(`[ATC] CSV  -> ${csvOutputPath}`);
};

main();
