/**
 * Does the JavaScript pipeline compute the same detector the Python one does?
 *
 * Every accuracy figure this project reports was measured on the Python implementation. A
 * port that merely looks right would give the deployed application an evidence base it does
 * not have, so this compares the two, feature by feature, on real corpus documents, with the
 * observer's token stream held byte-identical between them.
 *
 *     node edge/test/parity.test.mjs [edge/test/parity-cases.json ...]
 *
 * Tolerances are tight on purpose. The two implementations use the same double-precision
 * arithmetic in the same order, so the only legitimate difference is the last-ulp behaviour
 * of `log`/`sqrt` and NumPy's pairwise summation. Anything larger is a porting bug, not
 * rounding, and the thresholds below are set where that distinction actually falls.
 */

import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

import { analyze, featuresFor } from '../src/analyze.js';
import { buildModels } from '../src/models.js';
import { documentGenreFeatures } from '../src/detect.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '..', '..');

const TOL = {
  feature: 1e-9,      // raw feature values
  probability: 1e-9,  // calibrated sentence probability
  logit: 1e-8,
  share: 1e-9,
  gate: 1e-9,
};

let failures = 0;
let checks = 0;
const worst = new Map();

function note(kind, delta, where) {
  const cur = worst.get(kind);
  if (!cur || delta > cur.delta) worst.set(kind, { delta, where });
}

function near(kind, got, want, tol, where) {
  checks += 1;
  const bothNull = (got === null || got === undefined || Number.isNaN(got)) &&
    (want === null || want === undefined || Number.isNaN(want));
  if (bothNull) return true;
  if (typeof got !== 'number' || typeof want !== 'number' ||
      !Number.isFinite(got) || !Number.isFinite(want)) {
    failures += 1;
    if (failures <= 25) console.log(`  MISMATCH ${where}: got ${got}, want ${want}`);
    return false;
  }
  const delta = Math.abs(got - want);
  note(kind, delta, where);
  if (delta > tol) {
    failures += 1;
    if (failures <= 25) {
      console.log(`  MISMATCH ${where}: got ${got}, want ${want} (delta ${delta.toExponential(3)})`);
    }
    return false;
  }
  return true;
}

function same(got, want, where) {
  checks += 1;
  if (got !== want) {
    failures += 1;
    if (failures <= 25) console.log(`  MISMATCH ${where}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
    return false;
  }
  return true;
}

// -- run ------------------------------------------------------------------------------

const files = process.argv.slice(2);
if (!files.length) files.push(resolve(HERE, 'parity-cases.json'));

const ngramPath = resolve(ROOT, 'edge', 'assets', 'ngram.bin');
if (!existsSync(ngramPath)) {
  console.error('missing edge/assets/ngram.bin -- run edge/scripts/build_ngram_bin.py');
  process.exit(1);
}
const raw = readFileSync(ngramPath);
const models = buildModels(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
console.log(`n-gram reference: ${models.reference.index.size} tokens, ` +
  `${models.reference.triHi.length} trigrams`);

let documents = 0;
let sentences = 0;

for (const file of files) {
  if (!existsSync(file)) {
    console.error(`missing ${file} -- run edge/scripts/export_parity.py`);
    process.exit(1);
  }
  const cases = JSON.parse(readFileSync(file, 'utf8'));
  console.log(`\n${file}: ${cases.length} documents`);

  for (const c of cases) {
    documents += 1;
    const where = `${c.source}#${c.id}`;
    const got = analyze(c.text, c.observation, models, { includeTokens: false });
    const exp = c.expected;

    // 1. Segmentation. Everything downstream addresses text by these offsets, so a single
    //    disagreement here would make the rest of the comparison meaningless.
    same(got.sentences.length, exp.sentences.length, `${where} nSentences`);
    const n = Math.min(got.sentences.length, exp.sentences.length);
    for (let i = 0; i < n; i += 1) {
      same(got.sentences[i].start, exp.sentences[i].start, `${where} s${i}.start`);
      same(got.sentences[i].end, exp.sentences[i].end, `${where} s${i}.end`);
      same(got.sentences[i].text, exp.sentences[i].text, `${where} s${i}.text`);
    }

    // 2. Every feature of every sentence, so a divergence names the feature responsible
    //    instead of showing up as an unexplained probability difference.
    const { features } = featuresFor(c.text, c.observation.tokens, models.reference);
    for (let i = 0; i < Math.min(features.length, exp.features.length); i += 1) {
      for (const [name, want] of Object.entries(exp.features[i])) {
        near('feature', features[i][name], want, TOL.feature, `${where} s${i}.${name}`);
      }
    }

    // 3. Sentence verdicts and the arithmetic the interface shows.
    for (let i = 0; i < n; i += 1) {
      const g = got.sentences[i];
      const e = exp.sentences[i];
      near('probability', g.probability, e.probability, TOL.probability, `${where} s${i}.p`);
      near('probability', g.smoothed, e.smoothed, TOL.probability, `${where} s${i}.smoothed`);
      near('logit', g.logit, e.logit, TOL.logit, `${where} s${i}.logit`);
      same(g.reliable, e.reliable, `${where} s${i}.reliable`);
      same(g.nWords, e.nWords, `${where} s${i}.nWords`);
      near('logit', g.evidenceRemainder, e.evidenceRemainder, TOL.logit, `${where} s${i}.remainder`);
      same(g.evidence.map((x) => x.name).join(','), e.evidence.map((x) => x.name).join(','),
        `${where} s${i}.evidenceOrder`);
      for (let k = 0; k < Math.min(g.evidence.length, e.evidence.length); k += 1) {
        near('logit', g.evidence[k].contribution, e.evidence[k].contribution, TOL.logit,
          `${where} s${i}.ev${k}.contribution`);
        same(g.evidence[k].measured, e.evidence[k].measured, `${where} s${i}.ev${k}.measured`);
      }
      sentences += 1;
    }

    // 4. Passages -- the "which parts" answer.
    same(got.passages.length, exp.passages.length, `${where} nPassages`);
    for (let i = 0; i < Math.min(got.passages.length, exp.passages.length); i += 1) {
      same(got.passages[i].start, exp.passages[i].start, `${where} p${i}.start`);
      same(got.passages[i].end, exp.passages[i].end, `${where} p${i}.end`);
      near('probability', got.passages[i].meanProbability, exp.passages[i].meanProbability,
        TOL.probability, `${where} p${i}.mean`);
    }

    // 5. Document verdict, including the seeded bootstrap interval -- which is why this
    //    port carries a bit-exact PCG64 rather than any convenient RNG.
    for (const k of ['machineShare', 'machineShareLow', 'machineShareHigh', 'anyMachineProbability']) {
      near('share', got.verdict[k], exp.verdict[k], TOL.share, `${where} verdict.${k}`);
    }
    for (const k of ['nSentences', 'nWords', 'nReliableSentences']) {
      same(got.verdict[k], exp.verdict[k], `${where} verdict.${k}`);
    }

    // 6. The genre gate: its document features, its probability, and the refusal itself.
    const gfeat = documentGenreFeatures(features);
    for (const [name, want] of Object.entries(exp.genre.features)) {
      near('gate', gfeat[name], want, TOL.feature, `${where} gate.${name}`);
    }
    near('gate', got.verdict.inDomainProbability, exp.genre.inDomainProbability, 1e-4,
      `${where} gate.p`);
    same(got.verdict.band === 'out_of_scope', !exp.genre.inDomain, `${where} gate.refused`);
  }
}

console.log(`\n${documents} documents, ${sentences} sentences, ${checks} comparisons`);
console.log('largest disagreement by kind:');
for (const [kind, { delta, where }] of [...worst].sort((a, b) => b[1].delta - a[1].delta)) {
  console.log(`  ${kind.padEnd(12)} ${delta.toExponential(3)}   ${where}`);
}

if (failures) {
  console.log(`\nFAIL: ${failures} mismatch(es)`);
  process.exit(1);
}
console.log('\nPASS: the JavaScript pipeline reproduces the Python one on every comparison');
