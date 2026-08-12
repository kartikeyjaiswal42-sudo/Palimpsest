/**
 * Segmentation and the model-free features, on inputs the essay corpus never produces.
 *
 * The document parity run compares real essays and passes on all of them; that is necessary
 * and not sufficient. Real admissions essays contain no accented contraction, no Arabic-Indic
 * digit, no `Ph.D.` mid-sentence — so several of the fiddliest lines in the port (the
 * hand-written Unicode word boundaries, `\p{Nd}` standing in for Python's `\d`, the
 * abbreviation walk-back, `str.strip()`'s character set) are untouched by it. A mutation test
 * proved the gap: reverting the contraction pattern to JavaScript's ASCII `\b\w+` left the
 * corpus run entirely green.
 *
 *     node edge/test/units.test.mjs
 */

import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

import { splitParagraphs, splitSentences, tokenizeWords } from '../src/segment.js';
import { extractContextFeatures, extractSurfaceFeatures } from '../src/features.js';
import { extractCorpusFeatures, NgramReference } from '../src/ngram.js';
import { documentGenreFeatures } from '../src/detect.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '..', '..');
const TOL = 1e-9;

let failures = 0;
let checks = 0;

const show = (v) => (typeof v === 'string' ? JSON.stringify(v) : String(v));

function same(got, want, where) {
  checks += 1;
  if (got !== want) {
    failures += 1;
    if (failures <= 40) console.log(`  MISMATCH ${where}: got ${show(got)}, want ${show(want)}`);
  }
}

function near(got, want, where) {
  checks += 1;
  const gNull = got === null || got === undefined || Number.isNaN(got);
  const wNull = want === null || want === undefined || Number.isNaN(want);
  if (gNull && wNull) return;
  if (gNull !== wNull || Math.abs(got - want) > TOL) {
    failures += 1;
    if (failures <= 40) console.log(`  MISMATCH ${where}: got ${show(got)}, want ${show(want)}`);
  }
}

const casesPath = resolve(HERE, 'unit-cases.json');
if (!existsSync(casesPath)) {
  console.error('missing edge/test/unit-cases.json -- run edge/scripts/export_unit_cases.py');
  process.exit(1);
}
const ngramPath = resolve(ROOT, 'edge', 'assets', 'ngram.bin');
const raw = readFileSync(ngramPath);
const reference = NgramReference.parse(
  raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength),
);

const cases = JSON.parse(readFileSync(casesPath, 'utf8'));

for (const c of cases) {
  const where = c.name;

  const paras = splitParagraphs(c.text);
  same(paras.length, c.paragraphs.length, `${where} nParagraphs`);
  for (let i = 0; i < Math.min(paras.length, c.paragraphs.length); i += 1) {
    same(paras[i].start, c.paragraphs[i].start, `${where} para${i}.start`);
    same(paras[i].end, c.paragraphs[i].end, `${where} para${i}.end`);
    same(paras[i].text, c.paragraphs[i].text, `${where} para${i}.text`);
  }

  const spans = splitSentences(c.text);
  same(spans.length, c.sentences.length, `${where} nSentences`);
  for (let i = 0; i < Math.min(spans.length, c.sentences.length); i += 1) {
    same(spans[i].start, c.sentences[i].start, `${where} s${i}.start`);
    same(spans[i].end, c.sentences[i].end, `${where} s${i}.end`);
    same(spans[i].text, c.sentences[i].text, `${where} s${i}.text`);
    // The invariant the segmenter exists to hold.
    same(c.text.slice(spans[i].start, spans[i].end), spans[i].text, `${where} s${i}.invariant`);
  }

  const words = tokenizeWords(c.text);
  same(words.join('|'), c.words.join('|'), `${where} words`);

  const base = spans.map((s) => ({
    ...extractSurfaceFeatures(s.text),
    ...extractCorpusFeatures(s.text, reference, -3.5),
  }));
  const context = extractContextFeatures(base, spans.length);
  const combined = base.map((b, i) => ({ ...b, ...context[i] }));

  for (let i = 0; i < Math.min(combined.length, c.features.length); i += 1) {
    for (const [name, want] of Object.entries(c.features[i])) {
      near(combined[i][name], want, `${where} s${i}.${name}`);
    }
  }

  const gfeat = documentGenreFeatures(combined);
  for (const [name, want] of Object.entries(c.genre)) {
    near(gfeat[name], want, `${where} genre.${name}`);
  }
}

console.log(`${cases.length} constructed cases, ${checks} comparisons`);
if (failures) {
  console.log(`FAIL: ${failures} mismatch(es)`);
  process.exit(1);
}
console.log('PASS: segmentation and model-free features match Python on every constructed case');
