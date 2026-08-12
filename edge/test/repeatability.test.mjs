/**
 * How reproducible is the deployed detector, run twice on the same essay?
 *
 * Everything downstream of the observer is deterministic arithmetic, so this measures one
 * thing: whether Workers AI returns the same log-probabilities for the same prompt. It is an
 * fp8 mixture-of-experts model and nothing promises that it does — `live.test.mjs` found one
 * document in six whose scoring moved.
 *
 * This matters to a user, so it should be a measured number rather than a caveat. A tool
 * that says "likely machine-written" today and "insufficient evidence" tomorrow about the
 * same essay is not reporting a property of the essay.
 *
 *     node edge/test/repeatability.test.mjs [--runs 3] [--docs 3]
 */

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const arg = (n, d) => { const i = args.indexOf(`--${n}`); return i >= 0 ? args[i + 1] : d; };
const BASE = arg('base', 'https://palimpsest.amitynoidalibrary.workers.dev');
const RUNS = Number(arg('runs', 3));
const DOCS = Number(arg('docs', 3));

const all = JSON.parse(readFileSync(resolve(HERE, 'parity-cases.json'), 'utf8'));
const seen = new Set();
const chosen = all.filter((c) => {
  if (c.expected.sentences.length < 8 || seen.has(c.source)) return false;
  seen.add(c.source);
  return true;
}).slice(0, DOCS);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failures = 0;
const measured = [];

for (const c of chosen) {
  const label = `${c.source.split('/').pop()}#${c.id}`;
  const runs = [];
  for (let r = 0; r < RUNS; r += 1) {
    const res = await fetch(`${BASE}/api/analyze`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ text: c.text, include_tokens: false }),
    });
    if (!res.ok) {
      console.log(`FAIL ${label}: HTTP ${res.status}`);
      failures += 1;
      break;
    }
    runs.push(await res.json());
    await sleep(11000);
  }
  if (runs.length < RUNS) continue;

  const first = runs[0];
  let maxProb = 0;
  let maxAny = 0;
  const bands = new Set();
  for (const r of runs) {
    bands.add(r.verdict.band);
    maxAny = Math.max(maxAny, Math.abs(r.verdict.anyMachineProbability - first.verdict.anyMachineProbability));
    if (r.sentences.length !== first.sentences.length) {
      console.log(`FAIL ${label}: sentence count changed between runs`);
      failures += 1;
      break;
    }
    for (let i = 0; i < r.sentences.length; i += 1) {
      maxProb = Math.max(maxProb, Math.abs(r.sentences[i].probability - first.sentences[i].probability));
    }
  }

  const stable = bands.size === 1;
  if (!stable) failures += 1;
  measured.push({
    source: c.source,
    id: c.id,
    runs: RUNS,
    nSentences: first.sentences.length,
    maxSentenceProbDelta: maxProb,
    maxConfidenceDelta: maxAny,
    bandStable: stable,
    bands: [...bands],
  });
  console.log(
    `${label.slice(0, 42).padEnd(44)}${RUNS} runs  ` +
    `max Δ sentence p ${maxProb.toFixed(4)}  max Δ confidence ${maxAny.toFixed(4)}  ` +
    `band ${stable ? `stable (${[...bands][0]})` : `CHANGED: ${[...bands].join(' / ')}`}`,
  );
}

// Written out so the limitation the interface shows is generated from this measurement
// rather than typed from memory -- the same rule api/app.py applies to its error rates.
if (measured.length) {
  const out = resolve(HERE, '..', 'artifacts');
  mkdirSync(out, { recursive: true });
  const summary = {
    measuredAt: new Date().toISOString(),
    base: BASE,
    runsPerDocument: RUNS,
    nDocuments: measured.length,
    nSentences: measured.reduce((a, m) => a + m.nSentences, 0),
    maxSentenceProbDelta: Math.max(...measured.map((m) => m.maxSentenceProbDelta)),
    maxConfidenceDelta: Math.max(...measured.map((m) => m.maxConfidenceDelta)),
    bandStableDocuments: measured.filter((m) => m.bandStable).length,
    documents: measured,
  };
  writeFileSync(resolve(out, 'repeatability.json'), JSON.stringify(summary, null, 1));
  console.log(`\nedge/artifacts/repeatability.json: ${summary.nDocuments} documents x ${RUNS} runs, ` +
    `max Δ sentence p ${summary.maxSentenceProbDelta.toFixed(4)}, ` +
    `max Δ confidence ${summary.maxConfidenceDelta.toFixed(4)}, ` +
    `band stable on ${summary.bandStableDocuments}/${summary.nDocuments}`);
}

console.log(failures
  ? `\nFAIL: ${failures} document(s) did not reproduce`
  : '\nPASS: repeated runs returned the same band for every document');
process.exit(failures ? 1 : 0);
