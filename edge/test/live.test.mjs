/**
 * End-to-end check against the deployed Worker.
 *
 * `parity.test.mjs` holds the observer's output fixed and asks whether the JavaScript
 * pipeline computes the same detector as the Python one. This asks the remaining question:
 * does the *deployed* service, scoring the essay for real, land on the same answers?
 *
 * The two can differ for a reason that is not a bug in the port — Workers AI serves an fp8
 * mixture-of-experts model, and nothing promises that two runs over the same prompt return
 * identical log-probabilities. So this run reports the observer's own reproducibility
 * separately from the pipeline's, because conflating them would let genuine drift hide
 * behind "floating point" and would equally let a porting bug hide behind "the model".
 *
 *     node edge/test/live.test.mjs [--n 6] [--base https://...]
 */

import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const arg = (name, dflt) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : dflt;
};
const BASE = arg('base', 'https://palimpsest.amitynoidalibrary.workers.dev');
const N = Number(arg('n', 6));

const casesPath = resolve(HERE, 'parity-cases.json');
if (!existsSync(casesPath)) {
  console.error('missing edge/test/parity-cases.json -- run edge/scripts/export_parity.py');
  process.exit(1);
}
const all = JSON.parse(readFileSync(casesPath, 'utf8'));

// Spread across sources, and prefer documents with enough sentences to have a real verdict.
const bySource = new Map();
for (const c of all) {
  if (c.expected.sentences.length < 6) continue;
  if (!bySource.has(c.source)) bySource.set(c.source, []);
  bySource.get(c.source).push(c);
}
const chosen = [];
let round = 0;
while (chosen.length < N && round < 50) {
  for (const rows of bySource.values()) {
    if (rows[round] && chosen.length < N) chosen.push(rows[round]);
  }
  round += 1;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fmt = (v) => (typeof v === 'number' ? v.toFixed(4) : String(v));

let failures = 0;
let tokenExact = 0;
const probDeltas = [];
const shareDeltas = [];

console.log(`${BASE}\n${chosen.length} documents\n`);

for (const c of chosen) {
  const res = await fetch(`${BASE}/api/analyze`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ text: c.text, include_tokens: true }),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    console.log(`FAIL ${c.source}#${c.id}: HTTP ${res.status} ${body.error ?? ''} ${body.detail ?? ''}`);
    failures += 1;
    await sleep(11000);
    continue;
  }

  const got = await res.json();
  const exp = c.expected;
  const label = `${c.source.split('/').pop()}#${c.id}`;

  // Did the observer return the same token stream as the cached run? This is the question
  // that separates "the port disagrees" from "the model is not deterministic".
  //
  // The API publishes each logprob rounded to 3 decimals (`analyze.js`), so the cached raw
  // value has to be rounded the same way before comparing. Comparing raw against rounded
  // manufactures a 5e-4 disagreement that says nothing about the model — which is exactly
  // what the first version of this test reported.
  const r3 = (v) => Math.round(v * 1000) / 1000;
  const cachedTokens = c.observation.tokens;
  const liveTokens = got.tokens ?? [];
  let sameTokens = cachedTokens.length === liveTokens.length;
  let maxLogprobDelta = 0;
  let sameRanks = sameTokens;
  if (sameTokens) {
    for (let i = 0; i < cachedTokens.length; i += 1) {
      if (cachedTokens[i].token !== liveTokens[i].text) { sameTokens = false; break; }
      if (cachedTokens[i].rank !== liveTokens[i].rank) sameRanks = false;
      maxLogprobDelta = Math.max(
        maxLogprobDelta, Math.abs(r3(cachedTokens[i].logprob) - liveTokens[i].logprob),
      );
    }
  }
  if (sameTokens && sameRanks && maxLogprobDelta === 0) tokenExact += 1;

  const nSame = got.sentences.length === exp.sentences.length;
  if (!nSame) {
    console.log(`FAIL ${label}: ${got.sentences.length} sentences live vs ${exp.sentences.length} expected`);
    failures += 1;
  }

  let maxProb = 0;
  if (nSame) {
    for (let i = 0; i < exp.sentences.length; i += 1) {
      if (got.sentences[i].start !== exp.sentences[i].start ||
          got.sentences[i].end !== exp.sentences[i].end) {
        console.log(`FAIL ${label}: sentence ${i} offsets differ`);
        failures += 1;
        break;
      }
      maxProb = Math.max(maxProb, Math.abs(got.sentences[i].probability - exp.sentences[i].probability));
    }
  }
  probDeltas.push(maxProb);

  const dShare = Math.abs(got.verdict.anyMachineProbability - exp.verdict.anyMachineProbability);
  shareDeltas.push(dShare);

  const bandExp = exp.genre.inDomain ? '(scored)' : 'out_of_scope';
  const bandOk = exp.genre.inDomain ? got.verdict.band !== 'out_of_scope' : got.verdict.band === 'out_of_scope';
  if (!bandOk) {
    console.log(`FAIL ${label}: gate disagrees — live band ${got.verdict.band}, expected ${bandExp}`);
    failures += 1;
  }

  console.log(
    `${label.slice(0, 42).padEnd(44)}` +
    `tokens ${(sameTokens && sameRanks ? 'same' : 'DIFFER').padEnd(6)}` +
    `Δlogprob ${maxLogprobDelta.toExponential(1).padEnd(9)}` +
    `Δp ${maxProb === 0 ? 'exact' : maxProb.toExponential(1)}   ` +
    `Δany ${dShare === 0 ? 'exact' : dShare.toExponential(1)}   ${got.verdict.band}`,
  );

  // Stay inside the deployed per-IP limit (6/minute) rather than testing the rate limiter
  // by accident.
  await sleep(11000);
}

const summarise = (xs) => {
  if (!xs.length) return 'n/a';
  const sorted = [...xs].sort((a, b) => a - b);
  return `median ${fmt(sorted[Math.floor(sorted.length / 2)])}, max ${fmt(sorted[sorted.length - 1])}`;
};

const exactP = probDeltas.filter((d) => d === 0).length;
const exactS = shareDeltas.filter((d) => d === 0).length;
console.log(
  `\nobserver reproduced its cached scoring exactly (same tokens, same ranks, same ` +
  `logprobs to the 3 published decimals) on ${tokenExact}/${chosen.length} documents`);
console.log(`every sentence probability equals Python's, to the 4 published decimals: ` +
  `${exactP}/${probDeltas.length} documents  [${summarise(probDeltas)}]`);
console.log(`document confidence equals Python's:                                     ` +
  `${exactS}/${shareDeltas.length} documents  [${summarise(shareDeltas)}]`);

if (failures) {
  console.log(`\nFAIL: ${failures} structural mismatch(es)`);
  process.exit(1);
}
console.log('\nPASS: production segments identically, the gate agrees, and no request errored');
