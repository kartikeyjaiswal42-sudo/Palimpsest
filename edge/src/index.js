/**
 * Palimpsest on Cloudflare: the whole application in one Worker.
 *
 *     browser ──► Worker ──► Workers AI (one forward pass, logits read)
 *                    │
 *                    └─► segment · 43 features · sentence model · document model
 *                        · genre gate · calibrated band          (all here, in JS)
 *
 * The detector is not reimplemented so much as re-hosted: every weight, scale, calibration
 * knot and threshold is the artifact `scripts/train.py` wrote, and `edge/test/parity.test.mjs`
 * checks this pipeline against the Python one on 145 real documents and 2,801 sentences.
 * That is the condition under which the accuracy figures in `docs/` describe what is
 * actually deployed.
 *
 * The model is an instrument, never an oracle. See `observer.js`.
 */

import { analyze } from './analyze.js';
import { buildModels, bundle, OBSERVER_MODEL } from './models.js';
import { MAX_OBSERVER_CHARS, observe } from './observer.js';
import { FEATURES, GROUPS } from './registry.js';

export { Budget } from './budget.js';

/** Generous; it exists so a pasted novel cannot occupy a request slot for minutes. */
const MAX_CHARS = 40_000;

/** Rough cost of one analysis in Workers AI neurons, measured over 400 cached calls
 *  (mean 2.45, median 2.53, max 5.85). Reserved up front, then corrected to the real cost. */
const NEURON_ESTIMATE = 3;

let MODELS = null;
let MODELS_ERROR = null;

/**
 * Load the n-gram reference once per isolate.
 *
 * 8.5 MB of typed arrays, fetched from the asset binding and kept in module scope, so the
 * cost lands on the first request an isolate serves and not on the rest. Parsing it per
 * request would be the single slowest thing in the pipeline.
 */
async function models(env) {
  if (MODELS) return MODELS;
  if (MODELS_ERROR) throw MODELS_ERROR;
  try {
    const res = await env.ASSETS.fetch(new URL('/ngram.bin', 'https://palimpsest.internal'));
    if (!res.ok) throw new Error(`ngram.bin: ${res.status}`);
    MODELS = buildModels(await res.arrayBuffer());
    return MODELS;
  } catch (err) {
    // Cached so a broken asset does not re-download 8.5 MB on every request.
    MODELS_ERROR = err;
    throw err;
  }
}

const json = (body, status = 200, headers = {}) =>
  new Response(JSON.stringify(body), {
    status,
    headers: {
      'content-type': 'application/json; charset=utf-8',
      // Application responses are never cached: a stale verdict is worse than a slow one.
      'cache-control': 'no-store',
      ...headers,
    },
  });

function budgetStub(env) {
  return env.BUDGET.get(env.BUDGET.idFromName('global'));
}

const INTERNAL = 'https://palimpsest.internal';

async function health(env) {
  let ready = true;
  let detail = null;
  try {
    await models(env);
  } catch (err) {
    ready = false;
    detail = String(err.message ?? err);
  }

  let usage = null;
  try {
    usage = await (await budgetStub(env).fetch(`${INTERNAL}/state`)).json();
  } catch {
    usage = null;
  }

  return json({
    status: ready ? 'ok' : 'degraded',
    detail,
    observer: OBSERVER_MODEL,
    device: 'workers-ai',
    artifactSet: bundle.suffix,
    corpusReferenceLoaded: ready,
    flagThreshold: bundle.detector.flag_threshold,
    documentThreshold: bundle.bands.tMachine,
    trainedOn: bundle.detector.metadata,
    // The central design claim of the project, stated in the payload.
    usesGenerativeModel: false,
    // And the one that stops being true the moment the observer is remote. It is remote
    // here for everyone, so this is `true` and the interface says so in words as well.
    textLeavesMachine: true,
    maxObserverChars: MAX_OBSERVER_CHARS,
    note:
      'The language model is read for token probabilities only -- a single forward pass ' +
      'over the essay, whose logits are then arithmetic. It is never prompted and never ' +
      'asked for a verdict. The observer runs on Cloudflare Workers AI, so THE ESSAY TEXT ' +
      'IS SENT to that service to be scored. It is not stored or logged by this ' +
      'application, but it does leave the browser.',
    usage,
  });
}

async function handleAnalyze(request, env, ctx) {
  const startedAt = Date.now();
  let body;
  try {
    body = await request.json();
  } catch {
    return json({ error: 'bad_json', detail: 'Request body must be JSON.' }, 400);
  }

  const text = typeof body?.text === 'string' ? body.text : '';
  if (!text.trim()) return json({ error: 'empty', detail: 'No essay was provided.' }, 400);
  if (text.length > MAX_CHARS) {
    return json({
      error: 'too_long',
      detail: `That is ${text.length} characters; the limit is ${MAX_CHARS}.`,
    }, 413);
  }

  const ip = request.headers.get('cf-connecting-ip') ?? '0.0.0.0';
  const stub = budgetStub(env);

  let reservation;
  try {
    reservation = await (await stub.fetch(`${INTERNAL}/reserve`, {
      method: 'POST',
      body: JSON.stringify({ ip, estimate: NEURON_ESTIMATE }),
    })).json();
  } catch (err) {
    // Fail CLOSED on the metered resource: an unreachable counter must not become an
    // unmetered spend of the account's daily allowance.
    return json({
      error: 'unavailable',
      detail: 'The usage counter is unavailable, so this request was refused rather than ' +
        'run unmetered. Please try again shortly.',
    }, 503);
  }

  if (!reservation.ok) {
    return json(
      { error: reservation.reason, detail: reservation.detail },
      reservation.reason === 'budget' ? 503 : 429,
      { 'retry-after': String(reservation.retryAfter ?? 60) },
    );
  }

  let loaded;
  try {
    loaded = await models(env);
  } catch (err) {
    // Give the reservation back. Nothing reached Workers AI on this path -- the corpus
    // reference failed to load before the observer was called -- so the three neurons
    // charged up front were never spent, and leaving them charged would let a broken asset
    // binding quietly eat the day's allowance one refusal at a time. An observer failure
    // below is deliberately NOT refunded: that call may or may not have been billed and a
    // metered resource is the one place to guess high.
    ctx.waitUntil(
      stub.fetch(`${INTERNAL}/settle`, {
        method: 'POST',
        body: JSON.stringify({ estimate: NEURON_ESTIMATE, neurons: 0 }),
      }).catch(() => {}),
    );
    return json({ error: 'artifacts', detail: String(err.message ?? err) }, 503);
  }

  let observation;
  try {
    observation = await observe(env.AI, text);
  } catch (err) {
    return json({
      error: 'observer',
      detail: `The observer could not score this text: ${String(err.message ?? err).slice(0, 300)}`,
    }, err.status ?? 502);
  }

  // Correct the reserved estimate to what the call actually cost. Not awaited: the answer
  // is already computed and the user should not wait on bookkeeping.
  ctx.waitUntil(
    stub.fetch(`${INTERNAL}/settle`, {
      method: 'POST',
      body: JSON.stringify({ estimate: NEURON_ESTIMATE, neurons: observation.neurons }),
    }).catch(() => {}),
  );

  const includeTokens = body?.include_tokens !== false;
  const result = analyze(text, observation, loaded, { includeTokens, startedAt });
  return json(result);
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    if (url.pathname === '/api/health') return health(env);

    if (url.pathname === '/api/features') {
      return json({
        groups: GROUPS,
        features: FEATURES.map((f) => ({
          name: f.name,
          group: f.group,
          label: f.label,
          description: f.description,
          unit: f.unit,
          expectedDirection: f.expectedDirection,
          // Which of the 43 the served detector actually reads. The remote observer cannot
          // supply entropy or curvature -- the API returns only the realised token, never
          // the full predictive distribution -- so three are measured as "not measured".
          used: bundle.detector.feature_names.includes(f.name),
        })),
        usedByServedDetector: bundle.detector.feature_names,
      });
    }

    if (url.pathname === '/api/analyze') {
      if (request.method !== 'POST') return json({ error: 'method', detail: 'POST only.' }, 405);
      return handleAnalyze(request, env, ctx);
    }

    // The detector's worst held-out mistakes, ranked by confident wrongness.
    //
    // Served from the REDACTED asset built by edge/scripts/build_failures.py, never from
    // artifacts/confident_failures.json. That artifact embeds the complete text of the
    // documents this panel is about, and on this corpus those are real second-language
    // students' essays. The redaction is a separate FILE rather than a branch here,
    // because a branch is one edit from being inverted with nothing to notice, whereas the
    // assets directory physically does not contain the full essays.
    //
    // Read-only on purpose. The local build accepts POST /api/failures/explanation so the
    // author can write up why the maths failed; a public endpoint that writes into a
    // served artifact would let any visitor edit it, so the hosted build does not offer it
    // and says so in the payload rather than failing silently.
    if (url.pathname === '/api/failures') {
      if (request.method !== 'GET') return json({ error: 'method', detail: 'GET only.' }, 405);
      const res = await env.ASSETS.fetch(
        new URL('/failures.json', 'https://palimpsest.internal'),
      );
      if (!res.ok) {
        return json(
          { error: 'not_found', detail: 'No failure analysis is bundled with this build.' },
          404,
        );
      }
      const payload = await res.json();
      payload.explanationsEditable = false;
      return json(payload);
    }

    if (url.pathname === '/api/failures/explanation') {
      return json(
        {
          error: 'read_only',
          detail:
            'The hosted build serves the failure analysis read-only. Explanations are ' +
            'written in the local build, which reads the unredacted artifact.',
        },
        405,
      );
    }

    if (url.pathname.startsWith('/api/')) {
      return json({ error: 'not_found', detail: `No such endpoint: ${url.pathname}` }, 404);
    }

    // The interface. Application code is served no-store for the reason api/app.py gives:
    // a cached stylesheet once hid a fix that was present on disk and correct on the wire.
    const asset = await env.ASSETS.fetch(request);
    const headers = new Headers(asset.headers);
    headers.set('cache-control', 'no-store, must-revalidate');
    return new Response(asset.body, { status: asset.status, headers });
  },
};
