/**
 * A remote observer for Palimpsest.
 *
 * Palimpsest's features are all arithmetic on one question: "how predictable was this token
 * here?" The answer is only as good as the model asked. Locally we ask GPT-2 (124 M, 2019),
 * and docs/08-cross-vendor.md records what that costs -- on Claude Opus essays the detector
 * flags 2.6% of sentences, because frontier prose is not distinctively surprising to a 2019
 * model. It is merely fluent, and so is good human prose.
 *
 * A 30 B observer cannot run on the 8 GB laptop this project is developed on. It runs here
 * instead. This Worker exists for exactly one reason: Workers AI accepts
 *
 *     { prompt, raw: true, max_tokens: 1, prompt_logprobs: 0 }
 *
 * which SCORES THE GIVEN TEXT instead of generating a continuation, and returns each
 * position's log-probability and true rank. `raw: true` is load-bearing -- without it the
 * text is wrapped in a chat template and the numbers describe a conversation, not the prose.
 *
 * This endpoint returns raw per-token numbers and nothing else. It forms no opinion, applies
 * no threshold and reports no verdict; every judgement stays in the Python pipeline where it
 * can be audited. That separation is deliberate.
 *
 * Access is gated by a shared secret. The free allowance is 10,000 neurons/day for the whole
 * account and an open endpoint would let a stranger drain it.
 */

const MAX_CHARS = 6000; // ~1,500 tokens; longer essays are scored in chunks by the client.

/** Models that may be requested. An allow-list, so a typo cannot bill an expensive model. */
const ALLOWED = new Set([
  '@cf/qwen/qwen3-30b-a3b-fp8',
  '@cf/mistralai/mistral-small-3.1-24b-instruct',
  '@cf/meta/llama-3.3-70b-instruct-fp8-fast',
  '@cf/meta/llama-3.1-8b-instruct-fp8',
]);

const DEFAULT_MODEL = '@cf/qwen/qwen3-30b-a3b-fp8';

/**
 * Walk the prompt_logprobs array and attach character offsets.
 *
 * Workers AI returns one entry per token position, keyed by token id, with the decoded
 * token text inside. Position 0 has no left context and carries no logprob -- it is skipped
 * rather than imputed, matching the local scorer.
 */
/**
 * Choose, for every scored position, WHICH candidate was the token the text actually used.
 *
 * With prompt_logprobs:0 this is free -- one candidate per position. With k>0 the entry
 * holds the top k plus the realised token, and per-position text matching cannot separate
 * them: at one position of a real essay the candidates included both " blue" (realised,
 * rank 3) and " blueprint" (rank 2), both of which continue "...the blueprints for...".
 * Taking the longer one consumed five characters too many and every subsequent offset was
 * wrong -- silently, because each later token still matched *somewhere*.
 *
 * The constraint that does separate them is global: the chosen tokens must tile the text
 * EXACTLY, from the end of the unscored first token to the final character, in exactly as
 * many steps as there are scored positions. That is a path problem, so it is solved as one
 * -- a forward sweep over reachable cursors with parent pointers, then a walk back from the
 * end of the text. Ambiguity at a position is fine as long as only one branch survives to
 * the end, which is what makes this exact rather than heuristic.
 */
function resolveTiling(entries, text) {
  // entries[i] = candidate list for the i-th SCORED position, in order.
  // layer[cursor] = index of the candidate chosen to arrive at `cursor`, plus its parent.
  let layer = new Map();
  const parents = [];

  // The first scored position predicts the SECOND token, so the first token occupies
  // text[0:start] and `start` is unknown. Seed every plausible landing point.
  for (let ci = 0; ci < entries[0].length; ci += 1) {
    const t = entries[0][ci].decoded_token;
    if (typeof t !== 'string' || !t.length) continue;
    const at = text.indexOf(t);
    if (at < 0 || at > 64) continue; // one token cannot be longer than this
    const end = at + t.length;
    if (!layer.has(end)) layer.set(end, { ci, prev: -1, start: at });
  }
  parents.push(layer);

  for (let i = 1; i < entries.length; i += 1) {
    const next = new Map();
    for (const cursor of layer.keys()) {
      for (let ci = 0; ci < entries[i].length; ci += 1) {
        const t = entries[i][ci].decoded_token;
        if (typeof t !== 'string' || !t.length) continue;
        if (!text.startsWith(t, cursor)) continue;
        const end = cursor + t.length;
        if (!next.has(end)) next.set(end, { ci, prev: cursor, start: cursor });
      }
    }
    if (next.size === 0) return null; // no tiling survives; caller falls back
    layer = next;
    parents.push(layer);
  }

  // Walk back from a complete tiling of the text.
  let cursor = layer.has(text.length) ? text.length : null;
  if (cursor === null) return null;
  const choice = new Map();
  for (let i = entries.length - 1; i >= 0; i -= 1) {
    const node = parents[i].get(cursor);
    if (!node) return null;
    choice.set(i, { cand: entries[i][node.ci], start: node.start });
    cursor = node.prev;
    if (i > 0 && cursor < 0) return null;
  }
  return choice;
}

function align(promptLogprobs, text) {
  const out = [];
  let cursor = 0;
  let misaligned = 0;

  // Pre-pass: collect the candidate lists in position order, then solve the tiling once.
  const entryList = [];
  for (const entry of promptLogprobs) {
    if (!entry) continue;
    const c = Object.values(entry).filter((e) => e && typeof e.logprob === 'number');
    if (c.length) entryList.push(c);
  }
  const tiling = entryList.length ? resolveTiling(entryList, text) : null;
  const choice = new Map();
  if (tiling) for (const [i, v] of tiling) choice.set(i, v.cand);

  let pos = -1;
  for (const entry of promptLogprobs) {
    if (!entry) continue; // position 0: nothing precedes it, so nothing was predicted
    const cands = Object.values(entry).filter((e) => e && typeof e.logprob === 'number');
    if (!cands.length) continue;
    pos += 1;

    // Identify the REALISED token. With prompt_logprobs:0 there is exactly one candidate
    // and this is trivial. With k>0 the entry holds the top k PLUS the realised token, and
    // the only thing that distinguishes it is that the text actually continues with it.
    //
    // Longest match, not first match: candidate strings are prefixes of one another all the
    // time (" ", " a", " an"), so `find(startsWith)` picks whichever the object happened to
    // enumerate first and then every later offset is wrong. That is precisely what the
    // previous k>0 branch did -- it returned plausible-looking tokens at scrambled offsets,
    // which is worse than an error because nothing downstream can see it.
    let chosen = choice.get(pos) ?? null;
    if (!chosen) {
      // The exact tiling did not resolve (or does not exist for this text). Fall back to the
      // per-position heuristic and COUNT it, so a caller can see that these offsets are not
      // trustworthy rather than discovering it in a feature three layers downstream.
      misaligned += 1;
      chosen = null;
      for (const e of cands) {
        const t = e.decoded_token;
        if (typeof t !== 'string' || !t.length) continue;
        const idx = text.indexOf(t, cursor);
        if (idx < 0) continue;
        if (chosen === null || idx < text.indexOf(chosen.decoded_token, cursor)) chosen = e;
      }
      if (!chosen) chosen = cands.reduce(
        (a, b) => ((b.rank ?? 1e9) < (a.rank ?? 1e9) ? b : a), cands[0]);
    }

    // The observer's own distribution at this position, over the candidates it returned.
    //
    //   m       total probability the top-k candidates hold  (EXACT, no approximation)
    //   q_i     those candidates renormalised to sum to 1
    //   mu      E_q[log p]      Fast-DetectGPT's conditional mean
    //   sigma2  Var_q[log p]    its conditional variance
    //
    // Truncation is real and is reported, not hidden: `mass` travels with every token so a
    // consumer can see how much of the distribution these numbers actually cover. Entropy
    // of the renormalised head simplifies to log(m) - mu, which is why it is not summed
    // separately.
    let m = 0;
    for (const e of cands) m += Math.exp(e.logprob);
    let mu = 0;
    for (const e of cands) mu += (Math.exp(e.logprob) / m) * e.logprob;
    let sigma2 = 0;
    for (const e of cands) sigma2 += (Math.exp(e.logprob) / m) * (e.logprob - mu) ** 2;

    const tok = chosen.decoded_token ?? '';
    const idx = tok ? text.indexOf(tok, cursor) : -1;
    const start = idx >= 0 ? idx : cursor;
    const end = start + tok.length;

    const row = {
      token: tok,
      logprob: chosen.logprob,
      rank: typeof chosen.rank === 'number' ? chosen.rank : 1,
      start,
      end,
    };
    // Only meaningful with k>0; with a single candidate m is just that token's probability
    // and mu is its own logprob, which would make curvature identically zero.
    if (cands.length > 1) {
      row.mass = Math.min(m, 1);
      row.mu = mu;
      row.sigma2 = sigma2;
      row.entropy = Math.log(m) - mu;
    }
    out.push(row);
    cursor = end;
  }

  return { tokens: out, misaligned };
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === '/health') {
      return Response.json({ ok: true, models: [...ALLOWED], default: DEFAULT_MODEL });
    }

    if (url.pathname !== '/probe' || request.method !== 'POST') {
      return new Response('not found', { status: 404 });
    }

    // Constant-ish comparison is overkill for a rate-limit secret, but an open endpoint
    // would let anyone spend the account's 10,000 free neurons/day.
    const auth = request.headers.get('authorization') ?? '';
    if (!env.PROBE_SECRET || auth !== `Bearer ${env.PROBE_SECRET}`) {
      return new Response('unauthorized', { status: 401 });
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return Response.json({ error: 'bad json' }, { status: 400 });
    }

    const text = typeof body?.text === 'string' ? body.text : '';
    if (!text.trim()) return Response.json({ error: 'empty text' }, { status: 400 });

    const model = ALLOWED.has(body?.model) ? body.model : DEFAULT_MODEL;
    const topk = Number.isInteger(body?.top_k) ? Math.max(0, Math.min(20, body.top_k)) : 0;
    const clipped = text.length > MAX_CHARS ? text.slice(0, MAX_CHARS) : text;

    try {
      const res = await env.AI.run(model, {
        prompt: clipped,
        raw: true, // score the prose, not a chat transcript
        max_tokens: 1, // we want the PROMPT scored; the continuation is discarded
        temperature: 0,
        prompt_logprobs: topk,
      });

      const pl = res?.prompt_logprobs ?? res?.choices?.[0]?.prompt_logprobs;
      if (!Array.isArray(pl)) {
        return Response.json(
          { error: 'model returned no prompt_logprobs', model },
          { status: 502 },
        );
      }

      // Diagnostic only: hand back the untouched prompt_logprobs array so the shape of a
      // top-k reply can be read off the wire rather than guessed. `align()` was written for
      // prompt_logprobs:0 -- exactly one candidate per position -- and its k>0 branch has
      // never been checked against a real response. Whether the alternatives come back at
      // all decides whether entropy and Fast-DetectGPT curvature are recoverable from this
      // API, which is the difference between 40 features and 43.
      if (body?.debug_raw === true) {
        return Response.json({
          model,
          topk,
          neurons: res?.usage?.neurons ?? null,
          nPositions: pl.length,
          raw: pl.slice(0, 6),
        });
      }

      const aligned = align(pl, clipped);
      return Response.json({
        model,
        clipped: clipped.length < text.length,
        neurons: res?.usage?.neurons ?? null,
        // Alignment is checkable, so it is checked and reported rather than assumed: a
        // token stream that does not reconstruct the text has silently invalid offsets,
        // and every sentence-level feature is computed by slicing on those offsets.
        misaligned: aligned.misaligned,
        tokens: aligned.tokens,
      });
    } catch (err) {
      return Response.json(
        { error: String(err?.message ?? err).slice(0, 300), model },
        { status: 502 },
      );
    }
  },
};
