/**
 * The observer: one forward pass over the essay, whose logits are read.
 *
 * Workers AI accepts
 *
 *     { prompt, raw: true, max_tokens: 1, prompt_logprobs: 0 }
 *
 * which SCORES THE GIVEN TEXT instead of continuing it, returning each position's
 * log-probability and its true rank in the model's ranked next-token distribution.
 *
 * `raw: true` is load-bearing. Without it the essay is wrapped in a chat template and the
 * numbers describe a conversation rather than the prose. `max_tokens: 1` is not us asking
 * for an answer -- the single generated token is discarded; the API simply requires a
 * generation budget, and we want the *prompt* scored.
 *
 * There is no prompt text, no instruction, no question and no place for an opinion to enter.
 * The model is never asked whether the essay is machine-written; that judgement is made
 * downstream, by arithmetic, in code that can be audited. `/api/health` states this and
 * `test/no-generative-calls.test.mjs` checks the scoring path cannot reach a chat API.
 *
 * This is a direct port of `observer-worker/src/index.js`, folded into the same Worker so a
 * request costs one Workers AI call rather than an extra network hop to a sibling Worker.
 */

export const OBSERVER_MODEL = '@cf/qwen/qwen3-30b-a3b-fp8';

/** Matches the Python client and the original observer Worker. Longer texts are clipped. */
export const MAX_OBSERVER_CHARS = 6000;

/**
 * Walk the `prompt_logprobs` array and attach character offsets.
 *
 * One entry per token position. Position 0 has nothing before it, so nothing was predicted
 * there; it is skipped rather than imputed, which is what the local scorer does too.
 */
export function align(promptLogprobs, text) {
  const out = [];
  let cursor = 0;

  for (const entry of promptLogprobs) {
    if (!entry) continue;
    const ids = Object.keys(entry);
    if (!ids.length) continue;

    // With prompt_logprobs:0 there is exactly one entry and it is the realised token. With
    // top-k > 0 the realised token is the one the text actually shows at the cursor.
    let chosen = entry[ids[0]];
    if (ids.length > 1) {
      const at = text.slice(cursor);
      chosen =
        ids
          .map((id) => entry[id])
          .find((e) => e?.decoded_token && at.startsWith(e.decoded_token)) ?? chosen;
    }
    if (!chosen) continue;

    const tok = chosen.decoded_token ?? '';
    const idx = tok ? text.indexOf(tok, cursor) : -1;
    const start = idx >= 0 ? idx : cursor;
    const end = start + tok.length;

    out.push({
      token: tok,
      logprob: chosen.logprob,
      rank: typeof chosen.rank === 'number' ? chosen.rank : 1,
      start,
      end,
    });
    cursor = end;
  }

  return out;
}

export async function observe(ai, text, model = OBSERVER_MODEL) {
  const clipped = text.length > MAX_OBSERVER_CHARS ? text.slice(0, MAX_OBSERVER_CHARS) : text;

  const res = await ai.run(model, {
    prompt: clipped,
    raw: true, // score the prose, not a chat transcript
    max_tokens: 1, // the continuation is discarded; we want the PROMPT scored
    temperature: 0,
    prompt_logprobs: 0,
  });

  const pl = res?.prompt_logprobs ?? res?.choices?.[0]?.prompt_logprobs;
  if (!Array.isArray(pl)) {
    const err = new Error('the observer returned no token probabilities');
    err.status = 502;
    throw err;
  }

  return {
    model,
    clipped: clipped.length < text.length,
    neurons: typeof res?.usage?.neurons === 'number' ? res.usage.neurons : null,
    tokens: align(pl, clipped),
  };
}
