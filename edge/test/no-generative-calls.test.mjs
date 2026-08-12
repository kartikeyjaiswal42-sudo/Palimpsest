/**
 * The line the brief draws: the model must not make the judgement call.
 *
 * `tests/test_no_generative_calls.py` enforces this on the Python side by forbidding imports
 * of `openai`, `anthropic`, `litellm` and friends. The Worker has no import graph to
 * inspect — it reaches a model through a runtime binding — so the equivalent check is
 * structural: there is exactly one call into `env.AI`, it lives in `observer.js`, it passes
 * `raw: true`, and the only text it sends is the essay itself.
 *
 * The failure this is guarding against is not subtle in review and is invisible in output:
 * a detector that quietly asks a chat model "is this AI-written?" and relays the answer
 * returns the same shape of JSON as one that does arithmetic on logits.
 *
 *     node edge/test/no-generative-calls.test.mjs
 */

import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';

const SRC = resolve(dirname(fileURLToPath(import.meta.url)), '..', 'src');

let failures = 0;
const fail = (msg) => {
  failures += 1;
  console.log(`  FAIL ${msg}`);
};
const pass = (msg) => console.log(`  ok   ${msg}`);

const files = readdirSync(SRC).filter((f) => f.endsWith('.js'));
const sources = new Map(files.map((f) => [f, readFileSync(join(SRC, f), 'utf8')]));

// 1. No chat/completion SDK or endpoint anywhere in the serving path.
const FORBIDDEN = [
  /\bopenai\b/i, /\banthropic\b/i, /\bcohere\b/i, /\blitellm\b/i, /\bollama\b/i,
  /generativelanguage/i, /\bchat\/completions\b/i, /api\.groq\.com/i,
];
for (const [name, src] of sources) {
  // The comments in observer.js discuss chat templates by name; strip comments first so the
  // check reads code rather than prose about code.
  const code = src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '');
  for (const re of FORBIDDEN) {
    if (re.test(code)) fail(`${name} references ${re} in code`);
  }
}
pass('no chat-completion SDK or endpoint appears in any source file');

// 2. Exactly one call site into the AI binding, and it is the observer.
const callSites = [];
for (const [name, src] of sources) {
  const code = src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '');
  for (const m of code.matchAll(/\b(\w+)\.run\s*\(/g)) {
    if (m[1] === 'ai' || m[1] === 'AI' || /env\.AI\s*$/.test(code.slice(0, m.index))) {
      callSites.push(`${name}:${m[1]}.run`);
    }
  }
  if (/env\.AI\b/.test(code) && name !== 'index.js') {
    fail(`${name} reaches env.AI directly; the binding should only be passed to observe()`);
  }
}
if (callSites.length !== 1) fail(`expected exactly one AI call site, found ${callSites.length}: ${callSites}`);
else if (!callSites[0].startsWith('observer.js')) fail(`the AI call is in ${callSites[0]}, not observer.js`);
else pass(`exactly one AI call site, in ${callSites[0]}`);

// 3. That call scores the prompt rather than generating from it.
const observer = sources.get('observer.js') ?? '';
const call = observer.slice(observer.indexOf('ai.run('));
for (const [needle, why] of [
  ['raw: true', 'without raw:true the essay is wrapped in a chat template and the numbers describe a conversation'],
  ['prompt_logprobs: 0', 'this is the parameter that scores the given text instead of continuing it'],
  ['max_tokens: 1', 'the generation budget is minimal; the continuation is discarded'],
]) {
  if (call.includes(needle)) pass(`observer passes ${needle}`);
  else fail(`observer does not pass ${needle} — ${why}`);
}

// 4. The prompt is the essay and nothing else: no instruction is prepended or appended.
const promptLine = call.match(/prompt:\s*([^,\n]+)/)?.[1]?.trim();
if (promptLine === 'clipped') pass('the prompt is the essay text alone, with no instruction attached');
else fail(`the prompt is \`${promptLine}\`, which is not the bare essay text`);

// 5. Nothing in the pipeline reads a model's words. The observer's reply is consumed as
//    numbers; `decoded_token` is used only to locate a token in the text.
const analyze = sources.get('analyze.js') ?? '';
if (/\bresponse\b|\bcontent\b|\bchoices\[0\]\.message\b/.test(analyze)) {
  fail('analyze.js reads generated text from the model reply');
} else {
  pass('analyze.js consumes only logprobs, ranks and offsets');
}

console.log(failures ? `\nFAIL: ${failures} check(s)` : '\nPASS: the model is an instrument, not the judge');
process.exit(failures ? 1 : 0);
