# Palimpsest on Cloudflare

**Live: https://palimpsest.amitynoidalibrary.workers.dev**

The same detector, hosted. One Worker serves the interface, runs the observer, and computes
the verdict; there is no Python process anywhere in the request path.

```
browser ──► Worker ──► Workers AI          one forward pass, raw:true, logits read
               │
               └────►  segment · 43 features · sentence model · document model
                       · genre gate · calibrated band            (all in the Worker)
```

---

## Why a port at all

The application is FastAPI, and FastAPI does not run on Workers. The alternatives were a
Python host somewhere (Render, Fly) or a JavaScript port, and the port won for one reason:
the observer already runs on Workers AI, so a Python host adds a second machine and a second
network hop to reach the model that Cloudflare would otherwise be calling from the same
datacentre. Hosting the whole thing in the Worker removes a hop rather than adding one.

The cost of a port is that it might not be the same detector. Every accuracy figure in
`docs/` was measured on the Python implementation, so a rewrite that is merely close would
give the deployment an evidence base it has not earned. That is what `test/parity.test.mjs`
is for, and it is the main thing in this directory.

---

## Parity

`edge/test/parity.test.mjs` runs both implementations over real corpus documents with the
observer's token stream held byte-identical between them, and compares **364,056 values**:
every sentence's 43 features, its logit, its calibrated probability, the ordering and
magnitude of its evidence bars, the passages, the document verdict including its seeded
bootstrap interval, and the genre gate's inputs and decision.

```
262 documents, 5,527 sentences, 364,056 comparisons

largest disagreement by kind:
  feature      6.217e-15
  gate         2.665e-15
  probability  0.000e+00      exact
  logit        0.000e+00      exact
  share        0.000e+00      exact
```

Every probability, logit, threshold decision and bootstrap endpoint is **exactly** equal.
The only non-zero disagreements are 5e-15 on two n-gram surprisal means — last-ulp
differences between C's `log` and V8's, well below anything that could move a verdict.

The bootstrap interval being exact is deliberate and was the most annoying part: NumPy seeds
it with PCG64, so `src/numpy_random.js` reimplements SeedSequence's entropy mixer, the
128-bit PCG64 LCG and `Generator.integers`' Lemire rejection sampling, and checks its output
stream against NumPy directly. A convenient RNG would have produced a statistically
equivalent interval printing different digits, and the deployment would then quietly
disagree with every figure in the docs.

**The test is proved able to fail.** Six mutations, each reverting a specific decision:

| mutation | result |
|---|---|
| `Z_CLIP` 5.0 → 6.0 | FAIL 1,435 |
| genre gate keeps a genuine `0.0` instead of Python's falsy-drop | FAIL 431 |
| bootstrap seed 0 → 1 | FAIL 121 |
| sentence splitter forgets abbreviations | FAIL 16,502 |
| smoothing loses the double weight on the sentence itself | FAIL 2,820 |
| contraction regex reverted to JavaScript's ASCII `\b\w+` | **PASS** |

The last one is why `test/units.test.mjs` exists. Real admissions essays contain no accented
contraction, no Arabic-Indic digit and no `Ph.D.` mid-sentence, so the corpus run cannot see
whether the hand-written Unicode word boundaries are right. 34 constructed cases cover them,
and they redden for the ASCII `\w`, for ASCII `\d`, for the ECMAScript whitespace set in
place of Python's, and for a missing initials guard.

That last check found a genuine bug in this port before it shipped: `\n\s*\n` is the
paragraph splitter, and JavaScript's `\s` both omits U+0085 and includes U+FEFF, so a
document containing either would have been split into different paragraphs — and therefore
different sentences, and therefore different highlights — from the Python build.

---

## Verified in production

`test/live.test.mjs` sends corpus documents to the deployed Worker and compares against
Python's stored answers. On 6 documents spanning all four verdict bands, 5 reproduced the
cached observer scoring exactly and every sentence probability matched to the 4 decimals the
API publishes; the sixth showed observer drift (below).

`scripts/verify_ui.cjs` — the project's existing browser check — passes **28/28 against the
live URL**, including that the evidence bars printed on screen sum to the verdict printed on
screen, that the documented failure case still fails in the documented way, that the footer's
privacy claim agrees with `/api/health` rather than with whatever the page was built with,
that an essay longer than the observer's window says so instead of presenting a verdict on
its opening as a verdict on the essay, that a span the tool refuses to score is neither
shaded nor captioned with a percentage, that a 138-word run-on is not explained to its author
as "too short", no horizontal overflow at 390 px, and no console errors.

---

## What is different from the Python build

Four things, all deliberate. Two are wording; two *were* corrections that have since been
pushed back into the Python build, because a correction that lives only on one side of a
port is how the two sides drift.

Three more used to be listed here and are gone, for that reason. The observer-clipping
notice, its stylesheet rule and a status line reading "Scoring…" instead of "Scoring with the
local model…" were all real fixes that existed only here, so `web/` went on telling anyone
running the project locally that their essay was staying on their machine, and went on
presenting a verdict on 6,000 characters as a verdict on a 40,000-character essay. They now
live in `web/`, both builds read them, and `sync_web.py` is down from five patches to two.

**1. The privacy claim.** The interface footer said *"Nothing you paste leaves this
machine."* That was true when the observer was GPT-2 in-process and stopped being true when
the default moved to Workers AI — `/api/health` has reported `textLeavesMachine: true` the
whole time, so the page and the API were contradicting each other. Fixing it here left the
Python build still saying it, so `web/` now states the remote case and narrows it to the
local claim only when `/api/health` confirms an in-process observer; a failed probe keeps
the stronger warning. What remains different here is only *wording* — this page addresses a
browser with no local process, so it says "your machine" and points at running the project
yourself rather than at restarting with an environment variable.

**2. The error rates shown to users.** `api/app.py` rendered the limitations panel from
`artifacts/evaluation.json` unconditionally, while serving whichever detector
`PALIMPSEST_SUFFIX` selects — by default `_remote`. The panel stated a **17.8%** TOEFL
false-positive rate where the served build measures **10.9%**, **45%** cross-model-family
recall where it measures **64%**, and two suites that were never re-run on this observer as
if they had been. Both builds now render from `palimpsest.limitations`: from
`evaluation{suffix}.json` where that run covers the set, and from the GPT-2 run only where it
does not — labelled, because a number measured on a different instrument is worth showing and
is not worth showing silently. `tests/test_limitations.py` asserts the two builds publish the
same panel.

This is the exact failure `_limitations()`'s own docstring warns about — "the model was
retrained, and the interface went on confidently displaying the previous run's error rates" —
reached by a different route.

**3. Scores are not perfectly repeatable here, and the interface now says so.** The Python
project scored each essay once and cached it, so it never had to confront this. Running the
observer live does: Workers AI serves an fp8 mixture-of-experts model and does not always
return identical log-probabilities for identical text. Measured over 6 documents × 3 runs
(`artifacts/repeatability.json`):

| | |
|---|---|
| max change in a sentence score | **8.7 percentage points** |
| max change in document confidence | **1.7 points** |
| verdict band unchanged | **6 / 6** |

The band — the thing a user acts on — held in every case, but an essay sitting near a
threshold can cross it between runs, and that is now one of the limitations shown under every
verdict.

**4. Offsets are UTF-16, not Unicode code points.** Python counts code points; JavaScript
counts UTF-16 units. They agree on everything in the Basic Multilingual Plane and disagree on
astral characters. This port is UTF-16 throughout, which makes it self-consistent with the
observer (whose offsets have always come from `String.indexOf` inside a Worker) and with the
browser that slices text to draw highlights. The Python build mixes the two conventions at
that seam. No corpus document contains an astral character, so nothing measurable changes.

---

## Not a wrapper

The observer is called once per document with `{ prompt, raw: true, max_tokens: 1,
prompt_logprobs: 0 }`, which **scores the given text** instead of continuing it. There is no
instruction, no question, and no generated text read anywhere in the pipeline. `raw: true` is
load-bearing: without it the essay is wrapped in a chat template and the numbers describe a
conversation.

`test/no-generative-calls.test.mjs` enforces this structurally, since a Worker has no import
graph to inspect: exactly one call site reaches `env.AI`, it is in `observer.js`, it passes
all three parameters, the prompt is the bare essay text, and `analyze.js` never reads a
`response`/`content`/`message` field.

---

## Protecting the allowance

The account's Workers AI allowance is 10,000 neurons a day; one analysis costs ≈ 2.1
(measured live: 43 analyses, 91.18 neurons). The original `observer-worker/` sat behind a
shared secret because an open endpoint would drain it. This one is open, so the protection
lives in `src/budget.js`, a Durable Object holding two counters:

* per-IP sliding window — **6/minute, 60/hour** (verified: the 6th request in a minute returns 429);
* a **global daily budget of 7,000 neurons**, leaving headroom for the other Workers on the account.

A Durable Object rather than KV because both counters are read-modify-write on a hot key —
KV's eventual consistency would make the limit advisory, and KV's free write allowance (1,000
a day) is *below* the neuron budget it would be protecting. The budget check fails **closed**
and the rate limit fails **open**: an unreachable counter must not become an unmetered spend.

`/api/health` reports live usage.

---

## The n-gram reference

`artifacts/ngram_reference.json` is 18.6 MB — 16,818 unigrams, 263,395 bigrams, 512,992
trigrams, every key a string. Parsed into JavaScript objects that is well past a Worker's
128 MB isolate, and almost all of the weight is string keys that no lookup needs.

`scripts/build_ngram_bin.py` interns the vocabulary (16,819 tokens fit in a uint16 id), turns
every n-gram into integer ids, sorts, and binary-searches at query time: **8.5 MB** of typed
arrays, fetched from the asset binding once per isolate and held in module scope.

It is a re-encoding, not an approximation — no pruning, no hashing, no Bloom filter, because
`novel_trigram_rate` is a membership test whose false positives would silently shift a feature
the classifier has a weight for. The compiler reloads the binary and checks all 793,205
entries round-trip before writing the file.

---

## Build and deploy

**Pushing to `main` deploys.** `.github/workflows/deploy.yml` runs exactly the
sequence below and then checks the live `/api/health`, so a change by any
collaborator reaches https://palimpsest.amitynoidalibrary.workers.dev without
anyone running wrangler. It needs one repository secret,
`CLOUDFLARE_API_TOKEN`; the account id is already in `wrangler.jsonc`.

By hand, if you need to:

```bash
.venv/bin/python edge/scripts/build_ngram_bin.py    # 18.6 MB JSON -> 8.5 MB binary
.venv/bin/python edge/scripts/build_artifacts.py    # detector, gate, bands, limitations
.venv/bin/python edge/scripts/sync_web.py           # web/ -> assets/, with corrections
cd edge && npm install && npx wrangler deploy
```

**If you change anything the build reads, commit the regenerated files too.**
The workflow rebuilds and then diffs the result against the committed
`edge/src/artifacts.js` and `edge/assets`, and stops if they differ. That gate
exists because of a real incident: `highlight_disclosure()` reads
`artifacts/polish_head.json` and returns an empty list when the file is
missing, so a build without it produced a perfectly valid `artifacts.js` that
was simply missing a limitation, and the deploy published a tool disclosing
less about its own false positives than this repository said it did. Nothing
errored — the statement just stopped existing. Any input the build reads must
therefore be committed, and any generated file must be reproducible from what
is committed.

`sync_web.py` is the only thing that writes `edge/assets/`. Every difference from `web/` is a
named patch with a reason, and **a patch whose target text is missing aborts the build** — so
editing the interface in a way that invalidates a correction fails loudly instead of quietly
shipping the old claim, which is the failure being corrected in the first place.

## Tests

```bash
cd edge
node test/parity.test.mjs test/parity-cases.json test/parity-cases-live.json   # vs Python
node test/units.test.mjs                     # constructed inputs the corpus never produces
node test/no-generative-calls.test.mjs       # the model is an instrument, not the judge
node test/live.test.mjs                      # against the deployed Worker (spends neurons)
node test/repeatability.test.mjs             # observer determinism  (spends neurons)
```

The two parity fixture files are generated and gitignored:

```bash
.venv/bin/python edge/scripts/export_parity.py --limit 140
.venv/bin/python edge/scripts/export_parity.py --live --limit 12 --out edge/test/parity-cases-live.json
.venv/bin/python edge/scripts/export_unit_cases.py
```

`--live` scores un-flattened, multi-paragraph documents for real, because the corpus was
scored through `flatten()` and flattening removes the only thing `split_paragraphs` exists to
handle. Without it a port could get paragraph segmentation completely wrong and every cached
case would still pass.

---

## What this deployment does not change

The detector itself. Nothing here is refitted, retuned or rethresholded; `src/artifacts.js`
is a copy of what `scripts/train.py`, `fit_bands.py` and `fit_genre_gate.py` wrote. Every
limitation in [`../PROJECT.md`](../PROJECT.md) §10 still holds — frontier prose is still
undetectable by this method, paraphrase attacks are still untested, and the tool still cannot
tell you that a person wrote something.
