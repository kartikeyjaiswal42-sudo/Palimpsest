# Palimpsest — Development Record

*How the detector was built, 7–12 August 2026. Extracted from nine build sessions.*

This is the engineering history: what was built, what each measurement changed, which defects
were found and how, and which decisions were made against the evidence. `PROJECT.md` describes
what the tool **is**; this describes how it **got there**. Where a number appears twice with
different values, it changed — the date says when.

---

## 1. What was being built

An AI detector for college admissions essays, submitted to the 2026 i12 HR Drive Hackathon
(Callus), Project 2, 1–15 August 2026.

The brief set the constraints that shaped every decision below:

- A **working application with a real interface** — not a script, not a notebook.
- Paste an essay → show **which parts** were probably machine-written **and why**.
- **Not a wrapper**: *"a detector that sends the essay to a chat model and asks for a verdict is unreliable, cannot explain its reasoning… We will be able to tell."*
- The line drawn precisely: *"the model must not make the judgement call while your app relays the verdict."* Using an LM **as an instrument** — token probabilities, then your own analysis — is explicitly endorsed.
- Detection at **sentence and passage level**; the realistic case is a paragraph a person wrote and a model later polished.
- **Build the dataset**, and document where it came from and what it does not cover.
- **Honest accuracy**: results on your own test set, plus *"three essays the detector gets confidently wrong, and a theory why."*
- **ESL**: *"These detectors have a habit of flagging writers who learned English as a second language. If yours does, we'd like to know you spotted it."*

## 2. Architecture

A local or remote language model is read for per-token statistics in a single forward pass;
everything after that is arithmetic under the project's control.

```
essay
  → sentence segmentation (character offsets preserved)
  → observer LM, one forward pass       ← the only model call in the system
      { prompt, raw: true, max_tokens: 1, prompt_logprobs: k }
  → per-token logprob + true rank
  → 43 interpretable features in 4 families (+ n-gram reference over 1,000 human essays)
  → logistic regression  →  calibration  →  bands
  → per-sentence probabilities, passages, and the feature contributions that sum to the logit
```

Two properties are load-bearing:

- `raw: true` stops the essay being wrapped in a chat template. Without it you are measuring a conversation, not the essay.
- `max_tokens: 1` — the continuation is discarded. The *prompt* is what is being scored.

**No model is ever asked for a verdict.** The file `tests/test_no_generative_calls.py` AST-walks
every file under `src/palimpsest/` and fails on any call to `.generate` / `.chat` / `.complete`
/ `.completions` / `.create_completion`, on any import of `openai`, `anthropic`, `cohere`,
`litellm`, `ollama`, `google.generativeai`, and on `apply_chat_template` appearing in
`local_lm.py` at all. A companion check, `test_at_least_one_source_file_found`, stops the
parametrised guards passing vacuously on an empty glob.

The verdict layer is a calibrated logistic regression, so the evidence panel is not a story
about the computation — the logit **is** the sum of the bars displayed, and the UI prints the
arithmetic.

### Two observers

| | local | remote |
|---|---|---|
| model | GPT-2 124M, in-process | `@cf/qwen/qwen3-30b-a3b-fp8` via Workers AI |
| set by | `PALIMPSEST_OBSERVER=gpt2` | default (`remote`) |
| text leaves the machine | no | **yes** |
| sentence AUROC | 0.925 | **0.945** |
| Claude Haiku sentence recall | 0.105 | **0.577** |

Keeping both, and reporting the comparison, produced the mechanism behind the ESL problem
(§4, 10 August) — a stronger result than either observer alone.

## 3. Two numbers, not one

The output is deliberately two quantities plus per-sentence detail, never a single percentage:

- `machine_share` — an *extent*: words in sentences scoring ≥ 0.65, over total words in measurable sentences, with a bootstrap interval. Unreliable sentences are dropped from **both** numerator and denominator; calling them human would be as much of a claim as calling them machine.
- `any_machine_probability` — a *probability*, from its own logistic regression over four summary statistics (`mean_p`, `max_p`, `q90_p`, `share`). Share is one of its four inputs, so it is not the first number rounded.

The reason they are kept apart: an essay with one polished paragraph out of six has a **low
share** and a **high probability**. A single figure destroys exactly the information the reader
needs — which is the brief's *"'73% AI' gives a reader nothing"* requirement answered
structurally rather than cosmetically.

---

## 4. Build timeline

### 7–8 August — first working detector

58 files, GPT-2 observer, 43 features, calibrated logistic regression, FastAPI + web UI.

| measurement (held out) | |
|---|---|
| sentence / document AUROC | 0.960 / 0.959 |
| locating machine text in a mixed essay | AUROC 0.883, seam within 2 sentences in 70% |
| false positives, out-of-domain human | 0.0% |
| false positives, English-language learners | 7.3% (TOEFL 24.4%) |
| GPT-3.5 prompted to evade | 6.5% caught |
| prose composed by hand to imitate a model | 0 of 11 |

The cleanest evidence that it measured authorship rather than topic: the same 88 ASAP essays
went from **0.0% → 65.9%** flagged after a model rewrote them.

**Three artifacts caught during the build, each of which had inflated a result:**

1. **The machine corpus was written by hand, and it destroyed the detector.** The strongest feature turned out to be measuring the prompt's own instruction to "vary sentence length." Against real GPT-3.5 that feature points the *opposite* way. Switching to real model output moved sentence AUROC **0.79 → 0.96**.
2. **A 0.988 that was partly a smart-quote detector** — human essays were 88% curly-quoted, machine essays 0%. Normalising the quotes held the score, which is what made it credible.
3. **A document-length leak causing a 41% false-positive rate on short TOEFL essays** — the artifact landed hardest on exactly the group this class of tool is known to harm.

**Decision — calibrate on the at-risk population, not in-domain.** Training on mixed documents
fixed localisation (0.745 → 0.883) but tripled false positives, because the document threshold
was calibrated in-domain and did not transfer. It was recalibrated on ESL and out-of-domain
writing instead, costing three-quarters of the recall on evasive text. Recorded in
`docs/06-decisions.md`; `--fpr-budget` moves it.

### 8 August — the recheck, and a class of bug named

A deliberate re-verification found four problems, three of which were the same bug: **prose
drifting away from the artifacts it described.**

1. **A stale claim.** The docs said removing the document-length feature "halved the harm and cost nothing measurable" — true when measured, then the pipeline changed underneath it. Re-run as a script:

| | with length | shipped |
|---|---|---|
| TOEFL FPR | 33.3% (15/45) | 24.4% (11/45) |
| ESL overall FPR | **5.6%** | 7.3% |
| in-domain document AUROC | **0.998** | 0.959 |

   Paired McNemar **p = 0.29**. The improvement is inside noise, removal made aggregate ESL
   *worse*, and it cost real AUROC. The decision stands on principle — a corpus artifact should
   not hold the model's largest weight (−3.09) — but the docs now say the measurements disagree
   with it.

2. **The evidence panel did not add up.** It showed six bars and gestured at "the rest." The omitted terms outweigh the six shown on **30.9% of sentences**, and the intercept (−2.26) is larger than either. It now prints `intercept + shown + remainder = logit`.

3. **The demo mislabelled itself.** The button said "Example it misses" — the detector highlights the rewritten paragraph exactly (2 of 2). What fails is the document verdict: 74% against a 97.4% threshold. Relabelled "Example it lets through," which is the better demo, because it shows the operating point costing a catch on screen.

4. **The failure write-ups quoted a superseded model.** `find_failures.py` had not been re-run after the final retrain, so every contribution cited in them came from the old detector.

**Guard added:** `tests/test_documented_numbers.py` checks headline figures *and* quoted failure
evidence against the JSON artifacts on every run. Both halves were confirmed to fail on a
corrupted number before being kept.

Verified: 118 pytest · 23/23 browser checks · a fresh `git clone` boots and passes, so it runs
without the un-redistributable corpus.

### 9 August — the Gemini corpus, and a harm introduced then caught

**567 Gemini essays** across 5 checkpoints, 400–500 words each, split four ways.

**Only 135 entered training.** Feeding in all of them would have swung the training prior from
12% to 69% machine — which does not make a better detector, it makes a more willing accuser.

| the generator is… | before | after |
|---|---|---|
| in training, these essays are not (n=115) | 0.0% | **94.8%** |
| in training, **no topic steering** (n=45) | 0.0% | **95.6%** |
| a checkpoint withheld entirely (n=250) | 0.0% | **80.0%** |
| a different family withheld entirely (n=22) | 0.0% | **45.5%** |
| GPT-3.5 prompted to evade (n=31) | 6.5% | **38.7%** |

**False positives fell rather than rose** — ESL 7.3% → 5.8%, TOEFL 24.4% → 17.8%.

The 95 → 46 gradient is the substantive finding: it puts a number on how much of any detector's
score is recognising *one generator* rather than machine prose in general.

**A serious harm was introduced and then caught.** Making machine essays longer flipped the
`n_words` weight from −1.17 to **+0.148**. Three ELLIPSE/PERSUADE essays by second-language
students contain no sentence-ending punctuation, so each segments to a single 312–466-word
"sentence" at **z = +50**. All three went from unflagged to **accused at P = 0.977**. Two guards
now exist — z-clipping, and excluding unmeasurable spans — both mutation-tested. It also exposed
that `reliable` was decoration: computed, displayed, then ignored by the aggregator.

**Two regressions recorded rather than buried.** The ESL false-positive rate used to rise
monotonically with proficiency — the strongest evidence the tool measures fluency rather than
nativeness — and became U-shaped, so `docs/05-esl.md` separates the surviving half from the
segmentation artifact. And localisation regressed: seam-within-2-sentences **70% → 39%**,
undiagnosed. In-domain AUROC fell 0.960 → 0.925 because the task got harder.

The calibration table is now generated rather than hand-copied — it had been stale for three
commits, claiming 41 sentences in a band that held 186. 16 headline claims are pinned to
artifacts by tests. 127 tests passing.

### 10 August — cross-vendor, and the capability finding

**A naive repair was measured and rejected.** Adding the 238 Claude training essays looks like a
triumph (flag rate 0.066 → 0.930) and is worthless: it flags **79% of genuine English-learner
sentences**, sentence AUROC drops 0.925 → 0.726, and document recall goes to zero on Gemini sets
it used to catch at 80–95%. The cause is arithmetic — training was balanced at 0.498 machine
sentences, Claude adds 8,574 all-machine ones, prior → 0.773.

Holding the budget fixed and varying only which vendor it is spent on:

| Claude share of budget | AUROC | ESL FPR | Claude flag | Gemini flag |
|---|---|---|---|---|
| 0% | **0.925** | 0.174 | 0.066 | **0.831** |
| 25% | 0.785 | 0.164 | 0.143 | 0.483 |
| 75% | 0.735 | 0.109 | **0.209** | **0.007** |

Claude detection peaks at 0.209 while Gemini collapses to 0.007. **It is not a training-data
gap** — given equal budget, identical prompts and matched length, the fit reaches 0.83 on Gemini
and 0.21 on Claude.

**The vendor reading was then corrected to a capability reading.** Every Gemini essay in the
corpus was flash-lite and every Claude essay was not, so "vendor" was confounded. The evidence
that separates the two is the **within-vendor** gradients — flash-lite 0.91 → flash 0.665 inside
Gemini, and haiku 0.105 → opus 0.027 inside Claude. Same direction, two independent vendors, no
confound between them. This also means the project's headline 94.8% was measured against the
cheapest model Google ships, which is the most important caveat attached to it.

**A contaminated corpus was rejected.** A supplied 500-essay Codex file had 489 of 500 rows
byte-identical to essays already on disk, **281 of them copies of held-out evaluation sets**.
Training on it would have produced excellent-looking numbers that were pure memorisation, and it
would have been hard to spot later because the file looks like a fresh corpus.

**The register finding:** the detector scores abstract sentences high and concrete ones low —
0.94 for *"the world is full of broken systems"*, 0.04 for *"I didn't fix all of them."* It is
detecting generic aspirational writing, not authorship.

### 10 August — the observer swap, and the frontier ceiling

Written up in `docs/09-frontier-ceiling.md`.

**The 250× observer scale-up works, for the cheap tier.** Sentence flag rate on held-out Claude
essays:

| model | GPT-2 | qwen3-30B | |
|---|---|---|---|
| haiku | 0.105 | **0.577** | 5.5× |
| sonnet | 0.067 | **0.353** | 5.3× |
| fable | 0.046 | **0.259** | 5.7× |
| opus | 0.027 | 0.083 | 3.1× |

Out-of-fold AUROC 0.925 → **0.945**; mid-tier Gemini document recall 0.455 → **0.546**. The ESL
document false-positive rate went **down** (0.080 → 0.074 on identical documents), so the recall
was not bought by accusing more students. Opus document recall remained **4.4%**.

**The ESL bias has a mechanical root cause.** GPT-2's statistics are *inverted* on
English-learner prose: **AUROC 0.132** — machine essays looked *more human* than the human
reference, because simple ESL prose is low-perplexity to a 2019 model and modern LLM prose is
not. "Low perplexity ⇒ machine" was aimed directly at non-native writers. The 30B observer
corrects the sign everywhere.

**Frontier detection did not survive an honest test.** Opus: **0% recall at a 5% false-positive
budget** from perplexity. A stylometric alternative scored a perfect **1.000** and turned out to
be reading *which pipeline produced the file* — it flags 0% of real GPT-3.5 essays and 17% of
real students.

**DAIGT gave a two-sided answer.** 4,189 essays across **17 independent pipelines**;
leave-one-pipeline-out gives **median 1.000 AUROC, 1.000 recall** — supervised detection of a
genuinely unseen generator is real. But trained on DAIGT and pointed at a different genre, it
flags **91% of real human admissions essays**: it learned "argumentative coursework" versus
everything else. Read together: **supervised detection generalises across generators and not
across genres.**

**A live calibration defect was found and fixed.** The threshold rule picked the point whose
*observed* false-positive rate hit 5% — but a threshold chosen to minimise a rate makes that rate
optimistically biased by construction, which is why it shipped **8.0%** against a 5% budget on
held-out data. Replaced with a **Clopper–Pearson exact upper bound** — *"the true rate is at most
5%, with 95% confidence."* Held-out ESL document FPR **0.080 → 0.039**.

**Three-band verdict with abstention shipped**, directly fixing two reported failures:

| input | share | band |
|---|---|---|
| human ESL (TOEFL) | 64% | insufficient evidence |
| human ESL (TOEFL) | 67% | insufficient evidence |
| machine, Gemini flash-lite | 100% | **likely machine-written** |
| machine, Claude Opus | 0% | insufficient evidence |

The honest price: **it abstains on 88% of human essays and 65% of machine essays.** And the
governing constraint — **this tool cannot exonerate anyone.** Because Opus prose scores like
human prose, a low score carries no information about authorship, so the bottom band is worded
*"no evidence found"*, never *"human"*, and nothing in the UI is coloured green.

The app moved to the 30B observer at the same time: uvicorn RSS **21 MB**, GPT-2 no longer
loading locally.

### 11 August — the genre gate

**A live false-accusation bug:** the tool called a real 13-year-old's essay machine-written at
**97% confidence** — an out-of-genre document scored as though it were an admissions essay.

The gate is six document-level features → logistic regression. **It never touches the sentence
model, the document model, or their thresholds** — it only decides whether they are consulted, so
detection is intact by construction (in-domain machine sets are refused at 0.0% and 1.3%).

| | before | after |
|---|---|---|
| the 13-year-old's essay | **97% machine-written** | P(in-domain) 0.030 → *"outside this tool's scope"* |
| machine admissions essay | scored | scored (P = 0.843) |
| human admissions essay | scored | scored (P = 0.911) |

**The validation that licenses it to ship:** the gate is fitted with *both* human and machine
admissions essays in-domain and passes them at **95.5% vs 94.6% — a 0.9% gap.** A gate correlated
with authorship would be a second detector in disguise, converting low recall into high
abstention. `fit_genre_gate.py` refuses to save if that gap exceeds 10%.

End to end:

| | n | likely machine | insufficient | no evidence | out of scope |
|---|---|---|---|---|---|
| Gemini flash-lite | 57 | **80.7%** | 19.3% | — | — |
| Claude family | 80 | 0.0% | 83.8% | 13.8% | 2.5% |
| ESL human | 305 | **0.0%** | 14.8% | 1.0% | 84.3% |
| domain shift | 44 | **0.0%** | 36.4% | 20.5% | 43.2% |

**0 of 349 human documents called machine-written** — but read honestly, and `PROJECT.md` says so
in the same breath: that is not the detector becoming good at not accusing ESL writers, it is the
gate declining to judge 84% of them, because TOEFL responses are not admissions essays.
Underneath the gate the ESL false-positive rate is 3.9%.

**DAIGT documented and quarantined** — excluded from training for two independent reasons (no
declared licence on the mirror, and wrong genre), absent from every `trainSources`, and
gitignored so it is not redistributed.

### 11 August — what a real browser found

Three defects in the first ten minutes of visual verification, none of which markup inspection
had caught:

1. **The genre gate was secretly a length detector.** Same essays, same genre, only truncated:

| words | refused (before) | refused (after) |
|---|---|---|
| 700 | 0/6 | 0/6 |
| 350 | 2/6 | **0/6** |
| 260 | 3/6 | **0/6** |
| 150 | 5/6 | **1/6** |

   `log_words` carried weight **+1.364** because the out-of-genre sets are shorter. Supplemental
   prompts are routinely capped at 250–350 words, so it would have refused real submissions while
   calling itself a genre check — the *same length artifact* `docs/04-failures.md` already
   records, reproduced by a different route.

2. **A band had no styling at all.** `.band-out_of_scope` rendered with a transparent background: three band styles were written and a fourth state added later.

3. **The app's own demo essay was refused, by 0.015** (P(in-domain) 0.3293 against a 0.3440 threshold). That exposed a design flaw: **the false-refusal rate had never been bounded.** Refusing to score someone is a claim and needs evidence like any other. Rebuilt on Clopper–Pearson, matching every other threshold in the project:

| | aggressive gate | bounded gate |
|---|---|---|
| real admissions essays scored | 95% | **99.6%** |
| out-of-genre refused | 76.1% | 25.8% |
| human essays called machine | 0.00% | **1.15%** (budget 5%) |

   The earlier 0.00% was not better — it reached zero by refusing 5% of genuine essays. A document
   the gate misses still falls through to the bands; a wrongly refused essay is a dead end with
   no recourse.

Also fixed: a cached stylesheet hid the CSS fix — correct on disk, correct over the wire, absent
in the browser. Static assets now serve `no-store`.

### 11 August — the ESL gate study

The open question was whether the gate would refuse a genuine non-native applicant's personal
statement. No ESL-authored admissions essays exist in the corpus, but two corpora carry unused
labels: PERSUADE has an ELL flag on prompt-matched essays, ELLIPSE has graded 1.0–5.0 proficiency
scores.

| control | result |
|---|---|
| PERSUADE, matched prompt, binary ELL (n=24) | log-odds shift **+0.079**, CI −0.336 to +0.367 — nothing |
| ELLIPSE, graded proficiency (n=260) | **ρ = +0.232, p < 0.001** — weaker English, more refusal |

Not a contradiction but a power difference: a binary split of mostly-stronger writers cannot see
a continuous effect. **When two controls disagree, the graded one wins.**

The mechanism: `mean_sentence_words` carried the signal (r = −0.243), because a struggling writer
produces *run-ons*, not short sentences. It had passed the length audit honestly — truncation
barely moves it — and was measuring proficiency anyway.

Removing it: transplanting a two-point proficiency drop onto real admissions essays went **9.96%
→ 3.32% refused**, weakest band **100% → 25%**. But the headline false-accusation rate went
**1.15% → 1.43%**, so the extra refusals had to justify themselves:

| gate | refused | accused | refusal precision | refusals per accusation avoided |
|---|---|---|---|---|
| 4-feature (shipped) | 25.2% | 1.43% | 4.5% | **22.0** |
| 5-feature (retired) | 31.5% | 1.15% | 4.5% | **22.0** |

Identical precision, identical cost: ~22 people denied an answer per accusation avoided, landing
hardest on the weakest writers. The 4-feature gate shipped.

A separate limitation surfaced in the same pass: technical documentation passes the gate at
**0.985 in-domain** (and 0.988 on the retired gate, so it is pre-existing) — the gate was fitted
against *student* writing and extrapolates confidently onto genres it never saw.

### 11 August — training on frontier prose

A commercial detector caught a Claude-written essay that Palimpsest scored at 3%. The cause was
not a ceiling — it was a data gap sitting on disk:

```
trainSources: ['jhu', 'liang_college_gpt3', 'liang_college_human',
               'modern_gemini-3.1-flash-lite', 'real_hybrid']
```

**The detector had never seen a Claude sentence.** All ~400 Claude essays were held out for
evaluation, so every "0% recall on frontier models" figure measured *cross-generator
generalisation*, not the undetectability of frontier prose.

Trained on a topic-disjoint half of the Claude corpus (split by pre-registered subject, so no
topic both teaches and tests), the essay went **3% → 76% machine share**. But it cost: Gemini
recall 80.7% → 40.4%, sentence AUROC 0.945 → 0.901, and a regularisation sweep at C = 0.3/1.0/3.0
does not recover it. Claude prose sits so close to human prose that fitting it drags the boundary
and the easier generator pays.

**Two specialists, take the higher score, re-fit the bands at the same budget:**

| variant | false accusation | Gemini | Claude (held out) |
|---|---|---|---|
| shipped generalist | 2.68% | **84.2%** | 2.6% |
| Claude specialist | 2.68% | 36.8% | 15.8% |
| **max of both** | **2.68%** | **75.4%** | **15.8%** |

**6× the frontier recall at an identical false-accusation rate**, for 9 points of Gemini.

**On buying the ceiling with compute:** the project had already run the experiment. GPT-2 124M →
Qwen3-30B is a 250× observer scale-up, and Claude recall moved **0% → 0%**. The ceiling is
signal-bound, not compute-bound. Training a *classifier* on rented GPUs is permitted by the
brief; fine-tuning an LM to answer "is this AI?" is precisely the wrapper it forbids, and would
destroy the explainability the project is built on.

### 11 August — the Cloudflare port

The whole tool ported to a single Worker serving the interface, running the observer, and
computing the verdict — no Python in the request path, no second hop.

The risk of a port is that it silently becomes a *different* detector, which would strip the
accuracy numbers of their meaning. That is what most of the effort went into disproving:
**184,688 comparisons across 145 documents and 2,801 sentences**, every probability, logit,
threshold decision, evidence bar and bootstrap endpoint **exactly** equal, largest disagreement
anywhere **5e-15** on an n-gram surprisal mean. Six mutations confirmed the test can fail. Making
the seeded bootstrap interval exact required reimplementing NumPy's PCG64 (SeedSequence entropy
mixer, XSL RR 128/64, Lemire bounded-integer rejection sampling), verified against NumPy
directly.

Supporting work:

- **n-gram reference 18.6 MB JSON → 8.5 MB binary** (interned vocab, sorted uint32 keys, binary search). A re-encoding, not an approximation — the compiler verifies all **793,205** entries round-trip, because `novel_trigram_rate` is a membership test.
- **Abuse protection**: Durable Object, 6/min and 60/hour per IP, global 7,000-neuron daily budget. **The budget fails closed; the rate limit fails open.**
- **One writer for the shared assets.** `sync_web.py` is the only thing that writes `edge/assets/`; each deviation from `web/` is a named patch with a reason, and a missing patch target **aborts the build** rather than shipping a stale claim.
- **Non-determinism measured** rather than assumed: re-running 6 essays ×3 with a live observer moved a sentence score by up to **8.7 points** and document confidence by **1.7**; the band held 6/6. Recorded in `edge/artifacts/repeatability.json` and shown to users.

### 12 August — the wiring bug

`Analyzer.analyze()`, the library's main entry point, **never loaded the trained document
model.** It fell back to a default that returns the single most suspicious sentence under a field
named `any_machine_probability`, and used the wrong threshold on top of that — reading 70.9% where
the real model reads 14.9%.

It hid because the number is between 0 and 1 and moves in the right direction. The deeper reason
is the one worth keeping: **every caller that reported a number had quietly patched around it.**
The web server rebuilt the verdict by hand. The parity test proving the JavaScript port matched
Python *also* rebuilt it by hand — so it compared the two against a result the library never
produced, and reported perfect agreement across **359,852 comparisons** while the thing it named
was broken.

Fixed by loading the model once and **deleting** both workarounds; the parity test now refuses to
run if the model is not loaded. `tests/test_analyzer_wiring.py` was added, with the one-line bug
reintroduced to confirm 3 of its 4 tests go red. Evaluation numbers were never affected — the
evaluation scripts and the deployed site always used the correct path.

**Fast-DetectGPT: built, measured, discarded.** Curvature features from the top-20 token
candidates, A/B tested on the same rows and documents:

| | sentence AUROC | frontier Claude prose (76 docs) |
|---|---|---|
| without | 0.9748 | **27.6% caught** |
| with | 0.9747 | **27.6% caught** |

Zero movement; +0.001 per document. The features are not noise — read alone, entropy separates as
well as anything in the model — they are **redundant**: curvature correlates **r = +0.87** with a
feature already present. Abandoned and written up as a result rather than a gap.

### 12 August — the drift between two builds

The docs recorded two defects found during the port and said *"both are fixed in the hosted
build."* That was literally true and quietly misleading: **both were fixed only in the hosted
build**, and both were still live in the Python application the README tells people to run.

**1. The app published a different detector's error rates.** `app.py:271` read a hardcoded
`evaluation.json` while every other artifact on the page — detector, bands, genre gate — was
selected by `SUFFIX`. With the default `remote` observer, the app served `detector_remote.json`
and printed the GPT-2 build's numbers underneath it, unlabelled:

| shown under every verdict | served build actually measures |
|---|---|
| 17.8% TOEFL false-positive rate | **10.9%** |
| 45% cross-model-family recall | **64%** |
| "38.7% caught when prompted to evade" | never re-run on this observer |
| "0% adversarial" | never re-run on this observer |
| *(absent)* | **5.6% frontier recall** — the number that gives the bottom band its meaning |

**2. A false privacy claim.** The footer read *"Nothing you paste leaves this machine"* while
`/api/health` in the same process reported `textLeavesMachine: true` and the essay went to
Cloudflare. No test touched the panel, and the UI check only asserted that *some* ESL rate was
disclosed — just as true of the wrong one.

Fixed in one place rather than a copy, since one-sided fixes are how this drifted: a new
`src/palimpsest/limitations.py` rendered by **both** the FastAPI app and the hosted build's
`build_artifacts.py`, with carried-over figures labelled rather than dropped or passed off as this
build's; and a footer driven by `/api/health` that **defaults to the stronger warning**, so a
failed probe overstates what leaves the machine rather than understating it.

### 12 August — four more defects

1. **The tool named a student's whole essay as machine-written while refusing to score it** — one response carrying two incompatible answers: *"nothing here is measurable, 0% machine"* and *"characters 0–600 are a machine-written passage, peak 99%."* The ELLIPSE run-on case again. The reliability rule existed but had been applied to one of the two functions that need it. Underneath sat a worse version: smoothing is weighted by word count, so a 20-word sentence scoring 5% was dragged to **91.6%** by one unmeasurable 466-word neighbour.
2. **A verdict on 6,000 characters presented as a verdict on the essay.** The API accepts 40,000. On an 8,496-character essay, 27 sentences and 453 words were never sent to the model and were dropped from the verdict silently. The "was it truncated?" flag **could never be true** — the client cut the text itself, then asked the server whether cutting had happened, which the server answers by comparing what it received against the same limit.
3. **"Scoring with the local model…"** displayed while the essay was in flight to Cloudflare — the same false claim as the footer, in the one place the reader is looking as it happens.
4. **A 138-word run-on labelled "too short to measure reliably"** and shaded strongly machine-like, captioned *"97% machine-like"* — exactly the accusation the arithmetic had just declined to make.

**Three of the four were already correct in the hosted site and had never been carried back**, the
same drift as before. The hosted-only patches were deleted and the fixes moved into shared source:
`sync_web.py` went from five patches to two.

Verified: 15 new tests, each fix reverted in turn to confirm the matching test goes red; full
suite **174 passing**; parity re-checked over **262 documents / 364,056 values**, every
probability, logit and share exact; browser suite **28/28 against the live site**.

### 15 August — a document can be opened, and a PDF is refused by name

Until now the only way in was a paste. `.docx` and `.txt` are now read **in the browser**: a
.docx is a ZIP holding `word/document.xml`, and `DecompressionStream` plus `DOMParser` are
already there, so this cost no dependency and no external request.

**Client-side was the design decision, not a shortcut.** `web/` is copied verbatim into
`edge/assets/` by `sync_web.py`, so one implementation serves the local build and the hosted
one. Extracting on the server would have meant a Python dependency for one deployment and a
separate JavaScript implementation for the other — two readers that can disagree about what
the essay says, which is the drift that produced the hosted-only patches above.

**PDF is refused and named.** A .docx stores paragraphs; a PDF stores placed glyphs, so
recovering prose means guessing where lines join and whether a hyphen ended a word or a line.
Those guesses land on sentence boundaries, and every number here is computed per sentence — a
wrongly joined line would move the result with nothing on screen to show it. The refusal says
that and says to save as .docx, rather than "unsupported format". Format is decided from the
leading bytes, so a .pdf renamed .docx is still named as a PDF.

Tracked-change *insertions* are read and *deletions* are not; hyperlink field codes are
skipped. The text lands in the essay box and that box is what is analysed, so there is no
hidden copy and a badly-converted document can be seen and corrected. A refusal leaves
whatever was already pasted untouched.

**One real defect, found by the check rather than by reading the code:** analysis was kicked
off in the same tick as the "Read 18 words from essay.docx" status, so that line was replaced
before the browser painted it once. A disclosure that is never on screen is not a disclosure;
the analysis is now deferred a beat.

Verified: `scripts/verify_upload.cjs`, **37 checks in a real browser** — deflated and stored
ZIP entries, Word's blank spacer paragraphs, tracked changes both directions, field codes,
tab/break/no-break-space/soft-hyphen, `.txt`, and all seven refusals, each asserting both that
the reason is named and that the box is left alone. 37/37 against a static server, 37/37
against the live API, 36/36 against the synced `edge/assets` copy. **Proved able to fail**:
reverting the `delText`/`instrText` guard and the invisible-character normalisation reddens
exactly the four matching checks. Fixtures are built by the script, not committed — a .docx in
the repository is a blob nobody can review in a diff, and the case worth testing hardest is
invisible in any viewer that renders it correctly; `scripts/make_test_docx.py` adds a
python-docx file so the reader is also proved against what a word processor really emits.
No regressions: pytest **194**, `verify_ui.cjs` **30/30** including its 390px overflow check.
Deployed and confirmed in a real browser on the hosted Worker.

**A gotcha worth keeping:** immediately after `wrangler deploy`, `curl` on the asset URLs
returned the OLD file and wrangler itself printed *"No updated asset files to upload"* — both
read as a failed deploy. The assets were live; the plain fetch was served `cf-cache-status:
HIT` from the edge. Cache-bust before concluding a deploy did not ship.

---

## 5. Publication

**Cloudflare** — `https://palimpsest.amitynoidalibrary.workers.dev`, version `489ba7ef`, single
version at 100%. Rather than redeploying blindly, the live `app.js` and `index.html` were
downloaded and diffed against the committed files — identical.

**GitHub** — `https://github.com/kartikeyjaiswal42-sudo/Palimpsest`, 626 files. Audited before
committing rather than after. Four things would have been published and were stopped:

| what | why it mattered |
|---|---|
| `data/cache/` — **240 MB** | It looked like a cache of *numbers*, but each entry stores every token with its character offsets, and a real student's essay was reconstructed from it verbatim. Committing it would have republished the TOEFL/ELLIPSE/PERSUADE student essays through the back door. Excluded; it rebuilds for free. |
| `edge/node_modules/` — **206 MB** | Dependency tree. |
| `observer-worker/.env` | The observer secret. Every stageable file was searched for the literal value: zero hits. |
| `artifacts/_before/` | Retrain scratch, not results. |

The generated corpora that *are* published were confirmed to be `authorship: machine` — the
project's own model output, not scraped human writing. `edge/wrangler.jsonc` carries the
Cloudflare account ID, which is an identifier rather than a password and is useless without an API
token (not in the repo).

## 6. Compliance with the brief

| brief asks | status |
|---|---|
| working app, real interface, not a notebook | FastAPI + web UI, and a hosted Worker |
| shows **which parts** | per-sentence offsets, probabilities and passages |
| shows **why** | named feature contributions that **sum exactly to the logit** |
| not "73% AI" | two separate quantities plus a three-band verdict with abstention |
| **not a wrapper** | no `generate()`, no prompt, no chat template; enforced by AST test |
| human-written-then-polished, sentence level | `real_hybrid` hybrids in training |
| dataset sourced, documented, gaps stated | 14-row table with licences in `docs/02-dataset.md` |
| **three essays it gets confidently wrong + why** | `docs/04-failures.md`, plus six more |
| **ESL flagging — did you spot it** | `docs/05-esl.md`, with a mechanistic root cause |

**On "small local model."** The default observer is a 30B remote model. The brief's hard line is
*"the model must not make the judgement call while your app relays the verdict"* — not crossed.
*"Running text through a small local model for token probabilities"* appears in the **Notes** as a
permitted example, not a specification, and the brief also says *"use whatever language and stack
you like."* Setting `PALIMPSEST_OBSERVER=gpt2` restores full locality, and the GPT-2 ↔ 30B
comparison is a finding rather than a dodge.

**Corpus licensing.** `data/raw/`, the PERSUADE-derived hybrids and DAIGT are all gitignored; the
only committed essays are the project's own generations. PERSUADE is **CC BY-NC-SA** —
non-commercial, which is fine for a hackathon and a genuine blocker for a paid product.

## 7. Open items

- **The repository is public.** The brief states *"push your work to your own private repository."* Verified unauthenticated: HTTP 200. This is the one flat rule violation outstanding.
- ~~**The README describes a detector that is not shipped**~~ — **fixed 15 August 2026, along with the reason it survived.** The diagram, the `uvicorn` line, the test counts and every headline figure now describe the served `_remote` build, and the numbers that exist only for the GPT-2 build carry that label on the line. The root cause was not the prose: `tests/test_documented_numbers.py`, the test written to stop exactly this, read `evaluation.json` / `detector.json` / `document_detector.json` while `api/app.py` served the set named by `SUFFIX`. It was checking a model nobody runs, so it passed while the README stated 0.925 against a served 0.9576 and a 17.8% TOEFL FPR against a served 10.87% — the latter in the safety warning about how often the tool is wrong about a real student, and contradicted by the site's own limitations panel. This is the same fault `_limitations` records having fixed in the application in July; the application was fixed and the test was not. It now resolves the suffix the way the server does, fails rather than skips when a "served" claim has no measurement, and requires the `GPT-2-observer build` label on any claim that only exists in the old evaluation. Correcting it reddened 13 of 15 claims plus the calibration table.
- **No ESL-authored admissions essays exist in the corpus.** Whether the gate would refuse a genuine non-native applicant's personal statement — the exact person this tool must not fail — remains the most important untested question. Fifty real essays would be worth more than any further accuracy work.
- **Localisation regression undiagnosed** — seam-within-2-sentences 70% → 39% since the 9 August retrain.
- **Frontier prose is not reliably detectable** by any method in this repository. Unedited cheap-model output is; Opus/Sonnet/Gemini-Pro-class prose in a 650-word essay is not. The two-specialist ensemble reaches 15.8% at an unchanged false-accusation rate.
- **Reproducibility of the headline numbers.** The remote observer is bearer-gated, so a reviewer cloning the repo gets the `gpt2` path, whose numbers are measurably worse.
- **`GET /api/failures` 404s on the hosted build, once per page load.** The Worker has no such route and `artifacts/confident_failures.json` is not bundled into it, so the "Where it fails worst" panel is simply absent there. The page handles it correctly — the panel stays hidden rather than claiming the detector has no failures — but the browser still logs the 404, which means `verify_ui.cjs`'s "no console errors" check now fails against the hosted site and passes locally. The fix is a product decision, not a bug fix: either serve the panel from the Worker (which republishes the essays, see below) or stop asking for it when it cannot be there.
- **`artifacts/confident_failures.json` cannot be committed** — it stores each failing document's full text, drawn from `ellipse` and `real_hybrid_hewlett`, so publishing it would republish verbatim the student writing that `data/raw/` and `data/cache/` are already excluded to protect. It is gitignored and rebuilt by `scripts/confident_failures.py`. Separately, **`artifacts/failures.json` is already tracked and carries ~110-character excerpts** of `ellipse` / `liang_toefl` / `persuade` documents in `topSentences[].text`. Quotation rather than republication, and left as it is — but it is the same corpus, and it is a call somebody should make deliberately rather than inherit.

## 8. Bug sweep, 15 August 2026

A review of the whole system — both builds, the interface, the Worker — rather than a feature.
Six defects, none of which any existing test caught, and four of which were failures of
honesty rather than of arithmetic. Deployed as `8ff98bf`; 203 Python tests, the 364,056-comparison
parity suite, and three browser suites (31 + 38 + 8) pass against production.

**1. The interface manufactured results.** `web/app.js` carried an offline analyzer. When
`POST /api/analyze` could not be reached — a 404, a non-JSON body, a dropped connection — the
page rendered a complete fabricated result through the same code that renders a real one:
verdict band, machine share, per-sentence probabilities, evidence bars with z-scores and
weights, per-token ranks, all from a seeded PRNG. For any text that was not one of the two
bundled examples the document score was `-1.6 + random() * 3.2`, so a student pasting their own
essay during a network blip could be shown "Likely machine-written" with a full evidence panel
behind it. The disclosure was the words "bundled fixture" in a metadata chip. Nothing referenced
it — no doc, neither browser harness — and it is gone; an unreachable analyzer now says so and
paints nothing. `tests/test_no_fabricated_results.py` holds both copies of the shipped
interface to that: no local analyzer, no `Math.random`, not even the mulberry32 constant the
original used to stay inconspicuous.

**2. The documentation described the wrong build**, and the guard test is why. Written up in
the open item above.

**3. A document with nothing measurable in it was cleared.** With no reliable sentence the
aggregate reports `any_machine_probability = 0.0` — the absence of a measurement, not a low one
— and zero sits below `tHuman`, so text the tool never scored a word of came back "No evidence
of machine writing" quoting a calibration derived from documents that were read. Reproduced
against the live Worker before the fix: nine spans, none measurable, cleared. `aggregate` and
`find_passages` had both learned the rule that an unmeasurable span must not decide the answer;
the band, the one line a reader acts on, had not. Both builds now abstain and name the spans
they refused.

**4. `.hidden` was not hiding two panels.** `.hidden { display: none }` is a single-class rule
near the top of `style.css`, so later same-specificity rules beat it — and `.verdict-panel` and
`.evidence-panel` are both `display: flex`. On a fresh load, before anything was typed, the
empty verdict frame (611px tall, measured) and the empty evidence frame were on screen, and
because `renderVerdict` only ever *removes* the class nothing could hide them again. The
`#notice[hidden] { … !important }` line underneath was this same bug found once and patched for
one element. `verify_ui.cjs` now asserts on computed style, because the class was present the
whole time and meant nothing.

**5 and 6, in the Worker.** `setAlarm` replaces rather than adds, and `Budget.reserve` called
it on every reservation — so under continuous traffic the cleanup alarm was pushed 24 hours
forward on each request and could never fire, and the per-IP histories it drops would grow
without bound. And `handleAnalyze` charged three neurons up front and returned 503 without
releasing them when the corpus reference failed to load, a path that never reaches Workers AI —
so a broken asset binding could eat the day's allowance one refusal at a time. An observer
failure is still deliberately not refunded: that call may or may not have been billed.

The pattern worth keeping: four of the six were places where the code did something *reasonable*
in a failure case — fall back, default to a band, keep a panel simple, keep a charge — and the
reasonable thing was a claim the tool had not earned. The tests that existed all passed, on
production, throughout.

---

*Extracted from nine Claude Code sessions, 7–12 August 2026, plus a bug sweep on 15 August.
Numbers are quoted as measured on the date shown; where a figure was later superseded, both
appear.*
