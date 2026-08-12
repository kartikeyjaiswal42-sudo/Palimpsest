# Palimpsest — an AI detector for college admissions essays

*A working application. Paste an essay in; it shows which sentences read as machine-written,
and the arithmetic that made it say so.*

---

## 1. What it is

A white-box detector for **college admissions personal statements**. It runs as a local web
application: paste an essay, get a per-sentence heat map, a calibrated verdict band, and for
every sentence the named feature contributions that sum — exactly — to the number the
classifier produced.

Three commitments distinguish it from a percentage:

**It says where.** Detection is per sentence, because the realistic case is not a wholly
machine essay. It is a paragraph a person wrote and a model later polished. Hybrid documents
with known machine spans are in the training set for that reason.

**It says why.** Every flagged sentence carries its evidence: `fluency_typicality_gap
+2.389`, `machine_word_rate +1.415`, and so on. Those contributions plus the intercept plus
a stated remainder **reconstruct the logit to floating-point equality**. It is the
computation, not a paraphrase of it.

**It refuses.** The tool answers in bands — *likely machine* / *insufficient evidence* / *no
evidence* — and refuses outright on writing outside its genre. Both refusals are calibrated
against measured error budgets, not chosen by feel.

---

## 2. Run it

**Hosted: https://palimpsest.amitynoidalibrary.workers.dev** — the whole application in one
Cloudflare Worker, observer included. See [edge/README.md](edge/README.md).

```bash
pip install -e .                                    # or: uv sync
uvicorn palimpsest.api.app:app --port 8123          # http://127.0.0.1:8123
pytest                                              # 149 tests
```

By default the observer runs remotely (below). For a fully offline, fully local run:

```bash
PALIMPSEST_OBSERVER=gpt2 uvicorn palimpsest.api.app:app --port 8123
```

### The hosted build is the same detector, and that is checked rather than claimed

FastAPI does not run on Workers, so the serving path is ported to JavaScript. Nothing is
refitted: every weight, scale, calibration knot and threshold is the artifact `train.py`
wrote. `edge/test/parity.test.mjs` compares the two implementations over **145 real
documents, 2,801 sentences, 184,688 values** — every feature, logit, calibrated probability,
evidence bar, passage, gate decision and bootstrap endpoint — with the observer's token
stream held identical between them. Every probability, logit and threshold decision is
**exactly** equal; the largest disagreement anywhere is 5e-15 on an n-gram surprisal mean.
Six mutations confirm the test can fail. In production, `scripts/verify_ui.cjs` passes 24/24
against the live URL.

**Building it found two things wrong with this project, both in the direction of overclaiming:**

* the interface footer said *"Nothing you paste leaves this machine"* while `/api/health`
  reported `textLeavesMachine: true` — the claim stopped being true when the default observer
  moved to Workers AI, and nothing caught it;
* `api/app.py` rendered the limitations panel from `evaluation.json` regardless of which
  detector it was serving, so the app stated a **17.8%** TOEFL false-positive rate where the
  served `_remote` build measures **10.9%**, and **45%** cross-model-family recall where it
  measures **64%** — plus two suites never re-run on this observer at all, presented as
  though they had been. This is the failure `_limitations()`'s own docstring warns about,
  reached by a different route: not stale prose, a stale *file*.

**Both are now fixed in both builds, and that took a second pass.** Fixing them only in the
hosted build left the Python application — the one `README` tells you to run — still
claiming locality while shipping essays to Cloudflare, and still publishing another
detector's error rates. Two identical defects fixed on one side of a port is how a port
fails, so neither correction lives in a copy any more: the panel renders from
`palimpsest.limitations`, shared by both callers, and the footer is driven by `/api/health`
rather than hardcoded, in the direction that overstates what leaves the machine if the probe
fails. `tests/test_limitations.py` (4 checks) and a 24th check in `verify_ui.cjs` fail on the
old behaviour.

A third limitation is new and only exists in the hosted build: the
observer is an fp8 mixture-of-experts model on shared hardware and is **not bit-deterministic**
— re-running 6 essays 3 times each moved a sentence score by up to **8.7 points** and document
confidence by up to **1.7**, though the verdict band held 6/6. It is measured
(`edge/artifacts/repeatability.json`) and shown to users.

---

## 3. Architecture

```
essay text
    │
    ├─► segment into sentences                          text/segment.py
    │
    ├─► OBSERVER: one forward pass, read the logits      scorer/local_lm.py   (GPT-2, local)
    │      per token: logprob, rank, entropy, μ, σ²      scorer/remote_lm.py  (qwen3-30B, remote)
    │      ── never prompted, never asked for a verdict ──
    │
    ├─► 43 FEATURES per sentence                         features/
    │      likelihood 6 · rank 6 · corpus 4 · rhythm 9
    │      register 10 · context 6 · composite 2
    │
    ├─► SENTENCE MODEL: standardise → logistic regression → calibration
    │                                                     detect/classifier.py
    ├─► DOCUMENT MODEL: aggregate sentence probabilities  detect/document.py
    │
    ├─► GENRE GATE: is this an admissions essay at all?   detect/genre.py
    │      no  → "Outside this tool's scope", no verdict
    │
    └─► BANDS: likely machine / insufficient / no evidence
           thresholds from Clopper-Pearson error budgets  scripts/fit_bands.py
```

### Components

| module | role |
|---|---|
| `scorer/local_lm.py` | GPT-2 124 M read for per-token statistics. One forward pass, sliding window, no `generate()` |
| `scorer/remote_lm.py` | Same contract against a 30 B model via Workers AI `prompt_logprobs`; disk-cached |
| `scorer/ngram.py` | N-gram reference built from applicant prose — the corpus the essay is compared *against* |
| `features/` | 43 features in 7 groups, each with a registered label, unit and expected direction |
| `detect/classifier.py` | Sentence-level logistic regression + probability calibration |
| `detect/document.py` | Document aggregation: machine share, any-machine probability, passages |
| `detect/genre.py` | The scope gate (§6) |
| `api/app.py` | FastAPI: `/api/analyze`, `/api/health`, `/api/features`; serves `web/` |
| `web/` | The interface: heat map, evidence bars, band |

26 scripts (fetch, generate, train, evaluate, and the experiment harnesses), 8 test modules,
9 design documents.

### The observer is an instrument, not an oracle

This is the line the brief draws and it is drawn deliberately here.

The language model is called **once per document, in a single forward pass, and its logits
are read**. There is no prompt, no chat template, no `generate()` call, and no place for an
opinion to enter. `raw: true` on the remote path is load-bearing: without it the essay is
wrapped in a chat template and the numbers would describe a conversation.

Enforced, not asserted — [`tests/test_no_generative_calls.py`](tests/test_no_generative_calls.py)
forbids importing `openai`, `anthropic`, `cohere`, `litellm`, `ollama`,
`google.generativeai`, and `/api/health` reports `usesGenerativeModel: false`.

**Two observers, and the comparison is a finding.** GPT-2's statistics turn out to be
*inverted* against English-learner writing (AUROC **0.132** — it ranks non-native prose as
*more* machine-like than actual machine prose). That is the mechanism behind the
false-positive problem the whole field has. A 30 B observer corrects the sign everywhere and
lifts Claude Haiku sentence recall from 0.105 to 0.577. The remote observer sends essay text
to Cloudflare, which `/api/health` states explicitly (`textLeavesMachine: true`); `gpt2`
restores full locality at a measured cost.

---

## 4. How it was built

**The corpus is part of the work.** ~12 sources, each with provenance, count, licence and a
statement of what it does not cover — [docs/02-dataset.md](docs/02-dataset.md).

* **Human, in domain:** Liang et al. real admissions essays (70), Johns Hopkins *Essays That
  Worked* (31, filtered to publication ≤ 2022-11-30 — the year tags encode graduating class,
  not publication date, and filtering on them would mislabel the corpus by four years).
* **Machine, in domain:** Liang et al. GPT-3.5 essays (31), plus 135 generated with
  gemini-3.1-flash-lite and ~400 with four Claude checkpoints, on subjects **pre-registered
  before generation** so no subject crosses a split.
* **Hybrid:** part-human/part-machine documents with known character spans — the realistic
  attack, and the reason detection is per sentence.
* **Held out:** an ESL false-positive study (TOEFL, ELLIPSE, PERSUADE with its ELL flag), a
  domain-shift set, an adversarial set, and a control that holds the generator fixed while
  removing subject steering.

**Machine text is real model output, never our imitation of it.** The first machine corpus
was hand-composed and it was worthless — [docs/04-failures.md](docs/04-failures.md#4).

**Signals were selected by measurement.** Perplexity alone does not work and was measured not
assumed. Feature groups are reported separately, and single-feature AUROCs are published so a
reader can see which signals carry weight and which are decoration.

---

## 5. How good it is

### End to end — where documents actually land

Held-out sets only, the odd half (the even half calibrated every threshold):

| set | n | likely machine | insufficient | no evidence | out of scope |
|---|---|---|---|---|---|
| **MACHINE** | | | | | |
| Gemini flash-lite | 57 | **80.7%** | 19.3% | 0.0% | 0.0% |
| Gemini flash (unseen family) | 11 | 9.1% | 90.9% | 0.0% | 0.0% |
| Claude (haiku/sonnet/opus/fable) | 80 | 0.0% | 85.0% | 15.0% | 0.0% |
| **HUMAN** | | | | | |
| ESL (TOEFL + ELLIPSE + PERSUADE) | 305 | **1.6%** | 67.2% | 3.0% | 28.2% |
| domain shift | 44 | **0.0%** | 65.9% | 29.5% | 4.5% |

**Human essays called machine-written: 5 / 349 = 1.43%**, against a stated 5% budget.

An earlier build of the gate reached 0.00% here, and it was not better. It hit zero by
refusing 5% of *genuine* admissions essays — including the application's own showcase
example, which lost by 0.015 of a threshold. Refusing to score somebody is a claim too, so
the gate's false-refusal rate is now bounded the same way every other threshold in this
project is: **99.6% of real admissions essays are scored, 95% upper bound on wrongly
refusing one held under 2%.** The cost is that fewer out-of-genre documents are refused
(22.5%, down from 76.1%) and 1.43% of human essays reach a machine verdict — inside budget,
and the right direction to err, because a document the gate misses still falls through to the
bands while a wrongly refused essay is a dead end with no recourse. §8 shows the measurement
that justified paying the last 0.28 points of that.

### Classifier metrics

| | GPT-2 observer | 30 B observer |
|---|---|---|
| out-of-fold sentence AUROC | 0.925 | **0.945** |
| ESL document FPR (identical documents) | 0.080 | **0.074** → 0.039 after recalibration |
| Claude Haiku sentence recall | 0.105 | **0.577** |
| Claude Opus sentence recall | 0.027 | 0.083 |

### The ceiling, stated plainly

**Frontier prose is not detectable by this method and we measured how badly.** With a 250×
larger observer, Claude Opus reaches AUROC 0.637 and **0% recall at a 5% false-accusation
budget**. The capability gradient is monotone — haiku 0.955 → sonnet 0.735 → opus 0.631 —
and it survived every repair attempted. [docs/09-frontier-ceiling.md](docs/09-frontier-ceiling.md).

Consequence built into the product: **a low score is not evidence of a human author.** The
bottom band is worded *"no evidence of machine writing"*, never *"human"*, nothing in the UI
is coloured green, and `canExonerate` is `false` on every response.

---

## 6. The genre gate

The detector fails *confidently* outside its genre, not gracefully. Two measurements:

* trained on argumentative coursework and asked about admissions essays, a classifier flags
  **91.4% of real human ones**;
* pointed the other way, the shipped detector called a real school student's essay
  machine-written at **97% confidence** — an essay that opens *"If extracurricular activities
  were mandatory what would you do? I would play a sport like baseball or basketball."*

Simple, formulaic, repetitive prose is low-perplexity prose, and low perplexity is what the
detector reads as machine. The writers who produce it are younger, weaker, and second-language
writers — so an out-of-domain input yields a wrong answer aimed at whoever can least contest it.

The gate is four document-level features (distance from the applicant-prose reference, novel
trigram rate, first-person rate, specificity) → logistic regression, thresholded so the 95%
upper bound on refusing a *genuine* admissions essay stays under 2%. It scores **99.6% of
known admissions essays** and refuses **22.5% of other genres**.

**Two features were removed, each after a measured failure, and both removals are guarded by
a test that fails on the old gate.**

*Document length.* The first version included it, where it took the third-largest weight
because the out-of-domain sets are much shorter than admissions essays. A browser caught the
consequence: truncating genuine essays — same author, same genre, fewer words — flipped the
gate from 0/6 refused at 700 words to 5/6 at 150. Supplemental prompts are routinely capped
at 250–350 words, so that gate would have refused real submissions while calling itself a
genre check. It is the same length artifact [docs/04-failures.md](docs/04-failures.md)
already records, reached by a different route.

*Sentence rhythm.* `mean_sentence_words` survived the length audit honestly — truncation
barely moves it — but [`scripts/esl_gate_probe.py`](scripts/esl_gate_probe.py) found it was
the feature most correlated with measured **English proficiency** (r = −0.243 within a single
genre), because a struggling writer produces run-ons, not short sentences. It was importing
proficiency into a gate that claims to read genre. See §8 for what settled the trade.

**It cannot weaken detection**: it never touches the sentence model, document model or their
thresholds — it only decides whether they are consulted, and **0.0%** of both in-domain
machine sets are refused.

**What it does not catch, found in a browser.** The gate is fitted to separate admissions
essays from *student* writing — argumentative coursework and TOEFL responses — because that
is the out-of-domain data we hold. Given a genre it has never seen it extrapolates
confidently and wrongly: a paragraph of transformer documentation scores **0.985 in-domain**.
The retired five-feature gate scores the same text **0.988**, so this is not a cost of the
removal above; it is a property of fitting a logistic model against two genres and deploying
it against all of them. Nothing downstream misfires — the bands still answer on the evidence —
but "out of scope" should be read as *"unlike the student writing we tested"*, not as a
general-purpose genre classifier.

**The validation that licenses it to ship:** the gate is fitted with *both* human and machine
admissions essays in-domain, and its pass rates for the two are **100.0% vs 99.5%, a 0.5%
gap**. A gate correlated with authorship would be a second detector wearing a scope label,
laundering low recall into high abstention. `fit_genre_gate.py` refuses to save if that gap
exceeds 10%.

---

## 7. Where it fails

[docs/04-failures.md](docs/04-failures.md) opens with **three essays it gets confidently
wrong**, each with the mechanism:

1. a one-sentence ESL essay called machine at **P = 0.977**;
2. a letter to a teacher, **P = 0.976**;
3. a TOEFL essay on leadership, **P = 0.953**.

Then six larger mistakes, including two that invalidated earlier results: a **0.988 AUROC
that was partly a smart-quote detector**, and a length artifact that landed on exactly the
wrong people. Both were found by us and both are recorded rather than quietly fixed.

A third, from this build: a supervised stylometric classifier scored **AUROC 1.000** on
held-out Claude Opus and detected nothing — it flagged 0% of real GPT-3.5 essays from another
collection and 17% of real students. It had learned *which pipeline produced a file*. Four
successive controls (typography, length, corpus, genre) failed to break it; only refusing to
grade it on our own corpus did.

---

## 8. Fairness

Yes, it flags English-learner writing, and we found out why rather than only that.

[docs/05-esl.md](docs/05-esl.md) reports false positives by language background, by measured
proficiency (ELLIPSE holistic score), and in a **matched control** — PERSUADE, same prompts,
where the ELL flag is the only difference. This build adds the mechanism: GPT-2's likelihood
and rank statistics are *inverted* against ESL prose, so "low perplexity ⇒ machine" points
directly at non-native writers. Replacing the observer corrects the sign.

### The missing cell, and how far it could be closed

We hold native-authored admissions essays and we hold English-learner writing, but **no
ESL-authored personal statements** — the cell where those meet is empty. An earlier draft of
this document called that untestable and stopped. That was too quick a surrender: the *effect*
of low proficiency can be measured in the data we do hold, then applied to the admissions
essays we do hold. [`scripts/esl_gate_probe.py`](scripts/esl_gate_probe.py) does this three
ways, in increasing order of what they assume.

**1. A matched control finds nothing.** PERSUADE has the same prompts written by ELL-flagged
and non-ELL students, so differencing *within* a prompt holds genre, topic and task constant.
The gate's log-odds shift is **+0.079 (95% CI −0.336 to +0.367)** — indistinguishable from
zero. But this splits a binary flag over 24 documents, and low power is not evidence of
absence.

**2. A graded score finds something.** ELLIPSE carries a holistic 1.0–5.0 proficiency score
over 260 documents of one genre. P(in-domain) tracks it: **Spearman ρ = +0.169, permutation
p = 0.008**. The two results are not in conflict — a binary split of mostly-stronger writers
would miss a continuous effect, which is exactly the pattern seen.

**3. The transplant, which is the product question.** Regressing each gate feature on
proficiency and shifting real admissions essays down two points asks: if a personal statement
carried the signature of a much weaker English writer, would it still be scored? **3.32%
refused (95% CI 1.85–4.80) against a 0.74% baseline.** This assumes additivity and
cross-genre transfer, neither verifiable here. It bounds the risk; it does not measure it.

**What this changed.** On the retired five-feature gate the same transplant refused **9.96%**,
and **100%** of the weakest proficiency band. Removing `mean_sentence_words` cut those to
3.32% and 25%.

**And what it cost, stated plainly.** That removal moved the headline false-accusation rate
from 1.15% to **1.43%** — documents the gate used to refuse now reach the detector, and some
are flagged. Trading refusals for accusations inside the population this project exists to
protect is not self-evidently an improvement, so it was settled by measurement rather than
preference. [`scripts/gate_selectivity.py`](scripts/gate_selectivity.py) asks what the extra
refusals *buy*: of the documents a gate refuses, how many were heading for a false accusation?

| gate | refused | accused | refusal precision | refusals per accusation avoided |
|---|---|---|---|---|
| 4-feature (shipped) | 25.2% | 1.43% | 4.5% | 22.0 |
| 5-feature (retired) | 31.5% | 1.15% | 4.5% | 22.0 |

Against a **2.58%** base rate, both gates refuse at the *same* precision and the *same* cost.
The extra 6.3 points of refusal are not selective — they buy roughly one avoided accusation
per 22 people denied an answer, and they fall hardest on the weakest-English writers. That is
what decided it.

**The honest residue.** Refusal precision of 4.5% against a 2.58% base rate means the gate is
only weakly aimed. Its real justification is the cross-genre case in §6 — turning confident
wrong answers into honest refusals — not efficient harm prevention on ESL prose. And the empty
cell is still empty: no arithmetic here creates an ESL-authored admissions essay. Sourcing
even fifty would tell us more than any of the above.

---

## 9. Compliance with the brief

| requirement | status |
|---|---|
| Working application, real interface — not a script or notebook | ✅ FastAPI + web UI |
| Paste an essay; shows **which parts** | ✅ per-sentence spans, probabilities, passages |
| …and **why** | ✅ named contributions reconstructing the logit exactly |
| Not "73% AI" | ✅ bands + per-sentence evidence + explicit abstention |
| **Not a wrapper** — model must not make the judgement call | ✅ single forward pass, logits only; enforced by test |
| Detection at sentence/passage level (human text later polished) | ✅ hybrids with known spans in training |
| Dataset sourced, documented, gaps stated | ✅ [docs/02-dataset.md](docs/02-dataset.md), licences per source |
| **Three essays it gets confidently wrong, with reasons** | ✅ [docs/04-failures.md](docs/04-failures.md) Part 1 |
| **ESL false positives — did you spot them** | ✅ [docs/05-esl.md](docs/05-esl.md) + mechanism |

**On "a small local model".** The brief permits a language model as an instrument and draws
one line: *the model must not make the judgement call while the app relays the verdict.* That
line is not crossed — the observer is never asked anything. The default observer is remote for
measured accuracy reasons; `PALIMPSEST_OBSERVER=gpt2` gives the fully local configuration, and
the comparison between them is reported rather than hidden.

**Licensing.** Every source carries a licence in the dataset doc. PERSUADE is CC BY-NC-SA 4.0
(**non-commercial** — fine for evaluation, a genuine constraint on any paid product). DAIGT
declares **no licence** on its mirror; it is used only as a diagnostic, appears in no
`trainSources`, and is **not redistributed** in this repository.

---

## 10. What this does not cover

* **Frontier models.** Opus/Sonnet/Gemini-Pro-class prose: 0% recall at a defensible budget.
* **Paraphrase and "humanizer" attacks.** Not in the corpus. Published benchmarks find they
  hurt badly, so every number above is an **upper** bound on adversarial performance.
* **Genres other than admissions essays.** Refused by design — but the gate was fitted
  against *student* writing, and extrapolates confidently on genres it never saw (§6).
* **ESL-authored admissions essays.** No data. §8 bounds the risk by transplanting a measured
  proficiency effect (3.32% refused at a two-point drop); it does not measure it.
* **Languages other than English.**
* **Any claim that a person wrote something.** The tool cannot exonerate, and says so.

---

## 11. Documents

| | |
|---|---|
| [01-approach.md](docs/01-approach.md) | why these signals |
| [02-dataset.md](docs/02-dataset.md) | every source, count, licence, gap |
| [03-evaluation.md](docs/03-evaluation.md) | protocol and results |
| [04-failures.md](docs/04-failures.md) | three confident errors + six larger mistakes |
| [05-esl.md](docs/05-esl.md) | the false-positive study |
| [06-decisions.md](docs/06-decisions.md) | choices and what they cost |
| [07-ai-usage.md](docs/07-ai-usage.md) | how AI was used to build this |
| [08-cross-vendor.md](docs/08-cross-vendor.md) | it detects small models, not machines |
| [09-frontier-ceiling.md](docs/09-frontier-ceiling.md) | the ceiling, measured |
