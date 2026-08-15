# What was built, what was measured, and what was thrown away

*A record for anyone assessing this project. Every number here comes from an artifact in
`artifacts/` that a script regenerates; none is typed by hand. Where a claim could not be
checked, it says so.*

---

## 1. The short version

Three new signal families were built and measured. **One works, one is a length feature in
disguise, and one is a near-null.** All three are reported, because a project that only
publishes the family that worked is not measuring, it is selecting.

| family | verdict | evidence |
|---|---|---|
| **Structural** — dependency-tree depth, stop-word load, POS rhythm | **works**, and survives four controls | `artifacts/syntax_probe.json`, `syntax_controls.json` |
| **POS trigram entropy** — explicitly requested | **fails**: r = +0.88 with sentence length, R² = 0.872 against existing features. Length wearing an information-theory label. | `artifacts/syntax_probe.json` |
| **Cross-perplexity (Binoculars)** — the strongest published frontier method | **near-null**: 0.542 strength, recall *below* its own false-positive budget | `artifacts/binoculars_probe.json` |

---

## 2. The result that survived

Eleven structural features, joined onto the existing feature matrices so the 43 values that
produced every previously published number are untouched.

**The honest headline is the cross-pipeline number, not the in-pipeline one:**

| | AUROC gain |
|---|---|
| in-pipeline (grouped CV, our own generators) | +0.0124 |
| **cross-pipeline (fit on one pipeline, tested on another's machine text)** | **+0.0083**, 95% CI +0.0059 to +0.0109 |

And the direction that matters more than accuracy — false accusations of second-language
writers, measured at a matched 5% budget on 3,034 held-out ESL documents:

| | base | + structural |
|---|---|---|
| ESL essays **accused** | 14.80% | **6.43%** |
| ESL essays **highlighted** (what a reader actually sees) | 45.35% | **34.71%** |
| Gemini recall at *identical* ESL harm | 83.5% | **98.3%** |
| Claude (frontier) | 0.693 | 0.647 — no help, slightly worse |

Fewer innocent students accused **and** more machine text caught, at the same harm level.
That combination is unusual enough to be worth doubting, which is why §3 exists.

---

## 3. The controls, which could have killed it

`docs/12` records what happened last time this project found a large gain: a cross-observer
signal moved frontier prose from 0% recall to 100%, survived a typography control at 0.967,
and then **fell to 0.490 — chance — on machine essays somebody else generated.** It had
learned our generation pipeline.

So the structural block was graded the same way, by `scripts/syntax_controls.py`:

| control | result |
|---|---|
| **Cross-pipeline** (the decisive one) | **survives.** Fit on Liang's GPT-3.5 essays, tested on ours: +0.0083 (CI excludes zero). Nothing collapsed; both arms stay above 0.91. |
| **Collinearity** | the features carrying the signal are the ones *not* already in the model — `pos_trigram_surprisal` R² 0.22, `local_depth_burstiness` 0.33. Meanwhile `pos_trigram_entropy` sits at **0.872**, confirming from a third angle that it is length. |
| **Length** | truncate every document to a common 24 sentences: gain is +0.0099 against +0.0102. Essentially untouched. |
| **Foreign human** | on human prose from a collection never fitted on, the augmented arm flags **3.03% vs 4.57%**. Fewer false accusations, not more. |

**What the controls do not settle:** the ESL fairness figures were measured in-pipeline like
everything else. The foreign-human control points the right way but is a different *genre*
as well as a different collection, so it bounds the risk rather than isolating it.

---

## 4. Three results this project killed itself

The honesty case does not rest on the successes.

1. **A 0.988 AUROC that was substantially a smart-quote detector** (`docs/04`) — 83% of human
   documents carried curly apostrophes against 0% of every machine set.
2. **A 0.960 AUROC cross-observer signal, killed by its own control battery** (`docs/12`) —
   0.490 on a foreign corpus. Chance.
3. **Binoculars, run for the first time and reported as a null** (`docs/13`) — 0.542 strength,
   and 4.4% machine recall at a 5% false-accusation budget, meaning **it accuses machines
   less often than it accuses people.**

The third is the most useful of the three, for a reason worth stating: the first two were
*fitted*, so overfitting was always an available explanation for their collapse. **Binoculars
is fitted to nothing** — no training set, no learned threshold, no coefficients. It hit the
same frontier wall anyway. So the ceiling this project keeps reporting is **not a fitting
artefact**, a claim neither `docs/09` nor `docs/12` could make on their own.

A fourth finding in the same spirit: the run put **frontier Gemini below the human sources**
and made **TOEFL — real second-language student writing — the most machine-like source of
all**, from an instrument that never saw our corpus or our labels. The fairness problem now
has corroboration from a method with no way to have learned it.

---

## 5. A bug found in our own shipped detector

Building the failure panel surfaced it. The single worst false accusation in the corpus — a
second-language student's essay called machine-written at **P = 1.000** — is driven by one
feature contributing **+55.87**, roughly six times the entire rest of the sum.

`features/context.py` clips every in-document z-score to ±6, with a stated reason: *"one
feature should not be able to dominate the logit."* `style_gap_from_doc` is computed in the
same function a few lines below and **is not clipped.** Measured over 50,875 ESL sentences:

| feature | max observed |
|---|---|
| `logprob_z_in_doc` | 6.00 |
| `len_z_in_doc` | 6.00 |
| `tree_depth_z_in_doc` | 6.00 |
| **`style_gap_from_doc`** | **64.67** |

It is deliberately **not fixed**: clipping it changes shipped output, which invalidates the
364,056-value parity test, every published number and the fitted bands. That is a decision
with a measurement in front of it, not a patch. The evidence is recorded so the decision can
be made deliberately.

---

## 6. Seven mistakes made during this work, and corrected

Included because a record that contains no errors is not a record of real work.

| mistake | how it surfaced |
|---|---|
| ESL audit compared two differently-calibrated models at a *fixed* threshold, reporting a real gain as a "threshold move" | corrected to matched-harm recall + threshold-free AUROC |
| A test asserting "z-scores are capped" **passed with the cap deleted** — its baseline was all-identical, hitting a different code branch | rebuilt on a baseline with real spread; mutation-verified |
| The Colab notebook offered a `LOAD_IN_8BIT` flag that was dead code — the scorer takes no quantization parameter | removed rather than wired up, with the numerical reason stated |
| Three places told the reader to run `syntax_probe.py` after joining Binoculars — that script never reads `binoculars_score` | all three repointed; it would have printed a real number about a different question |
| The Colab guide said **"if the source medians come out mixed, abort"** — they came out mixed and the mixing *was* the result | advice deleted; only a known-answer probe now gates run validity |
| The redaction guard **rejected a correct build**, asserting no essay slice appeared in the output — but the quoted sentence *is* a slice by design | rewritten to measure *how much* survives, not whether any does |
| The controls verdict failed on a bare sign test, calling a **−0.0026 at a 0.9927 ceiling across 31 documents** a failed control | now reports headroom, sample size and bootstrap CIs; distinguishes a ceiling from a collapse |

The last three were caught by guards written earlier in the same work.

---

## 7. How it was verified

| | |
|---|---|
| Python test suite | **194 passing** |
| New tests for the structural block | 18, **proved able to fail** via three separate mutations |
| Python ↔ JavaScript parity | **364,056 comparisons**, 262 documents, 5,527 sentences — every feature, logit, probability and threshold decision |
| Real browser | local and hosted builds walked at 1280px and 390px, 0 console errors |
| Production | deployed, rollout confirmed single-version-100% before testing, all routes verified on the live URL |

Tests are **proved able to fail** rather than assumed to work: deleting the z-score cap,
returning 0.0 for a degenerate baseline, and replacing NaN with 0.0 each redden exactly the
checks that should catch them.

---

## 8. What is deliberately not shipped

**The structural block is measured and not wired into the live detector.** Not an unfinished
edge — a constraint:

The hosted build is **2,318 lines of hand-written JavaScript with zero dependencies**,
reproducing the Python pipeline to 15 decimal places. The structural features need a
dependency parse from spaCy's neural model. There is no JavaScript equivalent, and spaCy
cannot run in a Cloudflare Worker. Shipping the block locally would leave the laptop app and
the hosted app as **two different detectors** — the exact drift `PROJECT.md` §2 records four
separate incidents of.

This follows the precedent `docs/12` already set for the polish head: a measured gain, left
out, because a second accusation surface needs its own error budget in front of the reader
before it is allowed to accuse anyone.

**Also not established:** these features were built on the hypothesis that structure survives
paraphrase better than word choice. **That hypothesis remains untested** — the corpus holds
no humanizer or paraphrase attack, so every number here is an upper bound on adversarial
performance, and the motivating claim is not evidence.

---

## 9. Scale

| | |
|---|---|
| new code in this stream | ~3,400 lines across 12 modules, scripts and test files |
| corpus | 8,234 documents / 3.81 M words, composition regenerated by `scripts/dataset_report.py` |
| documents scored for the Binoculars run | 3,860 documents / 69,147 sentence spans |
| measurement artifacts | 54 JSON files, each regenerated by a named script |
| dataset integrity findings | 2 unreadable corpus sources and 525 duplicate texts, neither previously recorded |

The dataset report has a `--check` mode that **fails the build when the counts no longer
match the corpus on disk** — because the failure it exists to prevent already happened here
once: the interface published a 17.8% false-positive rate for a build that measured 10.9%.

---

## 10. Where to look

| | |
|---|---|
| [13-structural-features.md](13-structural-features.md) | the full account of all three families |
| [12-consensus-and-polish.md](12-consensus-and-polish.md) | the result that was killed by its own controls |
| [04-failures.md](04-failures.md) | three essays the detector gets confidently wrong |
| [05-esl.md](05-esl.md) | the false-positive study by language background |
| [09-frontier-ceiling.md](09-frontier-ceiling.md) | the ceiling, measured |
| `artifacts/syntax_controls.json` | the four controls, with intervals |
| `/api/failures` on the live build | the three worst mistakes, with the arithmetic that made them |
