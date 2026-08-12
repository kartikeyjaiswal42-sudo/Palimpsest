# A bigger observer fixes the cheap tier and does not fix the frontier

docs/08-cross-vendor.md ended with a diagnosis: the detector reads *register*, not authorship,
and the variable that predicts detectability is model **capability**. It left the obvious
repair untested — the instrument is GPT-2 (124 M, 2019), so perhaps the signal was never
measured rather than never present.

This document reports that test. The answer is half yes, and the half that is no is the half
that decides what can be sold.

## What was built

`observer-worker/` — a Cloudflare Worker exposing one thing: Workers AI's `prompt_logprobs`,
which **scores a given text** instead of generating one, returning each position's
log-probability and true rank. `raw: true` is load-bearing; without it the essay is wrapped
in a chat template and the numbers describe a conversation.

`src/palimpsest/scorer/remote_lm.py` — its client, with an on-disk cache keyed by (model,
text) so iterating on features never re-spends the allowance.

The observer is **qwen3-30b-a3b**, roughly 250× GPT-2. It cannot run on the 8 GB laptop this
project is developed on, which is the whole reason it runs on Cloudflare. A local
Binoculars-style two-model pair (Qwen3-0.6B base + instruct, `scorer/binoculars.py`) was
written first and abandoned unrun: the machine was found sitting at 12.4 GB of swap before
loading anything, and a 0.6 B observer is only 5× GPT-2 — too small to be worth the risk.
The module is kept because it is correct and will be useful on a machine with headroom.

Cost: **3.1 neurons per 571-word essay**, against a free allowance of 10,000/day. About
3,200 essays a day, free.

## Result 1 — GPT-2's statistics are *inverted*, and that is the ESL bug

Same 192 essays, both observers, one raw statistic at a time, **nothing fitted**. AUROC
against a human reference that deliberately pools native college essays with TOEFL writing.

| generator | GPT-2 124 M | qwen3-30 B |
|---|---|---|
| claude_haiku | 0.707 | **0.955** |
| gemini_flash_lite | **0.132** | 0.895 |
| claude_sonnet | 0.427 | 0.795 |
| gemini_flash | 0.175 | 0.742 |
| claude_opus | 0.458 | 0.695 |

The sub-0.5 entries are not noise. They are *backwards*: under GPT-2, machine essays look
**more human** than the human reference. The mechanism is the one docs/05-fairness.md
worried about, caught in the act — simple ESL prose is low-perplexity to a 2019 model,
modern LLM prose is not, so "low perplexity ⇒ machine" points at the TOEFL writers.

This is the root cause of the false-positive problem, and it means the shipped detector was
never using its model-based features the way the literature assumes. It works because the 29
surface features (`tricolon`, `hedge_rate`, `machine_phrase_rate`, …) compensate — which is
an independent confirmation of the register diagnosis in docs/08.

The 30 B observer corrects the sign everywhere.

## Result 2 — fusion recovers the cheap tier and not the frontier

`scripts/fusion_probe.py`. Seven statistics, logistic regression, **leave-one-generator-out**:
the generator being scored is held out entirely, because in production the next model is
always one you did not fit on. Significance by permuting test labels against fixed scores.

| held-out generator | AUROC | **TPR @ 5% FPR** | p |
|---|---|---|---|
| claude_haiku | 0.949 | **0.800** | <0.0001 |
| gemini_flash_lite | 0.749 | **0.000** | 0.0005 |
| claude_opus | 0.637 | **0.000** | 0.044 |
| claude_sonnet | 0.616 | **0.000** | 0.067 |
| gemini_flash | 0.604 | **0.000** | 0.124 |

**Read the third column, not the second.** AUROC asks whether machine essays rank above
human ones on average. The product question is whether the detector catches anything at a
false-accusation rate a customer could defend. Those two answers disagree here, and only one
of them is the product.

Claude Opus at AUROC 0.637 is statistically distinguishable from human writing (p = 0.044)
and commercially indistinguishable from it (0% recall at 5% FPR). Both facts are true and
only stating the first would be a lie by selection.

## The verdict

A 250× larger observer bought a genuine, large improvement **at the cheap end**: Haiku went
from roughly a tenth of sentences flagged to 80% document recall at a 5% false-positive
budget, on a generator never seen in training. That is a real detector and it is honestly
sellable.

It bought **nothing usable at the frontier**. Opus, Sonnet and mid-tier Gemini all sit at 0%
recall at 5% FPR. The capability gradient from docs/08 survived the repair that was supposed
to explain it away, which promotes it from "strongly supported" to the working model of what
this technique can do.

So the ceiling is not an artifact of a weak instrument. Perplexity-family detection of
frontier prose in a ~650-word essay is at or near its limit, and this is consistent with the
formal result that detectability degrades toward chance as a generator approaches the human
distribution, with **document length** the main remaining lever.

## Result 3 — the stylometric detector scores 1.000 and detects nothing

Statistics are a narrow model class, so the obvious next question is whether a classifier
reading raw text finds a fingerprint the statistics cannot express. `scripts/stylometry_
probe.py`: TF-IDF character 2-4-grams and a function-word-only control, logistic regression,
leave-one-generator-out.

It reported **AUROC 1.000 and 100% recall at 5% FPR on held-out Claude Opus**.

That number is worthless, and the sequence of controls it took to prove that is the most
transferable thing in this document.

1. **Typography.** 90% of Liang's human college essays contain a curly apostrophe and 80%
   contain curly quotes. **0%** of every generated set does — our generation pipeline folded
   them to ASCII. A character n-gram model finds `’` immediately. Fixed by normalising both
   classes identically.
2. **Length.** TOEFL essays run ~104 words, our generations ~640. The function-word
   representation encodes token count directly. Fixed by truncating every document to 300
   words.
3. **Corpus.** Holding out a second *Liang file* proves nothing, because the collection's
   conventions remain in training. Fixed by holding out PERSUADE — an independent collection.
4. **Genre.** Every remaining human corpus is school coursework; every machine essay is a
   polished personal-narrative admissions essay. Fixed by scoring against
   `liang_college_human`, real admissions essays, alone.

After all four controls the number was *still* 1.000. What finally broke it was refusing to
grade the classifier on this corpus at all:

| probe, threshold at 5% FPR on held-out human admissions essays | n | flagged machine |
|---|---|---|
| **MACHINE — Liang's own GPT-3.5 essays (foreign pipeline)** | 10 | **0.0%** |
| MACHINE — Liang's GPT-3.5, prompted to evade | 5 | 0.0% |
| HUMAN — hewlett, other genre | 65 | 7.7% |
| HUMAN — ellipse (ESL) | 999 | **15.2%** |
| HUMAN — persuade | 795 | **17.4%** |

A detector that flags 0% of real machine essays and 17% of real students is not a detector.
It learned **which pipeline produced a file**, and every internal split preserved that cue
because every machine essay in this project came from one generation pipeline.

The lesson generalises past this script: **on this corpus, a supervised model over raw text
cannot be trusted at any accuracy, because provenance is perfectly confounded with label.**
Only cross-pipeline evaluation can tell detection from bookkeeping. Any future supervised
attempt must be graded on machine text somebody else generated.

## What follows from this

Three things are worth doing and one is not.

**Not worth doing:** buying a bigger observer still. 124 M → 30 B moved Opus from 0.458 to
0.695 AUROC and 0% recall to 0% recall. A 70 B model will not close a gap that shape.

**Worth doing, in order:**

1. **Ship the cheap tier honestly, with mandatory abstention.** Most real-world essay
   cheating is free-tier output pasted unedited, which is the column that works. The product
   must have an explicit *insufficient evidence* band, and frontier-quality text must land in
   it rather than being scored confidently at 27% and read as an acquittal — which is what
   the live UI did to the two essays that prompted this work.

2. **Change the question from attribution to consistency.** "Was this written by a machine?"
   is the hard question. "Was this written by the same person who wrote that?" is a much
   easier one, it is robust to model capability because it never models the machine at all,
   and it matches what a school or admissions office actually holds: prior work by the same
   student. This is the most promising untried direction in the project.

3. **Use more text per decision.** Length is the one lever the theory leaves open. A single
   650-word essay is near the floor; a portfolio is not.

## Result 4 — the 30 B rebuild: a real gain at mid tier, none at the frontier

The shipped detector was refitted on the remote observer. 40 features rather than 43
(`mean_entropy`, `entropy_sd`, `curvature` are dropped, not imputed — see
`scorer/remote_lm.REMOTE_UNAVAILABLE`), same `Analyzer`, same training sets, same protocol.
Out-of-fold sentence AUROC 0.925 → **0.945**.

Sentence flag rate on held-out Claude essays, by generating model:

| model | GPT-2 | qwen3-30 B | |
|---|---|---|---|
| haiku | 0.105 | **0.577** | 5.5× |
| sonnet | 0.067 | **0.353** | 5.3× |
| fable | 0.046 | **0.259** | 5.7× |
| opus | 0.027 | **0.083** | 3.1× |

Document recall: `modern_unseen_family` (mid-tier Gemini) 0.455 → **0.546**;
`modern_holdout` (cheap Gemini, seen generator) 0.948 → 0.922; `modern_claude` 0.000 →
0.044.

False positives, both detectors scored on the SAME 175 held-out ESL documents — the first
comparison attempted was invalid, having run 395 documents against 175 because the remote
feature build was capped to save neurons:

| observer | sentence FPR | document FPR |
|---|---|---|
| GPT-2 124 M | 0.212 | 0.080 |
| qwen3-30 B | 0.279 | **0.074** |

So the rebuild buys a 3–5× sentence-recall gain across every Claude tier at slightly *lower*
document-level false-positive cost. **Opus document recall is still 4.4%, i.e. nothing.** The
ceiling in Result 2 stands.

One thing this exposes that is not about observers: **both detectors miss the 5% document
FPR budget on held-out ESL writing** (8.0% and 7.4%). The operating point is calibrated on
the even half of the at-risk sets and does not hold on the odd half — the calibration sample
is too small for the quantile to be stable. That is a live defect in the shipped product,
independent of everything else in this document.

## Result 5 — supervised stylometry works inside a genre and collapses across one

`scripts/fetch_external.py` pulls DAIGT: **4,189 student essays on PERSUADE prompts from 17
independent generation pipelines** (`Intel-neural-chat`, `kingki19_palm`, `text-davinci-001`,
`darragh_claude_v6/v7`, `Mistral7B_CME_v7`, …), each contributed by a different person. This
is the corpus Result 3 said was missing.

**A. Leave-one-pipeline-out, within DAIGT.** Median AUROC **1.000**, median TPR@5%FPR
**1.000** on character n-grams; 0.94–0.99 on function words alone. The held-out producer is
genuinely independent, so — unlike Result 3 — the provenance shortcut is gone by
construction. Supervised detection of an unseen generator is real.

**B. Train on DAIGT only, test on our corpus.** Nothing shared with training at all:

| probe | flagged machine | expected |
|---|---|---|
| MACHINE gemini_flash_lite | 100.0% | high |
| MACHINE claude_opus | 34.8% | high |
| **HUMAN liang_college_human** | **91.4%** | **≤5%** |
| HUMAN ellipse_esl | 74.2% | ≤5% |

It flags 91% of real human admissions essays. The model learned *argumentative coursework
written by a student* versus *everything else*, and our human essays are a different genre,
so they fall on the machine side.

**Both halves matter and they must be read together.** A supervised detector generalises
across *generators* far better than the perplexity route does — and it does not survive a
change of *genre* at all. So a supervised detector is only deployable on the exact genre it
was fitted to, and shipping one general model across admissions essays, coursework and lab
reports would produce the 91% number above on whichever genre it was not trained on.

That is a product constraint, not a research curiosity: **the system must know what genre it
is looking at and refuse the ones it was not fitted for.**

## What is still not true

* This measured **document-level** separation on 192 essays with a ~25-document human
  reference per split. The 5% FPR threshold is read off ~25 points and is correspondingly
  noisy; treat 0.800 as "large" and 0.000 as "nothing", not as three-decimal quantities.
* No adversarial condition was tested here. Paraphrasers and "humanizer" tools are not in
  this corpus, and every published benchmark finds they hurt badly. The frontier numbers
  above are therefore an **upper** bound on adversarial performance.
* The human reference is Liang's college essays plus TOEFL. Neither is admissions prose from
  2026, and `docs/06-decisions.md`'s warning about domain shift applies to this document too.

---

## The ceiling in one document

Aggregates make a ceiling easy to nod at. This is the same ceiling arriving as a single
essay, which is harder to wave away.

A user supplied a Gemini-written admissions essay and expected it to be caught, because the
Gemini holdout recall is 98.2%. The shipped detector returns **insufficient evidence**:
machine share 13%, document confidence 15%, 4 of 27 sentences flagged. The obvious reading
is that the detector failed. It did not fail quietly, and the reason is measurable.

Document-averaged features against the training distributions (mean ± SD):

| feature | this essay | human essays | machine essays | z vs human | z vs machine |
|---|---|---|---|---|---|
| `mean_logprob` | −3.426 | −2.796 ± 0.279 | −2.080 ± 0.337 | **−2.26** | −4.00 |
| `mean_log_rank` | 1.853 | 1.369 ± 0.192 | 0.938 ± 0.193 | **+2.52** | +4.73 |
| `frac_rank_top1` | 0.377 | 0.439 ± 0.038 | 0.522 ± 0.055 | −1.60 | −2.64 |
| `lrr` | 1.876 | 2.105 ± 0.135 | 2.313 ± 0.179 | −1.70 | −2.44 |
| `specificity_rate` | 2.396 | 2.827 ± 2.104 | 0.550 ± 0.690 | −0.20 | +2.68 |
| `novel_trigram_rate` | 0.880 | 0.825 ± 0.031 | 0.804 ± 0.036 | +1.76 | +2.10 |

Read the first two rows carefully. On the two load-bearing observer statistics this essay is
**further from the machine distribution than the human essays are**. The observer finds it
*harder* to predict than typical human admissions prose. `specificity_rate` sits squarely
inside the human distribution (z = −0.20) and 2.68 SD above the machine one.

The cause is not mysterious. The essay is model output carrying real, specific, personal
content — `TheNectar`, the UP Trade Show, a direct-DC solar induction calculation. Rare
proper nouns are genuinely improbable tokens, and generic model prose is characterised by
their absence. The detector is reading the content, and on the axis it measures, this
content is human-shaped.

**So there is no threshold that catches this essay and leaves the human essays alone.**
Turning up sensitivity until it is flagged would flag the human essays that look most like
it, which are the ones with the most specific personal detail. That is the wrong trade for a
tool whose false positives are accusations, and it is why the answer here is a refusal
rather than a number. Full numbers in `artifacts/missed_essay_diagnosis.json`.

## Fast-DetectGPT curvature was implemented, measured, and dropped

The obvious response to a frontier ceiling is a better statistic, and the best-known
candidate is Fast-DetectGPT's conditional curvature: the observer's own mean and variance of
`log p` at each position, rather than just the realised token's. Workers AI turns out to
expose what that needs — `prompt_logprobs: 20` returns the top-20 candidates **plus** the
realised token at **identical neuron cost** — so it was built: an exact-tiling aligner to
identify the realised token among the candidates, then `mu`, `sigma2` and entropy per token,
measured in 98.4% of rows.

It buys nothing. A controlled A/B on identical rows, identical documents and identical
fitting, differing only in whether the four new features are present:

| | sentence AUROC | document AUROC | frontier Claude prose (76 docs in both builds) |
|---|---|---|---|
| without | 0.9748 | 0.9983 | **27.6%** caught |
| with | 0.9747 | 0.9980 | **27.6%** caught |

Per document the new features move the score by a mean of **+0.001**. They nudge 61 of 76
documents upward, by far too little to cross anything.

The features are not noise — read alone, `mean_entropy` separates as well as anything in the
model (AUROC 0.244, against `mean_log_rank`'s 0.238). They are **redundant**. Curvature
computed over a truncated head correlates at r = +0.87 with `mean_logprob` and −0.84 with
`logprob_sd` (r² of 0.76 and 0.71) — two columns the model already reads. The 39-feature fit
had already extracted that information.

Two honest qualifications. This is **not** a refutation of Fast-DetectGPT: it is defined over
the full distribution and is strongest when the observer *is* the generator, whereas here the
top-20 head holds a median 88% of the mass and the observer is a different model from the one
being detected. And the in-domain half of the A/B had no headroom — 0.975 AUROC is saturated.
The frontier set was the half that could have moved, and it did not.

The direction was abandoned rather than shipped. Completing the corpus rebuild would have
spent hours and a large share of the daily neuron allowance to ship a measured zero.
Numbers in `artifacts/topk_ablation.json`.
