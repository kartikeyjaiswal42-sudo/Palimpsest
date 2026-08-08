# Approach: which signals, and why those

## The line the brief draws

> *"the model must not make the judgement call while your app relays the verdict."*

Palimpsest reads a language model the way a spectrometer reads light. GPT-2 runs in a single
forward pass and we take the logits. There is no `generate()` call, no prompt, no chat
template, and no path by which a model opinion could enter the score.
`tests/test_no_generative_calls.py` greps the scoring path for generation APIs and fails if
one appears.

Everything after that is arithmetic we can show you.

## Two observers, because they know different things

**A local causal LM (GPT-2, 124M).** Small enough to run on a laptop CPU in ~2 s per essay,
and — usefully — trained on pre-2019 text, so it cannot have memorised the output of the
models we are trying to detect. Its job is not to know what AI text looks like. It is a fixed,
neutral yardstick for *how predictable is this word, here*. From one pass we take four things
that mean genuinely different things:

| statistic | what it captures |
|---|---|
| log P(token) | the classic "too fluent" signal |
| rank of the observed token | robust to how well-calibrated the probabilities are; what GLTR used |
| entropy of the predictive distribution | separates "confident and right" from "no idea, and the word was common" |
| mean and variance of log P under the model's own distribution | makes Fast-DetectGPT's curvature computable analytically in one pass instead of ~100 |

**An n-gram model of real admissions essays.** GPT-2 knows what English looks like in
general; it has no idea what a seventeen-year-old writing about their grandmother sounds
like. The brief points at this directly — some differences *"only appear when a passage is
compared against a body of other writing"* — so we fit an interpolated trigram model on 1,000
human student essays and ask a second question: *is this how applicants write?*

It is fitted on **human text only**, never on a human-vs-machine likelihood ratio. A ratio
would score better on our own test set and be a worse detector in the world: it would encode
the quirks of the particular generator we happened to have and quietly stop working on a new
one. We pay for that in raw separation and buy generator-independence with it.

The disagreement between the two observers is itself a feature. `fluency_typicality_gap` is
positive when a passage is *smoother than general English yet less typical of the genre* —
the polished-text signature. Non-native writing usually shows the opposite sign, which is
part of how the two cases stay distinguishable.

## The four feature families

43 features, each with a registry entry in `features/registry.py` recording what it measures,
its units, and **which direction we predicted before fitting**.

1. **Likelihood and rank** (13) — surprisal, GLTR rank buckets, entropy, DetectLLM's LRR,
   Fast-DetectGPT curvature.
2. **Corpus-relative** (4) — surprisal against the human-essay reference, the rate of
   three-word phrases nobody else wrote, and the fluency/typicality gap.
3. **Surface style** (19) — rhythm, punctuation, clause structure, function-word rate, a
   curated stock-phrase lexicon, and two constructions machine prose overproduces (the "A, B,
   and C" tricolon and the "not just X but Y" antithesis).
4. **In-document context** (6) — how far this sentence sits from *the rest of its own essay*.

### Family 4 is the one that matters

The brief names the realistic case: *"a paragraph a person wrote and a model later polished"*.
That case is nearly invisible to absolute thresholds — plenty of humans write smooth prose —
but it is glaring as a *discontinuity*. A polished paragraph is unremarkable in isolation and
remarkable sitting inside four paragraphs by the same person that read nothing like it.

Every statistic in this family is leave-one-out: a sentence is compared against the median of
the *other* sentences, using median-absolute-deviation rather than standard deviation, so a
long inserted passage cannot drag the baseline it is being measured against.

There is a second reason we prioritised it. Comparing an author to themselves removes the
native-speaker norm from the comparison, which is the mechanism behind the documented ESL
bias in this class of tool. It reduced the bias measurably. It did not remove it —
[05-esl.md](05-esl.md) reports how much.

**These features have to be trained for.** With only all-human and all-machine documents in
the training pool they have nothing to detect, and the fit learns to ignore them. Adding real
mixed documents took localisation AUROC from 0.745 to 0.878.

## Why logistic regression

A gradient-boosted tree scores a couple of AUROC points higher. We chose against it.

In a linear model the explanation is not a story told about the computation — it *is* the
computation. The logit is literally a sum of per-feature terms, so the bars in the interface
are arithmetically identical to the number they explain, and the panel prints the sum so a
reader can check the parts add up. SHAP values on an ensemble would give a *model of* the
model's reasoning, and not always a faithful one. For a tool whose entire purpose is showing
its evidence, that gap is unaffordable.

Second reason: with 201 essays, a flexible model will memorise our particular generator. The
linear model's rigidity is doing real regularisation work.

**One honest caveat.** With correlated features, individual weights are not interpretable —
the fit splits a shared signal across collinear columns and signs flip. 25 of our 40
signed features matched the direction we predicted, and the 15 that did not are mostly
collinearity artifacts rather than discoveries. The per-feature AUROCs in
[03-evaluation.md](03-evaluation.md) are the trustworthy reading, and the interface shows
contributions rather than weights for exactly this reason.

## Two numbers, never one

> *"'73% AI' gives a reader nothing they can act on and nothing they can argue with."*

That number is useless because it fuses two different questions. We keep them apart:

- **`machineShare`** — how much of the essay reads as machine-written, as a fraction of its
  words, with a bootstrap interval because it is estimated from a few dozen noisy sentence
  scores.
- **`anyMachineProbability`** — confidence that *any* of it is machine-written.

An essay with one polished paragraph has a low share and a high probability. A single
percentage destroys exactly the distinction a reader needs.

## Things deliberately not done

- **No perplexity threshold.** Measured, and it does not work alone: `mean_logprob` reaches
  AUROC 0.724 in isolation, nowhere near enough to act on.
- **No DetectGPT perturbation.** ~100 extra forward passes per document for a statistic
  Fast-DetectGPT gets analytically in one.
- **No document-length feature.** It leaked, and it leaked onto short ESL essays.
  ([04-failures.md](04-failures.md#6))
- **No continuous colour gradient in the UI.** Calibration is reliable at the ends and poor
  in the middle, so scores are banded into five buckets. A reader should not be invited to
  distinguish 0.55 from 0.62 when the model cannot.
