# The detector does not detect machine text. It detects small models.

> **This title was "It detects Gemini" and the correction is recorded below rather than
> quietly applied.** The vendor reading came first because the Claude corpus was the first
> thing held out. Checking which *checkpoints* the Gemini corpus actually uses showed the
> whole of it is flash-lite/flash tier, and that detectability falls with model capability
> inside each vendor separately. See "The variable is capability, not vendor".

This is the largest thing measured in this project so far, and it was not visible until a
corpus existed from a lab whose models are in none of the training data.

## The measurement

Every held-out machine set, at the shipped operating point:

| set | what is held out | documents | document recall | median machine-share |
|---|---|---|---|---|
| `modern_holdout` | unseen essays, **generator seen** in training | 115 | **94.8%** | 0.970 |
| `modern_control` | same, no subject steering | 45 | **95.6%** | 0.943 |
| `modern_unseen` | a Gemini checkpoint withheld entirely | 250 | **80.0%** | 0.881 |
| `modern_unseen_family` | a different Gemini *tier* | 22 | **45.5%** | 0.709 |
| `modern_claude` | **a different vendor** (Anthropic) | 48 | **0.0%** | **0.007** |

Read the last column, not the last-but-one. Machine-share is the fraction of the essay the
tool reports as machine-written, and on the Claude essays its median is **0.007**. The
detector is not scoring them as borderline and losing them to a strict threshold. It is
reporting that they contain essentially no machine text at all. The sentence flag rate is
6.1%, which is well under the 15.4% false-positive rate it has on genuine human school
essays — so by its own numbers it is *more* confident these machine essays are human than it
is about actual humans.

No threshold fixes this. There is nothing to move a threshold onto.

## It is not one bad checkpoint

Broken out by generating model, machine sentences only:

| model | sentences | documents | sentence flag rate | median p |
|---|---|---|---|---|
| Haiku | 471 | 10 | 11.0% | 0.053 |
| Fable | 172 | 8 | 9.3% | 0.054 |
| Sonnet | 169 | 9 | 7.1% | 0.043 |
| **Opus** | 852 | 21 | **2.6%** | 0.039 |

All four are invisible, so this is not a weakness against one checkpoint that a little more
data would patch. It is the vendor.

The ordering inside the failure is worth its own line: **the most capable model is the least
detectable**, by a factor of four over the smallest. Whatever residual signal the detector is
picking up is a capability artifact, and it thins out as models improve. That is the opposite
of the direction a detector needs to age in.

## It is not our own prompting either

Broken out by prompting style:

| style | sentences | flag rate |
|---|---|---|
| `plain` (no style instruction at all) | 852 | 8.5% |
| `constrained` | 50 | 6.0% |
| `evasive` (explicitly told to evade detection) | 174 | 5.2% |
| `persona` | 332 | 4.5% |
| `notes` | 256 | 1.2% |

This is the check that stops the result being an artifact of how the corpus was commissioned.
`plain` carries no instruction about voice whatsoever — it is the model writing an admissions
essay, not the model performing an adjective we handed it — and it is the *most* detected
slice, at 8.5%. The essays explicitly told to evade detection do only slightly better than
that. If recall had lived in `evasive` alone, the honest reading would have been that the
detector had learnt our steering vocabulary; it does not, so the failure is on ordinary
unstyled model prose.

## What it means

The four Gemini rows describe a clean, unremarkable decay: 94.8% on a generator it has seen,
80.0% on a withheld checkpoint, 45.5% on a further-out tier. That curve is what a detector
generalising imperfectly-but-really looks like, and reported on its own it tells a reassuring
story about degrading gracefully as generators advance.

The Claude row says the curve was never about generator *recency*. It was about generator
*identity*. Held-out-checkpoint evaluation — the standard way this is measured, and what
`modern_unseen` and `modern_unseen_family` are — systematically overstates what a detector can
do, because a withheld checkpoint still shares pretraining data, tokeniser, post-training
method and house style with the ones in training. A model from another lab shares none of
that, and performance does not degrade. It collapses.

The honest statement of what this detector currently is: **a Gemini detector with a 0.0%
false-positive rate on other vendors' output.** That is a much narrower product than the
evaluation table suggested before this set existed.

## Why this set is the strict one

`modern_unseen_family` is named for the test it was meant to be, but every essay in it is
Gemini — `gemini-3-flash-preview`, `gemini-3.5-flash`, `gemini-3.6-flash` against the
`flash-lite` checkpoints in training. It varies the tier, not the family. `modern_claude` is
the only set in the project where the vendor changes, which makes it the only measurement of
the thing that actually matters when next year's model arrives from somewhere else.

## The corpus behind the number

`scripts/plan_claude_corpus.py` plans 500 essays and writes the plan to disk *before*
generation, because every choice in it is a way to measure ourselves instead of the model —
the mistake docs/04-failures.md #4 records, where a hand-built machine corpus scored document
AUROC 0.500 and its strongest feature was reading back our own instruction.

Four things the plan fixes in advance:

- **Length is matched to the human corpus.** Generators overshoot a requested word count by a
  measured 6.8% (mean 1.068, p10 1.032, p90 1.080 over the first 32 essays), so the plan asks
  for less than it wants. Uncorrected, the corpus landed 40 words above the human median,
  which `assemble_claude_corpus.report_lengths` flagged as separable-on-length-alone at n=31.
  docs/06-decisions.md #6 is what happens when that is missed: length took a weight of −3.09
  and the model's strongest belief became "short means machine".
- **Style is mostly not steered.** 47% of the plan carries a shape-only instruction that says
  nothing about voice and never says "sound human". The steered variants exist as a labelled
  minority so recall can be reported per style rather than averaged.
- **Subject never crosses the train/held-out boundary.** The split is by subject, decided
  before generation. No held-out essay shares a subject with a training one.
- **Four checkpoints, not one**, so recall can be reported per model.

## The finding replicates at n = 160

The corpus reached 398 gate-valid essays (238 training, 160 held out), and the held-out set
was re-measured at 160 documents rather than 48. Nothing softened:

| | n = 48 | n = 160 |
|---|---|---|
| sentence flag rate on machine sentences | 0.061 | 0.066 |
| documents called machine | 0.000 | 0.000 |
| median machine-share reported | 0.007 | 0.010 |

The per-model cells are now 30–46 documents each, and **the Opus-is-hardest ordering
survives** — it is no longer only a hypothesis:

| model | docs | sentence flag rate |
|---|---|---|
| opus | 46 | **0.027** |
| fable | 30 | 0.046 |
| sonnet | 42 | 0.067 |
| haiku | 42 | 0.105 |

The most capable model is the least detectable, by roughly 4×, at four times the sample size
that first suggested it.

## The fix was measured. It does not work.

This is the experiment the corpus was built to run, and the answer is negative in a way that
is more interesting than a positive would have been.

**Attempt 1 — add the training split to `train`.** On its face it works spectacularly: the
flag rate on held-out Claude goes 0.066 → 0.930 and the median machine-share 0.010 → 0.959.
Every other number says it is worthless:

| | before | after |
|---|---|---|
| out-of-fold sentence AUROC | 0.925 | **0.726** |
| ESL sentence false-positive rate | 0.174 | **0.791** |
| localisation AUROC (mixed documents) | 0.808 | **0.538** |
| document recall, *every* set incl. Gemini | 0.80–0.95 | **0.000** |

It flags 79% of genuine English-learner sentences. It is not detecting Claude; it is flagging
everything, and the "recall" is that indiscriminacy seen from a flattering angle. The document
operating point — calibrated to hold at-risk human writing under a 5% false-positive budget —
was forced to P ≥ 0.991 and took document recall to zero on sets it used to catch.

The mechanism is the class prior, and it is arithmetic. The training set was **balanced at
0.498 machine sentences**; the Claude split adds 8,574 sentences all labelled machine, taking
it to **0.773**. There are only 3,544 human sentences in the project, so there is no headroom
to add a machine vendor and stay balanced. This is docs/06-decisions.md's willing-accuser
failure reached literally.

**Attempt 2 — spend a fixed budget differently** (`scripts/vendor_swap_sweep.py`). Hold the
machine class at the size it already was and draw a fraction of it from Claude instead of
Gemini. The prior stays at 0.498 by construction, so anything that moves is about *which
vendor the budget is spent on* and nothing else:

| claude share | prior | sentence AUROC | ESL FPR | claude flag | gemini flag |
|---|---|---|---|---|---|
| 0% | 0.498 | **0.925** | 0.174 | 0.066 | **0.831** |
| 25% | 0.498 | 0.785 | 0.164 | 0.143 | 0.483 |
| 50% | 0.499 | 0.737 | 0.083 | 0.104 | 0.079 |
| 75% | 0.499 | 0.735 | 0.109 | **0.209** | **0.007** |

Claude detection never arrives. It peaks at 0.209 — against 0.831 for Gemini at the same
budget — while Gemini detection is destroyed on the way, ending at 0.007. Sentence AUROC falls
off at any non-zero share. There is no point on this curve worth shipping, and the shipped
detector remains the 0% one.

## The variable is capability, not vendor

The title of this document is wrong, and the correction matters more than the original claim.

Every "modern" essay in this project comes from a **flash-lite or flash tier** model. There is
not one Pro-tier Gemini essay anywhere in the corpus:

| set | model | sentence flag rate |
|---|---|---|
| `modern_train` / `modern_holdout` / `modern_control` | gemini-3.1-**flash-lite** | 0.91–0.93 |
| `modern_unseen` | gemini-3.5-**flash-lite** | 0.831 |
| `modern_unseen_family` | gemini-3-flash-preview / 3.5-**flash** / 3.6-**flash** | 0.665 |
| `modern_claude` | opus · sonnet · fable · haiku | 0.066 |

So the 94.8% headline was measured against the cheapest model Google ships, and
`modern_unseen_family` — read above as "a different family" — differs from the training set
mainly by being **bigger**, not by being a different lab.

Detectability falls with model capability, and it does so *within each vendor separately*:

- inside Gemini: flash-lite 0.91 → flash 0.665
- inside Claude: haiku 0.105 → sonnet 0.067 → fable 0.046 → opus 0.027

Two independent within-vendor gradients pointing the same way is what distinguishes a
capability effect from a house-style effect. The Gemini-versus-Claude gap is confounded —
every Gemini essay here is small and every Claude essay is not — but the within-vendor
gradients are not confounded, and they carry the same sign.

**Informal confirmation, worth replacing with a measurement.** A full Gemini 3.1 essay (not
flash-lite) scored a 0.371 machine share, below the flag threshold; an Opus 5 statement of
purpose scored 0.000. Both were entirely machine-written. Neither is in any evaluation set,
so these are anecdotes — but they are the two the model predicts.

The decisive experiment this corpus cannot run: **generate the Gemini corpus again at Pro
tier.** If flag rate collapses there too, vendor is eliminated and capability is the whole
story. Until then the capability reading is strongly supported and not proven.

## What it actually detects

The per-sentence evidence on that Gemini essay shows the mechanism. 8 of 24 sentences were
flagged, and they are the abstract ones:

| p | sentence |
|---|---|
| 0.943 | "I want to study mechanical engineering because the world is full of broken systems…" |
| 0.915 | "Instead, I saw a system of components." |
| 0.042 | "I didn't fix all of them." |
| 0.040 | "In fact, I probably ruined a few beyond recognition." |

Concrete narrative sentences score near zero; generic aspirational ones score near one. That
is a **register** detector wearing an authorship label, and it explains the Opus result
without any appeal to vendor: that essay is almost entirely concrete particulars — a named
technician, a jute bag of spanners, a 1.5% heat-rate deviation — and a register detector has
nothing to bite on.

It also explains why `specificity_rate` carries a human-ward weight, and why truepen
(`truepen/HISTORY.md`) hit the same wall from a completely different direction: it measured
that well-written frontier prose is not reliably detectable, with high perplexity being the
tell it could not use. Two projects, two methods, one conclusion.

## What that changes about the claim

The title of this document said the detector reads Gemini rather than machine-ness. That is
still true, but the reason is not the one implied. It is **not a training-data gap.** Given a
balanced budget and 238 Claude essays written to the same plan, on the same prompts, at the
same length distribution as the Gemini essays it learns to 0.83, the fit reaches 0.21.

Under these 43 features and a GPT-2 observer, Claude admissions prose sits roughly where
*human* admissions prose sits. Gemini prose does not. The 0.0% cross-vendor recall is a
statement about the feature space, not about how much data was collected — and collecting
more Claude essays is therefore not the repair. Whatever separates these models from human
writing is not what these features measure.

## What is still not true

- **The sweep has one held-out vendor and it is Claude.** With GPT-3.5, Gemini and Claude all
  either trained on or measured here, the project has no fourth lab left to hold out, so
  "does a fourth vendor generalise" remains unanswerable with this corpus.
- **The corpus is 398 of 500** — 85 essays ungenerated and 17 rejected by the gate for
  overshooting the ±8% band (13 of them haiku). The train/held-out split is by subject and
  pre-registered, so the missing essays cost statistical power, not validity.
- **These essays are harder than the ones already in `adversarial`.** That set of 21 older
  Claude documents flags at 21.4% of sentences; this one flags at 6.6%. Some of that is newer
  models and some is the plan's deliberate lack of style steering, and the two are still not
  separated.
- **`data/generated/palamiassist_college_essays_500.jsonl` must not be trained on.** It is a
  relabelled re-export of the existing corpus: 489 of its 500 rows are byte-identical to
  essays already on disk, and 281 of those come from held-out evaluation sets (103 from
  `claude_modern_heldout`, 98 from `modern_unseen`, 42 from `modern_holdout`, 24 from
  `modern_control`, 14 from `modern_unseen_family`). Using it would put the test sets into
  training and turn every number above into a memorisation score wearing a generalisation
  label.
