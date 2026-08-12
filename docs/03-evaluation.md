# Evaluation

Every number here is on data the model never saw during fitting. Regenerate all of it with:

```bash
python scripts/train.py && python scripts/evaluate.py
```

Raw output lands in `artifacts/evaluation.json`.

## How the splits work, and why it matters

Folds are split **by essay, never by sentence**. Sentences from one essay resemble each other
far more than they resemble sentences from another essay; a random sentence split puts
near-duplicates on both sides of the boundary and reports an accuracy the detector does not
have. `GroupKFold` is enforced inside `SentenceDetector.fit`, not left to the caller.

Mixed (part-human/part-machine) documents are grouped with the human essay they were built
from, since they share most of their text with it.

## In-domain performance

Real human admissions essays (101), real GPT-3.5 admissions essays (31), 135 modern
Gemini-3 essays and 69 mixed documents. 7,937 sentences from 336 essays, 47.9% machine.

| | |
|---|---|
| Sentence AUROC (out-of-fold) | 0.925 |
| Sentence average precision | 0.915 (baseline 0.479) |
| Sentence Brier score | 0.107 |
| Document AUROC (out-of-fold) | 0.909 |

**These went DOWN when the modern corpus went in** — sentence AUROC was 0.960 and document
AUROC 0.959 against GPT-3.5 alone. Nothing regressed: the task got harder. Distinguishing
2022 output from human prose is an easier problem than distinguishing 2026 output from it,
and a headline that only ever covered the easy half was worth less than a lower number that
covers both. The numbers that decide whether the tool is usable are in
[Detection](#detection), and they went from 0% to 71.6%.

**The length guard, and a real cost.** Human and machine sentences in the training pool now
average 18.5 and 20.6 words, and sentence length alone gives **AUROC 0.598** — against 18.5
vs 17.4 words and AUROC 0.494 before. Sentence length used to carry no signal at all and now
carries a little. Some of that is real (modern models do write longer sentences) and some is
ours: the generation prompt asks for 400–500 words. It is under the 0.15-from-chance limit
`scripts/train.py` warns at, so no alarm fires, but it is a confound that did not exist
before and roughly a tenth of the separation could be length rather than authorship.

### Calibration

In each score band, how often the sentence really was machine-written:

| predicted | n | mean predicted | actual |
|---|---|---|---|
| 0.0–0.1 | 2,502 | 0.053 | 0.050 |
| 0.1–0.2 | 688 | 0.143 | 0.185 |
| 0.2–0.3 | 372 | 0.246 | 0.290 |
| 0.3–0.4 | 278 | 0.348 | 0.345 |
| 0.4–0.5 | 265 | 0.448 | 0.392 |
| 0.5–0.7 | 538 | 0.607 | 0.556 |
| 0.7–1.0 | 3,294 | 0.890 | 0.892 |

Bands holding fewer than 20 sentences are omitted; the rest account for all 7,937.

Good at the ends, softer in the middle — the 0.5–0.7 band predicts 0.607 on 538 sentences
that were machine-written 0.556 of the time. Calibration improved when the modern corpus went
in, because the training prior stopped being 12% machine. The interface therefore bands scores into five buckets
rather than showing a continuous gradient: a reader should not be invited to distinguish 0.55
from 0.62 when the model demonstrably cannot.

This table is written to `artifacts/detector.json` by `scripts/train.py` and checked against
this prose by `tests/test_documented_numbers.py`. An earlier version of it survived the
operating-point recalibration unchanged and spent three commits claiming 41 sentences in a
band that held 186 — which is why it is now machine-checked rather than hand-copied.

## The operating point is a choice, not a result

Two errors, and they are not symmetric. Missing machine text makes the tool less useful;
flagging a real student's essay is an accusation. So the document threshold is set to a
**false-positive budget** rather than to 0.5 or to best-F1.

Critically, it is calibrated on the human population most at risk — essays by
English-language learners and out-of-domain student writing — using half that data, with the
other half reserved for reporting. Calibrating on in-domain essays instead gave 5% there and
26–52% on ESL writing: **the operating point simply did not transfer.**

The shipped point is **P ≥ 0.807**, chosen against a 5% false-positive budget on at-risk
human writing and landing at 3.0% on the calibration half. It costs recall on evasive text
and we would defend that: a tool that is wrong about a student one time in three has no
business existing, while a tool that misses evasive text is merely limited.

## Held-out results at the shipped operating point

### Detection

| set | what it is | document recall |
|---|---|---|
| `modern_holdout` | unseen essays, generator IS in training (115 docs) | **94.8%** |
| `modern_control` | same generator, **no subject steering** (45 docs) | **95.6%** |
| `modern_unseen` | a checkpoint withheld from training entirely (250 docs) | **80.0%** |
| `modern_unseen_family` | a different model family, withheld entirely (22 docs) | **45.5%** |
| `unseen_prompting` | GPT-3.5 prompted to evade detection (31 essays) | **38.7%** |
| `adversarial` | prose composed by hand to imitate a model (21 docs) | **0.0%** |

**Read the first four rows as one result, not four.** 94.8 → 95.6 → 80.0 → 45.5 is what
happens as the generator moves away from the training pool, and it is the honest answer to
"will this catch next year's model": partly, and less than the headline suggests.

`modern_control` is the control for a confound the corpus creates. Every modern essay answers
one of 40 subjects chosen in `scripts/generate_modern.py`, while the human essays are about
whatever their authors chose, so "machine" and "those topics" are correlated throughout the
new data. The control holds the generator fixed and removes only the steering: **95.6% against
94.8%** — statistically indistinguishable, so the detector is responding to the prose and not to beekeeping. Without that row
the whole modern result would be uninterpretable.

`modern_unseen_family` is 22 documents. The 95% interval on 45.5% is roughly ±20 points, so
read it as "much worse, and we cannot say how much more precisely" — the quota-limited
checkpoints could not produce more (see [02-dataset.md](02-dataset.md)).

`adversarial` remains a complete failure and is discussed in [04-failures.md](04-failures.md).

### False positives on human writing

| set | documents | sentence FPR | document FPR |
|---|---|---|---|
| `domain_shift` (ASAP 8th-grade essays) | 44 | 15.4% | **0.0%** |
| `esl` overall | 395 | 17.4% | **5.8%** |
| ├─ ELLIPSE (all ELL, graded proficiency) | — | 12.2% | 4.6% |
| ├─ PERSUADE (matched ELL flag) | — | 20.7% | 4.0% |
| └─ Liang TOEFL (short, non-native) | — | 31.4% | **17.8%** |

**Document false positives fell and sentence false positives rose, and both matter.** The
document rate is what decides whether the tool accuses anybody, and it improved everywhere
that counts — ESL overall 7.3% to 5.8% and TOEFL 24.4% to 17.8%. The sentence rate is what a user
actually *sees*, because the interface highlights sentences, and it went from 11.4% to 17.4%
on ESL writing: roughly one sentence in six of a real student's essay now gets shaded. The
sentence threshold dropped from 0.613 to 0.339 to buy the recall above, and this is the bill.
Nobody is accused more often; everybody sees more yellow.

### Localisation inside mixed documents

70 held-out documents that are part human and part machine, with a known seam.

| | | was |
|---|---|---|
| Sentence AUROC within mixed documents | **0.808** | 0.883 |
| Precision / recall at threshold | 0.647 / 0.649 | 0.805 / 0.573 |
| Seam located at all | 66 of 70 documents | 57 of 70 |
| Median seam offset | **3 sentences** | 1 |
| Seam within 2 sentences | **39%** | 70% |

**This is the clearest regression from the retrain and it is not a rounding difference.**
Locating the seam within two sentences fell from 70% to 39%, and the median offset tripled.
The seam is found in *more* documents than before (66 of 70 against 57) but placed far less
precisely. The likely cause is the in-document context features: they measure each sentence
against the rest of its own essay, and the mixed documents are built from GPT-3.5 rewrites,
so a fit now dominated by modern prose reads those contrasts differently. It was not
diagnosed further, and localisation is now the weakest part of the tool.

By rewrite direction — the two are genuinely different problems:

| pair | rewrite | AUROC | recall |
|---|---|---|---|
| ASAP | model *simplified* the second half | 0.861 | 0.751 |
| TOEFL | model *polished* the second half | 0.769 | 0.455 |

**This capability had to be trained for.** With no mixed documents in the training pool, the
in-document context features had nothing to detect — every training document was entirely one
class — so the fit learned to ignore them:

| | without mixed training | with |
|---|---|---|
| Localisation AUROC | 0.745 | **0.883** |
| Recall | 0.200 | **0.573** |
| Seam within 2 sentences | 43% | **70%** |
| Median offset | 3 sentences | **1** |

(That comparison was measured before the modern corpus was added and has not been re-run.
It is kept because it justifies why mixed documents are in the training pool at all, but the
right-hand column is no longer the shipped detector's localisation performance — the table
above it is.)

## The controlled ablation

The strongest result in the project, because it holds author and content fixed and varies
only authorship of the surface.

| | flagged | was |
|---|---|---|
| 88 original ASAP student essays | **0.0%** | 0.0% |
| the same 88, rewritten by a model | **48.9%** | 65.9% |
| 91 original TOEFL essays | 15.4% | 18.7% |
| the same 91, polished by GPT-4 | 22.0% | 24.2% |

The ASAP pair is still clean evidence that we respond to machine rewriting rather than to
topic, author or corpus — 0% against 48.9% on identical content by identical authors. But the
gap narrowed: catching 48.9% of the rewrites where the old fit caught 65.9%. These rewrites
are GPT-3.5-era, and the retrained detector spends its capacity on modern prose. Both TOEFL
rows fell together, which is the false-positive improvement showing up on the same set.

## External comparison

Liang et al. (2023) scored seven commercial detectors on `TOEFL_real_91`.

| | false-positive rate on those 91 essays |
|---|---|
| Seven commercial detectors, average | 61.22% |
| Unanimously misclassified by all seven | 19.78% |
| **Palimpsest** | **17.8%** |

Substantially better, and still far too high to use against a person.

## What is not measured

- **Two vendors, and only one of them modern.** Training machine text is GPT-3.5 (2022) and
  Gemini 3 (2026). **OpenAI's current models, Claude, Llama, Mistral, DeepSeek and Qwen are
  entirely unmeasured** — as unmeasured as Gemini was before this, which is exactly the gap
  that produced a 0% miss on the first modern essay anyone pasted in. The
  `modern_unseen_family` row (45.5%) is the best available estimate of what happens on an
  unseen generator, and it is a within-vendor estimate, so it is probably optimistic.
- **The intended generation gradient could not be built.** Every Gemini 2.x checkpoint and
  every `pro` model returned 429 on the free tier, so there is no 2024-vs-2025-vs-2026 trend,
  only Gemini 3 flash and flash-lite.
- **Topic is confounded with authorship in the modern corpus.** All 522 modern essays answer
  one of 40 subjects we chose. `modern_control` (91.1%) is the evidence that this is not what
  the detector responds to, but it is one control on 45 documents from one checkpoint.
- **One language.** English only.
- **Essays, not other forms.** Everything was measured on 100–700-word student prose.
- **Localisation regressed** and was not diagnosed: seam-within-2-sentences 70% → 39%.
- **Sentence-level false positives rose** to 17.4% on ESL writing even though document-level
  ones fell. The interface highlights sentences, so this is the number a real student sees.
