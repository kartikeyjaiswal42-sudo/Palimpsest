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

Real human admissions essays (101) against real GPT-3.5 admissions essays (31), plus 69
mixed documents. 4,895 sentences from 201 essays, 15.5% machine.

| | |
|---|---|
| Sentence AUROC (out-of-fold) | 0.960 |
| Sentence average precision | 0.810 (baseline 0.155) |
| Sentence Brier score | 0.054 |
| Document AUROC (out-of-fold) | 0.959 |

**The length guard.** Human and machine sentences in the training pool average 18.5 and 17.4
words; sentence length alone gives AUROC 0.494. The classifier cannot reach the right answer
by measuring length. `scripts/train.py` prints this check on every run and warns if it drifts
past 0.15 from chance.

### Calibration

In each score band, how often the sentence really was machine-written:

| predicted | n | mean predicted | actual |
|---|---|---|---|
| 0.0–0.1 | 3,411 | 0.003 | 0.005 |
| 0.1–0.2 | 39 | 0.150 | 0.103 |
| 0.2–0.3 | 32 | 0.250 | 0.250 |
| 0.5–0.7 | 41 | 0.624 | 0.366 |
| 0.7–1.0 | 466 | 0.876 | 0.893 |

Good at the ends, unreliable in the middle — the 0.5–0.7 band is over-confident on 41
sentences. The interface therefore bands scores into five buckets rather than showing a
continuous gradient: a reader should not be invited to distinguish 0.55 from 0.62 when the
model demonstrably cannot.

## The operating point is a choice, not a result

Two errors, and they are not symmetric. Missing machine text makes the tool less useful;
flagging a real student's essay is an accusation. So the document threshold is set to a
**false-positive budget** rather than to 0.5 or to best-F1.

Critically, it is calibrated on the human population most at risk — essays by
English-language learners and out-of-domain student writing — using half that data, with the
other half reserved for reporting. Calibrating on in-domain essays instead gave 5% there and
26–52% on ESL writing: **the operating point simply did not transfer.**

| operating point | ESL doc FPR | domain-shift doc FPR | prompt-engineered recall | localisation AUROC |
|---|---|---|---|---|
| P ≥ 0.50, calibrated in-domain | 34.8% | 31.8% | 41.9% | 0.883 |
| **P ≥ 0.974, calibrated on at-risk writing (shipped)** | **7.3%** | **0.0%** | **6.5%** | **0.883** |

We ship the second row. It costs three quarters of the recall on evasive text. That is the
trade we chose and we would defend it: a tool that is wrong about a student one time in three
has no business existing, while a tool that misses evasive text is merely limited.

## Held-out results at the shipped operating point

### Detection

| set | what it is | document recall |
|---|---|---|
| `unseen_prompting` | GPT-3.5 prompted to evade detection (31 essays) | **6.5%** |
| `adversarial` | prose composed by hand to imitate a model (21 docs) | **0.0%** |

Both are honest failures and both are discussed in [04-failures.md](04-failures.md).

### False positives on human writing

| set | documents | sentence FPR | document FPR |
|---|---|---|---|
| `domain_shift` (ASAP 8th-grade essays) | 44 | 8.3% | **0.0%** |
| `esl` overall | 395 | 11.4% | **7.3%** |
| ├─ ELLIPSE (all ELL, graded proficiency) | — | 6.8% | 2.3% |
| ├─ PERSUADE (matched ELL flag) | — | 14.3% | 8.0% |
| └─ Liang TOEFL (short, non-native) | — | 23.8% | **24.4%** |

### Localisation inside mixed documents

70 held-out documents that are part human and part machine, with a known seam.

| | |
|---|---|
| Sentence AUROC within mixed documents | **0.883** |
| Precision / recall at threshold | 0.805 / 0.573 |
| Seam located at all | 57 of 70 documents |
| Median seam offset | **1 sentence** |
| Seam within 2 sentences | **70%** |

By rewrite direction — the two are genuinely different problems:

| pair | rewrite | AUROC | recall |
|---|---|---|---|
| ASAP | model *simplified* the second half | 0.912 | 0.635 |
| TOEFL | model *polished* the second half | 0.845 | 0.455 |

**This capability had to be trained for.** With no mixed documents in the training pool, the
in-document context features had nothing to detect — every training document was entirely one
class — so the fit learned to ignore them:

| | without mixed training | with |
|---|---|---|
| Localisation AUROC | 0.745 | **0.883** |
| Recall | 0.200 | **0.573** |
| Seam within 2 sentences | 43% | **70%** |
| Median offset | 3 sentences | **1** |

## The controlled ablation

The strongest result in the project, because it holds author and content fixed and varies
only authorship of the surface.

| | flagged |
|---|---|
| 88 original ASAP student essays | **0.0%** |
| the same 88, rewritten by a model | **65.9%** |
| 91 original TOEFL essays | 18.7% |
| the same 91, polished by GPT-4 | 24.2% |

The ASAP pair is clean evidence that we respond to machine rewriting rather than to topic,
author or corpus. The TOEFL pair moves much less, and the reason is visible in the baseline:
those originals were already being flagged at 18.7%, so there is less headroom. The same
asymmetry appears in Liang et al.'s published measurements.

## External comparison

Liang et al. (2023) scored seven commercial detectors on `TOEFL_real_91`.

| | false-positive rate on those 91 essays |
|---|---|
| Seven commercial detectors, average | 61.22% |
| Unanimously misclassified by all seven | 19.78% |
| **Palimpsest** | **24.4%** |

Substantially better, and still far too high to use against a person.

## What is not measured

- **One generator.** All real machine training text is GPT-3.5. Performance against GPT-4,
  Claude, Llama or Gemini is **unmeasured**, and the prompt-engineering result suggests it
  would be considerably worse. This is the single biggest gap.
- **One language.** English only.
- **Essays, not other forms.** Everything was measured on 100–700-word student prose.
- The middle probability band (0.5–0.7) is poorly calibrated on 41 sentences.
