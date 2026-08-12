# False positives on writing by English-language learners

The brief says: *"These detectors have a habit of flagging writers who learned English as a
second language. If yours does, we'd like to know you spotted it."*

Ours does. Here is the measurement, the size of it, and a theory of the mechanism that the
data supports.

## Headline

At the shipped operating point, on held-out human writing:

| population | documents | document FPR |
|---|---|---|
| ASAP 8th-grade essays (native, out-of-domain) | 44 | **0.0%** |
| ELLIPSE (100% English-language learners) | — | **4.6%** |
| PERSUADE (mixed, carries an ELL flag) | — | **4.0%** |
| Liang TOEFL (non-native, short) | — | **17.8%** |

For external context, Liang et al. (2023) measured **61.22%** average false-positive rate
across seven commercial detectors on that same TOEFL set, with 19.78% misclassified
unanimously by all seven. We are roughly 2.5 times better and still not good enough to be
used against a person.

## The measurement that changes the story

The naive reading is "the detector penalises non-native writers". Two controls say that is
wrong, and something more specific is going on.

### Control 1 — matched ELL flag, same prompts, same graders

PERSUADE is the best control available: US students, same 15 prompts, same scoring rubric,
same years. The only structured difference is the `ell_status` field.

| group | sentences | sentence FPR | document FPR |
|---|---|---|---|
| ELL | 282 | 15.6% | **5.6%** |
| non-ELL | 3,170 | 21.2% | **3.8%** |

**The ELL group is flagged less often than the native group.** If non-nativeness were the
trigger, this table would look the other way round.

### Control 2 — graded proficiency within an all-ELL corpus

ELLIPSE is 100% English-language learners, each essay scored 1.0–5.0 for holistic
proficiency. If the detector punished weak English, the false-positive rate would fall as
proficiency rises. Measured across the full corpus:

| proficiency | sentences | sentence FPR |
|---|---|---|
| 2.0 | 181 | 16.0% |
| 2.5 | 443 | 11.1% |
| 3.0 | 1,050 | 11.6% |
| 3.5 | 819 | 11.7% |
| 4.0 | 292 | 3.8% |
| 4.5 | 120 | **19.2%** |
| 5.0 | 52 | **53.8%** |

**This used to be a clean monotone rise and it no longer is.** Before the modern-generator
retrain the rate climbed steadily from 2.7% at proficiency 2.0 to 24.2% at 4.5, which was
the single strongest piece of evidence that the detector responds to fluency rather than to
non-nativeness. The shape is now U-shaped: highest at the top (53.8% at 5.0, on 52
sentences), lowest in the middle, and elevated again at the bottom (16.0% at 2.0).

The top half of the claim survives and is stronger than before — **the most proficient
learners are flagged most, by a wide margin**. The bottom half does not: the least proficient
band is no longer the safest. That new bottom-end harm has an identified cause, and it is
ours. Essays by the weakest writers often carry little sentence-ending punctuation, so the
segmenter returns enormous single spans on which every per-sentence feature is out of
distribution ([04-failures.md](04-failures.md) Part 1). Two guards were added and the three
worst cases stopped being scored, but the effect has not been removed, only reduced.

So the honest statement is: **fluency still explains the top of this table, and a
segmentation artifact explains the bottom.** One of those is a finding about writing and the
other is a bug we have partly fixed. They should not be reported as one gradient, and the
earlier version of this document did exactly that.

## The theory

The detector measures **fluency and predictability**, and it reads *taught structure* as
machine structure.

Direct evidence from the failure analysis: the single most confidently misclassified TOEFL
sentence is

> "Therefore, I prefer computer science."

flagged on **stock vocabulary +4.12**. Five words, of which the content is a discourse marker
and a restatement.

ESL writing instruction teaches an explicit connective inventory and an explicit essay
skeleton: *first of all, therefore, in addition, in conclusion*. Instruction-tuned language
models overproduce the same inventory, because it is what well-formed expository English
looks like in their training distribution and what RLHF rewards. Two populations converge on
the same surface for unrelated reasons, and a fluency detector cannot tell them apart.

That explains all three observations at once:

- **Why higher proficiency is worse.** A more proficient learner deploys the taught
  scaffolding more completely, so they look more like the model.
- **Why PERSUADE ELL is flagged less.** Those essays are classroom writing by younger
  students whose prose is messier and less scaffolded than TOEFL exam prose, which is
  explicitly coached toward the template.
- **Why TOEFL is the worst set.** It is short, formulaic, exam-coached writing — maximum
  taught structure, minimum room for the in-document baseline to establish a personal style.

The bias is not about where a writer is from. It is about **how recently and how explicitly
they were taught to write this way** — which correlates with being a language learner, and
also with being a diligent native student, which is why failure #1 in
[04-failures.md](04-failures.md) is a native speaker.

## What we changed because of this

1. **Removed the document-length feature.** It gave short documents a machine-ward push —
   the fit's largest document-level weight, −2.77 — and TOEFL essays are short. Honest
   caveat: re-measured on the current build the removal **hurts** TOEFL, 6.7% → 17.8%, which
   a paired test puts just short of significance (0 essays fixed, 5 broken, p = 0.062). We
   removed it on principle, and the numbers now argue against us.
   ([04-failures.md](04-failures.md#6))
2. **Calibrated the operating point on the at-risk population, not the convenient one.** The
   document threshold is chosen so the false-positive rate on ESL and out-of-domain human
   writing stays inside an explicit budget, using half that data with the other half reserved
   for reporting. Calibrating in-domain instead gave 5% there and 26–52% on ESL writing. This
   costs roughly three quarters of our recall on evasive text, and we pay it.
3. **Built self-relative features deliberately.** The in-document family compares a sentence
   to the rest of *its own essay* rather than to a corpus norm, which removes the
   native-speaker baseline from the comparison. It reduced the bias; it did not remove it,
   because the mechanism above operates *within* an essay too.
4. **Ship the number.** Every API response carries the ESL error rate, and the interface
   prints it under every result.
5. **Removed sentence rhythm from the genre gate** — see the next section, which is the one
   place in this project where the two controls above disagreed with each other.

## The gate had the same bias, and the controls disagreed about it

The genre gate ([detect/genre.py](../src/palimpsest/detect/genre.py)) refuses to score
writing outside admissions prose. It can inherit exactly the bias documented above, in a form
that is harder to see: instead of accusing a non-native writer, it tells them the tool was not
built for their writing. `scripts/esl_gate_probe.py` re-runs both controls against it.

**They gave opposite answers.** The PERSUADE matched control found a gate log-odds shift of
**+0.079 (95% CI −0.336 to +0.367)** — nothing. The ELLIPSE proficiency gradient found
**ρ = +0.232, p < 0.001** — weaker English, lower P(in-domain). This is not a contradiction
but a power difference: control 1 splits a binary flag over 24 documents, control 2 reads a
graded score over 260. A continuous effect is invisible to a binary split of mostly-stronger
writers, and that is the pattern observed. **When the two controls disagree, prefer the graded
one.**

**The mechanism.** Regressing each gate feature on proficiency within ELLIPSE — one genre
throughout, so genre is held constant by construction — `mean_sentence_words` carried the
signal (r = −0.243, far ahead of the rest). Lower proficiency means *longer* sentences,
because a struggling writer produces run-ons rather than short ones. The feature had passed
the length audit honestly; truncation barely moves it. It was measuring proficiency anyway.

**What it cost to remove.** Transplanting a two-point proficiency drop onto real admissions
essays: **9.96% refused → 3.32%**, and the weakest proficiency band **100% → 25%**. But the
headline false-accusation rate moved **1.15% → 1.43%**, because documents the gate used to
refuse now reach the detector. Trading refusals for accusations within this same population is
not obviously progress, so it was settled by `scripts/gate_selectivity.py`: of the documents a
gate refuses, how many were heading for a false accusation? **Both gates: 4.5% precision
against a 2.58% base rate, 22 refusals per accusation avoided.** The extra refusals were not
selective — they bought nothing the wider net does not catch, and they fell hardest on the
weakest writers. Removed.

**Still true after the fix:** ρ = +0.169, p = 0.008. The gradient is reduced, not eliminated.

## What we did not fix

The 17.8% on TOEFL. We know the mechanism and we know two things that would help — refusing
to score documents under about ten sentences, and modelling essay position so a formulaic
conclusion is expected rather than surprising — and we ran out of time to do either properly.
Reporting it is not a substitute for fixing it.

## Honest limits of this analysis

- TOEFL essays are short (median 104 words) *and* non-native. Length and language background
  are confounded in that set. The ELLIPSE proficiency gradient and the PERSUADE matched
  control are what let us separate the two, and both point away from nativeness as the cause.
- ELLIPSE and PERSUADE are argumentative school essays, not admissions essays. The register
  is close but not identical to the product's target domain.
- No corpus here labels a writer's first language, so "non-native" is a single undifferentiated
  category. A Mandarin L1 writer and a Spanish L1 writer may well be treated differently, and
  we cannot see it.
- **We hold no ESL-authored admissions essays.** Every number above about the gate's behaviour
  on a non-native applicant's *personal statement* is a transplant: an effect measured in
  argumentative coursework, extrapolated linearly, applied to native-authored essays. It
  assumes additivity and cross-genre transfer, neither of which we can check. It bounds the
  risk and it does not measure it. Fifty real ones would be worth more than all of it.
