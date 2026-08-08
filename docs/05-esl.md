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
| ELLIPSE (100% English-language learners) | — | **2.3%** |
| PERSUADE (mixed, carries an ELL flag) | — | **8.0%** |
| Liang TOEFL (non-native, short) | — | **24.4%** |

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
| ELL | 282 | 11.3% | **0.0%** |
| non-ELL | 3,170 | 14.6% | **8.9%** |

**The ELL group is flagged less often than the native group.** If non-nativeness were the
trigger, this table would look the other way round.

### Control 2 — graded proficiency within an all-ELL corpus

ELLIPSE is 100% English-language learners, each essay scored 1.0–5.0 for holistic
proficiency. If the detector punished weak English, the false-positive rate would fall as
proficiency rises. Measured across the full corpus:

| proficiency | sentences | sentence FPR |
|---|---|---|
| 2.0 | 371 | 2.7% |
| 2.5 | 882 | 4.3% |
| 3.0 | 1,923 | 7.4% |
| 3.5 | 1,447 | 9.7% |
| 4.0 | 1,002 | 11.9% |
| 4.5 | 231 | **24.2%** |
| 5.0 | 127 | 13.4% |

The rate **rises with proficiency**, by roughly a factor of nine from 2.0 to 4.5. The
strongest non-native writers are flagged most.

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
   the fit's largest document-level weight, −3.09 — and TOEFL essays are short. Honest
   caveat: re-measured on the current build, this helps TOEFL by 33.3% → 24.4%, which a
   paired test says is **within noise** (p = 0.29), and it slightly *raises* aggregate ESL
   false positives. We removed it on principle, not on the numbers.
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

## What we did not fix

The 24.4% on TOEFL. We know the mechanism and we know two things that would help — refusing
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
