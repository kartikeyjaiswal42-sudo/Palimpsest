# Failures

The brief asks for three essays the detector gets confidently wrong and a theory about why.
Below are those three, plus the three larger mistakes made building it — which were more
instructive.

Failures are selected mechanically by `scripts/find_failures.py`: highest-scoring human
documents and lowest-scoring machine documents across every held-out set. Nothing was
hand-picked to flatter the tool.

---

# Part 1 — three essays it gets confidently wrong

## 1. A real student essay on distance learning, called machine with P = 0.998

`persuade:b69e0a7afc9a` — PERSUADE corpus, written by a US school student between 2010 and
2020, years before ChatGPT existed. 25 sentences, 95% of them flagged.

> "Waking up late and enjoying your morning is better than having to wake up before the sun
> rises to go to school…"

The two sentences that carry the verdict, with the evidence the model actually summed:

| sentence | top contributions |
|---|---|
| "Distance learning is a way of going to school without actually going there physically…" | smoother than the author's baseline **+4.71**, curvature vs baseline **+1.55** |
| "Clearly, distance learning is a beneficial option for many students because it can help them get more sleep" | average predictability **+1.64**, local rhythm **+1.38** |

**Theory.** This is a five-paragraph school essay executed correctly. The definitional
sentence and the signposted conclusion are the two most formulaic sentences in it — and they
are formulaic *because the form requires them to be*. Our strongest feature asks how far a
sentence sits from the author's own baseline, and a competent five-paragraph essay is built
out of deliberate register shifts: narrative opening, flat definition, signposted conclusion.
We read the genre's architecture as a change of author.

This is the failure mode that worries us most, because it will fire hardest on students who
were taught the structure most rigidly.

## 2. A TOEFL essay, called machine with P = 0.998

`liang_toefl:0088` — 7 sentences, non-native writer.

> "If I am choosing among computer science, business, and photography classes, I will choose
> computer science for two main reasons. First of all…"

| sentence | top contributions |
|---|---|
| "Therefore, I prefer computer science." | **stock vocabulary +4.12**, smoother than baseline −3.81, average predictability +2.93 |

**Theory.** "Therefore, I prefer computer science" is a five-word sentence whose entire
content is a discourse marker plus a restatement. Our `machine_word_rate` lexicon contains
*therefore*, because instruction-tuned models overproduce it. So does ESL writing
instruction, which teaches exactly this connective inventory — *first of all, therefore, in
addition, in conclusion* — as the scaffolding of an argumentative essay.

The detector is not detecting non-native writing. It is detecting **taught structure**. Both
populations use it, one because it was optimised toward it and the other because it was
explicitly instructed in it. Analysed in full in [05-esl.md](05-esl.md).

There is a second, compounding problem: the essay is 7 sentences long. Our in-document
features compare a sentence against the rest of its essay, and with six other sentences that
baseline is extremely noisy. Short documents are where we are least trustworthy and we do not
currently refuse to score them, which we should.

## 3. Machine-polished prose it misses entirely, P = 0.031

`hybrid_claude:004` — a real human essay with a paragraph rewritten to be more polished.

Only one sentence in the document scored high (0.786: *"Good mortaring requires technical
precision, speed and strength"*), and the document verdict stayed at 0.031.

**Theory.** Two causes, and they compound. First, the rewrite was composed by hand rather
than sampled from a model, and part 2 below explains why that alone makes it nearly
invisible. Second, the document aggregate is dominated by the 21 human sentences around it —
one flagged sentence in twenty-two moves `share` to 0.01, and the document model reasonably
concludes the essay is human.

That is the correct answer to "is this essay machine-written" and the wrong answer to "did
something happen in paragraph two". The interface mitigates this by always showing the
per-sentence heat map rather than only the document verdict, but the document number itself
remains unable to express "one sentence, high confidence".

---

# Part 2 — three larger mistakes, and what they cost

## 4. We wrote the machine corpus ourselves. It was worthless.

The first machine corpus was eleven essays composed by hand in three prompt conditions to
read like model output. Trained on that, sentence AUROC was **0.793**, document AUROC was
**0.500 — chance.**

The diagnostic that found it: per-feature AUROC, split by which prompt condition produced the
text.

| feature | our "plain" | our "evasive" | **real GPT-3.5** |
|---|---|---|---|
| `mean_logprob` | 0.532 | 0.425 | **0.724** |
| `frac_rank_top10` | 0.496 | — | **0.683** |
| `mean_entropy` | 0.499 | 0.639 | **0.293** |
| `local_len_burstiness` | 0.505 | **0.840** | **0.279** |

The strongest feature in the whole model was `local_len_burstiness` at 0.840 — on the
`evasive` condition only, where the instruction had literally been *"vary your sentence
lengths sharply"*. It was measuring our own prompt. Against real GPT-3.5 the same feature
lands at 0.279, i.e. on the **opposite side of chance**.

**The lesson, which we did not expect.** Prose *composed* to sound machine-generated does not
carry the statistical signature of prose *sampled* from a model — even when the author is
itself a language model. The detectable signal lives in autoregressive sampling, not in
stylistic choices. A writer imitating machine prose makes conscious decisions about
vocabulary and rhythm; those decisions produce token statistics that look nothing like
temperature sampling from a distribution.

Fix: real GPT-3.5 output for training, hand-written essays demoted to a held-out adversarial
set. Sentence AUROC 0.793 → 0.958, document AUROC 0.500 → 0.961. The hand-written set is now
one of our reported failure modes: **0 of 11 caught.**

## 5. A 0.988 AUROC that was partly a smart-quote detector

The result looked too good, so we went looking for the artifact and found it in the
typography:

| character | human docs | GPT-3.5 docs |
|---|---|---|
| curly apostrophe `’` | 88.1% | 9.7% |
| curly quotes `“ ”` | 72.3% | **0.0%** |
| em dash `—` | 43.6% | **0.0%** |
| digits 0–9 | ≥10% | ≤1% |

The human essays were scraped from college web pages, which render smart quotes; the machine
essays came through an API as ASCII. Worse than a surface feature, this reaches the
likelihood features too — GPT-2 assigns a different token, and therefore a different
surprisal, to U+2019 than to `'`.

Fix: typography normalised at ingest for both classes. AUROC 0.988 → 0.988. **The artifact
was real and it was not what was carrying the result** — but we could not have known that
without removing it, and shipping the number without checking would have been indefensible.

Digits are deliberately left alone: their absence from machine text is a property of the
prose, not the pipeline.

## 6. A length artifact that landed on exactly the wrong people

The document model was given `log_sentences`. It learned a large negative weight — fewer
sentences means machine — because the machine essays in our corpus happen to be shorter
(median 261 words against 642). That is a fact about how Liang et al. generated their data,
not about machine writing.

Its consequence was not abstract. TOEFL essays are short, so the artifact landed hardest on
precisely the group this class of tool is already documented to harm.

**And then the interesting part happened.** We first measured this early, reported that
removing the feature took TOEFL false positives from 40.7% to 19.8% at no cost, and wrote
that down. Later changes — mixed documents added to training, the operating point moved onto
the at-risk population — made that measurement unreproducible, and the write-up went on
asserting it. We caught it by turning the ablation into a script
([`scripts/ablate_length.py`](../scripts/ablate_length.py)) and re-running it. The current
result is **substantially less flattering than what we had written**:

| | with `log_sentences` | without (shipped) |
|---|---|---|
| TOEFL document FPR | 33.3% (15/45) | **24.4% (11/45)** |
| ESL overall document FPR | **5.6% (22/395)** | 7.3% (29/395) |
| domain-shift document FPR | 0.0% | 0.0% |
| In-domain document AUROC | **0.998** | 0.959 |
| weight the fit gave it | **−3.09** | — |

Three things are true here and the first two argue *against* the decision we made:

1. **The TOEFL improvement is not statistically significant.** The same 45 essays are scored
   by both arms, so the honest test is paired. McNemar's exact test: 6 essays fixed by the
   removal, 2 broken, **p = 0.29**. On this evidence we cannot claim the removal helps TOEFL.
2. **Aggregate ESL false positives got *worse*,** 5.6% → 7.3%. Keeping the feature does not
   spread harm evenly — it concentrates it on short essays while lowering the average.
3. **It cost real in-domain accuracy**, document AUROC 0.998 → 0.959.

We removed it anyway, and the reason is not in the table. A weight of −3.09 on document
length means the model's single strongest document-level input was *"short essays are
machine-written"* — which is true of our corpus (Liang's machine essays run to a median 261
words against 642) and true of nothing else. It is a fact about a data-collection procedure,
being used as though it were a fact about writing. The population it misclassifies is not
random: it is exam-condition writing, which is short because the exam is timed, by students
least able to contest an accusation.

So the honest statement is: **the case for removing it is principled, and the measurements
mildly disagree.** We would rather ship 0.959 that generalises than 0.998 that is partly a
word-count detector. Anyone who thinks that trade is wrong can re-run the script and take the
other arm — which is precisely why the script exists.

---

## What we would do next

1. **A second generator.** Everything real in the training data is GPT-3.5. This is the
   largest gap and the prompt-engineering result suggests transfer is poor.
2. **Refuse to score short documents.** Below roughly ten sentences the in-document features
   are too noisy, and that is where the worst false positives cluster.
3. **Down-weight genre scaffolding.** Failures 1 and 2 share a cause: we read the required
   architecture of a school essay as a change of author. Explicitly modelling essay position
   (opening / body / conclusion) would let the model expect a formulaic conclusion instead of
   being surprised by one.
