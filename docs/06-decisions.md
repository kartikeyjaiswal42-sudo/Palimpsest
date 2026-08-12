# Decisions we would be asked to defend

Short records of choices where a reasonable person would have done something else. Each notes
what we gave up.

---

### 1. The model is an instrument, never a judge

**Decision.** GPT-2 is read for token probabilities in one forward pass. No `generate()`, no
prompt, no chat template anywhere in the scoring path, enforced by a test.

**Alternative.** Ask a frontier model "is this AI-written?" — an afternoon's work and probably
better raw accuracy on easy cases.

**Why not.** It cannot explain itself, it is not reproducible across model versions, and it
is the thing the brief explicitly rules out. **Cost:** we are limited to what a 124M-parameter
observer can see.

---

### 2. Logistic regression over a gradient-boosted tree

**Decision.** Linear model, 43 interpretable features.

**Alternative.** LightGBM, worth a couple of AUROC points.

**Why not.** In a linear model the explanation *is* the computation — the logit is a sum of
the terms the interface displays. SHAP on an ensemble is a model of the model's reasoning and
not always faithful. With 201 training essays, the rigidity is also doing real
regularisation. **Cost:** a few points of AUROC, and individual weights are still unreliable
under collinearity, which we state rather than hide.

---

### 3. The n-gram reference is fitted on human text only

**Decision.** The corpus observer measures distance from *human essay writing*, not a
human-vs-machine likelihood ratio.

**Why not the ratio.** It would score better here and generalise worse: it encodes the quirks
of the one generator we happen to have. **Cost:** measurably weaker separation, bought
generator-independence.

---

### 4. Trained on real model output, not on machine-style prose we wrote

**Decision.** The machine half of training is real GPT-3.5.

**History.** We started with hand-composed essays. Sentence AUROC 0.793, document AUROC 0.500
— chance. The strongest feature was measuring our own prompt.
([04-failures.md](04-failures.md#4)) **Cost:** only 31 machine essays, from one generator.

---

### 5. Typography normalised at ingest; digits left alone

**Decision.** Smart quotes, em dashes and ellipses mapped to ASCII across both classes.

**Why.** Human essays were 88% smart-quoted and machine essays 0% — a collection artifact
that also reaches the likelihood features, since GPT-2 tokenises `’` and `'` differently.
Digits stay because their absence from machine prose is a property of the writing.
**Cost:** we lose any genuine em-dash signal. Given a 43.6%/0.0% split we could not tell
signal from artifact, and preferred to lose it.

---

### 6. Document length is not visible to the model

**Decision.** `log_sentences` removed from the document model.

**Why.** The fit gave document length a weight of **−3.09** — its strongest document-level
input was "short essays are machine-written", learned from the fact that our corpus's machine
essays are shorter (median 261 words against 642). That is a property of how Liang et al.
generated their data, not of machine writing, and the essays it misreads are short because
they were written under exam conditions.

**Cost: real, and we report it because the measurements do not support us.** Re-running the
ablation on the current build ([`scripts/ablate_length.py`](../scripts/ablate_length.py)):
in-domain document AUROC **0.998 → 0.959**, aggregate ESL false positives **5.6% → 7.3%**,
and the TOEFL improvement (33.3% → 24.4%) is **not significant** under a paired McNemar test
(p = 0.29). We removed it on the principle rather than the numbers: a corpus artifact holding
the largest weight in the model will not survive contact with a different corpus. An earlier
version of this document claimed the removal was free; that claim was measured on a since-
superseded pipeline and was wrong. See [04-failures.md](04-failures.md#6).

---

### 7. Paragraph structure stripped from every document

**Decision.** Newlines removed corpus-wide before feature extraction.

**Why.** Our largest human source ships with paragraphs stripped and our generations had
them, making paragraph structure a source marker. **Cost:** measured — segmentation changes
for 0 of 11 machine essays and 4 of 31 JHU essays (those with unpunctuated headings). The
interface still shows the user's own paragraphs.

---

### 8. Mixed documents added to training, accepting a lower headline number

**Decision.** Half the real spliced hybrids go into the training pool.

**Why.** The in-document context features exist for the mixed case and had never seen one, so
the fit ignored them. Localisation AUROC 0.745 → 0.883, seam within two sentences 43% → 70%.
**Cost:** in-domain sentence AUROC fell 0.988 → 0.958. The task genuinely got harder; we
preferred the capability the brief asks for over the better-looking number.

---

### 9. The operating point is calibrated on the population most at risk

**Decision.** The document threshold is set to a false-positive budget measured on ESL and
out-of-domain human writing, using half that data with the other half reserved for reporting.

**Alternative.** P ≥ 0.5, or a threshold tuned in-domain — which gave 5% false positives
in-domain and 26–52% on ESL writing. The operating point did not transfer.

**Cost, and it is large.** Recall on prompt-engineered machine text falls to 38.7%.
We would defend it: a tool that is wrong about a student one time in three should not exist,
while a tool that misses evasive text is merely limited.

---

### 10. Two document numbers instead of one percentage

**Decision.** Report `machineShare` (with a bootstrap interval) and `anyMachineProbability`
separately.

**Why.** They answer different questions and an essay with one polished paragraph has a low
share and a high probability. **Cost:** more to explain, which the interface does in a
sentence.

---

### 11. Scores banded into five buckets in the interface

**Decision.** No continuous colour gradient.

**Why.** Calibration is good at the ends and softer in the middle (0.5–0.7 predicts 0.607,
actual 0.556). A gradient invites a reader to distinguish 0.55 from 0.62, which the model
cannot do. **Cost:** less pretty.

---

### 12. Human corpus not committed to the repository

**Decision.** Ship the source registry, the fetch script and SHA-256 hashes instead of the
text.

**Why.** Most of it is other people's writing under licences that forbid redistribution.
**Cost:** rebuilding requires network access and takes about ten minutes.


---

### 13. Only 135 of 567 modern essays enter training

**Decision.** The modern corpus is capped at 135 essays in the training pool
(`TRAIN_CAP` in `scripts/split_modern.py`). The other 432 are held out.

**Why.** A modern essay segments to about 19.5 sentences, and the pool it joins holds 3,544
human sentences against 469 machine — a prior of 11.7% machine. Adding everything available
would have looked like this:

| essays added | machine share of the training pool |
|---|---|
| 80 | 35% |
| **135 (shipped)** | **~48%** |
| 200 | 54% |
| 400 | 69% |

At 400 the fit is told that three documents in four are machine-written. That does not make a
better detector, it makes a more willing one, and the people a willing detector is wrong
about are real students — disproportionately those writing in a second language, which this
project spends a whole document ([05-esl.md](05-esl.md)) apologising to. The document
operating point is chosen against a false-positive budget and would absorb part of it, but
the sentence threshold is chosen on precision, and sentence highlighting is what a user sees.

**Cost.** Three quarters of the corpus does no training work. It is not wasted — it becomes
a 250-document held-out set on an unseen checkpoint, which is what makes the headline recall
figure precise to about ±3 points instead of ±20.

---

### 14. One generator withheld from training entirely, and a second family too

**Decision.** `gemini-3.5-flash-lite` (250 essays) never enters training, and neither do the
three thinking-heavy checkpoints (22 essays). Both are reported separately from the
same-generator holdout.

**Why.** A random split across generators can only answer "does it detect models it has
seen", and that is the question this project already answered wrongly: 0.960 in-domain, then
a complete miss on the first modern essay a user pasted in. Recall reads 93.0% on a
trained-on generator, 71.6% on a withheld checkpoint and 36.4% on a withheld family. Only the
last is an estimate of what happens when the next model ships, and reporting the first alone
would have been a 57-point overstatement.

**Cost.** The training pool ends up drawn from a single checkpoint, which is less diverse
than it could be. And the withheld family is only 22 documents — the intended design withheld
the newest model instead, which produced 8 essays in 45 minutes before its quota ran out.

---

### 15. Subject steered during generation, style never

**Decision.** Each generated essay is given a concrete subject drawn from a list of 40. No
instruction describes voice, register or sentence rhythm.

**Why.** Four prompts sampled 500 times produce 500 variations on robotics club and the soup
kitchen; the first 18 generations did exactly that. But steering *style* is how this project's
first machine corpus came to measure its own instructions rather than authorship
([04-failures.md #4](04-failures.md)), so the prompt says nothing about prose.

**Cost.** Topic becomes confounded with authorship: every modern essay is about one of our 40
subjects, every human essay about whatever its author chose. That is why `modern_control`
exists — same generator, bare prompts, no steering — and it scores 91.1% against the steered
set's 93.0%, so the confound is measured rather than assumed. One control, 45 documents.
