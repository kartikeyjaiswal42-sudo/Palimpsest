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

**Why.** It learned "shorter ⇒ machine" from a quirk of the corpus and produced a 41%
false-positive rate on short TOEFL essays. **Cost:** none measurable — in-domain AUROC was
unchanged at 1.000.

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
the fit ignored them. Localisation AUROC 0.745 → 0.878, seam within two sentences 43% → 66%.
**Cost:** in-domain sentence AUROC fell 0.988 → 0.958. The task genuinely got harder; we
preferred the capability the brief asks for over the better-looking number.

---

### 9. The operating point is calibrated on the population most at risk

**Decision.** The document threshold is set to a false-positive budget measured on ESL and
out-of-domain human writing, using half that data with the other half reserved for reporting.

**Alternative.** P ≥ 0.5, or a threshold tuned in-domain — which gave 5% false positives
in-domain and 26–52% on ESL writing. The operating point did not transfer.

**Cost, and it is large.** Recall on prompt-engineered machine text falls from 41.9% to 9.7%.
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

**Why.** Calibration is good at the ends and poor in the middle (0.5–0.7 predicts 0.624,
actual 0.366). A gradient invites a reader to distinguish 0.55 from 0.62, which the model
cannot do. **Cost:** less pretty.

---

### 12. Human corpus not committed to the repository

**Decision.** Ship the source registry, the fetch script and SHA-256 hashes instead of the
text.

**Why.** Most of it is other people's writing under licences that forbid redistribution.
**Cost:** rebuilding requires network access and takes about ten minutes.
