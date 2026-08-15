# Two attempts on the frontier ceiling, one live defect, and the control that killed the good one

docs/09-frontier-ceiling.md ends with three things worth doing and one not worth doing. This
document reports two of the three, and the result is one large negative, one modest positive,
and a defect in the shipped product that neither attempt was looking for.

The negative is the interesting half, because it arrived disguised as the largest result in the
project.

---

## 1. The attempt: read what independent observers DISAGREE about

Every statistic in this project so far reads ONE observer's ABSOLUTE surprisal. That family is
confounded by how hard the content is, and docs/09 closes with the document that proves it: a
Gemini-written essay whose `mean_logprob` sat **2.26 SD further from the machine distribution
than human essays do**, because it carried real proper nouns (`TheNectar`, a solar induction
calculation). The observer found it *harder* to predict than typical human prose, so every
absolute statistic voted "human".

Rare tokens are hard for *every* model. That is the opening. If content difficulty is
common-mode across independent observers, a statistic built from their DISAGREEMENT cancels it,
and what survives is not "was this surprising" but "was this surprising in a way idiosyncratic
to one model's view of English".

The hypothesis: **frontier machine prose sits where independent strong models agree, and human
prose contains choices no model would have made.**

This is not "buy a bigger observer", which docs/09 measured and correctly rejected (124 M → 30 B
moved Opus from 0.458 to 0.695 AUROC and 0% recall to 0% recall). A *ratio between* observers is
a different measurement, and it is the one Binoculars is built on.

### What was built

`scripts/consensus_probe.py`. Three Workers AI models return per-token log-probability **and
true rank**, each aligned to character offsets: qwen3-30B-a3b, llama-3.3-70B, mistral-small-24B.
Three vendors, three separate pretraining runs. `llama-3.1-8b` is in the observer's allow-list
but returns the legacy `{response}` shape with no logprobs, so it cannot be an observer.

Character offsets are load-bearing. The three tokenizers cut one 571-word essay into 666, 667
and 665 tokens, so nothing can be compared token-by-token; offsets are the common frame, and
words are the unit all three can be asked about.

Fourteen features, all statements about disagreement rather than surprisal: consensus top-1
rate, log-probability spread and range, rank spread, a hard-disagreement rate, mean pairwise
correlation, the per-word min/max envelope, and a Binoculars-family ratio dividing surprisal by
cross-observer spread. It is **not** Binoculars: that needs the full next-token distribution
from both models, and Workers AI gives the realised token plus at most a top-20 head.

Protocol identical to `fusion_probe.py`, imported rather than copied so the estimator, the null
and the metric cannot drift: leave-one-generator-out, permutation null on test labels,
TPR at a 5% false-accusation budget. Cost 3.1 neurons per essay for qwen, 18.0 for llama, 21.3
for mistral, against a free allowance of 10,000/day — so `--budget` is enforced by a pre-flight
estimate that refuses to start and a mid-run meter that stops.

### The result, before controls

Arm A is the control on the harness: the 7 single-observer statistics must reproduce
`fusion_probe`'s published numbers. They do, exactly.

| held-out generator | A baseline AUROC / TPR@5% | **B consensus AUROC / TPR@5%** |
|---|---|---|
| cheap Gemini | 0.749 / 0.000 | 0.940 / **0.867** |
| mid Gemini | 0.604 / 0.000 | 0.820 / **0.364** |
| cheap Claude | 0.949 / 0.800 | 0.996 / **1.000** |
| mid Claude | 0.616 / 0.000 | 0.955 / **0.700** |
| **frontier Claude** | 0.637 / **0.000** | 0.960 / **1.000** |

Frontier prose, 0% recall to 100% recall, on a generator held out entirely, p < 0.0001.

That is the largest number this project has ever produced and it is also the exact shape of its
two largest mistakes: an AUROC 0.988 that was substantially a smart-quote detector
(docs/04 #2), and a stylometric classifier at AUROC 1.000 and 100% recall on held-out frontier
Claude which flagged **0%** of real GPT-3.5 essays from another collection (docs/09 R3).

So it was attacked before it was believed.

---

## 2. Six controls. The first two passed and the result did not survive the rest

`scripts/consensus_controls.py`.

### C1 — typography. Passed.

83% of the human documents in this benchmark carry a curly apostrophe and 70% carry curly
quotes, against **0% of every machine set** — our generation pipeline folded them to ASCII.
Cross-observer features are *more* exposed to this than perplexity is, and mechanically so:
three tokenizers disagree about how to cut a curly apostrophe, so a document containing one has
genuinely higher cross-model disagreement, which is the quantity every feature here reads.

Folding both classes identically (reusing `stylometry_probe.normalise_typography`, not a second
copy) moved frontier Claude from 1.000 to **0.967**. The leak is real and it is not the
explanation. Normalisation slightly *strengthened* the disagreement signal, which says
typography was noise in the alignment rather than the signal.

### C6 — collinearity. Passed.

Each consensus feature regressed on the 7 baseline statistics: R² of 0.559 at the highest,
0.04–0.24 for the disagreement family. The features genuinely are new information relative to
the baseline. That turns out not to be the same thing as authorship information.

### C3 — ablation. Failed.

`x_min_lp` and `x_max_lp` are the per-word envelope across observers, which is absolute
surprisal wearing a hat, and they are the strongest single features.

| feature set | frontier Claude AUROC / TPR@5% |
|---|---|
| all 14 | 0.957 / 0.967 |
| **without the envelope (12)** | 0.760 / **0.000** |
| **envelope only (2)** | 0.681 / **0.000** |

Neither half carries it. The result needed the specific combination, which is the signature of
a fit rather than a signal.

### C2 — length. Failed.

The human reference pools 591-word college essays with 134-word TOEFL responses, and both a
correlation and a variance are length-sensitive. Restricting to a length-comparable human
reference makes the numbers *erratic* rather than lower: cheap Claude collapses to 0.684 / 0.000
while mid Claude rises to 1.000 / 1.000. Instability of that size across generators, on n=30
against a 25-document reference, is fitting noise being reported as capability.

### C4 — a human corpus we did not collect. Failed.

JHU *Essays That Worked*: real admissions essays, same genre, gathered by somebody else.
Holding out another of Liang's files proves nothing, because the collection's conventions stay
in training.

| false-positive reference | frontier Claude TPR@5% |
|---|---|
| Liang human (ours throughout) | 0.967 |
| **JHU human** | **0.167** |

And with the JHU reference the *baseline* beats consensus on four generators of five. The gain
was calibration against one collection's human prose, not detection.

### C5 — machine text somebody else generated. Failed, decisively.

This is the control docs/09 R3 names as the only one that distinguishes detection from
bookkeeping, and the reason is structural rather than suspicious: **every machine document in
this benchmark came from our generation harness** — both Gemini tiers and all four Claude
checkpoints, our prompts, our subject list, our post-processing. Leave-one-generator-out holds
out a *generator* and never holds out the *pipeline*. Five generators share whatever our
pipeline imprints, so holding one out cannot reveal it.

Fitted on humans plus all five of our generators, tested on Liang's own GPT-3.5 admissions
essays — same genre, 2023, no connection to our harness:

| test positives | human reference | AUROC | flagged @5% FPR |
|---|---|---|---|
| foreign GPT-3.5 | ours | 0.832 | 0.567 |
| **foreign GPT-3.5** | **JHU** | 0.686 | **0.000** |
| **foreign GPT-3.5, evade-prompted** | **JHU** | **0.490** | **0.000** |

Foreign machine text against a foreign human reference: **nothing flagged, AUROC at chance.**

### Verdict

The cross-observer result is a **negative**. It learned properties of our generation pipeline
and of Liang's human files. It reproduces docs/09 R3's failure by a new route — through model
logits rather than raw text — which is worth knowing, because logits felt like they should be
immune to a provenance shortcut and are not.

The standing rule is now stronger than docs/09 stated it. It is not only supervised models over
raw text that cannot be trusted on this corpus; it is **any** fitted combination, including one
over cross-model statistics. Neither half of the confound can be removed by holding out a
generator, and both halves have to be foreign before a number means anything.

Cost of learning this: about 5,100 neurons, inside one day's free allowance.

---

## 3. The attempt that partly worked: find the polish, using the document as its own reference

The brief names the realistic case: *"a paragraph a person wrote and a model later polished."*

Judging a document alone compares it to a *population*, and carries every confound this project
has documented — proficiency, genre, topic, how specific the content happens to be. The
frontier ceiling is the statement that frontier prose sits inside the human population on those
axes.

Judging a sentence *against the rest of its own document* compares it to one author. Topic,
genre, proficiency and register are held constant by construction. Different measurement,
different ceiling — and it had never been evaluated at the frontier here. The `localisation`
eval set is GPT-3.5/GPT-4-era rewrites; the ten frontier-polished hybrids sit in `adversarial`,
where only DOCUMENT recall was reported (0.0%).

`scripts/polish_probe.py`. Not new: `features/context.py` already ships six self-relative
features and already makes this argument. What is new is that only **three of the 43** features
are z-scored against the document, the rest can be for free from rows already on disk, and the
shipped classifier is fitted to do both jobs at once — a wholly machine document has no internal
discontinuity, so half its training signal actively teaches the context features to stay quiet.

**The leak this design had to avoid.** `build_real_hybrids.py` splices a human head to a machine
tail. The cut point moves, but the machine half is always LATER, so `rel_position` predicts the
label almost perfectly while learning nothing. It is excluded, and the exclusion is printed. The
frontier hybrids replace a span in the MIDDLE, so they test the leak as well as the frontier.

**Controls.** Grouped by base document. `domain_shift` is unusable as the human control — 86 of
its 88 documents ARE the hybrid bases — and 53 ESL documents are excluded for the same reason;
checked, not assumed. And the training and test rewrites come from **different pipelines by
construction**: fitted on Liang's GPT-era rewrites, tested on ours. That is the C5 control
Track 1 failed, satisfied here by the shape of the data.

### The operating point is the finding

| | GPT-era polish | **frontier polish** |
|---|---|---|
| document hit rate @ 5% **sentence** budget | 0.935 | **0.900** |
| document hit rate @ 5% **document** budget | 0.532 | **0.100** |

The first row is the tempting one and it is not a defensible number. A reader is handed a
document, not a sentence; an essay holds ~19 sentences, so a threshold that flags 5% of
sentences flags at least one sentence in **30.7% of unedited human essays**. Calibrating per
sentence and reporting per document is scoring the easy question and billing the hard one.

At the honest operating point the GPT-era gain is real and the frontier is not:

| arm | GPT-era docHit | docPrec | frontier docHit |
|---|---|---|---|
| absolute features only | 0.338 | 0.954 | 0.000 |
| **the 5 shipped context features** | **0.072** | 0.571 | **0.000** |
| absolute + all 37 new LOO z-scores | **0.532** | 0.956 | 0.100 |

So: a real 0.338 → 0.532 improvement on the case the brief names, at 0.956 precision, holding
the accusation rate at 5% of documents — and the five self-relative features currently shipped
contribute almost nothing on their own.

**And the ceiling holds.** Frontier polish reaches 0.100 — one document in ten, n=10, 95%
Clopper-Pearson interval roughly 0.006–0.445. This is a *stronger* statement of docs/09's
ceiling than docs/09 makes: frontier prose is not reliably detectable at a defensible
false-accusation budget **even when the author's own prose is in the same document as a
reference**.

Shipped as `artifacts/polish_head.json` via `scripts/fit_polish.py`, as a separate head that
never touches the sentence model, the document model or their thresholds. Feature count is
swept and printed rather than chosen: k=37 was the only value within one standard error of the
best.

---

## 4. The defect both attempts walked into

Not a claim the interface got wrong. A number it never reported.

**At the shipped sentence-flag threshold, unedited human documents carry flags at a rate the
product does not disclose.** 30.7% of clean human essays contain at least one sentence flagged
at a 5% per-sentence budget; on the ESL corpus, by measured proficiency, the rate runs
**50–79%**. It *rises* with proficiency — 79.3% at holistic 4.5 against 54.4% at 2.0 — because a
stronger writer's prose varies more from sentence to sentence, and this family reads variation.

That is the same class of defect as the `find_passages`/`aggregate` contradiction in PROJECT.md
§2, reached by a third route: the document verdict is calibrated at document level and the heat
map is calibrated at sentence level, and the reader consumes the heat map. A document can
honestly say *insufficient evidence* while painting a quarter of a second-language writer's
essay in machine colour.

Consequences taken:

* the polish head is calibrated on the per-document maximum, so 5% of clean documents carry a
  flag **by construction**;
* the measured per-document flag rate is stated in the interface rather than left implicit;
* the ESL-by-proficiency table above is published, including the direction, which is the
  opposite of the one this project has documented elsewhere.

---

## 5. What follows

Of docs/09's three suggestions, one is now measured to a wall and one partly delivered.

**Measured to a wall:** better statistics of the same kind. Absolute surprisal, cross-observer
disagreement, and Fast-DetectGPT curvature (docs/09) have each been built and each bought
nothing usable at the frontier.

**Partly delivered:** more reference per decision. Comparing a sentence to its own document is
the cheapest version of it and it works one era back, not at the frontier.

**Still untried, and now the only lever left:** comparison against *the same student's other
writing*. It never models the machine, so model capability cannot defeat it, and it is what a
school actually holds. It is untried here for a data reason and the reason should be stated
plainly: no corpus on hand carries multiple documents per identified author in this genre.
PERSUADE has prompt and ELL flags but no student identifier; ELLIPSE is one document per
writer. Sourcing even fifty matched pairs would test it, and nothing in this document
substitutes for that.

**Artifacts.** `consensus_probe.json` · `consensus_probe_normalised.json` ·
`consensus_controls.json` · `polish_probe.json` · `polish_head.json`.
