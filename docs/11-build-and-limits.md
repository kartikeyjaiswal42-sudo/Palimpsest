# Palimpsest — What It Was Trained On, What It Cannot Do, and How to Rebuild It

*A companion to PROJECT.md. Written for the question set: how many essays, why the frontier
models get through, where the tool is weak, why it is not a wrapper, how the whole thing is
built from nothing, and what would actually improve it.*

---

## 1. What it was trained on

### The short answer

**336 essays / 7,937 sentences.** Of those sentences, 4,138 are human and 3,799 are machine —
deliberately near-balanced. This is `artifacts/detector_remote.json`, the shipped model.

### That is not all the essays. Here is where every one of them went

About **1,465 machine essays were generated** for this project. Only 135 of them are in
training. That looks wrong until you see the ledger, so here is the whole thing:

| corpus | essays | where it went |
|---|---|---|
| Gemini flash-lite, training split | 135 | **in training** |
| Gemini, held-out split | 115 | evaluation — the generator is in training, these essays are not |
| Gemini, unseen checkpoint | 250 | evaluation — measures a withheld checkpoint |
| Gemini, no-topic-steering control | 45 | evaluation — controls for topic rather than authorship |
| Gemini, unseen family | 22 | evaluation — measures a withheld model family |
| Claude haiku / sonnet / opus / fable | 398 | **all held out** — never in the shipped model |
| "palamiassist" 500-essay file | 500 | **excluded: contaminated** (see below) |
| real hybrids (part-human, part-machine) | 139 | 69 in training, 70 held out |
| Liang et al. GPT-3.5 | 31 | **in training** |
| Johns Hopkins human essays | 31 | **in training** |
| Liang et al. human admissions essays | 70 | **in training** |
| TOEFL / ELLIPSE / PERSUADE / domain-shift | 5,179 | held out — the ESL and false-positive studies |
| Ghostbuster IvyPanda human essays | 1,000 | the n-gram reference corpus, not training rows |
| DAIGT | 4,189 | research probe only; excluded from training |

So: **135 of the generated essays train the model, ~830 measure it, and 500 were thrown away.**
Held out is not wasted — a held-out essay is the only kind that can prove anything. Every
recall figure in this project exists *because* those essays were kept out.

### Why the 500-essay file was thrown away

Of its 500 rows, **489 were byte-identical to essays already on disk, and 281 were copies of
held-out evaluation sets.** Training on it would have produced excellent-looking numbers that
were pure memorisation — the model would have been tested on essays it had been trained on —
and it would have been very hard to catch later, because the file looks like a fresh corpus.

### Why only 135 Gemini essays entered training

**Because the human side is the scarce resource, and the balance is what keeps the tool from
becoming an accuser.**

There are **101 real human admissions essays** in existence for this project (31 Johns Hopkins
+ 70 Liang et al.). Machine essays are free — a few API calls. Human admissions essays are not:
they have to be published, provenance-checked and licence-checked. Every machine essay added
past the point where the two sides balance shifts the model's prior toward "machine", and a
model with a machine-heavy prior does not detect better — it accuses more.

This was measured twice, not assumed:

| what was tried | training prior | result |
|---|---|---|
| shipped balance | 47.9% machine | sentence AUROC **0.9576** |
| add all 567 Gemini essays | 69% machine | rejected before fitting — the prior alone makes it an accuser |
| add 238 Claude essays | 77.3% machine | AUROC 0.925 → **0.726**, flags **79% of genuine English-learner sentences**, Gemini recall → 0 |

The Claude row is the important one. It is not a hypothetical: the fit exists, and it is worse
at everything.

### The frontier-trained variant exists, and it is measurably worse

`artifacts/detector_frontier_remote.json` — 420 essays, 10,854 sentences, Claude in training.
Out-of-fold sentence AUROC **0.9074** against the shipped model's **0.9576**. It catches more
Claude and less of everything else. It is kept as a measured alternative, not shipped.

### The honest reading

The 336 is a **balance ceiling set by the human corpus, not a shortfall of effort or essays.**
More machine essays do not lift it. More *human admissions essays* do, and that is the single
highest-value thing anyone could add to this project — see §6.

---

## 2. Why it does not violate the project guidelines

*This section covers the substantive rules in the brief — the wrapper line, the instrument
rule, evidence, dataset honesty. Repository visibility and hosting are handled separately in
the development record.*

### The rule, quoted exactly

> **Not a wrapper**: a detector that sends the essay to a chat model and asks for a verdict is
> unreliable, cannot explain its reasoning, and takes an afternoon to build. We will be able to
> tell.

> Using a language model **as an instrument** is fine. Running text through a small local model
> for token probabilities, then doing your own analysis on those numbers, is real work and it's
> how good detectors are actually built.

> One line is worth drawing carefully: **the model must not make the judgement call while your app relays the verdict.**

The test is the model's **role**, not its size, vendor, or location.

### What the model is actually asked for

One call, per document, in a single forward pass:

```
{ prompt: <the essay>, raw: true, max_tokens: 1, prompt_logprobs: k }
```

There is no question in that call. There is no instruction, no system prompt, no chat
template, and nothing generated is read — `max_tokens: 1` exists only because the API requires
a number, and the continuation is discarded. What comes back is, for each token of the essay,
its log-probability and its true rank. Those are measurements of the text, in the same sense
that a thermometer measures a room.

`raw: true` is load-bearing. Without it the essay is wrapped in a chat template and the numbers
would describe a conversation rather than the essay.

Everything after that call is arithmetic in this repository: 43 features, standardisation, a
logistic regression, a calibration map, and thresholds fitted on documented data. **The verdict
is a sum of terms the code computes.**

### The clean formulation

> If you deleted the language model and swapped in a different scorer, would this still be your
> detector?

Here, yes. The feature definitions, the fusion, the weights, the calibration, the segmentation,
the thresholds and the bands are all the project's own, and the observer is replaceable — which
is proved by the fact that it **was** replaced, from GPT-2 to a 30B model, with everything
downstream unchanged. A prompt-and-relay detector has nothing left when you remove the model.

### It is enforced, not asserted — which is the part that matters

`tests/test_no_generative_calls.py` walks the abstract syntax tree of every file under
`src/palimpsest/` and fails the build on:

- any call to `.generate`, `.chat`, `.complete`, `.completions`, or `.create_completion`
- any import of `openai`, `anthropic`, `cohere`, `litellm`, `ollama`, or `google.generativeai`
- the strings `.generate(` or `apply_chat_template` appearing in `local_lm.py` at all

plus `test_at_least_one_source_file_found`, so the parametrised guards cannot pass vacuously on
an empty file glob. The same guard exists on the JavaScript side as
`edge/test/no-generative-calls.test.mjs`. `/api/health` reports `usesGenerativeModel: false`.

The guard was verified by injecting `import openai` and confirming the suite goes red.

### The "small local model" phrase

The brief's Notes say *"running text through a small local model for token probabilities."* The
default observer here is a 30B model on Cloudflare. Three reasons this is not a violation:

1. That sentence appears in the **Notes** as an example of acceptable technique, not as a specification. The rule is the sentence about the judgement call, and it is not crossed.
2. The brief also says *"use whatever language and stack you like. We care about what it does, not what it's written in."*
3. `PALIMPSEST_OBSERVER=gpt2` restores a fully local, fully offline run at a measured cost, and **the comparison between the two is one of the project's better findings** — GPT-2's statistics are inverted on English-learner prose (AUROC 0.132), which is the mechanism behind the false-positive problem the whole field has.

The honest disclosure that goes with it: with the remote observer, essay text leaves the
machine. `/api/health` reports `textLeavesMachine: true`, the footer states it, and a
regression test fails if that note and the running observer ever disagree again.

### The rest of the brief, point by point

| the brief asks for | how it is met |
|---|---|
| a working application, real interface, not a notebook | FastAPI app plus a hosted Cloudflare Worker, both serving the same interface |
| show **which parts** | per-sentence character offsets, probabilities, and multi-sentence passages |
| show **why** | named feature contributions that reconstruct the logit to floating-point equality |
| never a bare "73% AI" | two distinct quantities plus a three-band verdict with explicit abstention |
| sentence and passage level, human-then-polished | hybrid documents with known character spans are in training |
| build and document the dataset | ~12 sources with provenance, counts, licences and stated gaps |
| three essays it gets confidently wrong, and why | `docs/04-failures.md`, with mechanisms, plus six more |
| ESL false positives — did you spot it | `docs/05-esl.md`, with a measured mechanistic root cause |

### One place the wrapper could creep back in, and why it does not

The tempting shortcut is to have a model phrase the "why" for each flagged sentence. That would
put a model's words in the explanation slot. It is not done here: the evidence panel prints the
feature contributions, which **are** the computation rather than a paraphrase of it. There is no
natural-language generation anywhere in the product.

---

## 3. Why it cannot detect frontier Claude prose

This is the project's central negative result, and it is measured rather than conceded.

### The measurement

With the 30B observer, on held-out Claude essays, the capability gradient is monotone:

| model | document AUROC | sentence recall | document recall at a 5% false-accusation budget |
|---|---|---|---|
| haiku | 0.955 | 0.577 | — |
| sonnet | 0.735 | 0.353 | — |
| fable | — | 0.259 | — |
| opus | 0.631 | 0.083 | **0%** |

The same gradient appears *inside* Gemini — flash-lite 0.91 → flash 0.665 — which is what rules
out "it is a Claude problem" or "it is a vendor problem." Two independent vendors, same
direction, no confound between them. **The variable is model capability, not vendor.**

### Why a bigger observer does not fix it

GPT-2 124M → Qwen3-30B is a **250× scale-up of the instrument**. It lifted Haiku sentence recall
5.5× (0.105 → 0.577) and mid-tier Gemini document recall from 0.455 to 0.546. Claude Opus
document recall went from ~0% to **4.4%**.

The ceiling is **signal-bound, not compute-bound.** A better instrument reads a faint signal
more precisely; it cannot read a signal that is not there.

### What the features actually measure, and why frontier prose escapes them

Every statistical signal in this class of detector is a proxy for one idea: *machine prose is
more predictable and more generic than human prose.* That is true of cheap-model output and it
stops being true as models improve.

Here is the ceiling as a single document. A user supplied a Gemini-written admissions essay and
expected a catch. Document-averaged features against the training distributions:

| feature | this essay | human essays | machine essays | z vs human | z vs machine |
|---|---|---|---|---|---|
| `mean_logprob` | −3.426 | −2.796 ± 0.279 | −2.080 ± 0.337 | **−2.26** | −4.00 |
| `mean_log_rank` | 1.853 | 1.369 ± 0.192 | 0.938 ± 0.193 | **+2.52** | +4.73 |
| `frac_rank_top1` | 0.377 | 0.439 ± 0.038 | 0.522 ± 0.055 | −1.60 | −2.64 |
| `lrr` | 1.876 | 2.105 ± 0.135 | 2.313 ± 0.179 | −1.70 | −2.44 |
| `specificity_rate` | 2.396 | 2.827 ± 2.104 | 0.550 ± 0.690 | −0.20 | +2.68 |
| `novel_trigram_rate` | 0.880 | 0.825 ± 0.031 | 0.804 ± 0.036 | +1.76 | +2.10 |

Read the first two rows carefully. **On the two load-bearing observer statistics, this machine
essay is further from the machine distribution than the average human essay is.** The observer
finds it *harder* to predict than typical human admissions prose.

The cause is not mysterious. The essay is model output carrying real, specific, personal
content — a named product, a named trade show, a named calculation. **Rare proper nouns are
genuinely improbable tokens**, and what characterises generic model prose is their absence. The
detector is reading the content, and on the axis it measures, this content is human-shaped.

The consequence is the important part:

> **There is no threshold that catches this essay and leaves the human essays alone.** Turning
> up sensitivity until it is flagged would flag the human essays that look most like it — which
> are the ones with the most specific personal detail. The students who wrote the most personal
> essays would be accused first.

That is why the answer is a refusal rather than a number.

### The register finding, stated plainly

The detector scores **abstract** sentences high and **concrete** ones low. In one essay: 0.94 for
*"the world is full of broken systems"*, 0.04 for *"I didn't fix all of them."* It is detecting
generic aspirational writing, not authorship. When a frontier model is prompted well enough to
write concretely, the signal disappears — because the signal was never authorship in the first
place.

### Every repair that was tried, and what it returned

| attempted repair | outcome |
|---|---|
| 250× larger observer | Haiku 5.5×, mid-tier Gemini +0.09; **Opus unmoved** |
| Fast-DetectGPT conditional curvature | **exactly zero** — AUROC 0.9748 → 0.9747, frontier 27.6% → 27.6%; the features correlate r = +0.87 with one already present |
| supervised stylometry | scored a perfect **1.000** and was reading *which pipeline made the file* — it flags 0% of real GPT-3.5 essays and 17% of real students |
| train on Claude directly | works (that essay 3% → 76%) but costs Gemini recall 80.7% → 40.4% and AUROC 0.945 → 0.901 |
| two specialists, take the higher score | **6× frontier recall at an identical false-accusation rate** — the best trade measured, and not yet shipped |

### The honest ceiling

**Unedited cheap-model output is reliably detectable. Opus / Sonnet / Gemini-Pro-class prose in
a 650-word essay is not**, by any method in this repository.

And the product consequence, which is built in rather than written in a footnote: **a low score
is not evidence of a human author.** The bottom band reads *"no evidence of machine writing"*,
never *"human"*, nothing in the interface is coloured green, and `canExonerate` is `false` on
every response. This tool can raise suspicion. It can never clear anyone.

---

## 4. What it lacks

Ordered by how much each one matters.

### 4.1 No ESL-authored admissions essays — the empty cell

The corpus has ESL writing (TOEFL, ELLIPSE, PERSUADE) and admissions essays (Liang, Johns
Hopkins), but **no admissions essays written by non-native speakers.** That is the exact person
this tool must not fail, and there is no measurement of what happens to them.

The nearest available evidence was assembled from proficiency labels and it is not reassuring:
refusal correlates with weaker English at ρ = +0.232 (p < 0.001, n = 260). One feature carrying
that signal was removed. The cell is still empty, and no arithmetic fills it.

### 4.2 Frontier recall is 15.8% at best

See §3. The shipped generalist catches **0%** of held-out Claude documents at the operating
point; the unshipped two-specialist ensemble reaches 15.8%.

### 4.3 It abstains a great deal

At the current bounds the tool declines to give a verdict on a large fraction of documents —
88% of human essays and 65% of machine essays under the build where this was measured. That is
the honest price of a bounded false-accusation rate *and* a bounded false-refusal rate. It is
correct, and it is also the number a paying user would argue about.

### 4.4 Machine text comes from few pipelines

Every machine essay was produced by this project's own generation code. Provenance is therefore
correlated with label, which means no internal split can fully validate a supervised result —
a model can learn "which pipeline made this file" instead of "is this machine-written," and the
stylometry probe proved that failure is real, not theoretical.

### 4.5 The genre gate extrapolates

It was fitted against *student* writing. Technical documentation passes as in-domain at 0.985.
On genres it never saw, it is confident without cause.

### 4.6 Localisation regressed and was never diagnosed

Finding the seam in a part-human, part-machine document: within 2 sentences **70% → 39%** after
the 9 August retrain. Undiagnosed.

### 4.7 Two evaluation claims were never re-run on the current observer

The "38.7% caught when prompted to evade" and "0% adversarial" figures were measured on the
GPT-2 build. They are labelled as carried over, not passed off as current, but they are not
measurements of the shipped detector.

### 4.8 Reproducibility of the headline numbers

The remote observer is bearer-gated to protect a daily allowance, so a reviewer who clones the
repository gets the `gpt2` path, whose numbers are measurably worse (sentence AUROC 0.925 vs
0.958). The headline configuration cannot be reproduced from a clean clone.

### 4.9 Non-determinism on the live observer

Re-running the same essay three times moved a sentence score by up to 8.7 points and document
confidence by 1.7. The band held 6/6, and the variation is measured and disclosed — but the tool
is not bit-deterministic in its default configuration.

### 4.10 Licensing blocks commercial use

PERSUADE is **CC BY-NC-SA** — non-commercial. Fine for a hackathon; a genuine blocker for a paid
product without replacing that corpus.

### 4.11 Scope

English only. One genre (undergraduate admissions essays). One document at a time, with no
account of a student's other writing and no access to draft history.

---

## 5. Building the whole thing from scratch

What follows is the complete path from an empty directory to a deployed detector. The order
matters: each stage consumes the previous stage's artifact.

### Stage 0 — environment

```bash
pip install -e .          # or: uv sync
pytest                    # the guards should pass before anything is built
```

### Stage 1 — the human corpus

Human essays are the constraint on everything downstream (§1), so they come first.

```bash
python scripts/fetch_corpus.py                  # every source
python scripts/fetch_corpus.py --only liang_college_human jhu
```

What is needed, and why each one exists:

- **In-domain human admissions essays** — the thing being protected. These are rare, and they set the size of the whole training set.
- **A large general human corpus** — 1,000 essays here, used to build the n-gram reference, which is the body of writing every sentence is compared *against*. This is what makes "unusual phrasing" measurable rather than a vibe.
- **An at-risk population** — English-learner writing, with proficiency labels if you can get them. This is not optional: it is where the operating point gets calibrated.
- **An out-of-genre set** — writing that is not admissions essays, for the genre gate.

Record for each source: where it came from, how many, its licence, and what it does not cover.

### Stage 2 — the machine corpus

```bash
python scripts/generate_modern.py --target 100
python scripts/generate_modern.py --models gemini-3.6-flash
```

Four rules learned the hard way, each of which cost a rebuild when broken:

1. **Never write the machine essays yourself.** The first corpus here was hand-composed and it was worthless: the strongest feature turned out to be measuring the prompt's own instruction to "vary sentence length," and against real model output that feature points the opposite way. Sentence AUROC went 0.79 → 0.96 on switching to real output.
2. **Pre-register the subject split before generating.** Decide which subjects are training and which are held out *first*, so no topic can both teach and test. Otherwise you measure topic recognition and call it detection.
3. **Match the length distribution to the human essays.** Not doing this here flipped the `n_words` weight from −1.17 to +0.148 and caused three real students to be accused at P = 0.977.
4. **Normalise typography before measuring anything.** An early 0.988 AUROC was partly a smart-quote detector: human essays were 88% curly-quoted, machine essays 0%.

Then the hybrids, which are the realistic attack — a human essay with a machine-polished
paragraph, at known character offsets:

```bash
python scripts/build_real_hybrids.py
```

### Stage 3 — the reference model

```bash
python scripts/fit_reference.py
```

Builds the n-gram reference from human prose. This backs the corpus-relative features, such as
`novel_trigram_rate`, which asks whether a phrasing appears anywhere in a body of real student
writing. Note that this is a **membership** test, so any later compression of it has to be exact
rather than approximate.

### Stage 4 — features

```bash
python scripts/build_features.py --sets train
python scripts/build_features.py --sets all
```

This is where the observer runs: one forward pass per document, reading per-token log-probability
and rank. 43 features in 7 groups — likelihood 6, rank 6, corpus 4, rhythm 9, register 10,
context 6, composite 2. Each carries a registered label, unit and expected direction, so a
feature whose sign is backwards is visible rather than silently absorbed into a weight.

Two things worth building in from the start:

- **A reliability flag per sentence.** Under ~5 tokens or over ~90 words, a sentence cannot be measured. Make it a first-class field and make sure *every* consumer honours it — here it was computed, displayed, then ignored by the aggregator, and it caused the worst bug in the project.
- **Cache the observer output on disk**, keyed by text hash. It makes retraining cheap. Be aware the cache then contains the essays themselves, token by token, and must be treated as personal data.

### Stage 5 — the model

```bash
python scripts/train.py
```

Standardisation, then logistic regression, then a calibration map. Two decisions that carry the
whole project:

- **Group your cross-validation by essay, never by sentence.** Sentences from one essay are not independent; splitting by sentence leaks and inflates every number.
- **Keep the classifier linear.** The evidence panel is only honest if the logit *is* the sum of the displayed contributions. A gradient-boosted model would score better and would make the "why" a story about the computation instead of the computation.

Then aggregate sentences into a document verdict, keeping the two quantities separate — extent
(`machine_share`) and probability (`any_machine_probability`). Do not collapse them: an essay with
one polished paragraph has a low share and a high probability, and one number hides exactly that.

### Stage 6 — the operating point

```bash
python scripts/fit_bands.py --suffix _remote
python scripts/fit_genre_gate.py --suffix _remote
```

This stage is where a detector becomes either a tool or a hazard.

- **Calibrate on the population most at risk**, not in-domain. A threshold calibrated on the essays it was trained on does not transfer.
- **Use an exact upper bound, not the observed rate.** A threshold chosen because its observed false-positive rate hits 5% makes that rate optimistically biased by construction — it shipped 8.0% here. A Clopper–Pearson upper bound gives *"the true rate is at most 5%, with 95% confidence"* and moved held-out ESL false positives from 0.080 to 0.039.
- **Bound the false-refusal rate too.** Refusing to score somebody is a claim and needs evidence like any other. Not bounding it made the gate refuse the application's own demo essay by 0.015.
- **Validate the gate against authorship.** Fit it with both human and machine in-domain essays and check it passes them at the same rate — 95.5% vs 94.6% here. A gate correlated with authorship is a second detector in disguise, converting low recall into high abstention. `fit_genre_gate.py` refuses to save if that gap exceeds 10%.

### Stage 7 — evaluation and honest reporting

```bash
python scripts/evaluate.py
python scripts/find_failures.py
python scripts/ablate_length.py
```

Report on held-out sets only, split so that the half used to calibrate thresholds is not the half
used to report. Publish single-feature AUROCs so a reader can see which signals carry weight and
which are decoration.

**Then pin the prose to the artifacts.** `tests/test_documented_numbers.py` compares every headline
figure *and* every quoted failure example against the JSON on each run. Three of the four problems
found in the first audit were prose that had drifted from artifacts it once described. Without
this test, that drift is invisible and permanent.

### Stage 8 — the interface

Per-sentence heat map over the original text, an evidence panel that prints
`intercept + shown + remainder = logit` so the arithmetic can be checked on screen, a three-band
verdict, and explicit wording for the bottom band that never says "human."

### Stage 9 — deployment, if you port it

The risk in a port is that it silently becomes a different detector, which strips the accuracy
numbers of their meaning. The defence is a parity harness: run both implementations over hundreds
of real documents and compare **every** feature, logit, probability, threshold decision, evidence
bar and bootstrap endpoint, with the observer's token stream held identical. Here: 262 documents,
5,527 sentences, 364,056 values, largest disagreement 6e-15. Confirm the harness can fail by
mutating the code.

### The rule that made the difference

**Every guard must be proved able to fail.** Reintroduce the bug, watch the test go red, then
remove it again. A test that has never failed proves nothing — and one of the tests in this
project was passing on a field name that did not exist, reading `null` and reporting success.

---

## 6. How it could be improved

Ranked by measured value per unit of effort.

### 6.1 Source real human admissions essays — including ESL-authored ones

This unlocks two things at once: it fills the empty cell in §4.1, and it raises the balance
ceiling in §1 so more machine essays can enter training. **Fifty real ESL-authored admissions
essays would be worth more than any further accuracy work.** Everything else on this list is
downstream of the human corpus being small.

### 6.2 Ship the two-specialist ensemble

Already measured: **6× frontier recall at an identical false-accusation rate**, for 9 points of
Gemini recall. The artifacts exist (`detector_frontier_remote.json`, `ensemble_probe.json`); what
remains is wiring the second detector into the serving path, refitting the bands at the same
budget, and updating the headline numbers everywhere they appear. This is the best trade measured
on the project and it is not yet in the product.

### 6.3 Ingest a multi-pipeline corpus (RAID / MAGE)

Free, on HuggingFace, and the precondition for any supervised result being trustworthy: it breaks
the confound between provenance and label described in §4.4. The caution learned from DAIGT is
that it must be used **within genre** — leave-one-pipeline-out gives a median 1.000 AUROC, but
trained on coursework and pointed at admissions essays it flags 91% of real human ones.

### 6.4 Generate more frontier training essays

400 Claude essays bought 15.8% recall in the ensemble, and that curve has not flattened. This is
API credit, not GPU time — and note the measured result that **compute does not help**: a 250×
larger observer moved Claude recall 0% → 0%. Spend on data, not on training.

### 6.5 Run Binoculars

`src/palimpsest/scorer/binoculars.py` is written and has **never been executed** — it was
abandoned because the observer/performer model pair would not fit in 8 GB of RAM. It is the
strongest published frontier-detection method. Honest expectation: published Binoculars numbers
also degrade on frontier prose, so expect it to move the ceiling a little rather than break it.
It is worth running precisely because it is a real experiment with a real chance of a negative
result.

### 6.6 Authorship consistency rather than authorship detection

The only untried idea rated for the frontier. Instead of asking *"was this written by a machine?"*
— which §3 shows is unanswerable for good models — ask *"is this the same person who wrote their
other work?"* That question is **capability-independent**: it does not degrade as models improve,
because it compares a document against a specific writer rather than against a generic human
distribution. It needs a second sample of the student's writing, which changes the product from a
paste box into something an institution integrates.

### 6.7 Per-genre models instead of one general model

The measured constraint is that supervised detection **generalises across generators and not
across genres**. One model shipped across admissions essays, coursework and lab reports will
produce a very high false-positive rate on whichever genre it was not fitted to. Either fit per
genre, or keep the gate and refuse out-of-genre documents — which is what happens now.

### 6.8 More calibration data

The bounds are conservative because the calibration sample is small, and a Clopper–Pearson bound
tightens as n grows. More at-risk essays buys recall for free — no change to the model, only to
the confidence interval around its threshold.

### 6.9 Close the reproducibility gap

Either ship a scoped token, or ship a response cache covering the evaluation corpus, so a
reviewer can reproduce the headline numbers rather than only the weaker local ones.

### 6.10 Diagnose the localisation regression

70% → 39% since the 9 August retrain, and the seam-finding case is the brief's realistic scenario.

---

Numbers in this document come from the repository's own artifacts and documents:
`artifacts/detector_remote.json`, `artifacts/detector_frontier_remote.json`,
`artifacts/missed_essay_diagnosis.json`, `docs/02-dataset.md`, `docs/04-failures.md`,
`docs/05-esl.md`, `docs/09-frontier-ceiling.md`, and `PROJECT.md`.
