# Palimpsest

**An AI-text detector for college admissions essays that shows its working.**

A palimpsest is a manuscript where the earlier writing shows through the later. That is the
problem this tool is built for: not "is this essay AI", but *which parts of it are, and what
is the evidence.*

```
                    ┌──────────────────────────────────────────────┐
   essay  ─────────▶│  segment into sentences (offsets preserved)  │
                    └──────────────────┬───────────────────────────┘
                                       ▼
              ┌────────────────────────────────────────────────┐
              │  TWO OBSERVERS, neither of which gives a verdict │
              │                                                  │
              │  local GPT-2 ──▶ per-token logprob, rank,        │
              │  (124M, on CPU)   entropy, curvature             │
              │                                                  │
              │  n-gram model ──▶ how typical is this OF REAL    │
              │  of 1,000 human   ADMISSIONS ESSAYS specifically │
              │  essays                                          │
              └────────────────────────┬─────────────────────────┘
                                       ▼
                    ┌──────────────────────────────────────┐
                    │  43 interpretable features, 4 families│
                    │  likelihood · rank · corpus · style   │
                    │  + how each sentence compares to the  │
                    │    REST OF ITS OWN ESSAY              │
                    └──────────────────┬───────────────────┘
                                       ▼
                    ┌──────────────────────────────────────┐
                    │  logistic regression → calibrated p   │
                    │  the logit IS the explanation:        │
                    │  a sum of per-feature terms           │
                    └──────────────────────────────────────┘
```

## The one line that matters

**No language model is ever asked for a verdict.** GPT-2 is read for token probabilities in a
single forward pass — there is no `generate()` call, no prompt, and no chat template anywhere
in the scoring path. Every number the interface shows is arithmetic we do on those
probabilities. `tests/test_no_generative_calls.py` asserts this against the source, so it
cannot quietly stop being true.

## Run it

```bash
uv venv && uv pip install -e ".[data,dev]"     # or: python -m venv .venv && pip install -e ".[data,dev]"
uvicorn palimpsest.api.app:app --port 8123     # first run downloads GPT-2 (~500 MB)
open http://127.0.0.1:8123
```

Paste an essay, or press **Example it catches** / **Example it lets through** — the demo
deliberately ships a failure alongside the success. The second one is the more instructive:
the highlighting lands on the rewritten paragraph exactly, and the tool *still declines to
flag the essay*, because 74% document confidence is under the 97.4% threshold we ship. That
is the operating point costing us a catch, visible on screen rather than buried in a table.

Click any sentence to see the evidence behind its score. Analysis takes about 2 seconds for a
650-word essay on a laptop CPU.

To rebuild everything from scratch:

```bash
python scripts/fetch_corpus.py        # rebuild the human corpus from its sources
python scripts/fit_reference.py       # fit the n-gram reference
python scripts/build_features.py --sets all
python scripts/train.py               # fit + report cross-validated performance
python scripts/evaluate.py            # every held-out set
python scripts/find_failures.py       # the essays it gets confidently wrong
python scripts/ablate_length.py       # re-run the document-length ablation
pytest                                # 118 tests, no network, no model download
node scripts/verify_ui.cjs            # 23 end-to-end browser checks (needs the server up)
```

## What it does, honestly

Every number below is on data the model never trained on. The full breakdown, including the
trade-off curve, is in [docs/03-evaluation.md](docs/03-evaluation.md).

| | |
|---|---|
| Sentence AUROC, out-of-fold | **0.960** |
| Document AUROC, out-of-fold | **0.959** |
| Locating machine text inside a mixed essay | **AUROC 0.883**, seam found within 2 sentences in **70%** of documents |
| False positives on out-of-domain human essays | **0.0%** of documents |
| False positives on essays by English-language learners | **7.3%** of documents |

And the part that a leaderboard would hide:

| | |
|---|---|
| GPT-3.5 **prompted to evade detection** | only **6.5%** of essays caught |
| Prose a careful writer composed to imitate a model | **0 of 11** caught |
| TOEFL essays by non-native writers, wrongly flagged | **24.4%** |

The operating point is deliberately tuned so the tool almost never accuses a real student,
and it pays for that in recall. That is a choice, not a result, and
[docs/03-evaluation.md](docs/03-evaluation.md) shows what the other settings cost.

**For context on that last number:** Liang et al. (2023) measured a **61.22%** average false
positive rate across seven commercial detectors on the same 91 TOEFL essays. Palimpsest
scores 24.4% on that set. Better, and still far too high to use as evidence against anyone.

## The cleanest evidence that it measures what it claims

The ASAP corpus ships 88 student essays together with a version of each that a model
rewrote. Same author, same content, same subject — only the surface differs.

| | flagged as machine |
|---|---|
| The 88 original student essays | **0.0%** |
| The same 88 essays, rewritten by a model | **65.9%** |

That is a controlled comparison, not a correlation, and it is the strongest thing in the
evaluation.

## Three things this project got wrong first

The interesting work was mostly in catching our own mistakes. Full write-ups in
[docs/04-failures.md](docs/04-failures.md).

1. **We wrote the machine essays ourselves, and it destroyed the detector.** Eleven essays
   composed to read like LLM output. The strongest feature that emerged was
   sentence-length burstiness — which was measuring *our own instruction to "vary sentence
   length"*, not machine authorship. Against real GPT-3.5, the same feature points the
   opposite way. Prose *composed* to sound machine-generated does not carry the signature of
   prose *sampled* from a model. We switched to real model output; sentence AUROC went from
   0.79 to 0.96.

2. **A 0.988 AUROC that was partly a smart-quote detector.** Human essays came from web
   pages (88% contained `’`, 72% contained `“”`); machine essays came from an API (9.7% and
   **0%**). We normalise typography at ingest now. The score barely moved, which is itself
   the evidence that the rest of the signal is real.

3. **A length artifact that landed on exactly the people it should not — and a claim about
   it that went stale.** The document model gave "fewer sentences ⇒ machine" its largest
   weight (−3.09), learned from the fact that the machine essays in the corpus happened to
   be shorter. We removed it, measured a large improvement on short TOEFL essays, and wrote
   that down. Later pipeline changes invalidated the measurement and the write-up kept
   asserting it. Re-running the ablation as a script showed the improvement is **within
   noise** (p = 0.29) and that removal *costs* in-domain AUROC. We still removed it, on the
   principle that a corpus artifact should not be the model's strongest input — but the
   [docs now say the numbers disagree with us](docs/04-failures.md).

## Why non-native writers get flagged, with a theory

The false positives are not random. Ranked by confidence, the top ones are TOEFL and
PERSUADE essays, and the evidence panel names the same culprit each time — the sentence
*"Therefore, I prefer computer science"* is flagged on **stock vocabulary**.

ESL writing instruction explicitly teaches the discourse markers and the essay skeleton that
instruction-tuned models overproduce: *first of all, therefore, in addition, in conclusion*.
The detector is not detecting non-native writing. It is detecting **taught structure**, which
non-native writers use more because they were taught it more recently and more explicitly.

The measurement that supports this: on ELLIPSE, where every writer is an English-language
learner and each has a graded proficiency score, the false-positive rate **rises with
proficiency** rather than falling. And on PERSUADE, where the ELL flag is the only difference
between otherwise matched essays, the ELL group is flagged *less* often (0.0%) than the
native group (8.9%). Fluency is what we measure, not nativeness.

Full analysis: [docs/05-esl.md](docs/05-esl.md).

## Repository

```
src/palimpsest/
  text/segment.py        sentence splitting, character offsets preserved exactly
  scorer/local_lm.py     THE INSTRUMENT: GPT-2 → logprob, rank, entropy, curvature
  scorer/ngram.py        the corpus reference — "is this how applicants write?"
  features/              43 features in 4 families, each with a registry entry
                         explaining what it measures and which way we expected it to point
  detect/classifier.py   calibrated logistic regression + per-feature contributions
  detect/document.py     two document numbers, kept deliberately separate
  analyze.py             the pipeline; the API and training call the SAME method
  api/app.py             FastAPI
web/                     the interface (no build step)
scripts/                 fetch → fit → features → train → evaluate → failures
docs/                    approach, dataset, evaluation, failures, ESL, decisions, AI use
tests/                   118 tests: no model is ever asked for a verdict, and
                         the documented numbers must match the artifacts
```

## Documentation

| | |
|---|---|
| [01-approach.md](docs/01-approach.md) | which signals, and why those |
| [02-dataset.md](docs/02-dataset.md) | provenance, licences, what the data does **not** cover |
| [03-evaluation.md](docs/03-evaluation.md) | every held-out number and the trade-off curve |
| [04-failures.md](docs/04-failures.md) | three confident failures, with theories |
| [05-esl.md](docs/05-esl.md) | the false-positive study on non-native writers |
| [06-decisions.md](docs/06-decisions.md) | decisions we would be asked to defend |
| [07-ai-usage.md](docs/07-ai-usage.md) | how AI tools were used building this |

## What this tool must not be used for

It is an instrument for looking at text, not evidence about a person. A 22% false-positive
rate on non-native writers means that using this to accuse a student of cheating would be
wrong roughly one time in five for exactly the students least able to contest it. The
interface says so on every result, and the API returns the error rates in the response body.

## Licence

MIT for the code. The corpus is **not** redistributed — see
[docs/02-dataset.md](docs/02-dataset.md) for each source's terms and
`scripts/fetch_corpus.py` to rebuild it.
