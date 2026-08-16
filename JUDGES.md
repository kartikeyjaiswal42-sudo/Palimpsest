# Evaluating this in fifteen minutes

**Live:** https://palimpsest.amitynoidalibrary.workers.dev

Palimpsest is an AI-text detector for college admissions essays. The brief it was built
against asks for an instrument that shows its evidence rather than one that borrows a verdict
from a chat model, and asks for an honest account of what it gets wrong. This page is the
shortest path to checking both.

---

## 1. Verify the central claim yourself (2 min)

**No language model is ever asked for a verdict.** The observer is read for token
probabilities in a single forward pass. There is no `generate()` call, no prompt and no chat
template anywhere in the scoring path. Every number the interface shows is arithmetic done on
those probabilities.

That is a claim about source code, so it is enforced by tests that read the source:

```bash
pytest tests/test_no_generative_calls.py     # the Python build
cd edge && npm run test:wrapper              # the Worker build
```

The deployed Worker also reports it, and the deploy pipeline fails if it ever returns `true`:

```bash
curl -s https://palimpsest.amitynoidalibrary.workers.dev/api/health
# → {"status":"ok","corpusReferenceLoaded":true,"usesGenerativeModel":false,...}
```

## 2. Watch it show its working (3 min)

Open the live URL and press **Example it catches**. Then click any flagged sentence.

The evidence panel is not a summary of the reasoning — it **is** the reasoning. The named
per-feature contributions plus the intercept plus a stated remainder reconstruct the
classifier's logit to floating-point equality. A sentence scores what it scores because those
numbers sum to that.

Then press **Example it lets through**, which is the more instructive button and the reason it
exists. The highlighting lands on the rewritten paragraph exactly — and the tool **still
declines to flag the essay**, because 38% document confidence sits under the 80.7% threshold we
ship. That is the operating point costing us a catch, on screen rather than buried in a table.

## 3. Read the numbers, including the ones that hurt (5 min)

The full table is in the [README](README.md#what-it-does-honestly); the breakdown is
[docs/03-evaluation.md](docs/03-evaluation.md). Three things are worth checking specifically.

**The gradient is the finding, not the headline AUROC.** Recall falls from **94.8%** when the
generator is in the training pool to **63.6%** for a withheld model family. That number puts a
figure on how much of any detector's score is recognising *one generator* rather than
recognising machine prose.

**The failures are stated at the same volume as the successes.** Prose from the strongest
current models is caught **5.6%** of the time. Prose a careful writer composed to imitate a
model: **0 of 11**. GPT-3.5 prompted to evade: **38.7%** — a GPT-2-observer-build figure,
labelled as such wherever it appears, per the rule below.

**Which build produced which number is labelled on every line.** There are two — a served
build reading qwen3-30B on Workers AI, and a local GPT-2 build — and figures that were only
ever measured on the GPT-2 build say so, because a number from a model nobody is running does
not belong unlabelled in a table about the one behind the button.

That labelling exists because it went wrong. `tests/test_documented_numbers.py` — the test
written specifically to stop the documentation drifting — read the *unshipped* artifact set
while the server read the one named by `SUFFIX`. It was checking a model nobody runs, so it
passed happily while the README claimed **0.925 AUROC against a served 0.9576**, and a **17.8%
TOEFL false-positive rate against a served 10.87%** — the latter inside the safety warning
about how often the tool is wrong about a real student, and contradicted by the live site's own
limitations panel. The test now resolves the suffix the way the server does, and **fails rather
than skips** when a claim about the served build has no measurement behind it. Correcting it
reddened 13 of 15 claims. ([docs/10-development-record.md](docs/10-development-record.md).)

## 4. Three essays it gets confidently wrong, with theories (5 min)

**[docs/04-failures.md](docs/04-failures.md) Part 1** — this is the section the brief asks for.

| what it is | called machine at | the theory |
| --- | --- | --- |
| A one-sentence ESL essay (`ellipse:00109`) | **P = 0.977** | It has almost no sentence-ending punctuation, so the segmenter returns the **whole essay as one "sentence"**. `root_ttr` is a per-sentence measure; applied to a hundred times the word count it was designed for, it is inflated by construction. The essay is not rich in vocabulary — it is long for the thing being measured |
| A letter to a teacher (`persuade:2246059bfc70`) | **P = 0.976** | Salutations and sign-offs are the most formulaic text a student ever writes, and predictability is our strongest signal. We are reading **a letter's furniture** as evidence of authorship — made worse by `STUDENT_NAME`-style anonymisation tokens the student never wrote |
| A TOEFL essay on leadership (`liang_toefl:0051`) | **P = 0.953** | A single **+11.26** contribution is the whole verdict, and it comes from a feature comparing a sentence to the rest of its own essay. With four other sentences, "the rest of the essay" is an estimate from four points, and one unusual sentence makes every other one look unusual. There is a guard for spans that are too long and too short — **none for a document with too few sentences to establish a baseline.** That is the obvious next fix and it is not done |

**All three are ESL essays, and the doc says so rather than leaving it to be noticed.** They
are wrong for structural reasons — segmentation, formulaic furniture, too few sentences to
compare against — rather than because the writing resembles a machine's.

Part 2 covers larger mistakes the project made. Two are worth reading in full:

- **It scored 0.96 and then missed a real Gemini essay completely** — eighteen sentences, none
  flagged, machine share 0.0% — because every machine essay in the training pool was 2022
  GPT-3.5 output, so the features had encoded a *2022* model's habits. Two learned priors were
  pointing the wrong way for modern prose.
- **Every caller repaired the library on the way past, so nobody noticed it was broken** (§7).
  `Analyzer.analyze()` — the entry point a reader of this repository would call — used argument
  defaults that were wrong for serving, so `any_machine_probability` quietly became the single
  strongest *sentence* rather than a document probability. On one real essay it read **70.9%**
  where the fitted model reads **14.9%**: opposite sides of the "likely machine" line, from the
  same artifacts, in the same process.

The false-positive analysis on non-native writers ([docs/05-esl.md](docs/05-esl.md)) is the
other half of that honesty, and it contains a measurement that argues **against** the
convenient reading: on ELLIPSE, where every writer is a language learner, the false-positive
rate is **highest at the top** of the proficiency scale — 11.6% at holistic 3.0, 53.8% at 5.0.
What is being measured is fluency and taught structure, not nativeness.

---

## What this must not be used for

A **10.9%** false-positive rate on TOEFL essays means using this as evidence against a student
would be wrong roughly one time in nine, for exactly the students least able to contest it. The
interface says so on every result and the API returns the error rates in the response body. For
context, Liang et al. (2023) measured **61.22%** across seven commercial detectors on the same
essays — better is not the same as safe.

## If you want to go further

| | |
| --- | --- |
| [docs/14-submission-record.md](docs/14-submission-record.md) | what was built, measured, and **thrown away** — including two signal families that failed and are reported anyway |
| [docs/06-decisions.md](docs/06-decisions.md) | decisions we would be asked to defend |
| [docs/07-ai-usage.md](docs/07-ai-usage.md) | how AI tools were used building this |
| [docs/02-dataset.md](docs/02-dataset.md) | provenance, licences, and what the data does **not** cover |
| [docs/README.md](docs/README.md) | everything else, indexed |

## Running the whole thing locally

```bash
uv venv && uv pip install -e ".[data,dev]"
uvicorn palimpsest.api.app:app --port 8123        # scores via Workers AI; no weights downloaded
pytest                                            # 203 tests, no network, no model download
```

Rebuilding every number from raw sources — corpus fetch through evaluation — is the script
sequence in the [README](README.md#run-it).
