# Dataset

The corpus is assembled from public sources and rebuilt with `scripts/fetch_corpus.py`.
Provenance for every source lives in code, in `src/palimpsest/data/sources.py`, with the
licence, the citation, whether the text provably predates ChatGPT, and what it does not
cover.

## What is and is not in this repository

**The human essay text is not committed.** Most of it is other people's writing under terms
that do not permit redistribution: published admissions essays are copyright the colleges
that published them, and PERSUADE and ELLIPSE are CC BY-NC-SA. What is committed is the
source registry, the fetch script, and a manifest with a SHA-256 of every document, so anyone
can rebuild the corpus and prove they rebuilt the identical one.

The machine essays we generated are ours, so they ship in `data/generated/`.

## Sources

### Human, in domain — used for training

| source | n | provable pre-ChatGPT? | licence |
|---|---|---|---|
| Liang et al. real college admissions essays | 70 | collected pre-Apr-2023; no per-essay dates | no LICENSE file; research use |
| Johns Hopkins "Essays That Worked" | 31 | **yes**, filtered to publication ≤ 2022-11-30 | © JHU, research use |

> **A trap worth recording.** The JHU year tags encode the *graduating class*, not the
> publication year — the tag `essays-that-worked-2020` returns posts published 2016-12-15.
> Filtering on the tag name to build a pre-ChatGPT set would silently mislabel the corpus by
> about four years. We filter on the API's `date` field.

### Machine — used for training

| source | n | generator |
|---|---|---|
| Liang et al. GPT-3.5 admissions essays | 31 | GPT-3.5, early 2023 |
| `modern_train` — ours, `scripts/generate_modern.py` | 135 | gemini-3.1-flash-lite, 2026 |

**Real model output, not an imitation.** Our first machine corpus was hand-composed and it
did not work; see [04-failures.md](04-failures.md#4-we-wrote-the-machine-corpus-ourselves-it-was-worthless).

**The modern corpus, and why it exists.** A detector fitted on GPT-3.5 alone returned 0.0%
machine share on a real Gemini essay ([04-failures.md #0](04-failures.md)). 567 essays were
generated across five Gemini-3 checkpoints — 522 subject-steered plus a 45-essay topic
control — answering the *same four prompts* as the Liang GPT-3.5 set, read out of that file
rather than rewritten, so the two machine sources differ only by generator. We wrote them, so
unlike the human corpus they are committed to this repository. Split four ways by
`scripts/split_modern.py`; only 135 enter training, for the reason in
[06-decisions.md](06-decisions.md).

**Length: a confound closed and a smaller one opened.** The human essays run to a median of
632–715 words and the GPT-3.5 machine essays to 261 — a 2.5× gap sitting exactly on the label,
and the reason `log_sentences` had to be deleted from the document model. The modern essays
are drawn at 400–500 words so the machine class sits inside the human range. But sentence
length, which used to carry nothing (18.5 vs 17.4 words, AUROC 0.494), now carries a little:
**18.5 vs 20.6 words, AUROC 0.598.** Modern models write longer sentences and our prompt asks
for 400–500 words; both contribute and they are not separated here.

### Mixed documents — used for training and localisation

139 documents built by `scripts/build_real_hybrids.py` from Liang's matched pairs: the same
essay in its original human form and after a model rewrote it. Splicing the human first half
to the machine second half gives a document whose two halves are both real, whose content is
continuous, and whose seam we know exactly. Split by base essay, half for training and half
for reporting.

### Human — held out for the false-positive study

| source | n | ESL information |
|---|---|---|
| Liang et al. TOEFL essays | 91 | all non-native; **median 104 words — length is confounded with language background here** |
| ELLIPSE | 3,888 | 100% English-language learners, each with a graded 1.0–5.0 proficiency score |
| PERSUADE 2.0 | 1,200 | matched `ell_status` flag, same prompts and graders — the best control available |
| Liang et al. ASAP 8th-grade essays | 88 | **provably pre-ChatGPT** (2012 competition); native, out-of-domain |

### DAIGT — research probe only, deliberately NOT in the detector

| source | n | role | licence |
|---|---|---|---|
| `daigt` — Kaggle "LLM · Detect AI Generated Text" community corpus, HF mirror `ramensoft/daigt_v3` | 4,189 (354 human / 2,344 machine over 17 pipelines) | diagnostic only | **none declared** on the mirror |

**Why it is here.** Every machine essay we generate comes from one pipeline: ours. That makes
provenance perfectly confounded with the label, and
[09-frontier-ceiling.md](09-frontier-ceiling.md) records a classifier that exploited it to
score AUROC 1.000 while flagging 0% of real GPT-3.5 essays and 17% of real students. DAIGT's
machine half was contributed by many different people using many different models, so
`meta.pipeline` identifies an independent producer and holding one out removes that shortcut
by construction. It is the only corpus here that can distinguish detection from bookkeeping.

**Why it is not in the detector, and this is the load-bearing part.** Two reasons, either of
which alone would be sufficient:

1. **The mirror declares no licence.** Its human half derives from PERSUADE (CC BY-NC-SA
   4.0) and its machine half was contributed by competition entrants under Kaggle rules.
   Absent an explicit grant we will not fold it into a shipped model. Using it to *measure*
   our own system and reporting what we found is a different act from redistributing a model
   fitted on it.
2. **It is the wrong genre.** DAIGT is argumentative coursework on PERSUADE prompts; this
   detector is for admissions personal statements. Training on it and scoring admissions
   essays flags 91.4% of real human ones — measured, not assumed.

`artifacts/detector.json` and `artifacts/detector_remote.json` both record `trainSources`,
and DAIGT appears in neither. It is fetched by `scripts/fetch_external.py` and consumed only
by `scripts/cross_pipeline_probe.py`. The raw file is **not redistributed in this
repository**; the fetch script re-downloads it.

### Reference corpus — never a labelled example

1,000 Ghostbuster/IvyPanda student essays (CC BY 3.0, the cleanest licence here) fit the
n-gram model and play no other role. They carry a mild post-ChatGPT contamination risk: the
upstream dataset was assembled 2023-01-23 from a back-catalogue with no per-essay dates.
Contamination in a background frequency model is tolerable in a way that contamination in
training labels is not — a few machine essays shift word frequencies slightly, they do not
teach the classifier a wrong answer.

## Normalisation applied to every document

**Typography.** Human essays came from web pages (88% contained `’`, 72% `“”`, 44% `—`);
machine essays came from an API (9.7%, 0%, 0%). Both classes are mapped onto one convention
at ingest. Digits are deliberately left alone — their absence from machine prose is a
property of the writing, not the pipeline. Full numbers in
[04-failures.md](04-failures.md#5).

**Paragraph structure.** Our largest human source ships with every newline stripped while our
generations had paragraph breaks, which makes paragraph structure a source marker. It is
removed from every document in both classes. Measured cost: flattening changes sentence
segmentation for 0 of 11 machine essays and 4 of 31 JHU essays, all of which contain
unpunctuated headings.

## What this corpus does not cover

Read this before trusting any number computed on it.

- **Two vendors, one of them four years old.** Machine examples are GPT-3.5 (2022) and
  Gemini 3 (2026). **OpenAI's current models, Claude, Llama, Mistral, DeepSeek and Qwen are
  entirely unmeasured** — the same sentence appeared here when Gemini was the unmeasured one,
  and it cost a 0% miss on the first modern essay a user tried. Held-out recall on a
  different Gemini family is 36.4% (22 documents), which is the best available guess for an
  unseen generator and is a within-vendor guess.
- **The generation gradient could not be built.** Every Gemini 2.x checkpoint and every `pro`
  model returned 429 on the free-tier key, so there is no 2024/2025/2026 trend — only Gemini
  3 flash and flash-lite. The thinking-heavy checkpoints managed 7–8 essays each before
  stalling, which is why `modern_unseen_family` has 22 documents rather than 200.
- **Topic is confounded with authorship in the modern corpus.** All 522 subject-steered
  essays answer one of 40 subjects we chose (beekeeping, welding, a flooded darkroom), while
  the human essays are about whatever their authors chose. The 45-essay `modern_control` set
  answers the bare prompts with no steering and scores 91.1% against the steered set's 93.0%,
  which is the evidence that the detector responds to prose rather than to topic — but it is
  one control, on 45 documents, from one checkpoint.
- **Published essays are a biased sample of human writing.** The admissions essays are ones
  colleges chose to publish as exemplary — edited, successful, and unrepresentative of a
  typical applicant's draft.
- **Topic coverage is uncontrolled.** The brief warns that a detector trained only on sports
  essays behaves unpredictably elsewhere. Our essays span the usual admissions subjects, but
  we did not stratify by topic and cannot report per-topic performance.
- **English only, US-centric.** Both the essays and the lexicons.
- **Scale.** 336 essays in the training pool. Small enough that a flexible model would
  memorise it, which is part of why the classifier is linear.

## Dead ends, recorded so nobody repeats them

- **ICNALE** — the zip downloads openly (19.7 MB, HTTP 200) but every entry is
  password-encrypted and the password is issued only to registered users.
- **TOEFL11 / ETS (LDC2014T06)** — catalogue page loads; data is behind LDC membership fees.
- **The official PERSUADE 2.0 distribution** — the GitHub repo contains only a README and
  three rubric PDFs; the CSVs are on Google Drive behind a confirmation interstitial. The
  HuggingFace mirror is the working route.
- **Hamilton College** — 30 essays with the strongest provenance in the whole search (every
  archive provably pre-ChatGPT, with hometown metadata). Four rapid requests triggered a
  JavaScript bot-challenge that now returns HTTP 202 with an empty body to every header
  combination. Recoverable with a real browser; not currently in the corpus.
- **AuTexTification** — the HuggingFace rows API returns 501 for datasets that run arbitrary
  Python.
- **`ramensoft/daigt_v4`** — despite the DAIGT name it contains no human text at all; it is
  8,306 rows of Llama output.
