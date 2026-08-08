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

**Real model output, not an imitation.** Our first machine corpus was hand-composed and it
did not work; see [04-failures.md](04-failures.md#4). Sentence lengths in the training pool
match across classes (18.5 vs 17.4 words, AUROC of length alone 0.494), so the classifier
cannot win by measuring length.

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

- **One generator.** Every real machine example is GPT-3.5. GPT-4, Claude, Llama and Gemini
  are entirely unmeasured. This is the largest gap in the project.
- **Published essays are a biased sample of human writing.** The admissions essays are ones
  colleges chose to publish as exemplary — edited, successful, and unrepresentative of a
  typical applicant's draft.
- **Topic coverage is uncontrolled.** The brief warns that a detector trained only on sports
  essays behaves unpredictably elsewhere. Our essays span the usual admissions subjects, but
  we did not stratify by topic and cannot report per-topic performance.
- **English only, US-centric.** Both the essays and the lexicons.
- **Scale.** 201 essays in the training pool. Small enough that a flexible model would
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
