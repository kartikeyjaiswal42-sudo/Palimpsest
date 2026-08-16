# Documentation

**Evaluating this project?** [JUDGES.md](../JUDGES.md) is a fifteen-minute path through it.
[PROJECT.md](../PROJECT.md) is the full written report. Everything below is the working record
behind them.

## The argument

| | |
| --- | --- |
| [01-approach.md](01-approach.md) | which signals, and why those |
| [02-dataset.md](02-dataset.md) | provenance, licences, and what the data does **not** cover |
| [03-evaluation.md](03-evaluation.md) | every held-out number, and the trade-off curve behind the operating point |
| [04-failures.md](04-failures.md) | **three essays it gets confidently wrong, with theories** — then four larger mistakes the project made |
| [05-esl.md](05-esl.md) | the false-positive study on non-native writers, including the measurement that argues against the convenient reading |
| [06-decisions.md](06-decisions.md) | decisions we would be asked to defend |
| [07-ai-usage.md](07-ai-usage.md) | how AI tools were used building this |

## What the detector can and cannot reach

| | |
| --- | --- |
| [08-cross-vendor.md](08-cross-vendor.md) | how far skill transfers across model families |
| [09-frontier-ceiling.md](09-frontier-ceiling.md) | why the strongest current models are not reliably detected, measured rather than asserted |
| [13-structural-features.md](13-structural-features.md) | three new signal families: one works, one is a length feature in disguise, one is a near-null |
| [12-consensus-and-polish.md](12-consensus-and-polish.md) | consensus scoring, and detecting a polished passage rather than a generated one |

## The record

| | |
| --- | --- |
| [14-submission-record.md](14-submission-record.md) | what was built, what was measured, and **what was thrown away** — the single best summary for an assessor |
| [10-development-record.md](10-development-record.md) | the long chronological record, including the bugs and how each was found |
| [11-build-and-limits.md](11-build-and-limits.md) | how the thing is built and where its limits are |
| [DATASET-REPORT.md](DATASET-REPORT.md) | the corpus as it actually stands |
| [dataset-gaps.md](dataset-gaps.md) | the cells of the dataset that are empty, named |
| [design-brief.md](design-brief.md) | the interface brief |
| [exports/](exports/) | Word copies of the project documents, for submission |

## Reading order

**If you have ten minutes:** [JUDGES.md](../JUDGES.md).

**If you want to check whether the claims hold:** [03-evaluation.md](03-evaluation.md) →
[04-failures.md](04-failures.md) → [05-esl.md](05-esl.md). Accuracy, then what it gets wrong,
then who it gets wrong *about*.

**If you want to know whether the method has a future:**
[09-frontier-ceiling.md](09-frontier-ceiling.md) and [08-cross-vendor.md](08-cross-vendor.md)
are the two honest ceilings.

**If you are assessing the engineering:** [14-submission-record.md](14-submission-record.md),
then [06-decisions.md](06-decisions.md).

## A note on numbers in these documents

Every figure comes from an artifact in `artifacts/` that a script regenerates; none is typed by
hand. There are **two builds** — the served one (qwen3-30B on Workers AI, `artifacts/*_remote.json`)
and a local GPT-2 one — and any figure that exists only for the GPT-2 build carries that label on
the line. `tests/test_documented_numbers.py` enforces this against the build the server actually
resolves, and fails rather than skips when a claim about the served build has no measurement.
That test exists in that form because it was previously checking the unshipped build and let the
README drift; see [10-development-record.md](10-development-record.md).
