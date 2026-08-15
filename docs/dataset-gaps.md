<!-- Written by hand. scripts/dataset_report.py reads this file and never overwrites it. -->

# What this dataset does not cover

Each heading is a real limit on what any number measured on this corpus can mean. Delete a
line only when the gap is actually closed, not when it becomes inconvenient.

## Subject matter

The machine essays answer subjects chosen in `scripts/generate_modern.py` and pre-registered
in `scripts/plan_claude_corpus.py`. Human essays were written to whatever prompt their
applicant faced. "Machine" and "those subjects" are therefore correlated across most of the
modern corpus, which is why `modern_control` exists — same generator, no subject steering.
A reader should assume subject-matter bias anywhere that control was not run.

## Authors

No ESL-authored *admissions essays*. We hold native-authored personal statements and we hold
English-learner writing in other genres; the cell where they meet is empty. PROJECT.md §8
bounds the risk by transplanting a measured proficiency effect. It does not measure it.

## Attacks

No paraphrase or "humanizer" output. Published benchmarks find these hurt detectors badly,
so every accuracy number here is an **upper** bound on adversarial performance.

## Generators

All machine text comes from our own generation pipeline (plus one foreign GPT-3.5 set used
only as a control). docs/12 records a signal that scored 0.960 AUROC on our pipeline and
0.490 on somebody else's.

## Genre and language

Admissions essays, English only. The genre gate is fitted against *student* writing, so
"out of scope" means "unlike the student writing we tested", not "not an admissions essay".
