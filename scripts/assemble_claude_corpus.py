#!/usr/bin/env python
"""Turn generated essay files into a corpus, rejecting anything that isn't one.

    python scripts/assemble_claude_corpus.py             # validate + write the corpus
    python scripts/assemble_claude_corpus.py --missing   # list ids still to be generated
    python scripts/assemble_claude_corpus.py --strict    # exit non-zero if anything is rejected

Generation happens in workers that write one plain-text file per assignment into
``data/generated/claude_raw/``. This script pairs each file with its plan entry, checks that
what came back is actually an admissions essay, and writes the two corpus files.

**Why validation is a separate step with teeth.** A generating worker can fail in ways that
still produce a file: it can prepend "Here is the essay:", emit a markdown title, refuse the
task and explain why, or stop halfway through a sentence when its budget runs out. Every one
of those lands on disk looking exactly like success. If a truncated essay or a paragraph of
apology gets into the machine class, the detector learns that machine text is short and
apologetic, and the resulting recall number is measuring our own pipeline. So a file is
rejected unless it passes every check below, and rejected files are reported by name rather
than dropped quietly.

The word-count check is the one that catches the common failure. A worker terminated
mid-batch leaves a partial file, and a partial file is exactly the length artifact
docs/06-decisions.md #6 spent a whole decision removing.

**Two output files, because the split was pre-registered.** ``plan_claude_corpus.py`` decided
before generation which subjects are training and which are held out, and it split by subject
so no held-out essay shares a subject with a training one. That decision is honoured here and
is not re-derivable from anything in this file -- it is read from the plan.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palimpsest.data.fetch import Document, write_jsonl  # noqa: E402

PLAN = ROOT / "data" / "generated" / "plan" / "plan.json"
RAW = ROOT / "data" / "generated" / "claude_raw"

#: Where each split lands. The stems match ``build_features._STEMS``.
OUT = {
    "train": (ROOT / "data" / "generated" / "claude_modern.jsonl", "claude_modern", "train"),
    "heldout": (ROOT / "data" / "generated" / "claude_modern_heldout.jsonl",
                "claude_modern_heldout", "unseen-generator"),
}

#: A worker that refused, or narrated, rather than writing. Checked against the opening,
#: because a refusal announces itself immediately.
_PREAMBLE = re.compile(
    r"^\s*(here(?:'s| is)\b|sure[,!]|certainly[,!]|of course[,!]|i'?ll write|i'?ll compose|"
    r"below is\b|i cannot\b|i can'?t\b|i'?m unable\b|as an ai\b|i apologi[sz]e\b)",
    re.I)

#: Markdown the instruction explicitly forbade. Its presence means the instruction was not
#: followed, which makes the essay unrepresentative of what a student would paste in --
#: and it is the exact artifact truepen had to unpick, where formatting was detected instead
#: of prose.
_MARKDOWN = re.compile(r"(^\s{0,3}#{1,6}\s)|(^\s{0,3}[-*+]\s)|(\*\*.+?\*\*)|(^\s{0,3}\d+\.\s)",
                       re.M)

#: A trailing note the instruction forbade: word counts, sign-offs, self-assessment.
_TRAILER = re.compile(r"(\(?\s*(word count|approx\.?|~)\s*[:\-]?\s*\d{3,4}\s*(words)?\s*\)?|"
                      r"\[.*?\]|note\s*:\s|let me know if)", re.I)


def load_plan() -> dict[str, dict]:
    if not PLAN.exists():
        raise SystemExit(f"missing {PLAN.relative_to(ROOT)} -- run scripts/plan_claude_corpus.py")
    return {r["id"]: r for r in json.loads(PLAN.read_text(encoding="utf-8"))}


def check(text: str, row: dict) -> str | None:
    """None if the text is a usable essay, else the reason it is not."""
    if not text:
        return "empty"
    if _PREAMBLE.match(text):
        return "preamble/refusal"
    if _MARKDOWN.search(text):
        return "markdown"
    if _TRAILER.search(text):
        return "trailing note"

    words = len(text.split())
    target = row["target_words"]
    if words < 400:
        # The floor is independent of the target: below it the essay is either truncated or
        # not an admissions essay, whatever it was asked for.
        return f"under the 400-word floor ({words})"
    drift = (words - target) / target
    if abs(drift) > 0.25:
        # Deliberately generous. The per-essay target exists to shape the CORPUS
        # distribution, not because any single essay must hit a number -- an essay asked for
        # 450 words that came back at 550 is still squarely inside the human range and
        # throwing it away buys nothing. What this band is really for is catching the two
        # failures that would poison the class: a truncated file, and a runaway. The
        # distribution itself is checked at corpus level in ``report_lengths`` below, which
        # is where a systematic drift would actually show up.
        return f"{words} words vs target {target} ({drift:+.0%})"

    last = text.rstrip()[-1:]
    if last not in ".!?\"'":
        return "ends mid-sentence"
    # A title left on the first line: a short opening line with no terminal punctuation.
    first = text.split("\n", 1)[0].strip()
    if len(first.split()) <= 9 and first[-1:] not in ".!?\"'," and len(text.split("\n")) > 1:
        return f"looks like a title: {first!r}"
    return None


def report_lengths(docs: list[Document]) -> None:
    """Compare the finished machine corpus to the human one, decile by decile.

    This is the check that matters, and it is the reason the per-essay band above can afford
    to be loose. docs/06-decisions.md #6 records what happens when the two classes differ in
    length: the document model gave length a weight of -3.09 and its strongest single belief
    became "short means machine", learned entirely from the GPT-3.5 set being shorter than
    the human one. Length was removed from the document model to kill that, but the sentence
    features were never given the same surgery -- so a machine corpus with a distinctive
    length profile would put the artifact straight back, somewhere nobody is looking for it.

    A systematic drift shows up here and nowhere else. One essay 100 words over target is
    noise; a median 80 words above the human median is a confound.
    """
    human: list[int] = []
    for name in ("liang_college_human", "jhu"):
        path = ROOT / "data" / "raw" / f"{name}.jsonl"
        if path.exists():
            human += [json.loads(l)["n_words"] for l in path.open(encoding="utf-8")]
    if not human or not docs:
        return
    machine = [d.n_words for d in docs]

    def q(values: list[int], p: float) -> int:
        s = sorted(values)
        return s[int(p * (len(s) - 1))]

    print("\nlength vs the human corpus (the confound this corpus must not reintroduce):")
    print(f"  {'':8s} {'n':>4s} {'p10':>5s} {'p25':>5s} {'med':>5s} {'p75':>5s} {'p90':>5s} "
          f"{'mean':>5s}")
    for label, vals in (("human", human), ("machine", machine)):
        print(f"  {label:8s} {len(vals):4d} {q(vals, .10):5d} {q(vals, .25):5d} "
              f"{q(vals, .50):5d} {q(vals, .75):5d} {q(vals, .90):5d} "
              f"{sum(vals) / len(vals):5.0f}")
    gap = (sum(machine) / len(machine)) - (sum(human) / len(human))
    verdict = "OK" if abs(gap) <= 40 else "DRIFT -- machine class is separable on length alone"
    print(f"  mean gap {gap:+.0f} words   {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--missing", action="store_true", help="list ids not yet generated")
    ap.add_argument("--strict", action="store_true", help="exit non-zero on any rejection")
    args = ap.parse_args()

    plan = load_plan()
    present = {p.stem: p for p in RAW.glob("*.txt")} if RAW.exists() else {}

    if args.missing:
        missing = [i for i in plan if i not in present]
        by_batch: dict[int, list[str]] = {}
        for i in missing:
            by_batch.setdefault(plan[i]["batch"], []).append(i)
        print(f"{len(missing)} of {len(plan)} essays still to generate")
        for b in sorted(by_batch):
            model = plan[by_batch[b][0]]["model"]
            print(f"  batch {b:02d} [{model:6s}] {len(by_batch[b]):2d} missing")
        return 0

    kept: dict[str, list[Document]] = {"train": [], "heldout": []}
    rejected: list[tuple[str, str]] = []
    seen: dict[str, str] = {}

    for essay_id, row in plan.items():
        path = present.get(essay_id)
        if path is None:
            continue
        text = path.read_text(encoding="utf-8").strip()
        reason = check(text, row)
        if reason:
            rejected.append((essay_id, reason))
            continue
        doc = Document(
            id=f"{OUT[row['split']][1]}:{essay_id}",
            source_id=OUT[row["split"]][1],
            text=text,
            authorship="machine",
            role=OUT[row["split"]][2],
            meta={"model": row["model"], "style": row["style"], "frame": row["frame"],
                  "topic": row["topic"], "prompt_id": row["prompt_id"],
                  "prompt": row["prompt"], "target_words": row["target_words"],
                  "batch": row["batch"], "family": "claude"},
        )
        if doc.sha256 in seen:
            rejected.append((essay_id, f"duplicate of {seen[doc.sha256]}"))
            continue
        seen[doc.sha256] = essay_id
        kept[row["split"]].append(doc)

    total = sum(len(v) for v in kept.values())
    print(f"{len(present)} files on disk, {total} accepted, {len(rejected)} rejected\n")

    for split, docs in kept.items():
        path, source_id, role = OUT[split]
        if not docs:
            print(f"  {split:8s} 0 essays -- nothing written")
            continue
        write_jsonl(docs, path)
        words = sorted(d.n_words for d in docs)
        models = Counter(d.meta["model"] for d in docs)
        styles = Counter(d.meta["style"] for d in docs)
        print(f"  {split:8s} {len(docs):3d} essays -> {path.relative_to(ROOT)}")
        print(f"           median {words[len(words) // 2]} words, "
              f"range {words[0]}-{words[-1]}")
        print(f"           models {dict(models)}")
        print(f"           styles {dict(styles)}")

    report_lengths(kept["train"] + kept["heldout"])

    if rejected:
        print(f"\nrejected ({len(rejected)}):")
        for essay_id, reason in sorted(rejected)[:40]:
            print(f"  {essay_id}  {reason}")
        if len(rejected) > 40:
            print(f"  ... and {len(rejected) - 40} more")

    remaining = len(plan) - len(present)
    if remaining:
        print(f"\n{remaining} essays still to generate "
              f"(scripts/assemble_claude_corpus.py --missing lists them)")

    if args.strict and rejected:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
