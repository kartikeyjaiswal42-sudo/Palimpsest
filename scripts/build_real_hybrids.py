#!/usr/bin/env python
"""Construct part-human/part-machine documents from REAL matched pairs.

    python scripts/build_real_hybrids.py

Why this exists
---------------
Our first attempt at the "human draft, machine-polished paragraph" case was hand-written:
one of us rewrote a run of sentences from a real essay in a more polished register. The
evaluation showed that was worthless as training data -- see docs/04-failures.md. Prose
*composed* to sound machine-generated does not carry the statistical signature of prose
*sampled* from a model, which is the only thing our observer can see.

So we build hybrids out of text that is genuinely both. Liang et al. published matched
pairs: the same essay in its original human form and after a model rewrote it. Splicing the
first half of the human version to the second half of the machine version gives a document
whose two halves are real, whose content is continuous, and whose boundary we know exactly.

The seam is a real discontinuity of authorship, which is precisely what the in-document
context features are supposed to find.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palimpsest.data.fetch import Document, read_jsonl  # noqa: E402
from palimpsest.text import split_sentences  # noqa: E402

# (human source, machine-rewritten source, label). The two files are index-aligned: row i of
# each is the same underlying essay.
PAIRS = [
    ("liang_hewlett_human", "liang_hewlett_gptsimplify", "hewlett"),
    ("liang_toefl", "liang_toefl_gpt4polished", "toefl"),
]

MIN_SENTENCES = 6


def main() -> int:
    out: list[Document] = []
    for human_id, machine_id, tag in PAIRS:
        human = read_jsonl(ROOT / "data" / "raw" / f"{human_id}.jsonl")
        machine = read_jsonl(ROOT / "data" / "raw" / f"{machine_id}.jsonl")
        if len(human) != len(machine):
            print(f"! {tag}: {len(human)} vs {len(machine)} rows, not aligned; skipping")
            continue

        made = 0
        for i, (h, m) in enumerate(zip(human, machine, strict=True)):
            hs = split_sentences(h.text)
            ms = split_sentences(m.text)
            if len(hs) < MIN_SENTENCES or len(ms) < 3:
                continue

            # Keep the human opening, then switch to the machine rewrite for the remainder.
            # Alternating the cut point across documents stops the seam sitting at a fixed
            # relative position, which a model could otherwise learn instead of the signal.
            cut_h = len(hs) // 2 + (i % 3) - 1
            cut_h = max(2, min(cut_h, len(hs) - 2))
            cut_m = max(1, min(int(len(ms) * cut_h / len(hs)), len(ms) - 1))

            head = " ".join(s.text for s in hs[:cut_h])
            tail = " ".join(s.text for s in ms[cut_m:])
            if len(tail.split()) < 25 or len(head.split()) < 25:
                continue

            text = f"{head} {tail}"
            span = [len(head) + 1, len(text)]
            doc = Document(
                id=f"real_hybrid_{tag}:{i:04d}",
                source_id="real_hybrid",
                text=text,
                authorship="hybrid",
                role="localisation",
                meta={
                    "pair": tag,
                    "human_doc_id": h.id,
                    "machine_doc_id": m.id,
                    # Grouped with the human original so a split never separates them.
                    "base_doc_id": h.id,
                    "machine_spans": [span],
                    "cut_sentence": cut_h,
                    "generator": "gpt-3.5/gpt-4 (Liang et al. 2023)",
                    "style": "real-rewrite",
                },
            )
            sentences = split_sentences(doc.text)
            n_machine = sum(1 for s in sentences if span[0] <= (s.start + s.end) // 2 < span[1])
            if n_machine < 2 or n_machine >= len(sentences) - 1:
                continue  # need both classes genuinely present
            doc.meta["n_machine_sentences"] = n_machine
            doc.meta["n_total_sentences"] = len(sentences)
            out.append(doc)
            made += 1
        print(f"{tag}: {made} hybrids from {len(human)} pairs")

    path = ROOT / "data" / "generated" / "real_hybrid_essays.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for d in out:
            fh.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")

    tm = sum(d.meta["n_machine_sentences"] for d in out)
    tt = sum(d.meta["n_total_sentences"] for d in out)
    words = sorted(d.n_words for d in out)
    print(f"\n{len(out)} real hybrids -> {path.relative_to(ROOT)}")
    print(f"  {tm} machine / {tt} sentences ({100 * tm / max(tt,1):.1f}% machine)")
    print(f"  words min/med/max {words[0]}/{words[len(words)//2]}/{words[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
