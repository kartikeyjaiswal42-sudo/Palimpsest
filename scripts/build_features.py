#!/usr/bin/env python
"""Extract the sentence feature matrix for every corpus document.

    python scripts/build_features.py --sets train
    python scripts/build_features.py --sets all

This is the slow step: every document gets a GPT-2 forward pass. Results are cached to
``data/features/<set>.jsonl`` so training and evaluation can iterate without re-scoring.

Features are extracted through ``Analyzer.features_for`` -- the same method the API calls at
serve time. Training and serving must not have separate feature code, or every number we
report describes a system that is not the one we ship.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palimpsest.analyze import Analyzer  # noqa: E402
from palimpsest.data.fetch import Document, read_jsonl  # noqa: E402
from palimpsest.detect.classifier import SentenceDetector  # noqa: E402
from palimpsest.scorer.local_lm import get_scorer  # noqa: E402
from palimpsest.scorer.ngram import NgramReference  # noqa: E402

# Which raw files make up each named set, and what role each plays in the evaluation.
#
# The machine half of the TRAINING pool is real GPT-3.5 output, not text we composed to
# sound machine-generated. That distinction was not obvious to us at the start and it turned
# out to be the single most important decision in the project -- see docs/04-failures.md.
# Human and machine sentences in this pool have the same mean length (19.2 words), so the
# classifier cannot reach the right answer by measuring length.
SETS: dict[str, list[str]] = {
    "train": ["liang_college_human", "jhu", "liang_college_gpt3"],
    # Held out: the same generator, prompted to evade detection. Tests whether the detector
    # survives an adversary who knows it exists.
    "unseen_prompting": ["liang_college_gpt3_prompteng"],
    # Held out: human writing from another domain and school level. Tests false positives
    # when the input is nothing like what we trained on.
    "domain_shift": ["liang_hewlett_human"],
    # Held out: the false-positive study by language background.
    "esl": ["liang_toefl", "ellipse", "persuade"],
    # Held out: part-human/part-machine documents with a known seam.
    "localisation": ["real_hybrid"],
    # Held out: prose a careful writer composed to imitate a model. Our clearest failure.
    "adversarial": ["machine_claude", "hybrid_claude"],
    # Held out: identical content, one version rewritten by a model. Isolates what the
    # features respond to, holding the writer and the subject fixed.
    "ablation": ["liang_toefl_gpt4polished", "liang_hewlett_gptsimplify"],
}

GENERATED = {"machine_claude", "hybrid_claude", "real_hybrid"}
_STEMS = {
    "machine_claude": "machine_essays",
    "hybrid_claude": "hybrid_essays",
    "real_hybrid": "real_hybrid_essays",
}


def flatten(text: str) -> str:
    """Remove paragraph structure.

    Our largest human source ships with every newline stripped, while our generations have
    paragraph breaks. Left alone, paragraph structure is a source marker -- effectively a
    label leak -- so it is removed from every document in the corpus, both classes alike.
    Measured cost: flattening changes sentence segmentation for 0 of 11 machine essays and
    4 of 31 JHU essays, all of which contain unpunctuated headings. See docs/06-decisions.md.
    """
    return re.sub(r"\s*\n+\s*", " ", text).strip()


def load_set(name: str, limit_per_source: int | None) -> list[Document]:
    docs: list[Document] = []
    for source_id in SETS[name]:
        folder = "generated" if source_id in GENERATED else "raw"
        stem = _STEMS.get(source_id, source_id)
        path = ROOT / "data" / folder / f"{stem}.jsonl"
        if not path.exists():
            print(f"  ! missing {path.relative_to(ROOT)}, skipping")
            continue
        rows = read_jsonl(path)
        if limit_per_source:
            rows = rows[:limit_per_source]
        docs.extend(rows)
    return docs


def sentence_label(doc: Document, start: int, end: int) -> int:
    """1 if this sentence is machine-written.

    Hybrids carry explicit character spans for the machine-written run, so their sentences
    are labelled individually by whether the sentence's midpoint falls inside a span.
    """
    if doc.authorship == "human":
        return 0
    if doc.authorship == "machine":
        return 1
    spans = doc.meta.get("machine_spans") or []
    mid = (start + end) // 2
    return int(any(s <= mid < e for s, e in spans))


def group_key(doc: Document) -> str:
    """The unit that must never straddle a train/test split.

    A hybrid shares roughly 80% of its text with the human essay it was built from. If the
    two land on opposite sides of a split, the test set contains hundreds of words the model
    trained on and every metric is inflated. Hybrids are therefore grouped with their base.
    """
    return doc.meta.get("base_doc_id") or doc.id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", nargs="*", default=["train"], help="set names, or 'all'")
    ap.add_argument("--limit-per-source", type=int, default=None)
    ap.add_argument("--observer", default="gpt2")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    names = list(SETS) if args.sets == ["all"] else args.sets
    ref_path = ROOT / "artifacts" / "ngram_reference.json"
    reference = NgramReference.load(ref_path) if ref_path.exists() else None
    if reference is None:
        print("! no n-gram reference; corpus features will be NaN. Run fit_reference.py.")

    analyzer = Analyzer(SentenceDetector(), get_scorer(args.observer, args.device), reference)
    out_dir = ROOT / "data" / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        docs = load_set(name, args.limit_per_source)
        if not docs:
            print(f"\n=== {name}: no documents")
            continue
        print(f"\n=== {name}: {len(docs)} documents")
        started = time.perf_counter()
        path = out_dir / f"{name}.jsonl"
        n_sentences = 0
        with path.open("w", encoding="utf-8") as fh:
            for i, doc in enumerate(docs):
                text = flatten(doc.text)
                spans, features, _ = analyzer.features_for(text)
                for j, (span, feat) in enumerate(zip(spans, features, strict=True)):
                    fh.write(
                        json.dumps(
                            {
                                "doc_id": doc.id,
                                "source_id": doc.source_id,
                                "group": group_key(doc),
                                "authorship": doc.authorship,
                                "sentence_index": j,
                                "start": span.start,
                                "end": span.end,
                                "label": sentence_label(doc, span.start, span.end),
                                "n_doc_sentences": len(spans),
                                "doc_meta": doc.meta,
                                "features": {k: _clean(v) for k, v in feat.items()},
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    n_sentences += 1
                if (i + 1) % 25 == 0:
                    rate = (i + 1) / (time.perf_counter() - started)
                    print(f"    {i + 1}/{len(docs)} docs  {rate:.1f} docs/s")
        elapsed = time.perf_counter() - started
        print(
            f"    {n_sentences} sentences in {elapsed:.1f}s "
            f"({len(docs) / elapsed:.1f} docs/s) -> {path.relative_to(ROOT)}"
        )
    return 0


def _clean(v: float) -> float | None:
    """JSON has no NaN. Undefined features are written as null and re-read as NaN."""
    return None if v != v else float(v)


if __name__ == "__main__":
    raise SystemExit(main())
