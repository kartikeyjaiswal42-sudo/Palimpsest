#!/usr/bin/env python
"""Add the structural feature block to feature matrices that already exist.

    python scripts/build_syntax_features.py --sets train_remote esl_remote
    python scripts/build_syntax_features.py --sets all

Why this joins rather than rebuilds
-----------------------------------
``build_features.py`` runs the observer over every document, which is the expensive step and
which needs Workers AI credentials for the ``_remote`` sets. The structural features need no
language model at all -- only a dependency parse -- so they can be computed from the raw text
and joined onto the rows that are already on disk by ``(doc_id, start, end)``.

That is not merely a shortcut. It means the augmented matrix contains *the identical 43
existing feature values* that produced every published number, so a measured difference is
attributable to the new block and not to a re-scoring run that drifted. The join is checked,
not assumed: a row whose span cannot be recovered from the source text is a hard error, and
the script refuses to write a file where any row failed to match.

Reference fitting and leakage
-----------------------------
``pos_trigram_surprisal`` needs a reference distribution, and the only defensible one is the
*human* half of the *training* pool. Fitting it on everything would let held-out machine
essays inform the distribution they are then scored against, which is the leak that makes a
detector look good and generalise badly -- the same class of error docs/04-failures.md
records for the smart-quote artifact. The fitted reference is written to
``artifacts/pos_trigram_reference.json`` and reused by every later set.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palimpsest.data.fetch import read_jsonl  # noqa: E402
from palimpsest.features.syntax import (  # noqa: E402
    PosTrigramReference,
    extract_syntax_context_features,
    extract_syntax_features,
    get_parser,
    pos_tags,
)

FEATURE_DIR = ROOT / "data" / "features"
REFERENCE = ROOT / "artifacts" / "pos_trigram_reference.json"

#: Sources whose sentences may fit the POS reference: human, in-domain, training only.
#: `jhu` and `liang_college_human` are the human half of the training pool.
REFERENCE_SOURCES = ("liang_college_human", "jhu")


def load_texts() -> dict[str, str]:
    """Every document's text by id, from both the raw and generated corpora."""
    texts: dict[str, str] = {}
    for folder in ("raw", "generated"):
        for path in sorted((ROOT / "data" / folder).glob("*.jsonl")):
            try:
                for doc in read_jsonl(path):
                    texts[doc.id] = doc.text
            except Exception as exc:  # a partial corpus is normal; a silent one is not
                print(f"  ! could not read {path.name}: {exc}", file=sys.stderr)
    return texts


def fit_reference(texts: dict[str, str]) -> PosTrigramReference:
    """Fit the POS trigram model on human training prose only."""
    if REFERENCE.exists():
        ref = PosTrigramReference.from_dict(json.loads(REFERENCE.read_text()))
        print(f"reference: reusing {REFERENCE.relative_to(ROOT)} ({ref.total:,} tags)")
        return ref

    train = FEATURE_DIR / "train_remote.jsonl"
    if not train.exists():
        train = FEATURE_DIR / "train.jsonl"
    if not train.exists():
        raise SystemExit("no training feature file to fit the POS reference from")

    print(f"reference: fitting on human sentences in {train.name} ...")
    sequences, seen = [], 0
    for line in train.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row["source_id"] not in REFERENCE_SOURCES or row["label"] != 0:
            continue
        text = texts.get(row["doc_id"], "")[row["start"]:row["end"]]
        tags = pos_tags(text)
        if tags:
            sequences.append(tags)
        seen += 1
        if seen % 1000 == 0:
            print(f"  {seen:,} sentences ...")

    if not sequences:
        raise SystemExit(
            "fitted no POS sequences -- is spaCy installed? "
            "python -m spacy download en_core_web_sm"
        )
    ref = PosTrigramReference.fit(sequences)
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE.write_text(json.dumps(ref.to_dict()), encoding="utf-8")
    print(f"reference: {len(sequences):,} sentences, {ref.total:,} tags "
          f"-> {REFERENCE.relative_to(ROOT)}")
    return ref


def augment(name: str, texts: dict[str, str], ref: PosTrigramReference) -> bool:
    """Add the structural block to one feature file, in place. Returns False if skipped."""
    path = FEATURE_DIR / f"{name}.jsonl"
    if not path.exists():
        print(f"{name}: no such file, skipped")
        return False

    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if not rows:
        print(f"{name}: empty, skipped")
        return False
    if any(k in rows[0]["features"] for k in ("tree_depth_max", "stopword_ratio")):
        print(f"{name}: already has the structural block, skipped "
              f"(delete and rebuild to refresh)")
        return False

    # Group by document, preserving order, so the leave-one-out context features see the
    # essay rather than a shuffled bag of sentences.
    by_doc: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_doc.setdefault(r["doc_id"], []).append(i)

    t0 = time.time()
    missing_text: set[str] = set()
    done = 0

    for doc_id, idx in by_doc.items():
        idx.sort(key=lambda i: rows[i]["sentence_index"])
        text = texts.get(doc_id)
        if text is None:
            missing_text.add(doc_id)
            for i in idx:
                rows[i]["_syntax_missing"] = True
            continue

        base = [
            extract_syntax_features(text[rows[i]["start"]:rows[i]["end"]], ref)
            for i in idx
        ]
        ctx = extract_syntax_context_features(base, len(base))
        for i, b, c in zip(idx, base, ctx, strict=True):
            rows[i]["features"].update(b)
            rows[i]["features"].update(c)

        done += 1
        if done % 50 == 0:
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  {done}/{len(by_doc)} documents ({rate:.1f}/s)")

    if missing_text:
        # Refuse rather than write a file whose new columns are quietly NaN for some rows:
        # a partly-populated column produces a measurement that silently describes a subset.
        n = sum(len(by_doc[d]) for d in missing_text)
        raise SystemExit(
            f"{name}: source text unavailable for {len(missing_text)} documents "
            f"({n} rows) -- e.g. {sorted(missing_text)[:3]}. "
            "Fetch the corpus first (scripts/fetch_corpus.py); refusing to write a "
            "matrix whose new columns are populated for only part of the set."
        )

    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    tmp.replace(path)
    print(f"{name}: {len(rows):,} rows in {len(by_doc)} documents "
          f"({time.time() - t0:.0f}s) -> {path.relative_to(ROOT)}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True,
                    help="feature file stems, or 'all' for every *.jsonl in data/features")
    args = ap.parse_args()

    if get_parser() is None:
        raise SystemExit(
            "spaCy en_core_web_sm is required to build structural features.\n"
            "  uv pip install spacy && python -m spacy download en_core_web_sm"
        )

    names = args.sets
    if names == ["all"]:
        names = sorted(p.stem for p in FEATURE_DIR.glob("*.jsonl"))

    texts = load_texts()
    print(f"corpus: {len(texts):,} documents on disk")
    ref = fit_reference(texts)

    for name in names:
        augment(name, texts, ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
