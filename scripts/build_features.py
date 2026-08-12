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
    # `modern_train` is Gemini-3 output, added because the detector fitted on GPT-3.5 alone
    # missed a 2026 essay outright: 18 sentences, none flagged, median sentence probability
    # 0.019. That is not a threshold that needs moving, it is a generator the features had
    # never seen. See scripts/split_modern.py for what is deliberately NOT in here.
    "train": ["liang_college_human", "jhu", "liang_college_gpt3", "modern_train"],
    # `claude_modern` is deliberately NOT in `train`. Adding it was measured and it makes the
    # detector worse -- see scripts/vendor_swap_sweep.py and docs/08-cross-vendor.md. Build
    # its features with this set instead; the sweep reads them from train_with_claude.jsonl.
    "train_with_claude": ["liang_college_human", "jhu", "liang_college_gpt3", "modern_train",
                          "claude_modern"],
    # Held out: the same generator, prompted to evade detection. Tests whether the detector
    # survives an adversary who knows it exists.
    "unseen_prompting": ["liang_college_gpt3_prompteng"],
    # Held out: unseen essays from the modern models that ARE in training. The easier of the
    # two modern questions -- the fit has seen these generators' habits, just not these
    # essays.
    "modern_holdout": ["modern_holdout"],
    # Held out: every essay from one modern checkpoint kept out of training entirely.
    "modern_unseen": ["modern_unseen"],
    # Held out: a different model family, also entirely. The hardest of the three modern
    # questions and the one that predicts what happens when next year's model arrives.
    "modern_unseen_family": ["modern_unseen_family"],
    # Held out CONTROL: same generator as modern_holdout, but written to the bare prompts
    # with no subject steering. Every other modern set answers one of 40 subjects chosen in
    # generate_modern.py, so "machine" and "those topics" are correlated across the whole
    # modern corpus. Holding the generator fixed and removing only the steering is what
    # separates "it learned machine prose" from "it learned beekeeping".
    "modern_control": ["modern_control"],
    # Held out: human writing from another domain and school level. Tests false positives
    # when the input is nothing like what we trained on.
    "domain_shift": ["liang_hewlett_human"],
    # Held out: the false-positive study by language background.
    "esl": ["liang_toefl", "ellipse", "persuade"],
    # Held out: part-human/part-machine documents with a known seam.
    "localisation": ["real_hybrid"],
    # Held out: prose a careful writer composed to imitate a model. Our clearest failure.
    "adversarial": ["machine_claude", "hybrid_claude"],
    # Held out: admissions essays written by four checkpoints of the CLAUDE family, on
    # subjects that appear in no training essay. The split is by subject and was
    # pre-registered in scripts/plan_claude_corpus.py before a word was generated.
    #
    # This is the only set in the project that changes VENDOR. Everything the detector is
    # fitted on comes from OpenAI (GPT-3.5) or Google (`modern_train`), and `modern_unseen_
    # family` -- despite its name -- is Gemini throughout, so it varies the checkpoint tier
    # and not the family. A model from a different lab has different pretraining data,
    # different post-training and different habits, which makes this the strictest
    # generalisation question available: does the detector read machine-ness, or does it
    # read Google?
    #
    # It stays held out whatever happens to `claude_modern`. No subject crosses the
    # boundary, so the number it produces is a generalisation measurement and not a
    # memorisation one.
    "modern_claude": ["claude_modern_heldout"],
    # Held out: identical content, one version rewritten by a model. Isolates what the
    # features respond to, holding the writer and the subject fixed.
    "ablation": ["liang_toefl_gpt4polished", "liang_hewlett_gptsimplify"],
    # Student essays on PERSUADE prompts, machine half contributed by MANY independent
    # people using many models (`meta.pipeline` names each one). This is the only corpus in
    # the project where machine text was not produced by our own generation pipeline, which
    # is what makes it the only one that can distinguish detection from bookkeeping --
    # docs/09-frontier-ceiling.md records a classifier that scored AUROC 1.000 here and was
    # really recognising our own file conventions. Fetched by scripts/fetch_external.py.
    "daigt": ["daigt"],
}

GENERATED = {"machine_claude", "hybrid_claude", "real_hybrid",
             "modern_train", "modern_holdout", "modern_unseen", "modern_unseen_family",
             "modern_control", "claude_modern", "claude_modern_heldout"}
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
    ap.add_argument("--observer", default="gpt2",
                    help="'gpt2' (local) or 'remote' (Workers AI 30 B, see observer-worker/)")
    ap.add_argument("--top-k", type=int, default=0,
                    help="remote only: ask the observer for its k most likely candidates at "
                         "each position, not just the realised token. k>0 makes entropy and "
                         "Fast-DetectGPT curvature computable at IDENTICAL neuron cost -- it "
                         "is the same forward pass. See scorer/remote_lm for what the top-k "
                         "head does and does not measure.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-suffix", default="",
                    help="appended to each output filename, e.g. '_remote', so the two "
                         "observers' feature files never overwrite each other")
    args = ap.parse_args()

    names = list(SETS) if args.sets == ["all"] else args.sets
    ref_path = ROOT / "artifacts" / "ngram_reference.json"
    reference = NgramReference.load(ref_path) if ref_path.exists() else None
    if reference is None:
        print("! no n-gram reference; corpus features will be NaN. Run fit_reference.py.")

    # The remote observer is a 30 B model on Cloudflare rather than GPT-2 on this machine.
    # docs/09-frontier-ceiling.md measures why: GPT-2's statistics are not merely weak on
    # modern prose, they are INVERTED against ESL writing (AUROC 0.132), which is the
    # mechanism behind the project's false-positive problem. Both observers go through the
    # same Analyzer so training and serving cannot drift apart.
    if args.observer == "remote":
        from palimpsest.scorer.remote_lm import RemoteObserverScorer

        scorer = RemoteObserverScorer(top_k=args.top_k)
        print(f"observer: REMOTE {scorer.model_name} top_k={args.top_k} "
              f"(no local model is loaded)")
    else:
        scorer = get_scorer(args.observer, args.device)
    analyzer = Analyzer(SentenceDetector(), scorer, reference)
    out_dir = ROOT / "data" / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        docs = load_set(name, args.limit_per_source)
        if not docs:
            print(f"\n=== {name}: no documents")
            continue
        print(f"\n=== {name}: {len(docs)} documents")
        started = time.perf_counter()
        path = out_dir / f"{name}{args.out_suffix}.jsonl"
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
