#!/usr/bin/env python
"""Fit the human-essay n-gram reference used by the corpus-relative features.

    python scripts/fit_reference.py

Fitted on the Ghostbuster IvyPanda essays and nothing else. That set plays no other role --
it is never a labelled training example and never appears in any test set -- which keeps
the reference independent of everything we measure. Two reasons that matters:

* If the reference were fitted on the same human essays we train on, the corpus features
  would be memorising those specific essays and would look far stronger in evaluation than
  they are in reality.
* The Ghostbuster set has a mild post-ChatGPT contamination risk (see docs/02-dataset.md).
  Contamination in a background frequency model is tolerable in a way that contamination in
  training labels is not: a handful of machine-written essays shift word frequencies
  slightly, they do not teach the classifier a wrong answer.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palimpsest.data.fetch import read_jsonl  # noqa: E402
from palimpsest.scorer.ngram import NgramReference  # noqa: E402


def main() -> int:
    src = ROOT / "data" / "raw" / "ghostbuster_ivypanda.jsonl"
    if not src.exists():
        print(f"missing {src}; run scripts/fetch_corpus.py first")
        return 1

    docs = read_jsonl(src)
    print(f"fitting reference on {len(docs)} essays ({sum(d.n_words for d in docs):,} words)")
    started = time.perf_counter()
    ref = NgramReference.fit([d.text for d in docs], min_count=2)
    out = ROOT / "artifacts" / "ngram_reference.json"
    ref.save(out)
    size_mb = out.stat().st_size / 1e6

    print(
        f"  vocabulary {ref.vocab_size:,} | unigrams {len(ref.unigrams):,} | "
        f"bigrams {len(ref.bigrams):,} | trigrams {len(ref.trigrams):,}"
    )
    print(f"  {time.perf_counter() - started:.1f}s -> {out.relative_to(ROOT)} ({size_mb:.1f} MB)")

    # Sanity check: prose from the same genre should be less surprising than prose from a
    # different one. If this ever inverts, the reference is broken.
    essay = "I spent that summer working at my grandmother's shop, learning how to talk to people."
    alien = "Pursuant to subsection 4(b), the aforementioned party shall indemnify the licensor."
    import numpy as np

    e, a = np.mean(ref.surprisals(essay)), np.mean(ref.surprisals(alien))
    print(f"  check: essay prose {e:.2f} vs legal prose {a:.2f} surprisal -> "
          f"{'OK' if e < a else 'SUSPICIOUS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
