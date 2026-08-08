"""A second observer: how typical is this passage *of human essay writing specifically*?

The brief notes that some differences "only appear when a passage is compared against a
body of other writing". The local LM cannot supply that. GPT-2 knows what English looks
like in general; it has no idea what a seventeen-year-old writing about their grandmother
sounds like. So we fit a small interpolated n-gram model on the *human* half of our corpus
and use it as a second, domain-specific yardstick.

One design decision is worth stating up front, because the obvious alternative is
tempting and wrong.

We fit this reference on the **human corpus only**, never on a human-vs-machine likelihood
ratio. A ratio would be a stronger signal on our own test set and a worse detector in the
world: it would encode the particular quirks of the particular models we happened to
generate with, and quietly stop working on a generator we never saw. Measuring "distance
from human essay writing" keeps the feature generator-agnostic. We pay for that choice in
raw separation and buy robustness with it -- ``docs/06-decisions.md`` records the trade.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ..text.segment import tokenize_words

__all__ = ["NgramReference"]

# Interpolation weights for trigram / bigram / unigram. Chosen by grid search on a
# held-out slice of the human corpus (see scripts/fit_ngram.py); the surface is flat,
# anything near these values performs the same.
_LAMBDA3, _LAMBDA2, _LAMBDA1 = 0.5, 0.3, 0.2
_BOUNDARY = "<s>"


@dataclass
class NgramReference:
    """An interpolated trigram model of human essay prose."""

    unigrams: Counter
    bigrams: Counter
    trigrams: Counter
    total_tokens: int
    vocab_size: int
    n_documents: int

    # -- fitting ------------------------------------------------------------------

    @classmethod
    def fit(cls, documents: list[str], min_count: int = 2) -> "NgramReference":
        """Fit on human documents. Words seen fewer than ``min_count`` times become UNK."""
        raw = Counter()
        for doc in documents:
            raw.update(tokenize_words(doc))
        vocab = {w for w, c in raw.items() if c >= min_count}

        uni, bi, tri = Counter(), Counter(), Counter()
        total = 0
        for doc in documents:
            words = [w if w in vocab else "<unk>" for w in tokenize_words(doc)]
            padded = [_BOUNDARY, _BOUNDARY, *words]
            uni.update(words)
            total += len(words)
            for i in range(2, len(padded)):
                bi[(padded[i - 1], padded[i])] += 1
                tri[(padded[i - 2], padded[i - 1], padded[i])] += 1
        return cls(uni, bi, tri, total, len(vocab) + 1, len(documents))

    # -- scoring ------------------------------------------------------------------

    def _p_unigram(self, w: str) -> float:
        # Add-one smoothing over the closed vocabulary.
        return (self.unigrams.get(w, 0) + 1.0) / (self.total_tokens + self.vocab_size)

    def _p_bigram(self, prev: str, w: str) -> float:
        denom = self.unigrams.get(prev, 0) if prev != _BOUNDARY else self.n_documents
        if denom == 0:
            return self._p_unigram(w)
        return self.bigrams.get((prev, w), 0) / denom

    def _p_trigram(self, a: str, b: str, w: str) -> float:
        denom = self.bigrams.get((a, b), 0)
        if denom == 0:
            return self._p_bigram(b, w)
        return self.trigrams.get((a, b, w), 0) / denom

    def logprob(self, a: str, b: str, w: str) -> float:
        """Interpolated log P(w | a, b), never -inf."""
        p = (
            _LAMBDA3 * self._p_trigram(a, b, w)
            + _LAMBDA2 * self._p_bigram(b, w)
            + _LAMBDA1 * self._p_unigram(w)
        )
        return math.log(max(p, 1e-12))

    def surprisals(self, text: str) -> list[float]:
        """Per-word negative log-probability of ``text`` under the human reference."""
        words = [w if w in self.unigrams else "<unk>" for w in tokenize_words(text)]
        padded = [_BOUNDARY, _BOUNDARY, *words]
        return [
            -self.logprob(padded[i - 2], padded[i - 1], padded[i]) for i in range(2, len(padded))
        ]

    def novel_trigram_rate(self, text: str) -> float:
        """Fraction of the passage's word-trigrams that the human corpus has never seen.

        Human writing is idiosyncratic and constantly produces trigrams no one else wrote.
        Polished machine prose reaches for constructions that are already common, so this
        rate tends to fall. It is one of the few features that is *high* for humans.
        """
        words = [w if w in self.unigrams else "<unk>" for w in tokenize_words(text)]
        if len(words) < 3:
            return float("nan")
        grams = [(words[i - 2], words[i - 1], words[i]) for i in range(2, len(words))]
        unseen = sum(1 for g in grams if g not in self.trigrams)
        return unseen / len(grams)

    # -- persistence --------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "unigrams": dict(self.unigrams),
            "bigrams": {" ".join(k): v for k, v in self.bigrams.items()},
            "trigrams": {" ".join(k): v for k, v in self.trigrams.items()},
            "total_tokens": self.total_tokens,
            "vocab_size": self.vocab_size,
            "n_documents": self.n_documents,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NgramReference":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            unigrams=Counter(payload["unigrams"]),
            bigrams=Counter({tuple(k.split(" ")): v for k, v in payload["bigrams"].items()}),
            trigrams=Counter(
                {tuple(k.split(" ")): v for k, v in payload["trigrams"].items()}
            ),
            total_tokens=payload["total_tokens"],
            vocab_size=payload["vocab_size"],
            n_documents=payload["n_documents"],
        )
