"""Structural features: dependency-tree shape, stop-word load, part-of-speech rhythm.

Why this module exists
----------------------
Every other feature family here reads *which words appeared* -- their probability under an
observer (``model_based``), their identity against a lexicon (``surface``), their frequency
against an applicant-prose reference (``corpus``). A rewrite that swaps vocabulary and
varies sentence length defeats all three at once, because all three are counting words.

This family counts *structure*: how deep and how evenly a sentence branches, what fraction
of it is grammatical scaffolding, and how varied its part-of-speech sequence is. The
hypothesis under test is that a paraphrase pass moves words far more than it moves the
shape of the clause -- so a signal built on shape should survive an edit that a signal built
on words does not.

**That hypothesis is not confirmed on this corpus and this module does not assert it.**
The corpus holds no humanizer or paraphrase attacks (PROJECT.md §10), so nothing here can
show a feature "surviving" one. What ``scripts/syntax_probe.py`` measures is the ordinary
question -- do these separate human from machine at all, and do they separate *ESL* prose
from machine prose, which is the direction that matters for false accusations. Read that
artifact, not this docstring.

The two traps this module is built around
-----------------------------------------
**1. "Standard deviation of tree depth" is ambiguous, and one reading is nearly empty.**
A sentence has one dependency tree, hence one depth; a standard deviation needs a
population. Two different features are hiding in the phrase and both are computed here:

* ``tree_depth_sd`` -- spread of *token* depths inside one sentence. Measures whether the
  sentence branches evenly (a balanced list of parallel clauses) or lopsidedly (one long
  trailing subordinate limb). This is per sentence and needs no neighbours.
* ``tree_depth_z_in_doc`` -- how far this sentence's depth sits from the median depth of the
  *other* sentences in the same essay, in robust standard deviations. This is the "AI keeps
  a metronome, humans meander" claim in its measurable form, and it belongs to the existing
  ``context`` group, which already holds the same leave-one-out comparison for likelihood
  and length. Uniform structure across an essay shows up here as values near zero, not in
  ``tree_depth_sd``.

**2. Part-of-speech trigram entropy is length-degenerate.** A 15-word sentence yields ~13
trigrams that almost never repeat, so its empirical Shannon entropy is log(n_trigrams)
almost exactly -- a re-encoding of sentence length wearing an information-theory label. It
is computed here because it was asked for and because the degeneracy is worth showing
rather than asserting (``artifacts/syntax_probe.json`` reports its correlation with length),
but the feature expected to carry the actual signal is its companion:

* ``pos_trigram_surprisal`` -- mean surprisal of this sentence's POS trigrams under a
  trigram model fitted on *human applicant prose*, exactly mirroring ``corpus_surprisal_mean``
  for words. "Is this an ordinary grammatical shape for this genre" is a question with a
  reference distribution behind it, which is what makes it answerable.

Both are shipped so the comparison is visible.

Tagger
------
spaCy ``en_core_web_sm``, loaded once and lazily, with the parser required -- ``senter`` and
``ner`` are disabled since segmentation is done upstream in ``text/segment.py`` and entities
are not read here. If spaCy is absent every feature returns NaN, which the classifier
imputes to the training mean and therefore contributes exactly zero to the logit: an
unavailable tagger degrades the detector to its other features rather than crashing it or,
worse, silently voting.
"""

from __future__ import annotations

import logging
import math
from collections import Counter

import numpy as np

__all__ = [
    "ALL_SYNTAX_FEATURE_NAMES",
    "MIN_SENTENCES",
    "MIN_TOKENS",
    "SYNTAX_CONTEXT_FEATURE_NAMES",
    "SYNTAX_FEATURE_NAMES",
    "PosTrigramReference",
    "extract_syntax_context_features",
    "extract_syntax_features",
    "get_parser",
    "parser_available",
]

log = logging.getLogger(__name__)

SYNTAX_FEATURE_NAMES: tuple[str, ...] = (
    "tree_depth_max",
    "tree_depth_mean",
    "tree_depth_sd",
    "branching_factor",
    "stopword_ratio",
    "content_function_ratio",
    "pos_trigram_entropy",
    "pos_trigram_surprisal",
)

#: Document-relative structural features. Deliberately parallel to ``len_z_in_doc`` and
#: ``local_len_burstiness`` in ``features/context.py``: same leave-one-out construction,
#: same robust scaling, applied to tree depth instead of sentence length. This is where the
#: "AI holds one structural rhythm across a whole essay" claim becomes measurable.
SYNTAX_CONTEXT_FEATURE_NAMES: tuple[str, ...] = (
    "tree_depth_z_in_doc",
    "local_depth_burstiness",
    "pos_surprisal_z_in_doc",
)

ALL_SYNTAX_FEATURE_NAMES: tuple[str, ...] = (
    *SYNTAX_FEATURE_NAMES,
    *SYNTAX_CONTEXT_FEATURE_NAMES,
)

#: Matches ``context.MIN_SENTENCES``: below this an essay has no stable internal baseline.
MIN_SENTENCES = 5

#: Matches ``context._DEGENERATE_Z``. An in-document z-score is unbounded when the baseline
#: has no spread, and no single feature may be allowed to dominate the logit.
_DEGENERATE_Z = 6.0

#: Below this many parsed tokens the tree statistics describe punctuation more than syntax.
#: Matches MIN_WORDS in surface.py and MIN_TOKENS in model_based.py: short spans return NaN
#: rather than a confident-looking number computed from three tokens.
MIN_TOKENS = 4

_MODEL = "en_core_web_sm"
_nlp = None
_load_failed = False


def get_parser():
    """Return the shared spaCy pipeline, or None if spaCy/the model is unavailable.

    Loaded once per process. A failure is logged once and cached, so a missing model costs
    one warning rather than one per sentence.
    """
    global _nlp, _load_failed
    if _nlp is not None or _load_failed:
        return _nlp
    try:
        import spacy

        # The parser is the point; sentence segmentation is upstream and NER is unused.
        _nlp = spacy.load(_MODEL, disable=["ner", "senter", "lemmatizer"])
        log.info("syntax: loaded %s", _MODEL)
    except Exception as exc:  # pragma: no cover - exercised by the no-spacy path
        _load_failed = True
        log.warning(
            "syntax: %s unavailable (%s); syntax features will be NaN and contribute "
            "nothing to the logit. Install with: python -m spacy download %s",
            _MODEL, exc, _MODEL,
        )
    return _nlp


def parser_available() -> bool:
    """Whether real parses are being produced. Reported by /api/health."""
    return get_parser() is not None


def _depths(doc) -> np.ndarray:
    """Depth of every non-punctuation token, measured as edges from its sentence root.

    Walks ``head`` pointers with a visited set. spaCy guarantees a tree, but a guarantee is
    a poor reason to write a loop that cannot terminate on malformed input.
    """
    out = []
    for tok in doc:
        if tok.is_punct or tok.is_space:
            continue
        d, cur, seen = 0, tok, {tok.i}
        while cur.head.i != cur.i:  # the root is its own head
            cur = cur.head
            if cur.i in seen:  # cycle: refuse rather than spin
                return np.zeros(0, dtype=np.float64)
            seen.add(cur.i)
            d += 1
            if d > 200:
                return np.zeros(0, dtype=np.float64)
        out.append(d)
    return np.asarray(out, dtype=np.float64)


class PosTrigramReference:
    """A trigram model over part-of-speech tags, fitted on human applicant prose.

    The word-level twin of ``scorer.ngram.NgramReference``, and it exists for the same
    reason: "how surprising is this shape" is only answerable against a stated reference.
    The reference here is the *human* half of the training pool, so the question the feature
    asks is "is this an ordinary grammatical shape for a real applicant", not "is this
    grammatical English".

    Stupid-backoff smoothing (trigram -> bigram -> unigram -> uniform). The tag vocabulary
    is ~18 tags, so counts are dense and elaborate smoothing buys nothing measurable.
    """

    __slots__ = ("alpha", "bi", "n_tags", "total", "tri", "uni")

    def __init__(self, tri=None, bi=None, uni=None, total=0, n_tags=18, alpha=0.4):
        self.tri = tri or {}
        self.bi = bi or {}
        self.uni = uni or {}
        self.total = int(total)
        self.n_tags = int(n_tags)
        self.alpha = float(alpha)

    @classmethod
    def fit(cls, tag_sequences) -> PosTrigramReference:
        tri: Counter = Counter()
        bi: Counter = Counter()
        uni: Counter = Counter()
        for tags in tag_sequences:
            padded = ["<s>", "<s>", *tags, "</s>"]
            uni.update(padded)
            for i in range(len(padded) - 1):
                bi[(padded[i], padded[i + 1])] += 1
            for i in range(len(padded) - 2):
                tri[(padded[i], padded[i + 1], padded[i + 2])] += 1
        vocab = {t for t in uni if t not in ("<s>", "</s>")}
        return cls(dict(tri), dict(bi), dict(uni), sum(uni.values()), max(len(vocab), 1))

    def surprisal(self, tags: list[str]) -> float:
        """Mean negative log2 probability per tag. NaN on an empty sequence."""
        if not tags:
            return float("nan")
        padded = ["<s>", "<s>", *tags, "</s>"]
        total = 0.0
        for i in range(2, len(padded)):
            ctx2, ctx1, tag = padded[i - 2], padded[i - 1], padded[i]
            c3 = self.tri.get((ctx2, ctx1, tag), 0)
            c2 = self.bi.get((ctx2, ctx1), 0)
            if c3 and c2:
                p = c3 / c2
            else:
                b2 = self.bi.get((ctx1, tag), 0)
                b1 = self.uni.get(ctx1, 0)
                if b2 and b1:
                    p = self.alpha * (b2 / b1)
                else:
                    u = self.uni.get(tag, 0)
                    p = (self.alpha ** 2) * (u / self.total) if u and self.total else 0.0
            if p <= 0.0:
                p = (self.alpha ** 3) / max(self.n_tags, 1)
            total += -math.log2(p)
        return total / (len(padded) - 2)

    def to_dict(self) -> dict:
        return {
            "tri": {"\t".join(k): v for k, v in self.tri.items()},
            "bi": {"\t".join(k): v for k, v in self.bi.items()},
            "uni": self.uni,
            "total": self.total,
            "nTags": self.n_tags,
            "alpha": self.alpha,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PosTrigramReference:
        return cls(
            {tuple(k.split("\t")): v for k, v in d["tri"].items()},
            {tuple(k.split("\t")): v for k, v in d["bi"].items()},
            d["uni"], d["total"], d.get("nTags", 18), d.get("alpha", 0.4),
        )


def pos_tags(text: str) -> list[str]:
    """Coarse POS tags for one span, punctuation and whitespace dropped. [] if no parser."""
    nlp = get_parser()
    if nlp is None or not text.strip():
        return []
    return [t.pos_ for t in nlp(text) if not (t.is_punct or t.is_space)]


def extract_syntax_features(
    text: str, reference: PosTrigramReference | None = None
) -> dict[str, float]:
    """Compute the structural features for one span of text.

    Returns every name in ``SYNTAX_FEATURE_NAMES``, NaN where unmeasurable -- no parser, an
    empty span, or fewer than ``MIN_TOKENS`` real tokens.
    """
    nan = dict.fromkeys(SYNTAX_FEATURE_NAMES, float("nan"))
    nlp = get_parser()
    if nlp is None or not text.strip():
        return nan

    doc = nlp(text)
    real = [t for t in doc if not (t.is_punct or t.is_space)]
    n = len(real)
    if n < MIN_TOKENS:
        return nan

    depths = _depths(doc)
    if depths.size < MIN_TOKENS:
        return nan

    # Branching: children per non-leaf token. A flat sentence of coordinated clauses hangs
    # many children off few heads; a deeply nested one chains them.
    child_counts = [sum(1 for _ in t.children) for t in doc if not (t.is_punct or t.is_space)]
    non_leaf = [c for c in child_counts if c > 0]

    tags = [t.pos_ for t in real]
    stop = sum(1 for t in real if t.is_stop)
    # Open-class = content. spaCy's coarse tag set, so this is stable across models.
    content = sum(1 for t in real if t.pos_ in ("NOUN", "PROPN", "VERB", "ADJ", "ADV"))
    function = n - content

    trigrams = [tuple(tags[i:i + 3]) for i in range(len(tags) - 2)]
    if trigrams:
        counts = np.asarray(list(Counter(trigrams).values()), dtype=np.float64)
        probs = counts / counts.sum()
        entropy = float(-(probs * np.log2(probs)).sum())
    else:
        entropy = float("nan")

    return {
        "tree_depth_max": float(depths.max()),
        "tree_depth_mean": float(depths.mean()),
        # ddof=1: this is a sample of token depths, and with n small the population form is
        # biased low -- which would look like "machine-uniform" on short sentences.
        "tree_depth_sd": float(depths.std(ddof=1)) if depths.size > 1 else float("nan"),
        "branching_factor": float(np.mean(non_leaf)) if non_leaf else float("nan"),
        "stopword_ratio": float(stop / n),
        # Guarded: an all-content sentence ("Rain fell." after punctuation removal) would
        # otherwise divide by zero and arrive as inf, which standardises to garbage rather
        # than to the training mean.
        "content_function_ratio": float(content / function) if function else float("nan"),
        "pos_trigram_entropy": entropy,
        "pos_trigram_surprisal": (
            reference.surprisal(tags) if reference is not None else float("nan")
        ),
    }


# ---------------------------------------------------------------------------------------
# Document-relative structure
# ---------------------------------------------------------------------------------------


def extract_syntax_context_features(
    base: list[dict[str, float]], n_sentences: int
) -> list[dict[str, float]]:
    """Derive each sentence's structural features *relative to the rest of its essay*.

    ``base`` is the per-sentence output of ``extract_syntax_features``, in document order.

    Every statistic is leave-one-out, for the reason ``context.py`` gives: a sentence
    compared against a baseline it is itself part of shrinks its own deviation, which hides
    exactly the single inserted paragraph the tool exists to find.
    """
    if n_sentences < MIN_SENTENCES:
        return [dict.fromkeys(SYNTAX_CONTEXT_FEATURE_NAMES, float("nan")) for _ in base]

    depths = np.array([row.get("tree_depth_max", np.nan) for row in base], dtype=np.float64)
    return [
        {
            "tree_depth_z_in_doc": _loo_z(base, i, "tree_depth_max"),
            "local_depth_burstiness": _local_burstiness(depths, i),
            "pos_surprisal_z_in_doc": _loo_z(base, i, "pos_trigram_surprisal"),
        }
        for i in range(n_sentences)
    ]


def _loo_z(base: list[dict[str, float]], i: int, key: str) -> float:
    """Leave-one-out robust z-score of ``base[i][key]`` against the other sentences.

    Median/MAD rather than mean/sd: one inserted machine paragraph must not be allowed to
    move the baseline it is being measured against. Kept byte-for-byte equivalent to
    ``context._loo_z`` -- including the degenerate-spread branch, where a sentence differing
    from a perfectly uniform baseline is maximally unusual and must not report 0.0.
    """
    values = np.array([row.get(key, np.nan) for row in base], dtype=np.float64)
    here = values[i]
    if not np.isfinite(here):
        return float("nan")
    others = np.delete(values, i)
    others = others[np.isfinite(others)]
    if len(others) < 3:
        return float("nan")
    centre = float(np.median(others))
    mad = float(np.median(np.abs(others - centre))) * 1.4826
    if mad < 1e-9:
        spread = float(others.std(ddof=1))
        if spread < 1e-9:
            delta = here - centre
            if abs(delta) < 1e-9:
                return 0.0
            return float(np.sign(delta) * _DEGENERATE_Z)
        mad = spread
    return float(np.clip((here - centre) / mad, -_DEGENERATE_Z, _DEGENERATE_Z))


def _local_burstiness(values: np.ndarray, i: int, half_window: int = 2) -> float:
    """Coefficient of variation of tree depth in a five-sentence window centred on ``i``.

    The structural twin of ``context._local_burstiness``. Low values mean neighbouring
    sentences share one clause shape -- the parallel-construction habit this family was
    built to read. Local rather than whole-document because a polished *passage* sits inside
    an otherwise human essay, and a document-wide statistic would average that seam away.
    """
    lo = max(0, i - half_window)
    hi = min(len(values), i + half_window + 1)
    window = values[lo:hi]
    window = window[np.isfinite(window)]
    if len(window) < 3 or window.mean() < 1e-9:
        return float("nan")
    return float(window.std(ddof=1) / window.mean())
