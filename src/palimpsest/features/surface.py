"""Stylometric features that need no model at all.

These are cheap, fully transparent, and -- importantly -- they fail in *different* ways from
the likelihood features. A writer who defeats GPT-2's surprisal estimate by writing unusual
prose does not thereby stop using tricolons; a writer whose English is non-native trips the
likelihood features but not the construction ones. Combining families with uncorrelated
failure modes is most of what makes the ensemble worth having.
"""

from __future__ import annotations

import re

import numpy as np

from ..text.segment import tokenize_words
from .lexicons import (
    BOOSTERS,
    CONCRETE_MARKERS,
    DISCOURSE_MARKERS,
    FUNCTION_WORDS,
    HEDGES,
    MACHINE_LEANING_PHRASES,
    MACHINE_LEANING_WORDS,
)

__all__ = ["extract_surface_features", "SURFACE_FEATURE_NAMES", "MIN_WORDS"]

MIN_WORDS = 4

SURFACE_FEATURE_NAMES: tuple[str, ...] = (
    "n_words",
    "mean_word_len",
    "long_word_rate",
    "root_ttr",
    "comma_rate",
    "punct_variety",
    "em_dash_rate",
    "subordination_rate",
    "function_word_rate",
    "machine_phrase_rate",
    "machine_word_rate",
    "hedge_rate",
    "booster_rate",
    "discourse_marker_rate",
    "first_person_rate",
    "contraction_rate",
    "specificity_rate",
    "tricolon",
    "antithesis",
)

_CLAUSE_MARKERS = frozenset(
    """
    that which who whom whose because although though while whereas since unless
    until whether if when where after before as
    """.split()
)
_FIRST_PERSON = frozenset("i me my mine myself we us our ours".split())

_PUNCT = re.compile(r"[,;:—–\-()\"'!?.]")
_EM_DASH = re.compile(r"[—–]|(?<= )-{1,2}(?= )")
_CONTRACTION = re.compile(r"\b\w+['’](?:s|t|re|ve|ll|d|m)\b", re.IGNORECASE)
_DIGIT = re.compile(r"\d")
# A capitalised word that is not sentence-initial and is not the pronoun "I" -- a cheap
# proper-noun proxy that needs no part-of-speech tagger.
_MID_CAPS = re.compile(r"(?<=[a-z,;] )([A-Z][a-z]{2,})")
# "A, B, and C" -- the tricolon that polished machine prose reaches for constantly.
_TRICOLON = re.compile(r"\b[\w'’]+(?:\s+[\w'’]+){0,3},\s+[\w'’]+(?:\s+[\w'’]+){0,3},\s+and\s+")
# "not just X but Y" / "not only ... but also" / "it wasn't X, it was Y"
_ANTITHESIS = re.compile(
    r"\bnot (?:just|only|merely|simply)\b.{0,80}?\bbut\b"
    r"|\bwas ?n[o'’]t\b.{0,60}?\bit was\b"
    r"|\bmore than (?:just )?a\b",
    re.IGNORECASE | re.DOTALL,
)


def extract_surface_features(text: str) -> dict[str, float]:
    """Compute the model-free features for one span of text."""
    words = tokenize_words(text)
    n = len(words)
    if n < MIN_WORDS:
        return dict.fromkeys(SURFACE_FEATURE_NAMES, float("nan"))

    lower = text.lower()
    per100 = 100.0 / n

    lengths = np.array([len(w) for w in words], dtype=np.float64)
    phrase_hits = sum(1 for p in MACHINE_LEANING_PHRASES if p in lower)

    specific = (
        len(_MID_CAPS.findall(text))
        + len(_DIGIT.findall(text))
        + sum(1 for w in words if w in CONCRETE_MARKERS)
    )

    return {
        # --- shape
        "n_words": float(n),
        "mean_word_len": float(lengths.mean()),
        "long_word_rate": float((lengths > 6).mean()),
        # Root type-token ratio. Plain TTR falls mechanically as length grows, which would
        # make it a length feature in disguise; dividing by sqrt(n) largely removes that.
        "root_ttr": float(len(set(words)) / np.sqrt(n)),
        # --- punctuation and clause structure
        "comma_rate": text.count(",") * per100,
        "punct_variety": float(len(set(_PUNCT.findall(text)))),
        "em_dash_rate": len(_EM_DASH.findall(text)) * per100,
        "subordination_rate": sum(1 for w in words if w in _CLAUSE_MARKERS) * per100,
        "function_word_rate": sum(1 for w in words if w in FUNCTION_WORDS) * per100,
        # --- register and vocabulary
        "machine_phrase_rate": phrase_hits * per100,
        "machine_word_rate": sum(1 for w in words if w in MACHINE_LEANING_WORDS) * per100,
        "hedge_rate": sum(1 for w in words if w in HEDGES) * per100,
        "booster_rate": sum(1 for w in words if w in BOOSTERS) * per100,
        "discourse_marker_rate": sum(1 for w in words if w in DISCOURSE_MARKERS) * per100,
        "first_person_rate": sum(1 for w in words if w in _FIRST_PERSON) * per100,
        "contraction_rate": len(_CONTRACTION.findall(text)) * per100,
        # --- specificity: names, numbers, dates. Its ABSENCE is the informative case.
        "specificity_rate": specific * per100,
        # --- two constructions machine prose over-produces
        "tricolon": float(bool(_TRICOLON.search(text))),
        "antithesis": float(bool(_ANTITHESIS.search(text))),
    }
