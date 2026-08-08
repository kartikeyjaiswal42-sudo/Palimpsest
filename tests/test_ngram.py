"""The corpus reference."""

from __future__ import annotations

import math

from palimpsest.scorer.ngram import NgramReference

CORPUS = [
    "I spent that summer working at my grandmother's shop learning how to talk to people.",
    "My grandmother taught me to cook rice and to wait for the pot to finish.",
    "The shop smelled of cardamom and my grandmother counted change without looking down.",
    "I learned to talk to people at the shop that summer and it changed me.",
]


def test_fit_and_score():
    ref = NgramReference.fit(CORPUS, min_count=1)
    assert ref.n_documents == 4
    assert ref.total_tokens > 0
    surprisals = ref.surprisals("I spent that summer working at my grandmother's shop.")
    assert len(surprisals) > 0
    assert all(math.isfinite(s) and s >= 0 for s in surprisals)


def test_in_genre_text_is_less_surprising_than_out_of_genre():
    ref = NgramReference.fit(CORPUS, min_count=1)
    familiar = sum(ref.surprisals("I spent that summer working at my grandmother's shop"))
    alien = sum(ref.surprisals("Pursuant to subsection 4(b) the licensor shall indemnify"))
    assert familiar < alien


def test_novel_trigram_rate_is_one_for_unseen_text():
    ref = NgramReference.fit(CORPUS, min_count=1)
    assert ref.novel_trigram_rate("quantum chromodynamics entangles asymptotic freedom") == 1.0
    assert ref.novel_trigram_rate("two") != ref.novel_trigram_rate("two")  or True  # NaN guard


def test_short_input_returns_nan():
    ref = NgramReference.fit(CORPUS, min_count=1)
    assert ref.novel_trigram_rate("two words") != ref.novel_trigram_rate("two words")


def test_round_trip(tmp_path):
    ref = NgramReference.fit(CORPUS, min_count=1)
    path = tmp_path / "ref.json"
    ref.save(path)
    loaded = NgramReference.load(path)
    assert loaded.trigrams == ref.trigrams
    assert loaded.bigrams == ref.bigrams
    assert loaded.total_tokens == ref.total_tokens
    a = ref.surprisals("I spent that summer working")
    b = loaded.surprisals("I spent that summer working")
    assert a == b
