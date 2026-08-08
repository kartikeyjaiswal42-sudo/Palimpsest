"""Sentence segmentation. The invariant that everything downstream depends on."""

from __future__ import annotations

import pytest

from palimpsest.text import split_paragraphs, split_sentences, tokenize_words

TEXTS = [
    'Dr. Smith arrived at 3.14 p.m. on Monday. He said, "This is fine." Then he left!',
    "I burned the rice. Again. My grandmother said nothing.\n\nThat was worse.",
    "Visit google.com for details. It works.",
    "J. K. Rowling wrote it. Approx. 500 pages, i.e. long.",
    "No terminal punctuation here",
    "",
    "   ",
    "Ellipsis... then more. And a question? Yes!",
]


@pytest.mark.parametrize("text", TEXTS)
def test_offsets_map_back_exactly(text):
    """text[span.start:span.end] == span.text, always. Highlighting depends on it."""
    for span in split_sentences(text):
        assert text[span.start : span.end] == span.text


@pytest.mark.parametrize("text", TEXTS)
def test_spans_are_ordered_and_disjoint(text):
    spans = split_sentences(text)
    for a, b in zip(spans, spans[1:], strict=False):
        assert a.end <= b.start


@pytest.mark.parametrize("text", TEXTS)
def test_no_words_are_lost(text):
    """Segmentation may drop whitespace but must never drop a word."""
    joined = " ".join(s.text for s in split_sentences(text))
    assert tokenize_words(joined) == tokenize_words(text)


def test_abbreviations_do_not_split():
    assert len(split_sentences("Dr. Smith arrived at 3.14 p.m. on Monday.")) == 1


def test_decimals_do_not_split():
    assert len(split_sentences("The value was 3.14 exactly.")) == 1


def test_initials_do_not_split():
    assert len(split_sentences("J. K. Rowling wrote it.")) == 1


def test_paragraph_break_forces_a_sentence_break():
    text = "First para has no full stop\n\nSecond para does."
    assert len(split_sentences(text)) == 2


def test_quoted_terminator_stays_attached():
    spans = split_sentences('He said, "This is fine." Then he left.')
    assert spans[0].text.endswith('"')
    assert len(spans) == 2


def test_empty_input():
    assert split_sentences("") == []
    assert split_paragraphs("") == []
