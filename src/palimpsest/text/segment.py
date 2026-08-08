"""Segmentation of an essay into paragraphs and sentences, preserving character offsets.

Every downstream layer -- feature extraction, scoring, and the highlighting in the web UI --
addresses text by ``(start, end)`` offsets into the *original* string. So the one invariant
this module must never break is:

    text[span.start:span.end] == span.text

We deliberately do not pull in spaCy or NLTK. Sentence splitting for essay prose is a small,
well-understood problem, and a 400 MB dependency whose model downloads at import time would
be the single largest thing in the project for the least benefit. The rules below are written
out explicitly so they can be argued with -- see ``tests/test_segment.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["Span", "split_sentences", "split_paragraphs", "tokenize_words"]


@dataclass(frozen=True, slots=True)
class Span:
    """A half-open character range ``[start, end)`` into a source string."""

    start: int
    end: int
    text: str

    def __len__(self) -> int:
        return self.end - self.start


# Tokens that end in '.' but almost never end a sentence. Kept lowercase; matched case-folded.
_ABBREVIATIONS = frozenset(
    """
    mr mrs ms dr prof sr jr st rev hon gen col capt lt sgt gov pres supt
    inc ltd co corp dept univ assn bros
    jan feb mar apr jun jul aug sep sept oct nov dec
    mon tue tues wed thu thurs fri sat sun
    vs etc al ca cf ed eds vol no pp fig figs approx
    """.split()
)

# Multi-dot abbreviations such as "e.g." / "U.S." / "Ph.D." / "a.m.".
_DOTTED_ABBREV = re.compile(r"(?:[A-Za-z]\.){2,}$")

# A sentence terminator, optionally followed by closing quotes/brackets.
_TERMINATOR = re.compile(r"[.!?…]+[\"'”’)\]]*")

# Paragraphs are separated by a blank line, or by a single newline when the text uses
# one-line-per-paragraph formatting (common when an essay is pasted out of a text box).
_PARA_SPLIT = re.compile(r"\n\s*\n")

_WORD = re.compile(r"[A-Za-zÀ-ɏ']+")


def split_paragraphs(text: str) -> list[Span]:
    """Split ``text`` on blank lines into non-empty paragraph spans."""
    spans: list[Span] = []
    cursor = 0
    for chunk in _PARA_SPLIT.split(text):
        start = text.index(chunk, cursor) if chunk else cursor
        stripped = chunk.strip()
        if stripped:
            offset = chunk.index(stripped)
            spans.append(Span(start + offset, start + offset + len(stripped), stripped))
        cursor = start + len(chunk)
    return spans


def _is_abbreviation(text: str, dot_index: int) -> bool:
    """True if the '.' at ``dot_index`` closes an abbreviation rather than a sentence."""
    # Walk back over the word that the dot terminates.
    i = dot_index
    while i > 0 and (text[i - 1].isalpha() or text[i - 1] == "."):
        i -= 1
    word = text[i : dot_index + 1]
    if _DOTTED_ABBREV.match(word):
        return True
    return word.rstrip(".").lower() in _ABBREVIATIONS


def _is_decimal(text: str, dot_index: int) -> bool:
    """True for the '.' in '3.14' -- digits on both sides."""
    return (
        dot_index > 0
        and dot_index + 1 < len(text)
        and text[dot_index - 1].isdigit()
        and text[dot_index + 1].isdigit()
    )


def _is_initial(text: str, dot_index: int) -> bool:
    """True for the '.' after a single capital letter, as in 'J. K. Rowling'."""
    if dot_index == 0 or not text[dot_index - 1].isupper():
        return False
    before = dot_index - 2
    return before < 0 or not text[before].isalpha()


def _boundary_is_real(text: str, end: int) -> bool:
    """True if a sentence genuinely ends at offset ``end`` (exclusive)."""
    if end >= len(text):
        return True
    # Require whitespace after the terminator; "google.com" must not split.
    if not text[end].isspace():
        return False
    # The next non-space character should start a new sentence. Lowercase usually means
    # we cut something we should not have (e.g. an abbreviation we do not know about),
    # but a newline is strong enough evidence on its own.
    rest = text[end:]
    if "\n" in rest[: len(rest) - len(rest.lstrip())]:
        return True
    nxt = rest.lstrip()
    if not nxt:
        return True
    return not (nxt[0].isalpha() and nxt[0].islower())


def split_sentences(text: str) -> list[Span]:
    """Split ``text`` into sentence spans.

    Sentences never cross a paragraph boundary: a paragraph break is always a sentence
    break, even if the writer omitted the full stop. Leading/trailing whitespace is
    excluded from every span so that ``span.text`` is exactly the sentence.
    """
    spans: list[Span] = []
    for para in split_paragraphs(text):
        spans.extend(_split_paragraph_sentences(para))
    return spans


def _split_paragraph_sentences(para: Span) -> list[Span]:
    body = para.text
    out: list[Span] = []
    start = 0

    for match in _TERMINATOR.finditer(body):
        dot = match.start()
        end = match.end()
        char = body[dot]
        if char == "." and (
            _is_decimal(body, dot) or _is_initial(body, dot) or _is_abbreviation(body, dot)
        ):
            continue
        if not _boundary_is_real(body, end):
            continue
        piece = body[start:end]
        stripped = piece.strip()
        if stripped:
            lead = piece.index(stripped[0]) if stripped else 0
            out.append(
                Span(para.start + start + lead, para.start + start + lead + len(stripped), stripped)
            )
        start = end

    tail = body[start:]
    stripped = tail.strip()
    if stripped:
        lead = tail.index(stripped[0])
        out.append(
            Span(para.start + start + lead, para.start + start + lead + len(stripped), stripped)
        )
    return out


def tokenize_words(text: str) -> list[str]:
    """Lowercased alphabetic word tokens. Used by the n-gram and lexicon features."""
    return [m.group(0).lower() for m in _WORD.finditer(text)]
