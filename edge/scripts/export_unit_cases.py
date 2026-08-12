"""Export Python's answers for constructed strings that the corpus never exercises.

The document-level parity run compares 145 real essays and passes, but real admissions
essays are ordinary text: they contain no accented contraction, no non-ASCII digit, no
`Ph.D.` mid-sentence, no run-on with a decimal in it. Several of the trickiest lines in the
port -- the hand-written Unicode word boundaries, `\\p{Nd}` for Python's `\\d`, the
abbreviation walk-back -- are therefore *not* covered by it, and a mutation test confirmed
that: reverting the contraction pattern to JavaScript's ASCII-only `\\b\\w+` leaves the
whole corpus run green.

So the awkward inputs are constructed here, and Python is asked what it says about them.

    .venv/bin/python edge/scripts/export_unit_cases.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "edge" / "test" / "unit-cases.json"

#: Each entry targets a specific line of the port. Where a case exists to defeat a plausible
#: shortcut, the shortcut is named.
CASES: list[tuple[str, str]] = [
    ("plain", "I grew up in Kanpur. My father repaired radios. I learned to listen."),
    # Unicode `\w` inside the contraction pattern: ASCII `\b\w+` misses these entirely.
    ("accented-contraction", "The café's owner and José's mother wouldn't agree. Renée's idea won."),
    ("curly-apostrophe", "It wasn’t her plan. She’d already left, and they’re still arguing."),
    ("naive-ascii-digits", "I scored ٤٥ and ９９ on the test. Also 3.14 and 42 appear here."),
    # `\d` on a str pattern is Unicode-decimal in Python; JavaScript's `\d` is ASCII.
    ("decimal-not-a-boundary", "Pi is 3.14 and e is 2.718. That ends the list."),
    ("abbreviations", "Dr. Rao met Mrs. Iyer at 5 p.m. on Sept. 3. They talked for hours."),
    ("dotted-abbrev", "She earned a Ph.D. in the U.S. before moving. I visited her there."),
    ("initials", "J. K. Rowling wrote it. R. R. Martin did not. Both sold well."),
    ("lowercase-after-dot", "I went to google.com. then i stopped. The rest is history."),
    ("no-terminal-punctuation", "this essay has no full stops at all and simply runs on forever"),
    ("paragraphs", "First paragraph here.\n\nSecond paragraph starts.\nThird line joins it.\n\n\nFourth."),
    ("single-newlines", "Line one.\nLine two.\nLine three.\nLine four.\nLine five."),
    ("leading-trailing-space", "   \n\n  Indented start. And an end.   \n\n   "),
    ("tricolon", "I brought books, a compass, and my grandfather's watch. It was enough."),
    ("tricolon-negative", "I brought books and a compass and my watch. No list here."),
    ("antithesis", "It was not just a hobby but a calling. It wasn't luck, it was practice."),
    ("antithesis-more-than", "The lab was more than just a room. It shaped me."),
    ("em-dashes", "The plan — ambitious, unfunded — collapsed. Then we rebuilt it - slowly - again."),
    ("mid-caps", "I met Priya in Delhi, and Arjun in Chennai; Meera came later."),
    ("machine-phrases", "It is important to note that this experience taught me a valuable lesson. In conclusion, it was a testament to perseverance."),
    ("punct-variety", "Why? Because: commas, semicolons; dashes—and (brackets) too! Yes."),
    ("quotes-after-terminator", 'She said "I will go." Then she left. He asked "Why?" and waited.'),
    ("ellipsis", "I waited… and waited. Nothing came. Then everything did."),
    ("very-short", "Hi. Ok. No."),
    ("one-long-runon", " ".join(["word"] * 200)),
    ("repeated-identical", "The same. The same. The same. The same. The same. The same."),
    ("hyphens", "A well-known, self-taught, ever-evolving student—me—applied anyway. It worked."),
    ("numbers-heavy", "In 2019 I logged 1,200 hours across 3 labs and 47 experiments. Then I stopped."),
    ("first-person-none", "The committee reviewed applications. Standards were high. Results followed. Many waited."),
    ("nonbreaking-space", "This sentence uses NBSP. This one uses thin space. Done."),
]


#: Whitespace where Python and JavaScript genuinely disagree about what whitespace *is*.
#: Written as escapes rather than literals so the file stays readable and greppable.
#:
#: `str.strip()` takes U+001C-U+001F and U+0085; `String.trim()` does not. `String.trim()`
#: takes U+FEFF; Python does not. And `\s` inside the paragraph splitter `\n\s*\n` inherits
#: both differences, which moves where paragraphs — and every downstream highlight — begin.
CASES += [
    ("py-only-whitespace", "  I began badly. And ended badly too. "),
    ("bom-not-whitespace", "﻿I start with a byte-order mark. It should survive."),
    ("para-split-nel", "First para ends here.\n\nSecond para starts. It runs on."),
    ("para-split-bom", "First para ends here.\n﻿\nNot a paragraph break for Python."),
]


def main() -> int:
    from palimpsest.features.surface import extract_surface_features
    from palimpsest.features.context import extract_context_features
    from palimpsest.scorer.ngram import NgramReference
    from palimpsest.text.segment import split_paragraphs, split_sentences, tokenize_words
    from palimpsest.features.corpus import extract_corpus_features
    from palimpsest.detect.genre import document_genre_features

    ref_path = ROOT / "artifacts" / "ngram_reference.json"
    reference = NgramReference.load(ref_path) if ref_path.exists() else None

    def clean(d: dict) -> dict:
        return {k: (None if isinstance(v, float) and v != v else v) for k, v in d.items()}

    out = []
    for name, text in CASES:
        sentences = split_sentences(text)
        # A fixed observer mean-logprob, so the corpus gap feature is exercised without
        # needing a real scoring call. The port must reproduce the same arithmetic.
        base = []
        for s in sentences:
            surface = extract_surface_features(s.text)
            corpus = extract_corpus_features(s.text, reference, -3.5)
            base.append({**surface, **corpus})
        context = extract_context_features(base, len(sentences))
        combined = [{**b, **c} for b, c in zip(base, context, strict=True)]

        out.append({
            "name": name,
            "text": text,
            "paragraphs": [{"start": p.start, "end": p.end, "text": p.text}
                           for p in split_paragraphs(text)],
            "sentences": [{"start": s.start, "end": s.end, "text": s.text} for s in sentences],
            "words": tokenize_words(text),
            "features": [clean(f) for f in combined],
            "genre": clean(document_genre_features(combined)),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out), encoding="utf-8")
    print(f"edge/test/unit-cases.json: {len(out)} constructed cases, "
          f"{sum(len(c['sentences']) for c in out)} sentences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
