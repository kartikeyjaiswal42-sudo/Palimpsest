"""What the tool does with text it has decided it cannot measure.

``aggregate`` learned this rule after it accused three real second-language students: a span
marked unreliable must not decide the answer. The rule was applied there and nowhere else,
so the same response could refuse to score an essay and simultaneously name the whole of it
as a machine-written passage. These tests hold every consumer of ``reliable`` to the rule,
and hold the interface to explaining the RIGHT reason -- the fixed phrase it used to print,
"too short to measure reliably", is precisely backwards for the 138- to 466-word run-ons the
rule exists to protect.
"""

from __future__ import annotations

import numpy as np
import pytest

from palimpsest.analyze import _why_unmeasurable
from palimpsest.detect.document import (
    MAX_SENTENCE_WORDS,
    SentenceVerdict,
    aggregate,
    find_passages,
    smooth_probabilities,
)

#: One unmeasurable 466-word span: an ELLIPSE essay written without sentence-ending
#: punctuation, which the segmenter returns whole.
RUNON = SentenceVerdict(0, 0, 600, "x " * 466, 0.99, 466, 0.99, False)


def ordinary(index: int, prob: float = 0.02, words: int = 20) -> SentenceVerdict:
    start = 1000 + index * 100
    return SentenceVerdict(index, start, start + 90, "a sentence.", prob, words, prob, True)


def test_an_unmeasurable_span_is_never_published_as_a_passage():
    """The exact contradiction: no verdict, but a passage naming the essay."""
    verdict = aggregate([RUNON], threshold=0.5)
    passages = find_passages([RUNON])

    assert verdict.n_reliable_sentences == 0
    assert verdict.machine_share == 0.0
    assert passages == [], (
        "the aggregator refused to score this span and find_passages published it as a "
        "machine-written passage anyway -- one response, two incompatible answers, on the "
        "essay of a second-language student"
    )


def test_a_passage_never_stretches_across_an_unmeasured_span():
    """Two flagged runs either side of a run-on must stay two runs, not become one."""
    left = SentenceVerdict(0, 0, 90, "a.", 0.99, 30, 0.99, True)
    right = SentenceVerdict(2, 700, 790, "b.", 0.99, 30, 0.99, True)
    middle = SentenceVerdict(1, 100, 690, "x " * 200, 0.99, 200, 0.99, False)

    passages = find_passages([left, middle, right])

    assert len(passages) == 2, (
        "an unmeasurable span was stepped over, so one passage was reported spanning "
        "characters the observer never scored"
    )
    assert [p.sentence_indices for p in passages] == [[0], [2]]


def test_an_unmeasurable_span_does_not_decide_its_neighbours_smoothing():
    """Smoothing is weighted by WORDS, and the unmeasurable spans are the long ones."""
    probs = np.array([0.05, 0.99, 0.05])
    words = np.array([20.0, 466.0, 20.0])
    reliable = np.array([True, False, True])

    unaware = smooth_probabilities(probs, words)
    aware = smooth_probabilities(probs, words, reliable)

    assert unaware[0] > 0.65, "this fixture no longer reproduces the contamination"
    assert aware[0] == pytest.approx(0.05, abs=1e-9), (
        f"a 20-word sentence scoring 5% was smoothed to {aware[0]:.1%} by an unmeasurable "
        "neighbour, which is what decides whether it is drawn as part of a machine passage"
    )
    assert aware[1] == pytest.approx(0.99), "an unreliable span keeps its own raw probability"


def test_smoothing_is_unchanged_when_everything_is_measurable():
    """The ordinary case must be bit-identical, or every published number moves."""
    rng = np.random.default_rng(7)
    probs = rng.random(40)
    words = rng.integers(5, 60, size=40).astype(float)

    assert smooth_probabilities(probs, words) == pytest.approx(
        smooth_probabilities(probs, words, np.ones(40, dtype=bool)), abs=0.0
    )


def test_a_reliable_flagged_run_is_still_reported():
    """The fix must not silence real findings."""
    run = [ordinary(i, prob=0.99, words=30) for i in range(3)]
    passages = find_passages([*run, ordinary(3, prob=0.01)])

    assert [p.sentence_indices for p in passages] == [[0, 1, 2]]
    assert passages[0].peak_probability == pytest.approx(0.99)


# ---------------------------------------------------------------------------------------
# Which reason the reader is given.


def test_the_reason_for_a_run_on_is_not_that_it_is_short():
    from palimpsest.scorer.local_lm import TokenScores

    scores = _stub_scores(clipped=False, last_char=5000)
    reason = _why_unmeasurable(
        n_tokens=180, n_words=MAX_SENTENCE_WORDS + 48, start=0, scores=scores
    )

    assert reason == "too_long", (
        "a 138-word run-on was explained to its author as 'too short to measure reliably'; "
        "these essays are written by second-language students, who are the people most "
        "likely to want to contest the result and least helped by a false explanation"
    )
    assert isinstance(scores, TokenScores)


def test_a_span_the_observer_never_read_says_so():
    scores = _stub_scores(clipped=True, last_char=6000)
    assert _why_unmeasurable(n_tokens=0, n_words=30, start=6200, scores=scores) == (
        "beyond_observer_window"
    ), "a sentence past the observer's window was reported as too short to measure"


def test_a_genuinely_short_span_still_says_short():
    scores = _stub_scores(clipped=False, last_char=5000)
    assert _why_unmeasurable(n_tokens=2, n_words=3, start=10, scores=scores) == "too_short"


def test_a_measurable_span_has_no_reason():
    scores = _stub_scores(clipped=False, last_char=5000)
    assert _why_unmeasurable(n_tokens=40, n_words=22, start=10, scores=scores) is None


def _stub_scores(clipped: bool, last_char: int):
    """A TokenScores carrying only what the reason-resolver reads."""
    from palimpsest.scorer.local_lm import TokenScores

    z_i = np.array([last_char], dtype=np.int32)
    z_f = np.zeros(1, dtype=np.float32)
    return TokenScores(
        tokens=["x"], char_start=z_i, char_end=z_i, logprob=z_f, rank=z_i,
        entropy=z_f, mu=z_f, sigma2=z_f, model_name="stub", device="stub", clipped=clipped,
    )


# ---------------------------------------------------------------------------------------
# The document band.
#
# `aggregate` and `find_passages` both learned the rule that an unmeasurable span must not
# decide the answer. The BAND did not, and the band is the one line a reader acts on. With
# no reliable sentence the aggregate reports `any_machine_probability = 0.0` -- the absence
# of a measurement, not a low one -- and zero sits below `tHuman`, so a document the tool
# never scored a word of came back "No evidence of machine writing", quoting a calibration
# ("N% of known machine essays land here") derived from documents that were actually read.


def test_a_document_with_nothing_measurable_is_not_given_a_calibrated_band():
    from palimpsest.api.app import _band, _unmeasurable_band

    verdict = aggregate([RUNON], threshold=0.5)
    assert verdict.n_reliable_sentences == 0
    assert verdict.any_machine_probability == 0.0

    # What the old code did with that zero.
    assert _band(verdict.any_machine_probability)["band"] == "no_evidence", (
        "this test no longer reproduces the original defect -- the bands moved, so the "
        "check below is no longer evidence of anything"
    )

    answer = _unmeasurable_band(verdict.n_sentences)
    assert answer["band"] == "insufficient_evidence"
    assert answer["canExonerate"] is False
    assert "could be scored" in answer["bandDetail"]
    # It must not read as a finding in either direction.
    assert "no evidence of machine writing" not in answer["bandLabel"].lower()


def test_the_unmeasurable_band_counts_the_spans_it_refused():
    from palimpsest.api.app import _unmeasurable_band

    assert "1 span " in _unmeasurable_band(1)["bandDetail"]
    assert "7 spans " in _unmeasurable_band(7)["bandDetail"]


def test_a_document_with_one_measurable_sentence_still_gets_a_real_band():
    """The guard is for zero, not for 'few'. Abstaining more than necessary is its own fault."""
    from palimpsest.api.app import _band

    verdict = aggregate([RUNON, ordinary(1, prob=0.99, words=30)], threshold=0.5)
    assert verdict.n_reliable_sentences == 1
    assert _band(verdict.any_machine_probability)["band"] in {
        "likely_machine", "insufficient_evidence", "no_evidence",
    }
