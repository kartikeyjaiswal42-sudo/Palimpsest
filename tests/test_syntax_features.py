"""Structural features: the properties that must hold for the block to mean anything.

These are contract tests, not accuracy tests. Whether the features *detect* anything is
measured in ``scripts/syntax_probe.py`` and recorded in ``artifacts/syntax_probe.json``; a
unit test that asserted an AUROC would be asserting a property of the corpus.

What is tested here is the set of things that, if they broke, would make every number in
that artifact quietly wrong: NaN discipline, the leave-one-out construction, the z-score
cap, and the reference model's arithmetic.
"""

from __future__ import annotations

import math

import pytest

from palimpsest.features.syntax import (
    ALL_SYNTAX_FEATURE_NAMES,
    SYNTAX_CONTEXT_FEATURE_NAMES,
    SYNTAX_FEATURE_NAMES,
    PosTrigramReference,
    extract_syntax_context_features,
    extract_syntax_features,
    get_parser,
    pos_tags,
)

needs_parser = pytest.mark.skipif(
    get_parser() is None, reason="spaCy en_core_web_sm not installed"
)

LONG = (
    "Although the rain fell hard that evening, I walked home slowly, thinking about "
    "what she had said to me in the kitchen."
)


def test_every_declared_feature_is_returned():
    """A feature named in the tuple but missing from the dict would be imputed to the
    training mean forever, silently, and nothing else would notice."""
    out = extract_syntax_features(LONG)
    assert set(out) == set(SYNTAX_FEATURE_NAMES)


def test_names_are_unique_and_disjoint():
    assert len(ALL_SYNTAX_FEATURE_NAMES) == len(set(ALL_SYNTAX_FEATURE_NAMES))
    assert not set(SYNTAX_FEATURE_NAMES) & set(SYNTAX_CONTEXT_FEATURE_NAMES)


@pytest.mark.parametrize("text", ["", "   ", "Yes.", "I ran."])
def test_short_or_empty_spans_are_nan_not_zero(text):
    """Zero is a measurement; NaN is the absence of one. A short span standardises to the
    training mean and contributes exactly nothing, which is the intended behaviour -- a
    zero would be a real value with a real coefficient and a vote it has not earned."""
    out = extract_syntax_features(text)
    assert set(out) == set(SYNTAX_FEATURE_NAMES)
    assert all(math.isnan(v) for v in out.values())


@needs_parser
def test_measured_values_are_in_range():
    out = extract_syntax_features(LONG)
    assert out["tree_depth_max"] >= out["tree_depth_mean"] > 0
    assert 0.0 <= out["stopword_ratio"] <= 1.0
    assert out["tree_depth_sd"] >= 0.0
    assert out["pos_trigram_entropy"] >= 0.0


@needs_parser
def test_content_function_ratio_guards_division_by_zero():
    """An all-content span must arrive as NaN, never as inf: inf standardises to garbage
    rather than to the training mean, so it would poison the whole feature column."""
    out = extract_syntax_features("Dogs chase cats endlessly.")
    v = out["content_function_ratio"]
    assert math.isnan(v) or math.isfinite(v)


# ---------------------------------------------------------------------------------------
# Document-relative features
# ---------------------------------------------------------------------------------------


def _rows(depths):
    return [{"tree_depth_max": d, "pos_trigram_surprisal": d} for d in depths]


def test_context_features_need_a_baseline():
    """Below MIN_SENTENCES there is no stable baseline, so the answer is 'unmeasurable'."""
    base = _rows([3.0, 4.0])
    out = extract_syntax_context_features(base, len(base))
    assert len(out) == 2
    assert all(math.isnan(v) for row in out for v in row.values())


def test_z_is_leave_one_out():
    """A sentence must never be compared against a baseline it is part of: including
    itself shrinks its own deviation, which is precisely how a single inserted machine
    paragraph would hide."""
    base = _rows([3.0, 3.0, 3.0, 3.0, 3.0, 12.0])
    out = extract_syntax_context_features(base, len(base))
    # The outlier is measured against the five identical neighbours, so it must be extreme.
    assert out[5]["tree_depth_z_in_doc"] > 0
    # The identical ones sit exactly on their own baseline.
    assert out[0]["tree_depth_z_in_doc"] == pytest.approx(0.0, abs=1e-9)


def test_degenerate_baseline_reports_maximally_unusual_not_typical():
    """When every other sentence is identical the spread is zero. A sentence differing
    from a perfectly uniform baseline is maximally unusual, NOT typical -- returning 0.0
    here would report the exact opposite of what the data says. Mirrors the same guard in
    context.py, which is there because an earlier version got it backwards."""
    base = _rows([5.0, 5.0, 5.0, 5.0, 5.0, 9.0])
    out = extract_syntax_context_features(base, len(base))
    assert out[5]["tree_depth_z_in_doc"] == pytest.approx(6.0)


def test_z_is_capped_both_directions():
    """Unbounded z-scores let one feature dominate the logit. artifacts show
    style_gap_from_doc reaching 64.7 for want of this cap; the block must not repeat it.

    The baseline here deliberately has REAL SPREAD. An all-identical baseline sends the
    calculation down the degenerate branch, which returns +/-6 by a different route -- so a
    version of this test built on identical values passes even with the clip deleted, and
    the first draft of it did exactly that.
    """
    spread = [1.0, 2.0, 3.0, 4.0, 5.0]
    hi = extract_syntax_context_features(_rows([*spread, 1e9]), 6)
    lo = extract_syntax_context_features(_rows([*spread, -1e9]), 6)
    assert hi[5]["tree_depth_z_in_doc"] == pytest.approx(6.0)
    assert lo[5]["tree_depth_z_in_doc"] == pytest.approx(-6.0)
    # And the cap must not be flattening ordinary values on the way past.
    mid = extract_syntax_context_features(_rows([*spread, 3.0]), 6)
    assert abs(mid[5]["tree_depth_z_in_doc"]) < 6.0


def test_burstiness_of_a_flat_window_is_zero():
    """Identical neighbouring depths = no structural variation. This is the 'AI holds one
    rhythm' signal, so its zero point has to be right."""
    out = extract_syntax_context_features(_rows([4.0] * 6), 6)
    assert out[2]["local_depth_burstiness"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------------------
# The POS trigram reference
# ---------------------------------------------------------------------------------------


def test_reference_surprisal_is_lower_for_what_it_was_fitted_on():
    """The whole point of a reference model: shapes it has seen must be less surprising
    than shapes it has not."""
    seen = [["PRON", "VERB", "NOUN"]] * 40
    ref = PosTrigramReference.fit(seen)
    assert ref.surprisal(["PRON", "VERB", "NOUN"]) < ref.surprisal(["ADP", "ADP", "ADP"])


def test_reference_surprisal_is_non_negative_and_finite():
    ref = PosTrigramReference.fit([["PRON", "VERB", "NOUN"], ["DET", "NOUN", "VERB"]])
    for tags in (["PRON", "VERB", "NOUN"], ["X", "X", "X"], ["NOUN"]):
        s = ref.surprisal(tags)
        assert math.isfinite(s) and s >= 0.0


def test_reference_empty_sequence_is_nan():
    ref = PosTrigramReference.fit([["PRON", "VERB"]])
    assert math.isnan(ref.surprisal([]))


def test_reference_round_trips_through_json():
    """The reference is written to disk and reused by every later set; a lossy round trip
    would silently change every pos_trigram_surprisal value built after the first run."""
    ref = PosTrigramReference.fit([["PRON", "VERB", "NOUN"], ["DET", "NOUN", "VERB"]])
    back = PosTrigramReference.from_dict(ref.to_dict())
    tags = ["PRON", "VERB", "NOUN"]
    assert back.surprisal(tags) == pytest.approx(ref.surprisal(tags))
    assert back.total == ref.total


@needs_parser
def test_pos_tags_drop_punctuation():
    assert "PUNCT" not in pos_tags("Hello, world! This is a sentence.")
