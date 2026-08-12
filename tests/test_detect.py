"""Classifier and document aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from palimpsest.detect import SentenceDetector, aggregate, find_passages, smooth_probabilities
from palimpsest.detect.classifier import choose_threshold
from palimpsest.detect.document import (
    DOC_FEATURES,
    DocumentDetector,
    SentenceVerdict,
    document_statistics,
)
from palimpsest.features.registry import FEATURE_NAMES


def toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    feats, labels, groups = [], [], []
    for i in range(n):
        machine = i % 4 == 0
        row = {name: float(rng.normal(1.0 if machine else 0.0, 1.0)) for name in FEATURE_NAMES}
        feats.append(row)
        labels.append(int(machine))
        groups.append(f"doc{i // 8}")
    return feats, np.array(labels), np.array(groups)


def test_fit_predict_and_contributions_sum_to_the_logit():
    """The explanation must BE the computation, not a story about it."""
    feats, y, g = toy()
    det = SentenceDetector().fit(feats, y, g)
    pred = det.predict(feats[0])
    total = sum(c.contribution for c in pred.contributions) + det.intercept
    assert abs(total - pred.logit) < 1e-9


def test_displayed_evidence_plus_remainder_reconstructs_the_logit():
    """What the interface SHOWS must add up to the verdict it shows next to it.

    The panel displays the six largest terms out of forty-three. That is a useful summary
    and a bad explanation on its own: the omitted terms are individually small but there are
    thirty-seven of them, and measured across the training corpus they outweigh the six
    shown on **30.9% of sentences** (median |remainder| 0.89, max 6.35). The intercept is
    larger still at -2.26 -- most sentences are human and the model knows it -- so a panel
    showing only feature bars omits the single biggest term in the sum.

    So the panel prints the intercept and the summed remainder too, and this asserts that
    those numbers really do reconstruct the logit rather than approximately gesturing at it.
    """
    feats, y, g = toy()
    det = SentenceDetector().fit(feats, y, g)
    for row in feats[:25]:
        pred = det.predict(row)
        shown = sum(c.contribution for c in pred.top(6))
        rebuilt = pred.intercept + shown + pred.remainder(6)
        assert abs(rebuilt - pred.logit) < 1e-9, (
            "intercept + shown + remainder does not equal the logit; the evidence panel "
            "would be printing arithmetic that does not close"
        )


def test_remainder_covers_every_feature_not_displayed():
    """No term may be counted twice or dropped between the bars and the remainder."""
    feats, y, g = toy()
    det = SentenceDetector().fit(feats, y, g)
    pred = det.predict(feats[0])
    for k in (0, 1, 6, len(pred.contributions)):
        shown = pred.top(k)
        assert len(shown) == min(k, len(pred.contributions))
        total = sum(c.contribution for c in shown) + pred.remainder(k)
        assert abs(total - sum(c.contribution for c in pred.contributions)) < 1e-9


def test_unmeasured_features_contribute_exactly_zero():
    """A NaN feature must not silently vote."""
    feats, y, g = toy()
    det = SentenceDetector().fit(feats, y, g)
    row = dict(feats[0])
    row["curvature"] = float("nan")
    pred = det.predict(row)
    c = next(c for c in pred.contributions if c.name == "curvature")
    assert not c.measured
    assert c.contribution == 0.0


def test_round_trip_through_json(tmp_path):
    feats, y, g = toy()
    det = SentenceDetector().fit(feats, y, g)
    det.flag_threshold = 0.42
    path = tmp_path / "d.json"
    det.save(path)
    loaded = SentenceDetector.load(path)
    assert loaded.flag_threshold == 0.42
    assert np.allclose(loaded.coef, det.coef)
    assert abs(loaded.predict(feats[3]).probability - det.predict(feats[3]).probability) < 1e-9


def test_unfitted_detector_refuses_to_predict():
    with pytest.raises(RuntimeError):
        SentenceDetector().predict({})


def test_choose_threshold_prefers_precision():
    y = np.array([0] * 90 + [1] * 10)
    p = np.concatenate([np.linspace(0, 0.6, 90), np.linspace(0.5, 0.99, 10)])
    t, precision, recall = choose_threshold(y, p, target_precision=0.8)
    assert 0.0 < t <= 1.0
    assert 0.0 <= recall <= 1.0


def test_smoothing_pulls_an_isolated_spike_down():
    probs = np.array([0.02, 0.02, 0.95, 0.02, 0.02])
    weights = np.full(5, 20.0)
    out = smooth_probabilities(probs, weights)
    assert out[2] < probs[2]
    assert out[2] > probs[1]


def test_passages_are_maximal_contiguous_runs():
    sentences = [
        SentenceVerdict(i, i * 10, i * 10 + 9, "x", p, 20, p, True)
        for i, p in enumerate([0.1, 0.9, 0.9, 0.1, 0.8])
    ]
    passages = find_passages(sentences, threshold=0.5)
    assert [p.sentence_indices for p in passages] == [[1, 2], [4]]


def test_document_model_never_sees_length():
    assert "log_sentences" not in DOC_FEATURES
    assert "n_sentences" not in DOC_FEATURES


def test_document_statistics_separate_share_from_peak():
    """One machine sentence among many: low share, high peak. That distinction is the point."""
    probs = np.array([0.01] * 19 + [0.95])
    words = np.full(20, 20.0)
    stats = document_statistics(probs, words, 0.5)
    assert stats["share"] < 0.1
    assert stats["max_p"] > 0.9


def test_aggregate_reports_an_interval():
    sentences = [
        SentenceVerdict(i, i * 10, i * 10 + 9, "x", 0.9 if i < 5 else 0.05, 20, 0.9 if i < 5 else 0.05, True)
        for i in range(10)
    ]
    v = aggregate(sentences, threshold=0.5)
    assert 0.0 <= v.machine_share_low <= v.machine_share <= v.machine_share_high <= 1.0
    assert v.n_sentences == 10


def test_aggregate_handles_empty_input():
    v = aggregate([])
    assert v.machine_share == 0.0 and v.n_sentences == 0


def test_unfitted_document_model_falls_back_to_peak():
    stats = document_statistics(np.array([0.1, 0.8]), np.array([10.0, 10.0]), 0.5)
    assert DocumentDetector().predict(stats) == pytest.approx(0.8)


# ---------------------------------------------------------------------------------------
# Unmeasurable spans must not decide a verdict.
#
# These exist because the tool accused three real students. ELLIPSE and PERSUADE contain
# essays written with no sentence-ending punctuation at all, so the segmenter returns the
# whole essay as one 312-, 313- or 466-word "sentence". Every per-sentence feature is then
# computed on a document: `n_words` landed 50 standard deviations above the training mean and
# all three were flagged as machine at P > 0.90. `reliable` was already being computed and
# reported at the time -- and then ignored by the aggregator, which made it decoration.


def test_an_unmeasurable_span_is_excluded_from_the_verdict():
    """A long run-on span must not drag the document verdict with it."""
    ordinary = [
        SentenceVerdict(i, i * 10, i * 10 + 9, "x", 0.02, 20, 0.02, True) for i in range(5)
    ]
    runon = SentenceVerdict(5, 60, 600, "x " * 466, 0.99, 466, 0.99, False)

    without = aggregate(ordinary, threshold=0.5)
    with_runon = aggregate([*ordinary, runon], threshold=0.5)

    assert with_runon.machine_share == pytest.approx(without.machine_share), (
        "an unreliable span changed machine_share; it must be excluded from both the "
        "numerator and the denominator"
    )
    assert with_runon.n_sentences == 6, "the span should still be counted and shown"
    assert with_runon.n_reliable_sentences == 5


def test_a_document_with_nothing_measurable_is_not_accused():
    """The three ESL run-on essays: one span, unreliable, nothing else to go on."""
    only_runon = [SentenceVerdict(0, 0, 600, "x " * 466, 0.99, 466, 0.99, False)]
    v = aggregate(only_runon, threshold=0.5)
    assert v.machine_share == 0.0
    assert v.any_machine_probability == 0.0, (
        "a document whose only span is unmeasurable was scored anyway -- this is the exact "
        "path that flagged three second-language students at P > 0.90"
    )
    assert v.n_reliable_sentences == 0


def test_standardised_features_are_clipped():
    """No single feature may win by arithmetic on an out-of-distribution value."""
    from palimpsest.detect.classifier import Z_CLIP

    feats, labels, groups = toy()
    det = SentenceDetector(feature_names=tuple(FEATURE_NAMES)).fit(feats, labels, groups)
    absurd = {name: 1e9 for name in FEATURE_NAMES}
    z = det.to_matrix([absurd])[0]
    assert np.all(np.abs(z) <= Z_CLIP + 1e-9), (
        f"a feature standardised past +/-{Z_CLIP}; a 466-word 'sentence' reached z=+50 and "
        "its 0.148 weight became the largest term in the model"
    )
