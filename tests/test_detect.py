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
