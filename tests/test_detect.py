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
