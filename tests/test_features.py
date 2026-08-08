"""Feature extraction: registry consistency, NaN discipline, and length invariance."""

from __future__ import annotations

import numpy as np
import pytest

from palimpsest.features import (
    CONTEXT_FEATURE_NAMES,
    CORPUS_FEATURE_NAMES,
    FEATURE_NAMES,
    FEATURES_BY_NAME,
    MODEL_FEATURE_NAMES,
    SURFACE_FEATURE_NAMES,
    extract_context_features,
    extract_corpus_features,
    extract_surface_features,
)
from palimpsest.features.model_based import extract_model_features
from palimpsest.scorer.local_lm import TokenScores


def fake_scores(n: int, seed: int = 0) -> TokenScores:
    rng = np.random.default_rng(seed)
    return TokenScores(
        tokens=["x"] * n,
        char_start=np.arange(n, dtype=np.int32) * 2,
        char_end=np.arange(n, dtype=np.int32) * 2 + 1,
        logprob=rng.normal(-3, 1, n).astype(np.float32),
        rank=rng.integers(1, 500, n).astype(np.int32),
        entropy=np.abs(rng.normal(3, 1, n)).astype(np.float32),
        mu=rng.normal(-3.2, 0.5, n).astype(np.float32),
        sigma2=np.abs(rng.normal(2, 0.3, n)).astype(np.float32),
        model_name="test",
        device="cpu",
    )


def test_registry_matches_extractors_exactly():
    produced = (
        set(MODEL_FEATURE_NAMES)
        | set(SURFACE_FEATURE_NAMES)
        | set(CORPUS_FEATURE_NAMES)
        | set(CONTEXT_FEATURE_NAMES)
    )
    assert produced == set(FEATURE_NAMES)


def test_every_feature_has_documentation():
    for name in FEATURE_NAMES:
        f = FEATURES_BY_NAME[name]
        assert f.label and f.description
        assert f.expected_direction in (-1, 0, 1)
        assert f.group


def test_short_spans_return_nan_not_a_confident_number():
    """Four tokens cannot support a surprisal variance. Saying so beats inventing one."""
    feats = extract_model_features(fake_scores(3))
    assert all(np.isnan(v) for v in feats.values())
    surface = extract_surface_features("Two words")
    assert all(np.isnan(v) for v in surface.values())


def test_model_features_are_finite_for_normal_input():
    feats = extract_model_features(fake_scores(40))
    for name, value in feats.items():
        assert np.isfinite(value), f"{name} was not finite"


def test_curvature_is_length_invariant():
    """The paper's form grows with length; ours must not, or it becomes a length feature."""
    short = extract_model_features(fake_scores(20, seed=1))["curvature"]
    long = extract_model_features(fake_scores(400, seed=1))["curvature"]
    assert abs(short - long) < 1.0


def test_surface_rates_are_per_hundred_words_not_counts():
    once = extract_surface_features("I think therefore I am, and moreover I write. " * 1)
    twice = extract_surface_features("I think therefore I am, and moreover I write. " * 6)
    assert abs(once["discourse_marker_rate"] - twice["discourse_marker_rate"]) < 2.0


def test_tricolon_and_antithesis_detection():
    assert extract_surface_features("It taught me patience, humility, and resolve today.")["tricolon"] == 1.0
    assert extract_surface_features("The lab was not just a room but a refuge for me.")["antithesis"] == 1.0
    assert extract_surface_features("I walked home slowly in the rain after school.")["tricolon"] == 0.0


def test_context_features_need_enough_sentences():
    base = [{"mean_logprob": -3.0, "n_words": 12.0} for _ in range(3)]
    out = extract_context_features(base, 3)
    assert all(np.isnan(v) for v in out[0].values())


def test_context_features_are_leave_one_out():
    """An inserted outlier must not drag the baseline it is measured against."""
    base = [{k: v for k, v in (("mean_logprob", -4.0), ("n_words", 12.0))} for _ in range(9)]
    base.append({"mean_logprob": -1.0, "n_words": 12.0})  # the smooth insertion
    out = extract_context_features(base, len(base))
    assert out[-1]["logprob_z_in_doc"] > 2.0
    assert abs(out[0]["logprob_z_in_doc"]) < 1.0


def test_corpus_features_degrade_to_nan_without_a_reference():
    out = extract_corpus_features("Some text here for testing.", None, -3.0)
    assert set(out) == set(CORPUS_FEATURE_NAMES)
    assert all(np.isnan(v) for v in out.values())
