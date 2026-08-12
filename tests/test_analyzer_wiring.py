"""The library's own entry point must produce the numbers the reports were measured with.

This file exists because of a bug that survived every other check in the project.

``Analyzer.analyze`` built its document verdict with ``aggregate(verdicts)``. Both of that
call's interesting arguments have defaults, and both defaults are wrong for serving:

* ``doc_model=None`` makes ``aggregate`` fall back to ``DocumentDetector()``, whose
  ``predict`` returns ``stats["max_p"]`` -- the single strongest sentence in the essay.
* ``threshold=FLAG_THRESHOLD`` is the 0.65 constant used to draw highlight passages, not the
  0.3004 cut-off the document model's ``share`` input was standardised at during fitting.

So the field named ``any_machine_probability`` stopped being a document probability and
became a sentence maximum. Nothing caught it: the value is bounded in [0, 1], it rises when
the essay looks more machine-written, and it is fed to the band thresholds without complaint.
On one real essay it read 70.9% where the fitted model reads 14.9% -- opposite sides of the
"likely machine" line.

It survived because every caller that reported a number rebuilt the verdict itself rather
than trusting the library: the FastAPI route did, and so did the parity exporter, which is
why the JavaScript port showed *exact* agreement with a Python path that no longer existed.
A reference implementation that each caller quietly repairs on the way past cannot be
checked by comparing against it.

The tests below therefore assert the wiring end to end, through ``analyze()``, with only the
observer stubbed out.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from palimpsest.analyze import Analyzer
from palimpsest.detect.document import FLAG_THRESHOLD, document_statistics
from palimpsest.scorer.local_lm import TokenScores

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
SUFFIX = "_remote"

#: Long enough that sentences clear the reliability floor and the document model sees a real
#: spread of sentence scores rather than one span.
ESSAY = " ".join(
    [
        "The doors of the metro do not merely close; they execute a distinct sequence.",
        "I do not ignore it, I measure the latency between the chime and the seal.",
        "That obsession with thresholds is what pushed me toward mechanical engineering.",
        "Last summer I took apart a counterfeit bottle cap on my kitchen table.",
        "The knurling was wrong, and the wrongness was measurable rather than aesthetic.",
        "My cousin and I spent four months building a cap that could not be refilled.",
        "Pitching it to buyers taught me that an elegant design nobody can source is useless.",
        "Negotiating with suppliers rearranged how I think about what counts as a solution.",
        "I want to spend my life closing the distance between a drawing and a delivered part.",
    ]
)


class _StubScorer:
    """A deterministic observer stand-in: no weights, no network, real character offsets.

    The point of these tests is the wiring between the detector, the document model and the
    verdict. Which observer produced the logprobs is irrelevant to that, and stubbing it
    keeps the test hermetic.
    """

    model_name = "stub"
    device = "cpu"

    def score(self, text: str) -> TokenScores:
        starts, ends, toks = [], [], []
        for m in re.finditer(r"\S+", text):
            toks.append(m.group())
            starts.append(m.start())
            ends.append(m.end())
        n = len(toks)
        rng = np.random.default_rng(11)
        return TokenScores(
            tokens=toks,
            char_start=np.array(starts, dtype=np.int32),
            char_end=np.array(ends, dtype=np.int32),
            # A spread wide enough that sentences do not all score identically; the actual
            # values carry no meaning and no assertion depends on them.
            logprob=rng.normal(-2.5, 1.5, n).astype(np.float32),
            rank=rng.integers(1, 60, n).astype(np.int32),
            entropy=np.full(n, np.nan, dtype=np.float32),
            mu=np.full(n, np.nan, dtype=np.float32),
            sigma2=np.full(n, np.nan, dtype=np.float32),
            model_name="stub",
            device="cpu",
        )


def _analyzer(**kw) -> Analyzer:
    a = Analyzer.from_artifacts(ARTIFACTS, observer="remote", suffix=SUFFIX)
    a.scorer = _StubScorer()
    for k, v in kw.items():
        setattr(a, k, v)
    return a


needs_artifacts = pytest.mark.skipif(
    not (ARTIFACTS / f"document_detector{SUFFIX}.json").exists()
    or not (ARTIFACTS / f"detector{SUFFIX}.json").exists(),
    reason="fitted artifacts not present",
)


@needs_artifacts
def test_from_artifacts_loads_a_fitted_document_model():
    """The load must happen once, in construction -- not be left to each caller."""
    a = Analyzer.from_artifacts(ARTIFACTS, observer="remote", suffix=SUFFIX)
    assert a.document_model is not None, "Analyzer did not load document_detector*.json"
    assert a.document_model.coef is not None, (
        "document model loaded but unfitted; aggregate would silently return max_p"
    )


@needs_artifacts
def test_verdict_comes_from_the_document_model_not_the_strongest_sentence():
    a = _analyzer()
    result = a.analyze(ESSAY)
    reliable = [s for s in result.sentences if s.reliable]
    assert len(reliable) >= 5, "stub produced too few scorable sentences to test anything"

    probs = np.array([s.probability for s in reliable], dtype=np.float64)
    words = np.array([s.n_words for s in reliable], dtype=np.float64)
    stats = document_statistics(probs, words, a.detector.flag_threshold)

    assert result.verdict.any_machine_probability == pytest.approx(
        a.document_model.predict(stats)
    ), "the verdict is not the fitted document model's output"

    # The exact failure mode: the unfitted fallback returns max_p under the same field name.
    assert result.verdict.any_machine_probability != pytest.approx(float(probs.max())), (
        "any_machine_probability equals the strongest sentence -- this is the unfitted "
        "DocumentDetector fallback, not a document score"
    )


@needs_artifacts
def test_share_is_computed_at_the_threshold_the_document_model_was_fitted_at():
    """``share`` is a standardised input, so the cut-off is part of the model, not a taste.

    ``document_detector*.json`` records it as ``sentenceThreshold``. Computing it at the 0.65
    highlighting constant instead feeds the fit a column it never saw during training.
    """
    a = _analyzer()
    result = a.analyze(ESSAY)
    reliable = [s for s in result.sentences if s.reliable]
    probs = np.array([s.probability for s in reliable], dtype=np.float64)
    words = np.array([s.n_words for s in reliable], dtype=np.float64)

    fitted_at = a.document_model.metadata.get("sentenceThreshold")
    assert fitted_at == pytest.approx(a.detector.flag_threshold), (
        "the detector's flag threshold no longer matches the one the document model was "
        "fitted at; retrain both together"
    )

    at_fitted = document_statistics(probs, words, a.detector.flag_threshold)["share"]
    assert result.verdict.machine_share == pytest.approx(at_fitted)

    at_highlight = document_statistics(probs, words, FLAG_THRESHOLD)["share"]
    if at_highlight != pytest.approx(at_fitted):
        assert result.verdict.machine_share != pytest.approx(at_highlight), (
            "machine_share was computed at the 0.65 passage-highlighting constant"
        )


@needs_artifacts
def test_an_unfitted_document_model_would_change_the_answer():
    """Guards the guard: proves the assertions above are capable of failing.

    If the fallback happened to agree with the fitted model on this essay, the tests would
    pass while asserting nothing. This shows the two paths genuinely disagree here.
    """
    fitted = _analyzer().analyze(ESSAY).verdict.any_machine_probability
    fallback = _analyzer(document_model=None).analyze(ESSAY).verdict.any_machine_probability
    assert fitted != pytest.approx(fallback), (
        "fitted and unfitted document models agree on this essay, so these tests cannot "
        "detect the regression they exist for -- change ESSAY"
    )
