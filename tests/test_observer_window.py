"""The remote observer reads 6,000 characters, and the reader must be told when that bit.

A verdict on the first 6,000 characters of a 9,000-character essay is a verdict on a
different document than the one on the screen, and the API accepts up to 40,000. The fact
was being computed and then lost three times over: the client pre-truncated the text and
then asked the SERVER whether truncation had happened (it could not know), the adapter to
``TokenScores`` dropped the flag, and ``_meta`` never published it. Each link is tested
here, because any one of them silently restores the original silence.
"""

from __future__ import annotations

import numpy as np
import pytest

from palimpsest.scorer.remote_lm import MAX_CHARS, RemoteLMScorer, RemoteObserverScorer


class _FakeRemote(RemoteLMScorer):
    """A scorer that never touches the network but keeps the real clipping logic."""

    def __init__(self):
        self.model = "stub"
        self.top_k = 0
        self.n_misaligned = 0
        self.sent: str | None = None

    def _check_alignment(self, text, payload, toks):
        return None

    def _fetch(self, text: str) -> dict:
        # Exactly what the real client sends, and exactly what the worker can answer: the
        # worker compares what it RECEIVED against its own 6,000 limit.
        self.sent = text[:MAX_CHARS]
        return {
            "tokens": [{"token": "x", "start": 0, "end": 1, "logprob": -1.0, "rank": 1}],
            "model": "stub",
            "clipped": len(self.sent) > MAX_CHARS,
        }


def test_the_client_reports_clipping_the_server_cannot_see():
    """The bug: the flag was structurally incapable of being true on this path."""
    scorer = _FakeRemote()
    long_essay = "a " * (MAX_CHARS)  # twice the window

    scores = scorer.score(long_essay)

    assert len(long_essay) > MAX_CHARS
    assert len(scorer.sent) == MAX_CHARS, "the client truncates before sending"
    assert scores.clipped is True, (
        "the client cut the essay to 6,000 characters and then asked the server whether "
        "anything had been cut -- a question the server answers by comparing what it "
        "received against the same 6,000, which this client has just guaranteed. The flag "
        "could never be true, so an essay scored on its opening reported no truncation"
    )


def test_a_short_essay_is_not_marked_clipped():
    scorer = _FakeRemote()
    scores = scorer.score("a short essay.")
    assert scores.clipped is False


def test_the_adapter_carries_clipping_into_the_type_the_analyzer_reads():
    """`Analyzer` consumes TokenScores; a fact that stops at the adapter cannot be shown."""
    adapter = RemoteObserverScorer.__new__(RemoteObserverScorer)
    adapter._remote = _FakeRemote()
    adapter.model_name = "stub"
    adapter.device = "remote"
    adapter.max_length = MAX_CHARS

    scores = adapter.score("a " * MAX_CHARS)

    assert scores.clipped is True, (
        "RemoteScores.clipped was computed and then dropped on the floor by the adapter"
    )


def test_meta_publishes_the_window_and_whether_it_was_hit():
    """The interface cannot say what it is not told."""
    from palimpsest.analyze import Analyzer
    from palimpsest.detect.classifier import SentenceDetector
    from palimpsest.scorer.local_lm import TokenScores

    adapter = RemoteObserverScorer.__new__(RemoteObserverScorer)
    adapter._remote = _FakeRemote()
    adapter.model_name = "stub"
    adapter.device = "remote"
    adapter.max_length = MAX_CHARS

    analyzer = Analyzer.__new__(Analyzer)
    analyzer.scorer = adapter
    analyzer.reference = None
    analyzer.document_model = None
    analyzer.detector = SentenceDetector(feature_names=("n_words",))
    analyzer.detector.metadata = {}

    z_i = np.array([0], dtype=np.int32)
    z_f = np.zeros(1, dtype=np.float32)
    clipped_scores = TokenScores(
        tokens=["x"], char_start=z_i, char_end=z_i, logprob=z_f, rank=z_i,
        entropy=z_f, mu=z_f, sigma2=z_f, model_name="stub", device="remote", clipped=True,
    )

    meta = analyzer._meta(0.1, clipped_scores)

    assert meta["clipped"] is True, "the response cannot tell the page a prefix was scored"
    assert meta["observerCharLimit"] == MAX_CHARS, (
        "the window must be published, or the interface has to hard-code 6,000 and drift "
        "from whatever the observer really used"
    )


def test_the_local_observer_is_never_marked_clipped():
    """GPT-2 slides a window over the whole document; nothing is cut."""
    from palimpsest.scorer.local_lm import TokenScores

    z_i = np.zeros(0, dtype=np.int32)
    z_f = np.zeros(0, dtype=np.float32)
    empty = TokenScores(
        tokens=[], char_start=z_i, char_end=z_i, logprob=z_f, rank=z_i,
        entropy=z_f, mu=z_f, sigma2=z_f, model_name="gpt2", device="cpu",
    )
    assert empty.clipped is False
    assert empty.select(0, 10).clipped is False, "select() must carry the flag through"


@pytest.mark.parametrize("limit", [MAX_CHARS])
def test_the_python_and_worker_limits_are_the_same_number(limit):
    """Two constants describing one window. If they drift, the client cuts at the wrong
    place and the disclosure names a boundary that is not where the text stopped."""
    worker = (
        __import__("pathlib").Path("edge/src/observer.js").read_text(encoding="utf-8")
    )
    assert f"MAX_OBSERVER_CHARS = {limit}" in worker
