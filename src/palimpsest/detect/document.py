"""From per-sentence probabilities to a document verdict and the passages behind it.

Two numbers, not one
--------------------
"73% AI" is the thing the brief singles out as useless, and the reason it is useless is
that it silently fuses two different questions. This module keeps them apart:

``machine_share``
    How much of this essay reads as machine-written, as a fraction of its words. This is a
    quantity, and it comes with a bootstrap interval because it is estimated from a few
    dozen noisy sentence scores.
``any_machine_probability``
    How confident we are that *any* of it is machine-written. This is a probability, and it
    is what a reader actually wants when the answer is "one paragraph".

An essay with one polished paragraph out of six has a low share and a high probability.
Collapsing those into a single percentage destroys exactly the information a reader needs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "DocumentDetector",
    "DOC_FEATURES",
    "document_statistics",
    "SentenceVerdict",
    "Passage",
    "DocumentVerdict",
    "aggregate",
    "find_passages",
    "FLAG_THRESHOLD",
]

#: Sentence probability at or above which we highlight. Chosen in scripts/train.py as the
#: point where precision on the held-out set reaches 0.90 -- we would rather miss machine
#: text than accuse a student. docs/03-evaluation.md shows the full trade-off curve.
FLAG_THRESHOLD = 0.65

#: A "sentence" longer than this many words is not scored into the document verdict.
#:
#: Defined here, in words, so the serving path and the evaluation scripts share one number.
#: The features are per-sentence; applied to a 466-word span every one of them is out of the
#: distribution it was fitted on. `n_words` lands 50 standard deviations high and `root_ttr`
#: -- distinct words over the square root of the count -- is inflated by construction.
#:
#: This is not hypothetical. Three ELLIPSE and PERSUADE essays by second-language writers
#: contain no sentence-ending punctuation at all, so each segments to a single 312-, 313- or
#: 466-word "sentence", and all three were flagged as machine at P > 0.90. They are the least
#: fluent writing in the corpus, by the students least able to contest an accusation.
#:
#: 90 words is far above any real sentence (the 99.9th percentile in the corpus is near 60)
#: and far below a run-on essay, so ordinary prose is untouched.
MAX_SENTENCE_WORDS = 90

#: A run of flagged sentences shorter than this is reported as a weaker "isolated sentence"
#: signal rather than a passage. Single-sentence flags are the noisiest thing we produce.
MIN_PASSAGE_WORDS = 25


@dataclass(frozen=True, slots=True)
class SentenceVerdict:
    index: int
    start: int
    end: int
    text: str
    probability: float
    n_words: int
    #: Probability after smoothing with the neighbours. Used for passage detection only.
    smoothed: float
    #: False when the sentence was too short to measure properly.
    reliable: bool


@dataclass(frozen=True, slots=True)
class Passage:
    start: int
    end: int
    sentence_indices: list[int]
    n_words: int
    mean_probability: float
    peak_probability: float


@dataclass(frozen=True, slots=True)
class DocumentVerdict:
    machine_share: float
    machine_share_low: float
    machine_share_high: float
    any_machine_probability: float
    n_sentences: int
    n_words: int
    n_reliable_sentences: int


def smooth_probabilities(probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Length-weighted moving average over a +/-1 sentence window.

    A single sentence is a small sample and its probability is correspondingly jumpy.
    Smoothing before thresholding is what stops the interface lighting up one clause in the
    middle of an obviously human paragraph. The unsmoothed value is still what the evidence
    panel reports, because that is the number the features actually explain.
    """
    n = len(probs)
    if n == 0:
        return probs
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo, hi = max(0, i - 1), min(n, i + 2)
        w = weights[lo:hi].astype(np.float64).copy()
        # The sentence itself counts double; neighbours inform, they do not outvote.
        w[i - lo] *= 2.0
        total = w.sum()
        out[i] = float((probs[lo:hi] * w).sum() / total) if total > 0 else probs[i]
    return out


def find_passages(
    sentences: list[SentenceVerdict], threshold: float = FLAG_THRESHOLD
) -> list[Passage]:
    """Maximal contiguous runs of sentences whose smoothed probability clears ``threshold``."""
    passages: list[Passage] = []
    run: list[SentenceVerdict] = []

    def flush() -> None:
        if not run:
            return
        words = sum(s.n_words for s in run)
        passages.append(
            Passage(
                start=run[0].start,
                end=run[-1].end,
                sentence_indices=[s.index for s in run],
                n_words=words,
                mean_probability=float(
                    np.average([s.probability for s in run], weights=[s.n_words for s in run])
                ),
                peak_probability=max(s.probability for s in run),
            )
        )

    for s in sentences:
        if s.smoothed >= threshold:
            run.append(s)
        else:
            flush()
            run = []
    flush()
    return passages


def aggregate(
    sentences: list[SentenceVerdict],
    threshold: float = FLAG_THRESHOLD,
    doc_model: "DocumentDetector | None" = None,
    rng_seed: int = 0,
) -> DocumentVerdict:
    """Combine sentence verdicts into the two document-level numbers."""
    if not sentences:
        return DocumentVerdict(0.0, 0.0, 0.0, 0.0, 0, 0, 0)

    # A sentence the tool has said it cannot measure must not decide the answer. `reliable`
    # used to be reported and then ignored here, which made it decoration: the three run-on
    # ESL essays above were each marked unreliable AND flagged as machine at P > 0.90 in the
    # same response. Unreliable spans are dropped from both the numerator and the denominator
    # -- counting them as human would be just as much of a claim as counting them as machine.
    scored = [s for s in sentences if s.reliable] or []
    if not scored:
        # Nothing measurable. Say so with a neutral verdict rather than a confident one
        # computed from spans we have just declared out of distribution.
        return DocumentVerdict(0.0, 0.0, 0.0, 0.0, len(sentences),
                               int(sum(s.n_words for s in sentences)), 0)

    probs = np.array([s.probability for s in scored], dtype=np.float64)
    words = np.array([s.n_words for s in scored], dtype=np.float64)
    total_words = float(words.sum())

    flagged = probs >= threshold
    share = float(words[flagged].sum() / total_words) if total_words > 0 else 0.0
    low, high = _bootstrap_share(probs, words, rng_seed, threshold)
    stats = document_statistics(probs, words, threshold)

    return DocumentVerdict(
        machine_share=share,
        machine_share_low=low,
        machine_share_high=high,
        any_machine_probability=(doc_model or DocumentDetector()).predict(stats),
        n_sentences=len(sentences),
        n_words=int(total_words),
        n_reliable_sentences=sum(1 for s in sentences if s.reliable),
    )


def document_statistics(probs: np.ndarray, words: np.ndarray, threshold: float) -> dict:
    """Aggregate sentence scores into the features the document model reads.

    Deliberately small and monotone. Each entry answers a different question: how machine-y
    is the document on average, how machine-y is its worst stretch, and how much of it is
    flagged at all. A one-paragraph insertion moves `max` and `share` while barely touching
    `mean`, which is exactly the case a single average would hide.
    """
    if len(probs) == 0:
        return {"mean_p": 0.0, "max_p": 0.0, "q90_p": 0.0, "share": 0.0, "log_sentences": 0.0}
    total = max(float(words.sum()), 1.0)
    flagged = probs >= threshold
    return {
        "mean_p": float(np.average(probs, weights=np.maximum(words, 1.0))),
        "max_p": float(probs.max()),
        "q90_p": float(np.percentile(probs, 90)),
        "share": float(words[flagged].sum() / total),
        "log_sentences": float(np.log1p(len(probs))),
    }


#: The document model reads ONLY these. `log_sentences` is deliberately absent.
#:
#: It was included at first and the fit gave it a weight of -3.09 -- by some margin the
#: largest document-level weight, and it says "fewer sentences means machine". It says that
#: because the machine essays in our training corpus happen to be shorter than the human ones
#: (median 261 words against 642). That is a fact about how Liang et al. generated their data,
#: not about machine writing, and the essays it misreads are short because they were written
#: under exam conditions, by the group this kind of detector is already known to harm.
#:
#: Removing it is NOT free, and scripts/ablate_length.py re-measures the cost on the current
#: build rather than trusting this comment: in-domain document AUROC 0.998 -> 0.959, and
#: aggregate ESL false positives actually rise (5.6% -> 7.3%). The TOEFL improvement that
#: motivated the removal (33.3% -> 24.4%) does not survive a paired test (McNemar p = 0.29).
#: We removed it on the principle -- a corpus artefact must not be the model's strongest
#: input -- while reporting that the measurements mildly disagree. docs/04-failures.md #6.
DOC_FEATURES: tuple[str, ...] = ("mean_p", "max_p", "q90_p", "share")


@dataclass
class DocumentDetector:
    """A five-feature logistic regression over the sentence-score summary.

    This replaces an earlier hand-derived formula that multiplied per-sentence
    probabilities under an independence assumption. Sentences in one essay are emphatically
    not independent -- same author, same prompt, same afternoon -- and the formula produced
    confident nonsense. Fitting five weights on document-level labels is both more honest
    and, unlike the formula, something we can show a calibration curve for.
    """

    coef: np.ndarray | None = None
    intercept: float = 0.0
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    #: Probability at or above which a DOCUMENT is called machine-written.
    #:
    #: Not 0.5. The two errors are not symmetric: missing machine text makes the tool less
    #: useful, while flagging a student's own essay is an accusation. This is chosen at
    #: training time as the lowest cut-off whose false-positive rate on held-out HUMAN
    #: essays stays inside an explicit budget, and the recall we get at that point is
    #: whatever it is. Reported in docs/03-evaluation.md as an operating point, not a score.
    threshold: float = 0.5
    metadata: dict = field(default_factory=dict)

    def _matrix(self, stats: list[dict], fit: bool = False) -> np.ndarray:
        x = np.array([[s[k] for k in DOC_FEATURES] for s in stats], dtype=np.float64)
        if fit:
            self.mean = x.mean(axis=0)
            sd = x.std(axis=0)
            self.scale = np.where(sd > 1e-9, sd, 1.0)
        return (x - self.mean) / self.scale

    def fit(self, stats: list[dict], labels: np.ndarray) -> "DocumentDetector":
        from sklearn.linear_model import LogisticRegression

        x = self._matrix(stats, fit=True)
        lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        lr.fit(x, np.asarray(labels).astype(int))
        self.coef = lr.coef_[0].astype(np.float64)
        self.intercept = float(lr.intercept_[0])
        return self

    def predict(self, stats: dict) -> float:
        if self.coef is None:
            # Unfitted fallback: the strongest single sentence, which is monotone and never
            # collapses to zero the way a thresholded count does.
            return float(stats.get("max_p", 0.0))
        z = self._matrix([stats])[0]
        return float(1.0 / (1.0 + np.exp(-np.clip(z @ self.coef + self.intercept, -30, 30))))

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "features": list(DOC_FEATURES),
            "threshold": self.threshold,
            "coef": None if self.coef is None else [float(v) for v in self.coef],
            "intercept": self.intercept,
            "mean": None if self.mean is None else [float(v) for v in self.mean],
            "scale": None if self.scale is None else [float(v) for v in self.scale],
            "metadata": self.metadata,
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "DocumentDetector":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        arr = lambda v: None if v is None else np.asarray(v, dtype=np.float64)
        return cls(coef=arr(d["coef"]), intercept=float(d["intercept"]),
                   mean=arr(d["mean"]), scale=arr(d["scale"]),
                   threshold=float(d.get("threshold", 0.5)), metadata=d.get("metadata", {}))


def _bootstrap_share(
    probs: np.ndarray, words: np.ndarray, seed: int, threshold: float, n_boot: int = 400
) -> tuple[float, float]:
    """Percentile interval for the machine share, resampling sentences with replacement."""
    n = len(probs)
    if n < 3:
        return (0.0, 1.0)
    rng = np.random.default_rng(seed)
    shares = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        p, w = probs[idx], words[idx]
        total = w.sum()
        shares[b] = (w[p >= threshold].sum() / total) if total > 0 else 0.0
    return (float(np.percentile(shares, 5)), float(np.percentile(shares, 95)))
