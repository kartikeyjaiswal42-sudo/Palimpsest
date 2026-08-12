"""The analysis pipeline: text in, an explained verdict out.

    segment -> score with the observers -> extract features -> classify -> aggregate

This is the only object the API and the CLI talk to. It holds the loaded model and is
cheap to call repeatedly.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .detect.classifier import Prediction, SentenceDetector
from .detect.document import (
    MAX_SENTENCE_WORDS,
    DocumentDetector,
    DocumentVerdict,
    Passage,
    SentenceVerdict,
    aggregate,
    find_passages,
    smooth_probabilities,
)
from .features import (
    extract_context_features,
    extract_corpus_features,
    extract_model_features,
    extract_surface_features,
)
from .features.model_based import MIN_TOKENS
from .scorer.local_lm import LocalLMScorer, TokenScores, get_scorer
from .scorer.ngram import NgramReference
from .text.segment import Span, split_sentences, tokenize_words

__all__ = ["Analyzer", "AnalysisResult", "TokenView", "SentenceReport"]

#: GLTR-style rank buckets. The boundaries are the ones the original paper used.
_RANK_BUCKETS = ((10, "top10"), (100, "top100"), (1000, "top1000"))


@dataclass(frozen=True, slots=True)
class TokenView:
    """One token, for the token-level evidence strip in the interface."""

    text: str
    start: int
    end: int
    logprob: float
    rank: int
    bucket: str


@dataclass
class SentenceReport:
    index: int
    start: int
    end: int
    text: str
    probability: float
    smoothed: float
    n_words: int
    reliable: bool
    features: dict[str, float]
    prediction: Prediction
    #: Why this span could not be measured, or None when it could. The interface used to
    #: supply this itself, as the single fixed phrase "too short to measure reliably" -- which
    #: is exactly backwards for the case the reliability rule exists to catch. A 138-word
    #: run-on was shown to the reader as "97% machine-like -- too short to measure reliably".
    #: Only the pipeline knows which condition actually failed, so it says.
    unreliable_reason: str | None = None

    def to_dict(self, top_k: int = 6) -> dict:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "probability": round(self.probability, 4),
            "smoothed": round(self.smoothed, 4),
            "nWords": self.n_words,
            "reliable": self.reliable,
            "unreliableReason": self.unreliable_reason,
            "logit": round(self.prediction.logit, 4),
            "evidence": [
                {
                    "name": c.name,
                    "label": c.label,
                    "group": c.group,
                    "description": c.description,
                    "value": None if not c.measured else round(c.value, 4),
                    "z": round(c.z, 3),
                    "weight": round(c.weight, 4),
                    "contribution": round(c.contribution, 4),
                    "toward": c.toward,
                    "measured": c.measured,
                }
                for c in self.prediction.top(top_k)
            ],
            # The displayed bars are the k largest terms, which is NOT the whole logit. The
            # 37 smaller terms outweigh the six shown on 30.9% of training sentences, and
            # the intercept (-2.26, the prior that most sentences are human) is larger than
            # either. Publishing both lets the interface print a sum that actually closes,
            # so a reader can check the explanation against the verdict instead of taking
            # "the rest come from the remaining features" on trust.
            "intercept": round(self.prediction.intercept, 4),
            "evidenceRemainder": round(self.prediction.remainder(top_k), 4),
            "nFeaturesShown": min(top_k, len(self.prediction.contributions)),
            "nFeaturesTotal": len(self.prediction.contributions),
        }


@dataclass
class AnalysisResult:
    text: str
    verdict: DocumentVerdict
    sentences: list[SentenceReport]
    passages: list[Passage]
    tokens: list[TokenView]
    meta: dict = field(default_factory=dict)

    def to_dict(self, include_tokens: bool = True) -> dict:
        out = {
            "verdict": {
                "machineShare": round(self.verdict.machine_share, 4),
                "machineShareLow": round(self.verdict.machine_share_low, 4),
                "machineShareHigh": round(self.verdict.machine_share_high, 4),
                "anyMachineProbability": round(self.verdict.any_machine_probability, 4),
                "nSentences": self.verdict.n_sentences,
                "nWords": self.verdict.n_words,
                "nReliableSentences": self.verdict.n_reliable_sentences,
            },
            "sentences": [s.to_dict() for s in self.sentences],
            "passages": [
                {
                    "start": p.start,
                    "end": p.end,
                    "sentenceIndices": p.sentence_indices,
                    "nWords": p.n_words,
                    "meanProbability": round(p.mean_probability, 4),
                    "peakProbability": round(p.peak_probability, 4),
                }
                for p in self.passages
            ],
            "meta": self.meta,
        }
        if include_tokens:
            out["tokens"] = [asdict(t) for t in self.tokens]
        return out


class Analyzer:
    """Holds the observers and the fitted detector; turns text into an explained verdict."""

    def __init__(
        self,
        detector: SentenceDetector,
        scorer: LocalLMScorer | None = None,
        reference: NgramReference | None = None,
        document_model: DocumentDetector | None = None,
    ) -> None:
        self.detector = detector
        self.scorer = scorer or get_scorer()
        self.reference = reference
        # The FITTED document model. Passing it is not optional in practice: without it
        # `aggregate` silently falls back to `max_p`, the single strongest sentence, and the
        # result is not a miscalibrated document score but a different statistic entirely.
        # This library forgot to load it for a while, and the failure was invisible because
        # `max_p` looks like a probability, moves in the right direction, and is bounded in
        # [0, 1]. On one real essay it read 70.9% where the fitted model reads 14.9% -- the
        # bands turn that difference into two different verdicts. Every script that reports
        # a number (fit_bands, evaluate, report_product) loaded the model itself, so the
        # corpus results were never affected; only callers of `Analyzer` were.
        self.document_model = document_model

    # -- construction ------------------------------------------------------------

    @classmethod
    def from_artifacts(
        cls,
        model_dir: str | Path = "artifacts",
        observer: str = "gpt2",
        device: str = "cpu",
        suffix: str = "",
    ) -> "Analyzer":
        """Load the fitted detector and n-gram reference from disk.

        ``observer="remote"`` scores through Workers AI (see ``observer-worker/``) instead of
        loading GPT-2 locally. The detector must have been FITTED on the same observer --
        ``suffix="_remote"`` selects it -- because the two instruments disagree about what a
        given sentence's surprisal is, and a detector scored with the wrong one is not
        miscalibrated, it is meaningless. docs/09-frontier-ceiling.md measures how far apart
        they are: on the same essays, one statistic's AUROC moves from 0.132 to 0.895.
        """
        model_dir = Path(model_dir)
        detector = SentenceDetector.load(model_dir / f"detector{suffix}.json")
        ref_path = model_dir / "ngram_reference.json"
        reference = NgramReference.load(ref_path) if ref_path.exists() else None
        doc_path = model_dir / f"document_detector{suffix}.json"
        document_model = DocumentDetector.load(doc_path) if doc_path.exists() else None
        if observer == "remote":
            from .scorer.remote_lm import RemoteObserverScorer

            scorer = RemoteObserverScorer()
        else:
            scorer = get_scorer(observer, device)
        return cls(detector, scorer, reference, document_model)

    # -- the pipeline ------------------------------------------------------------

    def features_for(self, text: str) -> tuple[list[Span], list[dict[str, float]], TokenScores]:
        """Segment ``text`` and build the full feature matrix. Shared by training and serving.

        Training must extract features exactly the way serving does, or every reported
        number is measuring a different system than the one that ships. Both paths call
        this method.
        """
        sentences = split_sentences(text)
        if not sentences:
            return [], [], self.scorer.score("")

        token_scores = self.scorer.score(text)

        base: list[dict[str, float]] = []
        for span in sentences:
            span_tokens = token_scores.select(span.start, span.end)
            model_feats = extract_model_features(span_tokens)
            surface_feats = extract_surface_features(span.text)
            corpus_feats = extract_corpus_features(
                span.text, self.reference, model_feats["mean_logprob"]
            )
            base.append({**model_feats, **surface_feats, **corpus_feats})

        context = extract_context_features(base, len(sentences))
        combined = [{**b, **c} for b, c in zip(base, context, strict=True)]
        return sentences, combined, token_scores

    def analyze(self, text: str) -> AnalysisResult:
        """Full analysis of one essay."""
        started = time.perf_counter()
        spans, features, token_scores = self.features_for(text)

        if not spans:
            return AnalysisResult(
                text=text,
                verdict=aggregate([]),
                sentences=[],
                passages=[],
                tokens=[],
                meta=self._meta(0.0, token_scores),
            )

        predictions = [self.detector.predict(f) for f in features]
        probs = np.array([p.probability for p in predictions], dtype=np.float64)
        word_counts = np.array([len(tokenize_words(s.text)) for s in spans], dtype=np.float64)

        token_counts = [len(token_scores.select(s.start, s.end)) for s in spans]
        # bool(), not the bare comparison: word_counts is a numpy array, so the comparison
        # yields numpy.bool_, which pydantic refuses to serialise and the API answers 500 on
        # every request. Nothing in the type annotation catches it.
        reliable_flags = [
            bool(token_counts[i] >= MIN_TOKENS and word_counts[i] <= MAX_SENTENCE_WORDS)
            for i in range(len(spans))
        ]
        # Smoothing is told which spans are measurable, because it is weighted by word count
        # and the unmeasurable spans are the long ones. See smooth_probabilities.
        smoothed = smooth_probabilities(probs, word_counts, np.array(reliable_flags, dtype=bool))

        reports: list[SentenceReport] = []
        verdicts: list[SentenceVerdict] = []
        for i, span in enumerate(spans):
            n_tokens = token_counts[i]
            reliable = reliable_flags[i]
            reports.append(
                SentenceReport(
                    index=i,
                    start=span.start,
                    end=span.end,
                    text=span.text,
                    probability=float(probs[i]),
                    smoothed=float(smoothed[i]),
                    n_words=int(word_counts[i]),
                    reliable=reliable,
                    features=features[i],
                    prediction=predictions[i],
                    unreliable_reason=_why_unmeasurable(
                        n_tokens, int(word_counts[i]), span.start, token_scores
                    ),
                )
            )
            verdicts.append(
                SentenceVerdict(
                    index=i,
                    start=span.start,
                    end=span.end,
                    text=span.text,
                    probability=float(probs[i]),
                    n_words=int(word_counts[i]),
                    smoothed=float(smoothed[i]),
                    reliable=reliable,
                )
            )

        elapsed = time.perf_counter() - started
        return AnalysisResult(
            text=text,
            # The threshold must be the one the document model was FITTED at
            # (`document_detector*.json` records it as `sentenceThreshold`), not the 0.65
            # module default that `find_passages` uses for highlighting. `share` is a
            # standardised input to that model: computing it at a different cut-off feeds
            # the fit a column it has never seen.
            verdict=aggregate(
                verdicts,
                threshold=self.detector.flag_threshold,
                doc_model=self.document_model,
            ),
            sentences=reports,
            passages=find_passages(verdicts),
            tokens=_token_views(token_scores),
            meta=self._meta(elapsed, token_scores),
        )

    def _meta(self, elapsed: float, scores: TokenScores | None) -> dict:
        return {
            "observer": self.scorer.model_name,
            "device": self.scorer.device,
            "hasCorpusReference": self.reference is not None,
            "nObserverTokens": 0 if scores is None else len(scores),
            "elapsedMs": round(elapsed * 1000, 1),
            # Whether the observer saw the whole document, and how much of it it can see at
            # once. Without these the interface cannot tell a verdict on an essay from a
            # verdict on its opening. The hosted build has published `clipped` since it
            # shipped; this build had the flag and threw it away.
            "clipped": bool(getattr(scores, "clipped", False)),
            "observerCharLimit": getattr(self.scorer, "max_length", None),
            "trainedOn": self.detector.metadata,
        }


def _why_unmeasurable(
    n_tokens: int, n_words: int, start: int, scores: TokenScores
) -> str | None:
    """Which reliability condition failed, or None if none did.

    Three different conditions were being reported to the reader as one sentence -- "too
    short to measure reliably" -- and for two of them that phrase is false. A 138-word
    run-on is not short; it is the opposite, and it is the exact shape of writing (ELLIPSE
    and PERSUADE essays with no sentence-ending punctuation) that the reliability rule was
    added to protect. Telling that student their sentence was too short is not a wording nit:
    it misdirects the one person most likely to want to contest the result.

    Order matters. "Never looked at" outranks the others, because a span outside the
    observer's window has no measurement to be out of distribution in the first place.
    """
    if scores.clipped and n_tokens == 0 and start >= _observed_chars(scores):
        return "beyond_observer_window"
    if n_words > MAX_SENTENCE_WORDS:
        return "too_long"
    if n_tokens < MIN_TOKENS:
        return "too_short"
    return None


def _observed_chars(scores: TokenScores) -> int:
    """The last character the observer actually read. 0 when it read nothing."""
    return int(scores.char_end[-1]) if len(scores) else 0


def _token_views(scores: TokenScores) -> list[TokenView]:
    views: list[TokenView] = []
    for i in range(len(scores)):
        rank = int(scores.rank[i])
        bucket = "tail"
        for limit, name in _RANK_BUCKETS:
            if rank <= limit:
                bucket = name
                break
        views.append(
            TokenView(
                text=scores.tokens[i],
                start=int(scores.char_start[i]),
                end=int(scores.char_end[i]),
                logprob=round(float(scores.logprob[i]), 3),
                rank=rank,
                bucket=bucket,
            )
        )
    return views
