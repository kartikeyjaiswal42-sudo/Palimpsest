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
    ) -> None:
        self.detector = detector
        self.scorer = scorer or get_scorer()
        self.reference = reference

    # -- construction ------------------------------------------------------------

    @classmethod
    def from_artifacts(
        cls,
        model_dir: str | Path = "artifacts",
        observer: str = "gpt2",
        device: str = "cpu",
    ) -> "Analyzer":
        """Load the fitted detector and n-gram reference from disk."""
        model_dir = Path(model_dir)
        detector = SentenceDetector.load(model_dir / "detector.json")
        ref_path = model_dir / "ngram_reference.json"
        reference = NgramReference.load(ref_path) if ref_path.exists() else None
        return cls(detector, get_scorer(observer, device), reference)

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
                meta=self._meta(0.0, 0),
            )

        predictions = [self.detector.predict(f) for f in features]
        probs = np.array([p.probability for p in predictions], dtype=np.float64)
        word_counts = np.array([len(tokenize_words(s.text)) for s in spans], dtype=np.float64)
        smoothed = smooth_probabilities(probs, word_counts)

        reports: list[SentenceReport] = []
        verdicts: list[SentenceVerdict] = []
        for i, span in enumerate(spans):
            n_tokens = len(token_scores.select(span.start, span.end))
            reliable = n_tokens >= MIN_TOKENS
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
            verdict=aggregate(verdicts),
            sentences=reports,
            passages=find_passages(verdicts),
            tokens=_token_views(token_scores),
            meta=self._meta(elapsed, len(token_scores)),
        )

    def _meta(self, elapsed: float, n_tokens: int) -> dict:
        return {
            "observer": self.scorer.model_name,
            "device": self.scorer.device,
            "hasCorpusReference": self.reference is not None,
            "nObserverTokens": n_tokens,
            "elapsedMs": round(elapsed * 1000, 1),
            "trainedOn": self.detector.metadata,
        }


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
