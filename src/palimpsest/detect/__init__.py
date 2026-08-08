"""Verdict layer: the interpretable sentence classifier and document aggregation."""

from .classifier import Contribution, Prediction, SentenceDetector
from .document import (
    FLAG_THRESHOLD,
    DocumentVerdict,
    Passage,
    SentenceVerdict,
    aggregate,
    find_passages,
    smooth_probabilities,
)

__all__ = [
    "FLAG_THRESHOLD",
    "Contribution",
    "DocumentVerdict",
    "Passage",
    "Prediction",
    "SentenceDetector",
    "SentenceVerdict",
    "aggregate",
    "find_passages",
    "smooth_probabilities",
]
