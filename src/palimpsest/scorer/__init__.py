"""Observers: the local language model and the human-corpus n-gram reference."""

from .local_lm import LocalLMScorer, TokenScores, get_scorer, select_device
from .ngram import NgramReference

__all__ = ["LocalLMScorer", "TokenScores", "NgramReference", "get_scorer", "select_device"]
