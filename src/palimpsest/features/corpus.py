"""Features that compare a passage against a body of human essay writing.

These need the fitted :class:`~palimpsest.scorer.ngram.NgramReference`. When no reference is
available (a fresh checkout before ``palimpsest fit``) every value is NaN and the pipeline
carries on with the other families -- the detector degrades, it does not crash.
"""

from __future__ import annotations

import numpy as np

from ..scorer.ngram import NgramReference

__all__ = ["extract_corpus_features", "CORPUS_FEATURE_NAMES"]

CORPUS_FEATURE_NAMES: tuple[str, ...] = (
    "corpus_surprisal_mean",
    "corpus_surprisal_sd",
    "novel_trigram_rate",
    "fluency_typicality_gap",
)


def extract_corpus_features(
    text: str, reference: NgramReference | None, mean_logprob: float
) -> dict[str, float]:
    """Compute corpus-relative features for one span.

    ``mean_logprob`` is the local LM's per-token mean log-probability for the same span; it
    is needed for the gap feature and is passed in rather than recomputed.
    """
    if reference is None:
        return dict.fromkeys(CORPUS_FEATURE_NAMES, float("nan"))

    surprisals = np.asarray(reference.surprisals(text), dtype=np.float64)
    if len(surprisals) < 4:
        return dict.fromkeys(CORPUS_FEATURE_NAMES, float("nan"))

    corpus_mean = float(surprisals.mean())

    # The two observers disagree in an informative way. The local LM measures "is this
    # fluent English"; the reference measures "is this how applicants actually write".
    # Polished machine prose scores well on the first and poorly on the second: it is
    # smoother than general English yet less typical of the genre than a real essay. A
    # non-native writer usually shows the opposite sign, which is why this feature helps
    # separate the two cases rather than conflating them.
    gap = corpus_mean - (-mean_logprob) if np.isfinite(mean_logprob) else float("nan")

    return {
        "corpus_surprisal_mean": corpus_mean,
        "corpus_surprisal_sd": float(surprisals.std(ddof=1)) if len(surprisals) > 1 else np.nan,
        "novel_trigram_rate": reference.novel_trigram_rate(text),
        "fluency_typicality_gap": float(gap),
    }
