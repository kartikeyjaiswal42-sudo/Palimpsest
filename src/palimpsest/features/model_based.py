"""Features computed from the local LM's per-token statistics.

Everything here is arithmetic on the arrays produced by :mod:`palimpsest.scorer.local_lm`.
No feature in this file asks a model a question; they only measure the numbers it produced.

Length invariance
-----------------
These features are compared across spans of very different sizes -- an eleven-word sentence
against a forty-word one. Two of the published statistics we use are defined as sums and
therefore grow with length, which would let the classifier learn "long sentence" instead of
"machine sentence". Both are converted to per-token forms here, and the deviation from the
original papers is deliberate and noted at each site.
"""

from __future__ import annotations

import numpy as np

from ..scorer.local_lm import TokenScores

__all__ = ["extract_model_features", "MODEL_FEATURE_NAMES", "MIN_TOKENS"]

# Below this many scored tokens the estimates are too noisy to mean anything. We return NaN
# rather than a confident-looking number computed from four tokens. The pipeline treats NaN
# as "not measured" and the UI says so.
MIN_TOKENS = 5

MODEL_FEATURE_NAMES: tuple[str, ...] = (
    "mean_logprob",
    "logprob_sd",
    "logprob_iqr",
    "frac_rank_top1",
    "frac_rank_top10",
    "frac_rank_top100",
    "frac_rank_tail",
    "mean_log_rank",
    "log_rank_sd",
    "mean_entropy",
    "entropy_sd",
    "lrr",
    "curvature",
    "surprisal_autocorr",
)


def extract_model_features(scores: TokenScores) -> dict[str, float]:
    """Reduce one span's token statistics to interpretable features."""
    n = len(scores)
    if n < MIN_TOKENS:
        return dict.fromkeys(MODEL_FEATURE_NAMES, float("nan"))

    lp = scores.logprob.astype(np.float64)
    rank = scores.rank.astype(np.float64)
    ent = scores.entropy.astype(np.float64)
    log_rank = np.log(np.maximum(rank, 1.0))

    q75, q25 = np.percentile(lp, [75, 25])

    # DetectLLM's Log-Likelihood Log-Rank Ratio. Both terms are means, so the ratio is
    # already length-invariant. Machine text is both more likely and more highly ranked,
    # but rank collapses faster than likelihood, which pushes the ratio up.
    #
    # The denominator is a mean of non-negative terms and is zero in exactly one case: the
    # observer ranked EVERY token in the span first. The ratio is then undefined, not
    # enormous, and the old guard -- dividing by 1e-6 -- claimed it was enormous. With
    # GPT-2 that case essentially never arose. With the 30 B observer 48% of all tokens are
    # rank 1, so short spans reach it routinely: 11 of 6,941 training sentences (0.16%),
    # scoring up to 7.4e5.
    #
    # Those 11 rows did not merely score wrongly. They set the column's standard deviation
    # to 11,625 (its true value is 0.356), so every OTHER sentence standardised to
    # z ~= -0.03 and lrr -- a feature carrying one of the largest fitted weights -- stopped
    # voting at all. The failure is silent and it is worst on exactly the text this project
    # struggles with: fluent prose produces rank-1 tokens, so a stronger generator makes the
    # degenerate case likelier and the feature deader.
    #
    # Undefined is reported as not-measured, which the classifier already handles by
    # imputing the training mean -- standardising to zero, contributing zero. See the module
    # docstring in detect/classifier.py: an unmeasured feature must not silently vote.
    mean_nll = float(-lp.mean())
    mean_log_rank = float(log_rank.mean())
    lrr = mean_nll / mean_log_rank if mean_log_rank > 0.0 else float("nan")

    # Fast-DetectGPT's conditional probability curvature: how far the observed tokens sit
    # above what the model itself expects to see, in units of its own standard deviation.
    #
    # The paper sums over tokens and divides by sqrt(sum of variances), which grows like
    # sqrt(n) for a consistently biased passage. We use the per-token form so a short
    # sentence and a long one land on the same scale. Same quantity, length-normalised.
    mu = scores.mu.astype(np.float64)
    var = scores.sigma2.astype(np.float64)
    denom = np.sqrt(max(float(var.mean()), 1e-9))
    curvature = float((lp.mean() - mu.mean()) / denom)

    return {
        # --- how predictable was the passage
        "mean_logprob": float(lp.mean()),
        "logprob_sd": float(lp.std(ddof=1)) if n > 1 else float("nan"),
        "logprob_iqr": float(q75 - q25),
        # --- where the observed tokens sat in the ranked distribution (GLTR buckets)
        "frac_rank_top1": float((rank <= 1).mean()),
        "frac_rank_top10": float((rank <= 10).mean()),
        "frac_rank_top100": float((rank <= 100).mean()),
        "frac_rank_tail": float((rank > 1000).mean()),
        "mean_log_rank": mean_log_rank,
        "log_rank_sd": float(log_rank.std(ddof=1)) if n > 1 else float("nan"),
        # --- how uncertain the model was, which contextualises the two above
        "mean_entropy": float(ent.mean()),
        "entropy_sd": float(ent.std(ddof=1)) if n > 1 else float("nan"),
        # --- composite statistics from the detection literature
        "lrr": lrr,
        "curvature": curvature,
        # --- rhythm in the surprisal signal itself: human writing alternates between
        #     predictable connective tissue and genuinely surprising content words, which
        #     shows up as negative lag-1 autocorrelation. Machine prose is flatter.
        "surprisal_autocorr": _autocorr(lp),
    }


def _autocorr(x: np.ndarray, lag: int = 1) -> float:
    """Lag-``lag`` Pearson autocorrelation of a 1-D signal."""
    if len(x) <= lag + 2:
        return float("nan")
    a, b = x[:-lag], x[lag:]
    a_c, b_c = a - a.mean(), b - b.mean()
    denom = np.sqrt((a_c**2).sum() * (b_c**2).sum())
    if denom < 1e-12:
        return float("nan")
    return float((a_c * b_c).sum() / denom)
